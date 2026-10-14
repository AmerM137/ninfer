#include "runtime/engine/model_instance.h"

#include "artifact/formats.h"
#include "artifact/reader.h"
#include "core/startup.h"
#include "models/qwen3_5/execution/profile.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/measurement.h"
#include "models/qwen3_5/program/planning/moe.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

namespace ninfer::runtime {
namespace {
using Clock = std::chrono::steady_clock;

void validate_options(const EngineOptions& options) {
    if (options.artifact_path.empty()) {
        throw std::invalid_argument("Engine artifact_path must not be empty");
    }
    if (options.artifact_path.extension() != ".ninfer") {
        throw std::invalid_argument("NInfer accepts only .ninfer artifacts");
    }
    if (options.max_context == 0) {
        throw std::invalid_argument("Engine max_context must be nonzero");
    }
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit:
        if (options.kv_capacity.explicit_tokens == 0) {
            throw std::invalid_argument("Engine explicit kv_capacity must be nonzero");
        }
        if (options.kv_capacity.automatic_headroom_bytes != 0) {
            throw std::invalid_argument(
                "Engine explicit kv_capacity must not carry automatic headroom");
        }
        break;
    case KvCapacityMode::Automatic:
        if (options.kv_capacity.explicit_tokens != 0) {
            throw std::invalid_argument(
                "Engine automatic kv_capacity must not carry explicit tokens");
        }
        break;
    default:
        throw std::invalid_argument("Engine kv_capacity mode is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }
    if (options.max_pending_requests == 0 || options.pending_timeout_ms == 0) {
        throw std::invalid_argument("Engine pending request capacity and timeout must be nonzero");
    }
    if (options.enable_vision && options.media_live_bytes == 0) {
        throw std::invalid_argument(
            "Engine media_live_bytes must be nonzero when Vision is enabled");
    }
    if (options.media_preprocess_threads > 64) {
        throw std::invalid_argument("Engine media_preprocess_threads must be in [0,64]");
    }
}

std::size_t current_free_device_bytes() {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return free_bytes;
}

std::size_t device_budget() {
    const auto available = current_free_device_bytes();
    const char* value    = std::getenv("NINFER_VRAM_BUDGET_MIB");
    if (!value) return available;
    const std::string_view text(value);
    std::uint64_t mib = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), mib);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || !mib ||
        mib > std::numeric_limits<std::size_t>::max() / (1024ULL * 1024)) {
        throw std::invalid_argument("NINFER_VRAM_BUDGET_MIB requires a positive MiB integer");
    }
    return std::min(available, static_cast<std::size_t>(mib * 1024 * 1024));
}

struct StartupPlan {
    models::qwen3_5::SequencePlan sequence;
    KvCapacityResolution capacity;
    bool host_experts;
    std::uint32_t chunk;
    std::shared_ptr<const models::moe::Plan> moe;
};

StartupPlan plan_startup(const models::qwen3_5::LoadPlan& source,
                         std::shared_ptr<const models::qwen3_5::execution::ModelProfile> profile,
                         DeviceContext& device, const EngineOptions& options, std::size_t budget) {
    namespace model     = models::qwen3_5;
    const auto resident = source.placement(device.device(), device.device());
    // Driver/module allocations are outside the explicit tensor layouts.
    const auto headroom =
        std::max<std::size_t>(256ULL * 1024 * 1024, options.kv_capacity.automatic_headroom_bytes);
    if (budget <= headroom)
        throw std::invalid_argument("VRAM budget is smaller than runtime headroom");
    auto policy                     = options.kv_capacity;
    policy.automatic_headroom_bytes = 0;
    const auto resident_chunk       = options.prefill_chunk ? options.prefill_chunk : 1024U;
    auto trial                      = options;
    trial.prefill_chunk             = resident_chunk;
    auto planner                    = model::make_sequence_planner(profile, device, trial);
    const auto& curve               = planner.capacity_curve();
    const auto required_pages =
        policy.mode == KvCapacityMode::Explicit
            ? static_cast<std::uint32_t>(
                  (std::uint64_t(policy.explicit_tokens) + curve.main_page_tokens - 1) /
                  curve.main_page_tokens)
            : curve.minimum_main_page_groups;
    const auto minimum = curve.reservation_bytes(required_pages);
    auto assignment    = model::detail::describe_moe_plan(source, *profile, options.moe);
    if (resident.device_capacity_bytes <= budget - headroom &&
        minimum <= budget - headroom - resident.device_capacity_bytes) {
        auto resolution =
            resolve_kv_capacity(policy, curve, budget - headroom - resident.device_capacity_bytes);
        resolution.automatic_headroom_bytes = headroom;
        auto sequence = std::move(planner).finalize(resolution.main_page_groups);
        return {std::move(sequence), resolution, false, resident_chunk, {}};
    }
    if (source.config().text.architecture != models::Architecture::Qwen3_5Moe) {
        throw std::invalid_argument(
            "VRAM budget cannot hold this model and its execution resources");
    }
    const auto host = source.placement(device.device(), Device::cpu());
    if (host.device_capacity_bytes >= budget - headroom) {
        throw std::invalid_argument("VRAM budget cannot hold fixed model weights");
    }
    const auto runtime_budget = budget - headroom - host.device_capacity_bytes;
    const auto chunk          = options.prefill_chunk ? options.prefill_chunk : 4096U;
    trial.prefill_chunk       = chunk;
    auto prototype            = std::make_shared<models::moe::Plan>(std::move(assignment));
    auto initial              = model::make_sequence_planner(profile, device, trial, prototype);
    auto limited              = policy;
    if (limited.mode == KvCapacityMode::Automatic) {
        limited = KvCapacityPolicy::explicit_capacity(initial.capacity_curve().resolved_tokens(
            initial.capacity_curve().minimum_main_page_groups));
    }
    const auto& initial_curve = initial.capacity_curve();
    const auto pages          = static_cast<std::uint32_t>(
        (std::uint64_t(limited.explicit_tokens) + initial_curve.main_page_tokens - 1) /
        initial_curve.main_page_tokens);
    if (initial_curve.reservation_bytes(pages) >= runtime_budget) {
        throw std::invalid_argument("VRAM budget cannot hold MoE execution resources");
    }
    auto base             = resolve_kv_capacity(limited, initial_curve, runtime_budget);
    const auto max_tokens = static_cast<std::int32_t>(
        std::max(std::min(chunk, options.max_context),
                 options.max_concurrency * (options.speculative.draft_tokens + 1)));
    CpuTopology topology;
    auto placement = model::detail::resolve_moe_plan(
        source, *profile, runtime_budget - base.runtime_reservation_bytes, options.moe, max_tokens,
        topology);
    if (!placement) {
        throw std::invalid_argument("VRAM budget cannot hold MoE execution resources");
    }
    auto moe        = std::make_shared<models::moe::Plan>(std::move(*placement));
    auto final      = model::make_sequence_planner(profile, device, trial, moe);
    auto resolution = resolve_kv_capacity(limited, final.capacity_curve(), runtime_budget);
    resolution.mode = policy.mode;
    resolution.automatic_headroom_bytes = headroom;
    return {std::move(final).finalize(resolution.main_page_groups), resolution, true, chunk,
            std::move(moe)};
}

} // namespace

EngineOptions normalize_engine_options(EngineOptions options) {
    switch (options.purpose) {
    case EnginePurpose::Generation:
        break;
    case EnginePurpose::CausalScoring:
        options.max_concurrency      = 1;
        options.max_pending_requests = 1;
        options.kv_capacity          = KvCapacityPolicy::explicit_capacity(options.max_context);
        options.speculative          = {};
        options.enable_vision        = false;
        options.use_cuda_graph       = false;
        options.context_cache        = ContextCacheOptions{
                   .enabled = false, .device_state_slots = 0, .host_capacity_bytes = 0};
        break;
    default:
        throw std::invalid_argument("Engine purpose is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }

    ContextCacheOptions& cache      = options.context_cache;
    const std::uint32_t concurrency = options.max_concurrency;
    cache.device_state_slots        = cache.device_state_slots.value_or(concurrency);
    const std::uint64_t total_device_state_slots =
        static_cast<std::uint64_t>(concurrency) + *cache.device_state_slots;
    if (total_device_state_slots > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("context cache Device state capacity exceeds uint32");
    }
    return options;
}

ModelInstance::ModelInstance(
    std::unique_ptr<models::qwen3_5::Model> source, const EngineOptions& options,
    std::shared_ptr<const models::qwen3_5::execution::ModelProfile> profile)
    : model(std::move(source)), parameters(*model, std::move(profile)),
      frontend(models::qwen3_5::make_frontend(
          model->resources(), {.chat_template_path       = options.chat_template_path,
                               .grammar_cache_bytes      = options.grammar_cache_bytes,
                               .architecture             = model->config().text.architecture,
                               .vision_enabled           = options.enable_vision,
                               .max_context              = options.max_context,
                               .media_cache_bytes        = options.media_cache_bytes,
                               .media_live_bytes         = options.media_live_bytes,
                               .media_preprocess_threads = options.media_preprocess_threads})),
      capacity(options.max_context) {}

ModelInstance::~ModelInstance() = default;

ConstructedModel construct_model(EngineOptions& options, DeviceContext& device) {
    validate_options(options);
    const auto start  = Clock::now();
    const auto budget = device_budget();
    StartupPhaseScope inspect(options.startup_observer, StartupPhase::ArtifactInspect);
    artifact::Reader reader(options.artifact_path);
    inspect.complete();
    StartupPhaseScope binding(options.startup_observer, StartupPhase::TargetPlan);
    auto plan             = models::qwen3_5::plan_load(reader, models::load_options(options));
    auto profile          = models::qwen3_5::execution::describe_model(plan);
    auto startup          = plan_startup(plan, profile, device, options, budget);
    options.prefill_chunk = startup.chunk;
    binding.complete();
    auto model = models::qwen3_5::materialize_model(
        models::qwen3_5::resolve_load(std::move(plan), device.device(),
                                      startup.host_experts ? Device::cpu() : device.device()),
        device, &options.startup_observer);
    device.synchronize();
    StartupPhaseScope frontend(options.startup_observer, StartupPhase::FrontendInitialize);
    auto instance = std::make_unique<ModelInstance>(std::move(model), options, std::move(profile));
    frontend.complete();
    StartupPhaseScope planning(options.startup_observer, StartupPhase::TargetFinalize);
    const auto signature =
        models::qwen3_5::prefill_signature(*instance->model, startup.moe.get(), startup.chunk);
    auto context_cost = resolve_context_machine_cost(
        {.hardware_class = context_cost_hardware_class(
             device.properties().name, device.properties().major, device.properties().minor),
         .prefill_signature = signature},
        options.context_cost.preset_path);
    auto resolution                          = startup.capacity;
    auto sequence                            = std::move(startup.sequence);
    resolution.available_after_weights_bytes = current_free_device_bytes();
    if (sequence.device_reservation_bytes() != resolution.runtime_reservation_bytes ||
        sequence.kv_capacity() != resolution.resolved_tokens) {
        throw std::logic_error("resolved KV capacity does not match the finalized Program plan");
    }
    options.context_cache.host_capacity_bytes = sequence.host_capacity_bytes();
    instance->kv_capacity_resolution          = resolution;
    instance->device_budget_bytes             = budget;
    planning.complete();
    StartupPhaseScope program(options.startup_observer, StartupPhase::ProgramInitialize);
    instance->program = models::qwen3_5::create_program(instance->parameters, std::move(sequence),
                                                        device, options.startup_observer);
    device.synchronize();
    program.complete();
    instance->kv_capacity_resolution.available_after_startup_bytes = current_free_device_bytes();
    const auto& stats = instance->model->storage_stats();
    LoadSummary summary;
    summary.architecture = models::architecture_name(instance->model->config().text.architecture);
    summary.model_name   = instance->model->info().name;
    summary.prefill_signature = signature;
    std::set<std::string> formats;
    for (const auto& weight : instance->model->weight_data()) {
        for (const auto& part : weight.view.parts) {
            formats.emplace(artifact::format_name(part.parent->geometry.format));
        }
    }
    summary.weight_formats.assign(formats.begin(), formats.end());
    summary.load_seconds         = std::chrono::duration<double>(Clock::now() - start).count();
    summary.upload_seconds       = stats.upload_seconds;
    summary.artifact_bytes_read  = stats.read_bytes;
    summary.host_to_device_bytes = stats.h2d_bytes;
    summary.peak_staging_bytes   = stats.peak_staging_bytes;
    summary.device_object_count  = stats.device_object_count;
    summary.host_object_count    = stats.host_object_count;
    summary.context_cost         = std::move(context_cost.summary);
    return {std::move(instance), std::move(summary), std::move(context_cost.model)};
}

} // namespace ninfer::runtime
