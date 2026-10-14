#include "runtime/engine/model_instance.h"
#include "runtime/engine/startup_memory.h"
#include "models/qwen3_5/program/graph_preparation.h"

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

std::size_t device_budget(std::size_t available) {
    const char* value = std::getenv("NINFER_VRAM_BUDGET_MIB");
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

struct WeightPlacement {
    std::size_t resident_bytes = 0, fixed_bytes = 0;
    std::optional<models::moe::Plan> offload;
    std::vector<ExpertBankCacheSpec> banks;
};

WeightPlacement describe_placement(const models::qwen3_5::LoadPlan& source,
                                   const models::qwen3_5::execution::ModelProfile& profile,
                                   Device device, const EngineOptions& options) {
    WeightPlacement out;
    out.resident_bytes = source.placement(device, device).device_capacity_bytes;
    auto offload       = models::qwen3_5::detail::describe_moe_plan(source, profile, options.moe);
    if (source.config().text.architecture != models::Architecture::Qwen3_5Moe) return out;
    out.fixed_bytes = source.placement(device, Device::cpu()).device_capacity_bytes;
    out.offload     = std::move(offload);
    for (const auto& bank : source.expert_banks())
        out.banks.push_back({bank.gate_up.layout, bank.down.layout, true});
    for (const auto& layer : out.offload->layers) out.banks[layer.spec.bank.index].resident = false;
    return out;
}

std::optional<StartupPlan>
plan_execution(const WeightPlacement& weights,
               std::shared_ptr<const models::qwen3_5::execution::ModelProfile> profile,
               DeviceContext& device, const EngineOptions& options, bool host_experts,
               std::size_t available) {
    namespace model  = models::qwen3_5;
    auto trial       = options;
    const auto chunk = options.prefill_chunk ? options.prefill_chunk : host_experts ? 4096U : 1024U;
    trial.prefill_chunk             = chunk;
    auto policy                     = options.kv_capacity;
    policy.automatic_headroom_bytes = 0;
    auto prototype = host_experts ? std::make_shared<models::moe::Plan>(*weights.offload) : nullptr;
    auto planner   = model::make_sequence_planner(profile, device, trial, prototype);
    const auto curve = planner.capacity_curve();
    const auto pages = policy.mode == KvCapacityMode::Explicit
                           ? static_cast<std::uint32_t>((std::uint64_t(policy.explicit_tokens) +
                                                         curve.main_page_tokens - 1) /
                                                        curve.main_page_tokens)
                           : curve.minimum_main_page_groups;
    if (curve.reservation_bytes(pages) > available) return std::nullopt;
    if (!host_experts) {
        auto resolution = resolve_kv_capacity(policy, curve, available);
        return StartupPlan{
            std::move(planner).finalize(resolution.main_page_groups), resolution, false, chunk, {}};
    }
    auto limited          = KvCapacityPolicy::explicit_capacity(curve.resolved_tokens(pages));
    auto base             = resolve_kv_capacity(limited, curve, available);
    const auto max_tokens = static_cast<std::int32_t>(
        std::max(std::min(chunk, options.max_context),
                 options.max_concurrency * (options.speculative.draft_tokens + 1)));
    CpuTopology topology;
    auto placement = models::moe::resolve_plan(*weights.offload, weights.banks,
                                               available - base.runtime_reservation_bytes,
                                               max_tokens, topology, options.moe.cpu_workers);
    if (!placement) return std::nullopt;
    auto moe   = std::make_shared<models::moe::Plan>(std::move(*placement));
    auto final = model::make_sequence_planner(profile, device, trial, moe);
    if (final.capacity_curve().reservation_bytes(pages) > available) return std::nullopt;
    auto resolution = resolve_kv_capacity(limited, final.capacity_curve(), available);
    resolution.mode = policy.mode;
    return StartupPlan{std::move(final).finalize(resolution.main_page_groups), resolution, true,
                       chunk, std::move(moe)};
}

struct CalibratedPlan {
    StartupPlan startup;
    GraphMemorySummary graph;
};

std::optional<CalibratedPlan>
calibrate_plan(const WeightPlacement& weights, ModelInstance& instance, DeviceContext& device,
               const EngineOptions& options, const StartupMemoryBudget& budget,
               StartupPlan candidate, std::vector<models::qwen3_5::GraphCalibration>& measured,
               std::size_t minimum_rebuild = 0) {
    namespace model = models::qwen3_5;
    for (;;) {
        auto probe_options          = options;
        probe_options.prefill_chunk = candidate.chunk;
        const auto key = model::describe_graph_preparation(probe_options, candidate.moe.get());
        auto found     = std::find_if(measured.begin(), measured.end(),
                                      [&](const auto& item) { return item.plan == key; });
        if (found == measured.end()) {
            measured.push_back(
                model::calibrate_graphs(instance.parameters, device, probe_options, candidate.moe));
            found = std::prev(measured.end());
        }
        const auto graph_bytes = std::max(found->rebuild_bytes, minimum_rebuild);
        const GraphMemorySummary graph{
            .rebuild_estimate_bytes      = graph_bytes,
            .calibration_increment_bytes = found->preparation_increment_bytes,
            .calibration_retained_bytes  = found->retained_increment_bytes,
        };
        const auto available = budget.available(device_memory_snapshot(device.device()));
        if (graph_bytes > available) return std::nullopt;
        auto next = plan_execution(weights, instance.parameters.profile, device, options,
                                   candidate.host_experts, available - graph_bytes);
        if (!next) return std::nullopt;
        const auto next_key = model::describe_graph_preparation(probe_options, next->moe.get());
        if (next_key == key) return CalibratedPlan{std::move(*next), graph};
        const bool revisited = std::any_of(measured.begin(), measured.end(),
                                           [&](const auto& item) { return item.plan == next_key; });
        if (revisited && candidate.sequence.device_reservation_bytes() <= available - graph_bytes) {
            // At a variant boundary choose the feasible measured side, then fill that interval.
            auto low  = candidate.sequence.device_reservation_bytes();
            auto high = available - graph_bytes;
            while (low < high) {
                const auto middle = low + (high - low + 1) / 2;
                auto trial = plan_execution(weights, instance.parameters.profile, device, options,
                                            candidate.host_experts, middle);
                if (trial &&
                    model::describe_graph_preparation(probe_options, trial->moe.get()) == key) {
                    low       = middle;
                    candidate = std::move(*trial);
                } else
                    high = middle - 1;
            }
            return CalibratedPlan{std::move(candidate), graph};
        }
        candidate = std::move(*next);
    }
}

std::optional<CalibratedPlan>
prepare_program(const WeightPlacement& weights, ModelInstance& instance, DeviceContext& device,
                EngineOptions& options, const StartupMemoryBudget& budget, CalibratedPlan plan,
                std::vector<models::qwen3_5::GraphCalibration>& measured, unsigned& attempts) {
    namespace model = models::qwen3_5;
    while (attempts < 2) {
        ++attempts;
        auto& startup             = plan.startup;
        const auto explicit_bytes = startup.sequence.device_reservation_bytes();
        const auto host_bytes     = startup.sequence.host_capacity_bytes();
        const auto retained =
            budget.retained(device_memory_snapshot(device.device()),
                            instance.model->storage_stats().device_capacity_bytes);
        if (explicit_bytes != startup.capacity.runtime_reservation_bytes ||
            startup.sequence.kv_capacity() != startup.capacity.resolved_tokens)
            throw std::logic_error("resolved capacity differs from the finalized Program plan");
        std::optional<DeviceMemorySnapshot> ready;
        try {
            StartupPhaseScope phase(options.startup_observer, StartupPhase::ProgramInitialize);
            instance.program = model::create_program(
                instance.parameters, std::move(startup.sequence), device, options.startup_observer);
            device.synchronize();
            ready = device_memory_snapshot(device.device());
            if (budget.fits(*ready)) {
                options.prefill_chunk                                    = startup.chunk;
                options.context_cache.host_capacity_bytes                = host_bytes;
                instance.kv_capacity_resolution                          = startup.capacity;
                instance.kv_capacity_resolution.automatic_headroom_bytes = budget.headroom();
                instance.device_budget_bytes                             = budget.limit();
                instance.startup_retained_bytes                          = retained;
                instance.cuda_graph = instance.program->memory_summary().cuda_graph;
                if (instance.cuda_graph) {
                    plan.graph.preparation_increment_bytes =
                        instance.cuda_graph->preparation_increment_bytes;
                    instance.cuda_graph = plan.graph;
                }
                phase.complete();
                return plan;
            }
        } catch (const DeviceAllocationError&) {
            // A failed preparation invalidates its observation; the next probe measures it again.
            auto probe_options          = options;
            probe_options.prefill_chunk = startup.chunk;
            const auto key = model::describe_graph_preparation(probe_options, startup.moe.get());
            std::erase_if(measured, [&](const auto& entry) { return entry.plan == key; });
        }
        instance.program.reset();
        device.synchronize();
        if (attempts == 2) return std::nullopt;
        const auto after_cleanup = device_memory_snapshot(device.device());
        const auto available     = budget.available(after_cleanup);
        const auto recovered     = ready ? device_memory_growth(after_cleanup, *ready) -
                                           static_cast<std::int64_t>(explicit_bytes)
                                         : 0;
        const auto reserve =
            std::max(plan.graph.rebuild_estimate_bytes,
                     static_cast<std::size_t>(std::max<std::int64_t>(0, recovered)));
        if (reserve > available) return std::nullopt;
        auto candidate = plan_execution(weights, instance.parameters.profile, device, options,
                                        startup.host_experts, available - reserve);
        if (!candidate) return std::nullopt;
        auto corrected = calibrate_plan(weights, instance, device, options, budget,
                                        std::move(*candidate), measured, reserve);
        if (!corrected) return std::nullopt;
        plan = std::move(*corrected);
    }
    return std::nullopt;
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
    const auto start   = Clock::now();
    const auto initial = device_memory_snapshot(device.device());
    const auto limit   = device_budget(initial.free_bytes);
    const auto headroom =
        std::max<std::size_t>(256ULL * 1024 * 1024, options.kv_capacity.automatic_headroom_bytes);
    StartupMemoryBudget budget(initial, limit, headroom);
    StartupPhaseScope inspect(options.startup_observer, StartupPhase::ArtifactInspect);
    artifact::Reader reader(options.artifact_path);
    inspect.complete();
    StartupPhaseScope binding(options.startup_observer, StartupPhase::TargetPlan);
    auto source        = models::qwen3_5::plan_load(reader, models::load_options(options));
    auto profile       = models::qwen3_5::execution::describe_model(source);
    const auto weights = describe_placement(source, *profile, device.device(), options);
    const auto initial_available = budget.available(initial);
    auto candidate               = weights.resident_bytes <= initial_available
                                       ? plan_execution(weights, profile, device, options, false,
                                                        initial_available - weights.resident_bytes)
                                       : std::nullopt;
    bool host_experts            = !candidate.has_value();
    if (host_experts && weights.offload && weights.fixed_bytes <= initial_available)
        candidate = plan_execution(weights, profile, device, options, true,
                                   initial_available - weights.fixed_bytes);
    if (!candidate)
        throw std::invalid_argument(
            "Device budget cannot hold model weights and required execution resources");
    binding.complete();
    std::unique_ptr<ModelInstance> instance;
    std::optional<CalibratedPlan> calibrated;
    std::vector<models::qwen3_5::GraphCalibration> measured;
    unsigned preparation_attempts       = 0;
    std::size_t available_after_weights = 0;
    LoadSummary summary;
    for (;;) {
        auto model = models::qwen3_5::materialize_model(
            models::qwen3_5::resolve_load(std::move(source), device.device(),
                                          host_experts ? Device::cpu() : device.device()),
            device, &options.startup_observer);
        device.synchronize();
        available_after_weights = device_memory_snapshot(device.device()).free_bytes;
        const auto& stats       = model->storage_stats();
        summary.upload_seconds += stats.upload_seconds;
        summary.artifact_bytes_read += stats.read_bytes;
        summary.host_to_device_bytes += stats.h2d_bytes;
        summary.peak_staging_bytes = std::max(summary.peak_staging_bytes, stats.peak_staging_bytes);
        {
            StartupPhaseScope phase(options.startup_observer, StartupPhase::FrontendInitialize);
            instance = std::make_unique<ModelInstance>(std::move(model), options, profile);
            phase.complete();
        }
        try {
            {
                StartupPhaseScope phase(options.startup_observer, StartupPhase::TargetFinalize);
                calibrated = calibrate_plan(weights, *instance, device, options, budget,
                                            std::move(*candidate), measured);
                if (calibrated) phase.complete();
            }
            if (calibrated)
                calibrated =
                    prepare_program(weights, *instance, device, options, budget,
                                    std::move(*calibrated), measured, preparation_attempts);
        } catch (const DeviceAllocationError&) {
            if (host_experts || !weights.offload) throw;
            calibrated.reset();
        }
        if (calibrated) break;
        instance.reset();
        device.synchronize();
        if (host_experts || !weights.offload || preparation_attempts == 2)
            throw std::invalid_argument(
                "Device budget cannot hold the calibrated runtime and requested KV capacity");
        host_experts         = true;
        const auto available = budget.available(device_memory_snapshot(device.device()));
        if (weights.fixed_bytes > available)
            throw std::invalid_argument("Device budget cannot hold fixed model weights");
        candidate = plan_execution(weights, profile, device, options, true,
                                   available - weights.fixed_bytes);
        if (!candidate)
            throw std::invalid_argument("Device budget cannot hold MoE execution resources");
        source = models::qwen3_5::plan_load(reader, models::load_options(options));
    }
    instance->kv_capacity_resolution.available_after_weights_bytes = available_after_weights;
    instance->kv_capacity_resolution.available_after_startup_bytes =
        device_memory_snapshot(device.device()).free_bytes;
    const auto signature = models::qwen3_5::prefill_signature(
        *instance->model, calibrated->startup.moe.get(), options.prefill_chunk);
    auto context_cost = resolve_context_machine_cost(
        {.hardware_class = context_cost_hardware_class(
             device.properties().name, device.properties().major, device.properties().minor),
         .prefill_signature = signature},
        options.context_cost.preset_path);
    const auto& stats    = instance->model->storage_stats();
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
    summary.load_seconds        = std::chrono::duration<double>(Clock::now() - start).count();
    summary.device_object_count = stats.device_object_count;
    summary.host_object_count   = stats.host_object_count;
    summary.context_cost        = std::move(context_cost.summary);
    return {std::move(instance), std::move(summary), std::move(context_cost.model)};
}

} // namespace ninfer::runtime
