#include "models/qwen3_5/program/moe_resources.h"
#include "models/qwen3_5/program/planning/moe.h"
#include "ops/moe/fixture.h"

#include <atomic>
#include <chrono>
#include <numeric>
#include <thread>

namespace {
using namespace ninfer;
using namespace ninfer::test;
namespace model = ninfer::models::qwen3_5;
using model::detail::MoePartition;
using model::detail::MoeBridge;
using Clock = std::chrono::steady_clock;
using Kind  = ops::ExpertSubsetKind;

void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}

void publish(ExpertCache& cache, const ExpertCacheRequirements& req,
             std::span<const ExpertKey> keys) {
    std::vector<ExpertPlacement> entries;
    std::size_t offset = 0;
    for (const auto key : keys) {
        entries.push_back({key, offset});
        offset += req.banks[key.bank.index].entry_bytes;
    }
    cache.replace({0, req.config.capacity_bytes, 256}, entries);
    cache.prepare({keys});
    const auto deadline = Clock::now() + std::chrono::seconds(10);
    while (!cache.progress().required_published) {
        if (Clock::now() > deadline) { throw std::runtime_error("MoE cache upload timed out"); }
        std::this_thread::yield();
    }
}

std::vector<std::uint64_t> offsets(const ExpertCache& cache, ExpertBankId bank) {
    return from_device<std::uint64_t>(cache.view(bank).gate_up.offsets, moe::kExperts);
}

struct Call {
    moe::Batch batch;
    moe::DeviceBatch device;
    GuardedDeviceBuffer output;
    Tensor residual;

    Call(moe::Fixture& fixture, int tokens, int pattern)
        : batch(fixture.batch(tokens, pattern)), device(batch), output(batch.residual.size() * 2),
          residual(output.span(), DType::BF16, {moe::kHidden, tokens}) {
        output.copy_from_host(batch.residual.data(), output.bytes());
    }

    int verify(const std::string& label) {
        return verify_reduction(label, from_device_bf16(residual.data, batch.complete.size()),
                                batch.complete, moe::kCompleteA16) +
               output.verify_guards(label);
    }
};

int qualify(QType down, const DeviceContext& device, const CpuTopology& topology) {
    moe::Fixture fixture(QType::Q4_G64_FP16, down);
    moe::HostBank native(fixture);
    native.register_upload(device.device());
    const std::array sources{native.source(), native.source()};
    auto description = describe_expert_cache(sources, {});
    ExpertCacheConfig config;
    config.capacity_bytes = description.banks[0].entry_bytes * moe::kExperts;
    description           = describe_expert_cache(sources, config);
    ExpertCache cache(sources, description, device, device.transfer_view());
    const auto& weights = fixture.weights();
    model::execution::MoeParameters parameters{
        fixture.profile(), weights.control, weights.shared, {0}, {}, {}};
    const auto team = plan_cpu_team(
        topology.info(), std::min<std::size_t>(4, cpu_worker_order(topology.info()).size()));
    std::vector<std::int32_t> ids(moe::kExperts), half;
    std::iota(ids.begin(), ids.end(), 0);
    for (int e : ids) {
        if (e % 2) { half.push_back(e); }
    }
    MoePartition mixed({0}, moe::kExperts, half, device), none({0}, moe::kExperts, {}, device),
        all({0}, moe::kExperts, ids, device), other({1}, moe::kExperts, ids, device);
    auto requirements =
        model::execution::describe_moe_execution(parameters.profile, Kind::MaskOnes, 1, 1024);
    for (auto kind : {Kind::All, Kind::None}) {
        requirements.include(
            model::execution::describe_moe_execution(parameters.profile, kind, 1, 1024));
    }
    const auto all_requirements =
        model::execution::describe_moe_execution(parameters.profile, Kind::All, 1, 128);
    require(all_requirements.pinned_frame_bytes == 0 && all_requirements.device_frame_bytes == 0 &&
                all_requirements.cpu_workspace_bytes == 0,
            "All-GPU MoE reserved CPU resources");
    MoeBridge resources(requirements, topology, team, device);
    auto execution = resources.execution_view();
    GuardedDeviceBuffer workspace_storage(requirements.gpu_workspace_bytes);
    MemoryArena workspace(workspace_storage.span());
    int failures       = 0;
    const auto execute = [&](Call& call, const MoePartition& partition,
                             model::execution::MoeExecutionView execution) {
        std::optional<ExpertCacheUse> use;
        if (partition.gpu_kind() != Kind::None) {
            use = cache.acquire(partition.gpu_keys(), execution.cuda);
            if (!use) throw std::logic_error("GPU assignment was not published");
        }
        try {
            model::execution::moe(call.device.x, parameters, partition.view(), sources[0],
                                  cache.view({0}), call.residual, workspace, execution);
        } catch (...) {
            if (use) cache.release(*use, execution.cuda);
            throw;
        }
        if (use) cache.release(*use, execution.cuda);
    };
    const auto run = [&](Call& call, const MoePartition& partition) {
        execute(call, partition, execution);
        require(workspace.used() == 0 && execution.host->workspace.used() == 0,
                "MoE leaked an arena scope");
    };

    // Allocate all outputs before submission so allocation does not accidentally synchronize reuse.
    std::vector<std::unique_ptr<Call>> calls;
    for (int tokens : {1024, 1, 128, 9, 48, 8, 1}) {
        calls.push_back(std::make_unique<Call>(fixture, tokens, calls.size() % 3));
    }
    publish(cache, description, mixed.gpu_keys());
    for (auto& call : calls) { run(*call, mixed); }
    resources.synchronize();
    for (auto& call : calls) {
        failures += call->verify("cached mixed T=" + std::to_string(call->device.x.ne[1]));
    }
    const auto first_offsets = offsets(cache, {0});
    for (int e : ids) {
        require((first_offsets[e] == kExpertAbsentOffset) == (e % 2 == 0),
                "test did not leave CPU-only experts absent on GPU");
    }

    // Rebuild the region with another bank, then restore the assignment at different offsets.
    publish(cache, description, other.gpu_keys());
    cuda_synchronize(device.transfer_stream());
    const auto occupied = offsets(cache, {1});
    const auto old      = first_offsets[half.front()];
    const auto keeper   = std::find(occupied.begin(), occupied.end(), old);
    require(keeper != occupied.end(), "replacement did not reuse the old expert slot");
    std::vector<ExpertKey> required(mixed.gpu_keys().begin(), mixed.gpu_keys().end());
    required.insert(required.begin(), {{1}, static_cast<std::uint32_t>(keeper - occupied.begin())});
    publish(cache, description, required);
    Call relocated(fixture, 9, 2);
    run(relocated, mixed);
    resources.synchronize();
    require(offsets(cache, {0})[half.front()] != old,
            "explicit placement did not change the indexed address");
    failures += relocated.verify("reloaded cached mixed");

    Call cpu_only(fixture, 8, 0);
    run(cpu_only, none);
    resources.synchronize();
    failures += cpu_only.verify("CPU routed plus GPU shared");

    // The same fixed partial partition sees batches with zero/all CPU assignments.
    std::vector<int> selected;
    for (int e : fixture.logical(0).route.ids) { selected.push_back(e); }
    MoePartition selected_partition({0}, moe::kExperts, selected, device);
    publish(cache, description, selected_partition.gpu_keys());
    Call empty_cpu(fixture, 1, 0);
    run(empty_cpu, selected_partition);
    resources.synchronize();
    failures += empty_cpu.verify("mixed partition with no CPU assignment");

    const auto unselected = std::find_if(ids.begin(), ids.end(), [&](int e) {
        return std::find(selected.begin(), selected.end(), e) == selected.end();
    });
    const std::array one_unselected{*unselected};
    MoePartition empty_gpu_partition({0}, moe::kExperts, one_unselected, device);
    publish(cache, description, empty_gpu_partition.gpu_keys());
    Call empty_gpu(fixture, 1, 0);
    run(empty_gpu, empty_gpu_partition);
    resources.synchronize();
    failures += empty_gpu.verify("mixed partition with no GPU assignment");

    publish(cache, description, all.gpu_keys());
    {
        MoeBridge gpu_only(all_requirements, topology, {}, device);
        require(!gpu_only.execution_view().host, "GPU-only resources created a CPU team/bridge");
        Call complete(fixture, 48, 1);
        execute(complete, all, gpu_only.execution_view());
        gpu_only.synchronize();
        failures += complete.verify("cached complete GPU MoE");
    }

    // Delay bridge progress, then queue another layer using the same frame immediately.
    publish(cache, description, mixed.gpu_keys());
    std::array<std::unique_ptr<Call>, 2> delayed{std::make_unique<Call>(fixture, 128, 1),
                                                 std::make_unique<Call>(fixture, 1, 2)};
    std::atomic<bool> open = false;
    cuda_check(cudaLaunchHostFunc(
                   execution.host->transfer.stream(),
                   [](void* pointer) {
                       auto& gate = *static_cast<std::atomic<bool>*>(pointer);
                       while (!gate.load(std::memory_order_acquire)) { std::this_thread::yield(); }
                   },
                   &open),
               "delay bridge");
    std::jthread opener([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        open.store(true, std::memory_order_release);
    });
    for (auto& call : delayed) { run(*call, mixed); }
    resources.synchronize();
    for (auto& call : delayed) { failures += call->verify("delayed bridge/reused frame"); }

    // An unpublished assignment fails without retaining a use or corrupting the next call.
    publish(cache, description, other.gpu_keys());
    cuda_synchronize(device.transfer_stream());
    Call missing(fixture, 1, 0);
    bool rejected = false;
    try {
        run(missing, mixed);
    } catch (const std::logic_error&) { rejected = true; }
    require(rejected, "unpublished GPU assignment was accepted");
    publish(cache, description, mixed.gpu_keys());
    run(missing, mixed);
    resources.synchronize();
    failures += missing.verify("reuse after missing cache assignment");

    Call teardown(fixture, 128, 1);
    {
        MoeBridge pending(requirements, topology, team, device);
        execute(teardown, mixed, pending.execution_view());
    }
    failures += teardown.verify("resource destruction after asynchronous return");
    failures += workspace_storage.verify_guards("MoE execution workspace");
    require(workspace.peak_used() <= requirements.gpu_workspace_bytes,
            "MoE GPU capacity query was too small");
    require(execution.host->workspace.peak_used() <= requirements.cpu_workspace_bytes,
            "MoE CPU capacity query was too small");
    return failures;
}

void qualify_allocation() {
    for (const auto heterogeneous : {false, true}) {
        model::detail::MoePlan plan;
        std::size_t total = 0, largest = 0;
        for (std::uint32_t i = 0; i < 4; ++i) {
            model::detail::MoeLayerPlan layer;
            layer.bank    = {i};
            layer.experts = 8;
            for (std::uint32_t e = 0; e < 8; ++e) layer.priority.push_back(7 - e);
            plan.layers.push_back(layer);
            ExpertBankCacheLayout bank;
            bank.entry_bytes = (heterogeneous ? 2 + i * i : 2) * 256;
            plan.cache.banks.push_back(bank);
            total += 8 * bank.entry_bytes;
            largest = std::max(largest, 8 * bank.entry_bytes);
        }
        for (std::size_t capacity = 0; capacity <= total + 256; capacity += 256) {
            plan.cache.config.capacity_bytes = capacity;
            const bool feasible              = model::detail::plan_moe_cache(plan);
            require(feasible == (capacity >= largest), "streaming capacity admission differs");
            if (!feasible) continue;
            const auto fixed        = plan.cache.config.fixed_bytes;
            std::size_t fixed_bytes = 0, target_bytes = 0;
            for (std::size_t i = 0; i < plan.layers.size(); ++i) {
                const auto& layer = plan.layers[i];
                const auto bytes  = plan.cache.banks[i].entry_bytes;
                require(layer.fixed_experts <= layer.cache_experts && layer.cache_experts <= 8,
                        "cache target does not extend its fixed prefix");
                fixed_bytes += layer.fixed_experts * bytes;
                target_bytes += layer.cache_experts * bytes;
                for (std::size_t w = 0; w < plan.window_count; ++w)
                    require((8 - layer.fixed_experts) * bytes <= plan.windows[w].bytes,
                            "streaming window cannot complete a bank");
                std::vector<bool> seen(8);
                for (std::uint32_t n = 0; n < layer.fixed_experts; ++n)
                    seen[layer.priority[n]] = true;
                for (const auto& entry : layer.stream_layout) {
                    require(!seen[entry.key.expert], "streaming layout duplicates a fixed expert");
                    seen[entry.key.expert] = true;
                }
                require(std::all_of(seen.begin(), seen.end(), [](bool v) { return v; }),
                        "fixed and streaming layouts omit an expert");
            }
            require(fixed_bytes == fixed && target_bytes == plan.target_bytes &&
                        target_bytes <= capacity,
                    "expert layout capacity accounting differs");
            std::size_t cursor = fixed;
            for (std::size_t w = 0; w < plan.window_count; ++w) {
                require(plan.windows[w].offset == cursor, "streaming windows leave a gap");
                cursor += plan.windows[w].bytes;
            }
            require(cursor == capacity, "expert budget was left outside the reusable windows");
            if (!heterogeneous) {
                const auto entry = plan.cache.banks[0].entry_bytes;
                // Four identical banks: two windows plus fixed copies cost (16 + 2*n)*entry.
                if (capacity >= 16 * entry && capacity < total) {
                    const auto minimum_count = (capacity / entry - 16) / 2;
                    for (const auto& layer : plan.layers)
                        require(
                            layer.fixed_experts >= minimum_count,
                            "fixed allocation reserved complete banks instead of missing copies");
                }
                require(target_bytes == std::min(total, capacity / entry * entry),
                        "small-T layout failed to fill whole-expert capacity");
            }
        }
    }
}

int qualify_schedule(const DeviceContext& device, const CpuTopology& topology,
                     std::uint32_t bank_capacity, bool one_large_layer = false) {
    moe::Fixture a(QType::Q4_G64_FP16, QType::Q5_G64_FP16),
        b(QType::Q4_G64_FP16, QType::Q6_G64_FP16);
    moe::HostBank ha(a), hb(b);
    ha.register_upload(device.device());
    hb.register_upload(device.device());
    const std::array sources{ha.source(), hb.source(), ha.source(), hb.source()};
    const std::array<moe::Fixture*, 4> fixtures{&a, &b, &a, &b};
    std::array<model::execution::MoeParameters, 4> parameters;
    std::array<const model::execution::MoeParameters*, 4> inputs;
    auto plan                = std::make_shared<model::detail::MoePlan>();
    const auto layout        = describe_expert_cache(sources, {});
    std::size_t largest_bank = 0;
    for (std::uint32_t i = 0; i < 4; ++i) {
        const auto& weights = fixtures[i]->weights();
        parameters[i] = {fixtures[i]->profile(), weights.control, weights.shared, {i}, {}, {}};
        inputs[i]     = &parameters[i];
        model::detail::MoeLayerPlan layer;
        layer.bank              = {i};
        layer.experts           = moe::kExperts;
        layer.hybrid_max_tokens = one_large_layer && i != 0 ? 1024 : 8;
        for (int e = 0; e < moe::kExperts; ++e) {
            if ((i == 0 && e % 2 == 0) || (i == 1 && e % 4 == 0) || i == 3 ||
                (one_large_layer && i != 0)) {
                layer.priority.push_back(e);
            }
        }
        for (int e = 0; e < moe::kExperts; ++e) {
            if (std::find(layer.priority.begin(), layer.priority.end(), e) == layer.priority.end())
                layer.priority.push_back(e);
        }
        const auto kind = Kind::MaskOnes;
        plan->bridge.include(model::execution::describe_moe_execution(parameters[i].profile, kind,
                                                                      1, layer.hybrid_max_tokens));
        plan->bridge.include(
            model::execution::describe_moe_execution(parameters[i].profile, Kind::All, 1, 1024));
        largest_bank = std::max(largest_bank, moe::kExperts * layout.banks[i].entry_bytes);
        plan->layers.push_back(std::move(layer));
    }
    ExpertCacheConfig config;
    // Exercise both one-window admission and double buffering.
    config.capacity_bytes     = bank_capacity * largest_bank;
    config.upload_slots       = 12;
    config.max_in_flight_uses = 5;
    plan->cache               = describe_expert_cache(sources, config);
    require(model::detail::plan_moe_cache(*plan), "schedule fixture has insufficient capacity");
    plan->team = plan_cpu_team(topology.info(),
                               std::min<std::size_t>(4, cpu_worker_order(topology.info()).size()));
    model::detail::MoeResources resources(plan, inputs, sources, device);
    require(resources.cache_usage().ready_cache_bytes == plan->target_bytes,
            "startup returned before the initial expert cache became ready");
    GuardedDeviceBuffer storage(plan->bridge.gpu_workspace_bytes);
    MemoryArena workspace(storage.span());
    int failures = 0;
    for (const int tokens : {1, 1, 8, 8, 9, 48, 1024, 1, 1, 8, 9, 1}) {
        std::array<std::unique_ptr<Call>, 4> calls;
        for (std::size_t i = 0; i < calls.size(); ++i) {
            calls[i] = std::make_unique<Call>(*fixtures[i], tokens, i % 3);
        }
        {
            const auto cpu_activity = resources.begin(tokens);
            for (std::size_t i = 0; i < calls.size(); ++i) {
                resources.before_layer(i);
                resources.execute(i, calls[i]->device.x, calls[i]->residual, workspace, {});
            }
        }
        resources.synchronize();
        for (auto& call : calls)
            failures += call->verify("scheduled MoE T=" + std::to_string(tokens));
        const auto usage = resources.cache_usage();
        require(usage.used_bytes <= config.capacity_bytes &&
                    usage.ready_cache_bytes <= usage.used_bytes,
                "schedule exceeded its physical capacity");
    }
    failures += storage.verify_guards("scheduled MoE workspace");
    return failures;
}
} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) { return 77; }
    try {
        qualify_allocation();
        CpuTopology topology;
        auto& device = cuda_environment().context;
        int failures = 0;
        for (const auto down : {QType::Q5_G64_FP16, QType::Q6_G64_FP16}) {
            failures += qualify(down, device, topology);
        }
        failures += qualify_schedule(device, topology, 1);
        failures += qualify_schedule(device, topology, 2);
        failures += qualify_schedule(device, topology, 3);
        failures += qualify_schedule(device, topology, 4);
        failures += qualify_schedule(device, topology, 2, true);
        std::cout << (failures ? "FAIL" : "OK") << " cached mixed MoE execution\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
