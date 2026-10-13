#include "models/qwen3_5/program/moe_resources.h"

#include "core/transfer.h"
#include "models/qwen3_5/program/planning/moe.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace ninfer::models::qwen3_5::detail {

MoePartition::MoePartition(ExpertBankId bank, std::int32_t experts,
                           std::span<const std::int32_t> gpu_experts, const DeviceContext& device)
    : bank_(bank), experts_(experts), kind_(ops::ExpertSubsetKind::None), mask_uploaded_(device) {
    if (bank.index == UINT32_MAX || experts <= 0) {
        throw std::invalid_argument("MoE partition requires a bank and positive expert count");
    }
    std::vector<std::int32_t> mask(experts);
    for (const auto e : gpu_experts) {
        if (e < 0 || e >= experts || mask[e]) {
            throw std::invalid_argument("MoE GPU assignment contains an invalid or duplicate ID");
        }
        mask[e] = 1;
    }
    keys_.reserve(experts);
    mask_        = PinnedHostBuffer(std::size_t(experts) * sizeof(std::int32_t));
    device_mask_ = DeviceBuffer(device.device(), mask_.size());
    std::copy(mask.begin(), mask.end(), static_cast<std::int32_t*>(mask_.data()));
    for (std::int32_t e = 0; e < experts; ++e) {
        if (mask[e]) { keys_.push_back({bank, static_cast<std::uint32_t>(e)}); }
    }
    if (keys_.size() == static_cast<std::size_t>(experts)) {
        kind_ = ops::ExpertSubsetKind::All;
    } else if (!keys_.empty()) {
        kind_ = ops::ExpertSubsetKind::MaskOnes;
        device_mask_.copy_from_host(mask_.data(), device_mask_.size());
    }
}

void MoePartition::update(std::span<const ExpertKey> ready, CudaExecutionView execution) {
    if (std::equal(keys_.begin(), keys_.end(), ready.begin(), ready.end())) return;
    if (upload_pending_) mask_uploaded_.synchronize();
    auto* mask = static_cast<std::int32_t*>(mask_.data());
    std::fill_n(mask, experts_, 0);
    for (const auto key : ready) {
        if (key.bank != bank_ || key.expert >= std::uint32_t(experts_) || mask[key.expert]) {
            throw std::invalid_argument("MoE snapshot contains an invalid or duplicate expert");
        }
        mask[key.expert] = 1;
    }
    keys_.assign(ready.begin(), ready.end());
    kind_           = keys_.empty()                           ? ops::ExpertSubsetKind::None
                      : keys_.size() == std::size_t(experts_) ? ops::ExpertSubsetKind::All
                                                              : ops::ExpertSubsetKind::MaskOnes;
    upload_pending_ = false;
    if (kind_ == ops::ExpertSubsetKind::MaskOnes) {
        copy_bytes_async(device_mask_.span(), mask_.span(), execution);
        mask_uploaded_.record(execution.stream());
        upload_pending_ = true;
    }
}

execution::MoePartitionView MoePartition::view() const {
    execution::MoePartitionView result{bank_, experts_, kind_, {}, {}};
    if (kind_ == ops::ExpertSubsetKind::MaskOnes) {
        result.host_mask =
            Tensor(const_cast<void*>(mask_.data()), Device::cpu(), DType::I32, {experts_});
        result.device_mask = Tensor(const_cast<void*>(device_mask_.data()), device_mask_.device(),
                                    DType::I32, {experts_});
    }
    return result;
}

struct MoeBridge::Host {
    CudaStream stream;
    CudaExecutionView transfer;
    DeviceBuffer device_frame;
    PinnedHostBuffer host_frame;
    HostBuffer workspace_storage;
    MemoryArena workspace;
    CpuExecutor executor;
    CudaCompletionEvent route_ready, host_ready, partial_ready;

    Host(const execution::MoeExecutionRequirements& requirements, const CpuTopology& topology,
         CpuTeamPlan team, const DeviceContext& device)
        : stream(device), transfer(device.execution_view(stream.stream())),
          device_frame(device.device(), requirements.device_frame_bytes),
          host_frame(requirements.pinned_frame_bytes),
          workspace_storage(requirements.cpu_workspace_bytes), workspace(workspace_storage.span()),
          executor(topology, std::move(team)), route_ready(device), host_ready(device),
          partial_ready(device) {}

    execution::MoeHostExecutionView view() {
        return {executor.execution_view(),
                transfer,
                device_frame.span(),
                host_frame.span(),
                workspace,
                route_ready,
                host_ready,
                partial_ready};
    }
};

MoeBridge::MoeBridge(const execution::MoeExecutionRequirements& requirements,
                     const CpuTopology& topology, CpuTeamPlan team, const DeviceContext& device)
    : cuda_(device.execution_view()) {
    if (requirements.pinned_frame_bytes != 0) {
        host_ = std::make_unique<Host>(requirements, topology, std::move(team), device);
    } else if (requirements.device_frame_bytes || requirements.cpu_workspace_bytes) {
        throw std::invalid_argument("MoE Host execution requirements are incomplete");
    }
}

MoeBridge::~MoeBridge() {
    try {
        synchronize();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "MoE resource cleanup failed: %s\n", error.what());
    }
}

execution::MoeExecutionView MoeBridge::execution_view() noexcept {
    execution::MoeExecutionView result{cuda_, {}};
    if (host_) { result.host.emplace(host_->view()); }
    return result;
}

void MoeBridge::synchronize() const {
    CudaDeviceScope scope(cuda_.device());
    // The compute stream may wait on the bridge; both have had all producers submitted.
    const auto transfer = host_ ? cudaStreamSynchronize(host_->transfer.stream()) : cudaSuccess;
    const auto compute  = cudaStreamSynchronize(cuda_.stream());
    const auto error    = transfer != cudaSuccess ? transfer : compute;
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("MoE stream synchronization: ") +
                                 cudaGetErrorString(error));
    }
}

struct MoeResources::Impl {
    struct Layer {
        std::unique_ptr<MoePartition> partition;
        std::vector<ExpertKey> keys, ready;
        std::optional<ExpertCacheUse> use;
    };

    std::shared_ptr<const MoePlan> plan;
    std::vector<const execution::MoeParameters*> parameters;
    std::span<const ExpertBankSource> sources;
    const DeviceContext& device;
    CpuTopology topology;
    CudaExecutionView cuda;
    CudaStream upload_stream;
    CudaExecutionView upload;
    ExpertCache cache;
    MoeBridge bridge;
    std::vector<Layer> layers;
    std::vector<ExpertKey> extra_keys;
    std::vector<std::size_t> needed;
    std::array<std::size_t, 2> window_layers{SIZE_MAX, SIZE_MAX};
    std::optional<std::thread::id> caller;
    std::int32_t tokens     = 0;
    std::size_t active_step = SIZE_MAX;
    bool cache_settled      = true;
    bool streaming          = false;

    Impl(std::shared_ptr<const MoePlan> plan,
         std::span<const execution::MoeParameters* const> parameters,
         std::span<const ExpertBankSource> sources, const DeviceContext& device)
        : plan(std::move(plan)), parameters(parameters.begin(), parameters.end()), sources(sources),
          device(device), cuda(device.execution_view()), upload_stream(device),
          upload(device.execution_view(upload_stream.stream())),
          cache(sources, this->plan->cache, device, upload, this->plan->fixed_layout),
          bridge(this->plan->bridge, topology, this->plan->team, device),
          layers(parameters.size()) {
        if (parameters.size() != this->plan->layers.size())
            throw std::invalid_argument("MoE schedule layer directory differs from its plan");
        needed.reserve(layers.size());
        std::vector<ExpertKey> targets;
        for (std::size_t i = 0; i < layers.size(); ++i) {
            const auto& layer = this->plan->layers[i];
            if (!layer.experts) continue;
            if (!parameters[i] || parameters[i]->bank != layer.bank ||
                parameters[i]->profile.routed.experts != layer.experts)
                throw std::invalid_argument("MoE schedule profile differs from its layer");
            auto& state     = layers[i];
            state.partition = std::make_unique<MoePartition>(
                layer.bank, layer.experts, std::span<const std::int32_t>{}, device);
            state.ready.reserve(layer.experts);
            for (std::uint32_t e = 0; e < std::uint32_t(layer.experts); ++e)
                state.keys.push_back({layer.bank, e});
            for (std::uint32_t n = 0; n < layer.cache_experts; ++n)
                targets.push_back({layer.bank, layer.priority[n]});
        }
        sort_keys(targets);
        for (const auto& placement : this->plan->extra_layout) extra_keys.push_back(placement.key);
        cache.replace(extra_region(), this->plan->extra_layout);
        cache.prepare({targets});
        while (!cache.progress({SIZE_MAX, SIZE_MAX}).required_published) cache.wait_for_progress();
        synchronize();
    }

    ~Impl() {
        try {
            release_snapshots();
            synchronize();
        } catch (const std::exception& error) {
            std::fprintf(stderr, "MoE schedule cleanup failed: %s\n", error.what());
        }
    }

    static void sort_keys(std::vector<ExpertKey>& keys) {
        std::sort(keys.begin(), keys.end(), [](auto a, auto b) {
            return a.bank.index != b.bank.index ? a.bank.index < b.bank.index : a.expert < b.expert;
        });
    }

    LayoutRegion extra_region() const {
        const auto fixed = plan->cache.config.fixed_bytes;
        return {fixed, plan->cache.config.capacity_bytes - fixed, 256};
    }

    std::size_t window_for(std::size_t layer) const {
        for (std::size_t w = 0; w < plan->window_count; ++w)
            if (window_layers[w] == layer) return w;
        return SIZE_MAX;
    }

    void load_window(std::size_t window, std::size_t layer) {
        if (window_layers[window] == layer) return;
        cache.replace(plan->windows[window], plan->layers[layer].stream_layout);
        window_layers[window] = layer;
    }

    void release(Layer& layer) {
        if (layer.use) {
            cache.release(*layer.use, cuda);
            layer.use.reset();
        }
    }

    void release_snapshots() {
        for (auto& layer : layers) release(layer);
    }

    void synchronize() const {
        CudaDeviceScope scope(device.device());
        const auto status = cudaStreamSynchronize(upload.stream());
        bridge.synchronize();
        CUDA_CHECK(status);
    }

    void snapshot(std::size_t index) {
        auto& layer = layers[index];
        if (layer.use) return;
        cache.ready_keys(plan->layers[index].bank, layer.ready);
        for (;;) {
            layer.use = cache.acquire(layer.ready, cuda);
            if (layer.use) break;
            cache.wait_for_progress();
        }
        layer.partition->update(layer.ready, cuda);
    }

    void prepare_step(std::size_t first) {
        const auto it = std::lower_bound(needed.begin(), needed.end(), first);
        if (it == needed.end()) return;
        const auto n = static_cast<std::size_t>(it - needed.begin());
        if (active_step != n) {
            const auto current = *it;
            const auto next =
                plan->window_count == 2 && n + 1 < needed.size() ? needed[n + 1] : SIZE_MAX;
            auto window = window_for(current);
            if (window == SIZE_MAX) {
                window = next != SIZE_MAX && window_for(next) == 0 ? 1 : 0;
                load_window(window, current);
            }
            cache.prepare({layers[current].keys});
            // Queue the current publication before reusing the other window. Its reader wait
            // must not get ahead of an unfinished current upload on the shared transfer stream.
            while (!cache.progress({SIZE_MAX, SIZE_MAX}).required_published)
                cache.wait_for_progress();
            if (next != SIZE_MAX) load_window(1 - window, next);
            cache.prepare({layers[current].keys, next != SIZE_MAX
                                                     ? std::span<const ExpertKey>(layers[next].keys)
                                                     : std::span<const ExpertKey>{}});
            active_step = n;
        }
        cache.progress();
    }

    void progress_cache() {
        if (!cache_settled && needed.empty())
            cache_settled = cache.progress({8 * 1024 * 1024, 1}).prefetch_ready;
    }
};

MoeResources::MoeResources(std::shared_ptr<const MoePlan> plan,
                           std::span<const execution::MoeParameters* const> parameters,
                           std::span<const ExpertBankSource> sources, const DeviceContext& device)
    : impl_(std::make_unique<Impl>(std::move(plan), parameters, sources, device)) {}

MoeResources::~MoeResources() = default;

void MoeResources::begin(std::int32_t tokens) {
    auto& state       = *impl_;
    const auto thread = std::this_thread::get_id();
    if (!state.caller) {
        if (!state.plan->caller_cpus.empty())
            bind_current_thread(state.topology, state.plan->caller_cpus);
        state.caller = thread;
    } else if (*state.caller != thread) {
        throw std::logic_error("MoE schedule has more than one control thread");
    }
    if (tokens <= 0) throw std::invalid_argument("MoE traversal requires positive token count");
    state.tokens = tokens;
    state.needed.clear();
    for (std::size_t i = 0; i < state.plan->layers.size(); ++i) {
        const auto& layer = state.plan->layers[i];
        if (layer.experts && layer.requires_full_bank(tokens) && !layer.stream_layout.empty())
            state.needed.push_back(i);
    }
    const bool streaming = !state.needed.empty();
    state.active_step    = SIZE_MAX;
    if (streaming != state.streaming) {
        state.release_snapshots();
        state.cache.replace(state.extra_region(),
                            streaming ? std::span<const ExpertPlacement>{}
                                      : std::span<const ExpertPlacement>(state.plan->extra_layout));
        state.window_layers.fill(SIZE_MAX);
        state.cache_settled = false;
        state.streaming     = streaming;
        if (!streaming) state.cache.prepare({{}, state.extra_keys});
    }
}

void MoeResources::before_layer(std::size_t layer) {
    impl_->prepare_step(layer);
    impl_->progress_cache();
}

void MoeResources::execute(std::size_t index, const Tensor& hidden, Tensor& residual,
                           MemoryArena& workspace, const ops::MoeExecutionHints& hints) {
    auto& state          = *impl_;
    const auto& p        = *state.parameters.at(index);
    const auto& selected = state.plan->layers.at(index);
    auto& layer          = state.layers.at(index);
    if (hidden.ne[1] != state.tokens) throw std::logic_error("MoE traversal token count changed");
    if (selected.requires_full_bank(state.tokens)) {
        while (!layer.use) {
            layer.use = state.cache.acquire(layer.keys, state.cuda);
            if (layer.use) break;
            const auto progress = state.cache.progress();
            if (!progress.required_published && progress.submitted_batches) continue;
            state.cache.wait_for_progress();
        }
        try {
            execution::moe(hidden, p, state.cache.view(p.bank), residual, hints, workspace,
                           state.cuda);
        } catch (...) {
            state.release(layer);
            throw;
        }
    } else {
        state.snapshot(index);
        try {
            execution::moe(hidden, p, layer.partition->view(), state.sources[p.bank.index],
                           state.cache.view(p.bank), residual, workspace,
                           state.bridge.execution_view(), hints);
        } catch (...) {
            state.release(layer);
            throw;
        }
    }
    if (!state.cache_settled) state.release(layer);
    state.prepare_step(index + 1);
    state.progress_cache();
}

void MoeResources::synchronize() const { impl_->synchronize(); }

std::uint64_t MoeResources::uploaded_bytes() const noexcept {
    return impl_->cache.uploaded_bytes();
}

ExpertCacheUsage MoeResources::cache_usage() const { return impl_->cache.usage(); }

const MoePlan& MoeResources::plan() const { return *impl_->plan; }

} // namespace ninfer::models::qwen3_5::detail
