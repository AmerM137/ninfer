#include "models/qwen4_exp/program/resources.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp {
namespace {
std::uint32_t count(std::uint64_t n) {
    if (n > UINT32_MAX) throw std::overflow_error("Qwen4Exp context descriptors exceed uint32");
    return static_cast<std::uint32_t>(n);
}

ExecutionPlan checked_plan(const execution::Parameters& p, ExecutionPlan plan) {
    if (plan.profile != p.profile || !plan.moe ||
        plan.capacity.runtime_reservation_bytes != plan.device_bytes())
        throw std::invalid_argument("Qwen4Exp resources require the finalized parameter plan");
    return plan;
}
} // namespace

ContextPool::ContextPool(MemorySpan backing, const QsaCacheLayout& layout, std::size_t host_bytes,
                         std::uint32_t history_count, context::HostKVExtentStore* host)
    : cache(backing, layout),
      pages(cache.pages(),
            count(layout.pages.spec.page_group_count +
                  host_bytes / plan_host_kv_page_layout(layout.pages.spec.geometry).page_stride)),
      histories(pages, cache.tables(), history_count, layout.tables.spec.logical_page_capacity,
                host) {}

Resources::Resources(const execution::Parameters& p, ExecutionPlan input, DeviceContext& device)
    : plan(checked_plan(p, std::move(input))), device_(device),
      persistent_(device.device(), plan.persistent.bytes),
      workspace_(device.device(), plan.workspace.bytes), ingress_(plan.pinned_io_bytes),
      egress_(plan.pinned_output_bytes), ple_stream_(device),
      ple_transfer_(device.execution_view(ple_stream_.stream())), workspace(workspace_.span()) {
    const auto backing = persistent_.span();
    states             = std::make_unique<StateViews>(backing, plan.persistent.state);
    std::vector<HostKVPageLayout> layouts{
        plan_host_kv_page_layout(plan.persistent.main.pages.spec.geometry)};
    if (plan.persistent.mtp)
        layouts.push_back(plan_host_kv_page_layout(plan.persistent.mtp->pages.spec.geometry));
    auto minimum_kv = layouts.front().page_stride;
    for (const auto& l : layouts) minimum_kv = std::min(minimum_kv, l.page_stride);
    const auto image_bytes = states->storage().image_bytes();
    const auto logical_states =
        count(states->storage().slot_count() + plan.host_context_bytes / image_bytes);
    const auto histories = count(4ULL * logical_states + 3ULL * plan.shape.max_concurrency + 2);
    if (plan.host_context_bytes) {
        host_context       = std::make_unique<HostContextArena>(plan.host_context_bytes,
                                                                std::min(minimum_kv, image_bytes));
        host_kv            = std::make_unique<HostKVArena>(*host_context, layouts);
        const auto extents = count(plan.host_context_bytes / minimum_kv);
        if (extents) host_extents = std::make_unique<context::HostKVExtentStore>(*host_kv, extents);
    }
    state_store = std::make_unique<context::StateImageStore>(states->storage(), host_context.get(),
                                                             logical_states);
    main = std::make_unique<ContextPool>(backing, plan.persistent.main, plan.host_context_bytes,
                                         histories, host_extents.get());
    if (plan.persistent.mtp)
        mtp = std::make_unique<ContextPool>(backing, *plan.persistent.mtp, plan.host_context_bytes,
                                            histories, host_extents.get());
    migration = std::make_unique<context::MigrationContext>(device, mtp ? 2 : 1);
    if (plan.persistent.buffers.gdn_records) {
        gdn_records.emplace(backing, *plan.persistent.buffers.gdn_records);
        gdn_fold.emplace(*gdn_records, states->gdn().all_layers_view());
    }
    std::vector<const moe::Parameters*> layers;
    for (const auto& layer : p.text.layers) layers.push_back(&layer.moe);
    experts    = std::make_unique<moe::Resources>(plan.moe, layers, p.model.expert_banks(), device);
    ple_upload = std::make_unique<ple::UploadSlots>(device, plan.ple.max_positions, 2);
    ple_prepare = std::make_unique<ple::disk::PrepareService>(
        p.model.ple_table(), p.model.config().text.ple.eos, plan.ple, ple_upload->host_slots());
    if (ple_prepare->memory_bytes() != plan.ple_prepare_bytes ||
        ple_upload->memory().device_bytes != plan.ple_upload.device_bytes)
        throw std::logic_error("PLE allocated capacity differs from the execution plan");
    states->storage().clear_all(device.execution_view());
    if (const auto& counts = plan.persistent.buffers.token_counts)
        CUDA_CHECK(
            cudaMemsetAsync(counts->bind(backing).data, 0, counts->region.bytes, device.stream()));
    device.synchronize();
}

Resources::~Resources() {
    try {
        ple_prepare.reset();
        ple_upload.reset();
        if (experts) experts->synchronize();
        device_.synchronize();
    } catch (...) { std::terminate(); }
}
} // namespace ninfer::models::qwen4_exp
