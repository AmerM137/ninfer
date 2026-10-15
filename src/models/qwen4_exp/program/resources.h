#pragma once

#include "models/qwen4_exp/program/planning.h"
#include "models/qwen4_exp/execution/parameters.h"
#include "models/common/context/migration.h"
#include "models/common/moe/resources.h"
#include "ninfer/ops/gdn_replay.h"

namespace ninfer::models::qwen4_exp {
struct ContextPool {
    QsaCache cache;
    context::LogicalKVPageStore pages;
    context::KVHistoryStore histories;

    ContextPool(MemorySpan, const QsaCacheLayout&, std::size_t host_bytes, std::uint32_t histories,
                context::HostKVExtentStore*);
};

// One Program's physical resources. Model/Parameters and DeviceContext outlive this owner;
// the Program retires snapshots, histories and migration batches before destroying it.
class Resources {
public:
    Resources(const execution::Parameters&, ExecutionPlan, DeviceContext&);
    ~Resources();
    Resources(const Resources&)            = delete;
    Resources& operator=(const Resources&) = delete;

    [[nodiscard]] MemorySpan persistent() { return persistent_.span(); }

    [[nodiscard]] MemorySpan ingress() { return ingress_.span(); }

    [[nodiscard]] MemorySpan egress() { return egress_.span(); }

    [[nodiscard]] MemorySpan sampling_ingress() { return sampling_.span(); }

    [[nodiscard]] CudaExecutionView ple_transfer() const { return ple_transfer_; }

    const ExecutionPlan plan;
private:
    DeviceContext& device_;
    DeviceBuffer persistent_, workspace_;
    PinnedHostBuffer ingress_, egress_, sampling_;
    CudaStream ple_stream_;
    CudaExecutionView ple_transfer_;
public:
    MemoryArena workspace;
    std::unique_ptr<HostContextArena> host_context;
    std::unique_ptr<HostKVArena> host_kv;
    std::unique_ptr<context::HostKVExtentStore> host_extents;
    std::unique_ptr<StateViews> states;
    std::unique_ptr<context::StateImageStore> state_store;
    std::unique_ptr<ContextPool> main, mtp;
    std::unique_ptr<context::MigrationContext> migration;
    std::optional<GdnReplayRecords> gdn_records;
    std::optional<ops::GdnReplayFoldPlan> gdn_fold;
    std::unique_ptr<moe::Resources> experts;
    std::unique_ptr<ple::UploadSlots> ple_upload;
    std::unique_ptr<ple::disk::PrepareService> ple_prepare;
};
} // namespace ninfer::models::qwen4_exp
