#pragma once

#include "core/buffer.h"
#include "core/expert_cache.h"
#include "models/qwen3_5/execution/moe.h"

#include <memory>
#include <optional>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

struct MoePlan;

// A layer snapshot shared by CPU and GPU. Updates and consumers use one compute stream.
class MoePartition {
public:
    MoePartition(ExpertBankId bank, std::int32_t experts, std::span<const std::int32_t> gpu_experts,
                 const DeviceContext& device);
    MoePartition(const MoePartition&)            = delete;
    MoePartition& operator=(const MoePartition&) = delete;

    void update(std::span<const ExpertKey> ready, CudaExecutionView execution);
    [[nodiscard]] execution::MoePartitionView view() const;

    [[nodiscard]] std::span<const ExpertKey> gpu_keys() const noexcept { return keys_; }

    [[nodiscard]] ops::ExpertSubsetKind gpu_kind() const noexcept { return kind_; }

private:
    ExpertBankId bank_;
    std::int32_t experts_;
    ops::ExpertSubsetKind kind_;
    PinnedHostBuffer mask_;
    DeviceBuffer device_mask_;
    CudaCompletionEvent mask_uploaded_;
    bool upload_pending_ = false;
    std::vector<ExpertKey> keys_;
};

// Program-owned CPU team and one reusable bridge frame.
// Only explicit synchronize/teardown waits for final GPU output; ordinary calls remain ordered
// by the borrowed compute stream. All external calls end before destruction.
class MoeBridge {
public:
    MoeBridge(const execution::MoeExecutionRequirements& requirements, const CpuTopology& topology,
              CpuTeamPlan team, const DeviceContext& device);
    ~MoeBridge();
    MoeBridge(const MoeBridge&)            = delete;
    MoeBridge& operator=(const MoeBridge&) = delete;

    [[nodiscard]] execution::MoeExecutionView execution_view() noexcept;
    void synchronize() const;

private:
    struct Host;
    CudaExecutionView cuda_;
    std::unique_ptr<Host> host_;
};

// Program-owned fixed experts, reusable streaming windows and CPU/GPU execution resources.
// Layer parameters and immutable sources outlive this owner; traversals have one Host caller.
class MoeResources {
public:
    MoeResources(std::shared_ptr<const MoePlan>,
                 std::span<const execution::MoeParameters* const> layers,
                 std::span<const ExpertBankSource> sources, const DeviceContext&);
    ~MoeResources();
    MoeResources(const MoeResources&)            = delete;
    MoeResources& operator=(const MoeResources&) = delete;

    // Retain the activity through the traversal; GPU-only streaming does not wake the team.
    [[nodiscard]] std::optional<CpuExecutionView::ActiveScope> begin(std::int32_t tokens);
    void before_layer(std::size_t layer);
    void execute(std::size_t layer, const Tensor& hidden, Tensor& residual, MemoryArena& workspace,
                 const ops::MoeExecutionHints& hints);
    void synchronize() const;
    [[nodiscard]] std::uint64_t uploaded_bytes() const noexcept;
    [[nodiscard]] ExpertCacheUsage cache_usage() const;
    [[nodiscard]] const MoePlan& plan() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::models::qwen3_5::detail
