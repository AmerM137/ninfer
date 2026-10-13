#pragma once

#include "ninfer/ops/moe.h"

namespace ninfer::models::qwen3_5::execution {

struct MoeParameters {
    ops::MoeProfile profile;
    Weight control;
    ops::SwiGLUWeights shared;
    ExpertBankId bank;
    ops::ExpertActivationDivisors activation_divisors;
    ExpertBankDeviceView resident;
};

[[nodiscard]] std::size_t moe_workspace_bytes(const ops::MoeProfile& profile, std::int32_t first,
                                              std::int32_t last);
void moe(const Tensor& hidden, const MoeParameters& parameters, const ExpertBankDeviceView& experts,
         Tensor& residual, const ops::MoeExecutionHints& hints, MemoryArena& workspace,
         CudaExecutionView execution);

struct MoePartitionView {
    ExpertBankId bank;
    std::int32_t experts           = 0;
    ops::ExpertSubsetKind gpu_kind = ops::ExpertSubsetKind::All;
    Tensor device_mask, host_mask;

    [[nodiscard]] ops::ExpertSubset gpu() const noexcept;
    [[nodiscard]] ops::ExpertSubset cpu() const noexcept;
};

struct MoeExecutionRequirements {
    std::size_t device_frame_bytes  = 0;
    std::size_t pinned_frame_bytes  = 0;
    std::size_t cpu_workspace_bytes = 0;
    std::size_t gpu_workspace_bytes = 0;

    void include(const MoeExecutionRequirements& other) noexcept;
};

// GPU subset is All, None or MaskOnes; the CPU always consumes its complement.
[[nodiscard]] MoeExecutionRequirements describe_moe_execution(const ops::MoeProfile& profile,
                                                              ops::ExpertSubsetKind gpu_subset,
                                                              std::int32_t min_tokens,
                                                              std::int32_t max_tokens);

struct MoeHostExecutionView {
    CpuExecutionView cpu;
    CudaExecutionView transfer;
    MemorySpan device_frame, host_frame;
    MemoryArena& workspace;
    CudaCompletionEvent& route_ready;
    CudaCompletionEvent& host_ready;
    CudaCompletionEvent& partial_ready;
};

struct MoeExecutionView {
    CudaExecutionView cuda;
    std::optional<MoeHostExecutionView> host;
};

// BF16 hidden and in/out residual are disjoint contiguous [H,T]. Required GPU keys have
// already been published and are retained by the caller through the last GPU consumer.
// A single caller reuses the frame on its fixed compute/bridge streams, outside capture.
// CPU work finishes before return; output and GPU workspace follow compute-stream ordering.
// Sources, partition and execution resources remain live through their asynchronous consumers.
// Returns whether a CPU contribution was evaluated, for the caller's waiting policy.
bool moe(const Tensor& hidden, const MoeParameters& parameters, const MoePartitionView& partition,
         const ExpertBankSource& source, const ExpertBankDeviceView& bank, Tensor& residual,
         MemoryArena& workspace, MoeExecutionView execution,
         const ops::MoeExecutionHints& hints = {});

} // namespace ninfer::models::qwen3_5::execution
