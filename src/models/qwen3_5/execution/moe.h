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

[[nodiscard]] std::size_t moe_workspace_bytes(const MoeParameters& parameters, std::int32_t first,
                                              std::int32_t last);
void moe(const Tensor& hidden, const MoeParameters& parameters, Tensor& residual,
         const ops::MoeExecutionHints& hints, MemoryArena& workspace, CudaExecutionView execution);

} // namespace ninfer::models::qwen3_5::execution
