#include "models/qwen3_5/execution/moe.h"

#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {

std::size_t moe_workspace_bytes(const MoeParameters& parameters, std::int32_t first,
                                std::int32_t last) {
    if (parameters.resident.device.type() != DeviceType::CUDA) {
        throw std::invalid_argument("resident MoE requires GPU expert sources");
    }
    return ops::topk_softmax_swiglu_moe_workspace_capacity_bytes(
        parameters.profile, ops::MoeEpilogue::AddResidual, first, last);
}

void moe(const Tensor& hidden, const MoeParameters& parameters, Tensor& residual,
         const ops::MoeExecutionHints& hints, MemoryArena& workspace, CudaExecutionView execution) {
    auto scope         = workspace.scope();
    const auto columns = hidden.ne[1];
    const auto storage = workspace.alloc_bytes(moe_workspace_bytes(parameters, columns, columns));
    MemoryArena scratch(storage);
    const ops::MoeWeights weights{parameters.control,
                                  {parameters.resident, parameters.activation_divisors},
                                  parameters.shared};
    ops::topk_softmax_swiglu_moe(hidden, parameters.profile, weights, ops::MoeEpilogue::AddResidual,
                                 residual, scratch, execution, hints);
}

} // namespace ninfer::models::qwen3_5::execution
