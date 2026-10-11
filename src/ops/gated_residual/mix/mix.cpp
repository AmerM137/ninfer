#include "ninfer/ops/gated_residual.h"
#include "ops/gated_residual/mix/launch.h"
#include <stdexcept>

namespace ninfer::ops {
namespace mix = detail::hc::mix;

namespace {
void execute_mix(mix::Weights weights, const Tensor& normalized, MemoryArena& workspace,
                 Tensor& output, Tensor* injection, CudaExecutionView execution) {
    if (workspace.device() != execution.device()) {
        throw std::invalid_argument("hc_mix: workspace belongs to another device");
    }
    const auto o = mix::validate(weights, normalized, output, injection, execution);
    const auto p = mix::select_plan(weights.kind, o.columns);
    auto scope   = workspace.scope();
    const auto scratch =
        mix::allocate_workspace(workspace, mix::layout(weights.kind, p, o.columns));
    mix::validate_scratch(weights, normalized, output, injection, scratch.storage);
    mix::launch(weights.kind, o, p, scratch, execution);
}
} // namespace

void hc_mix(const Tensor& normalized, const HcMixWeights& weights, MemoryArena& workspace,
            Tensor& block_input, Tensor& injection, CudaExecutionView execution) {
    execute_mix({mix::Kind::Block, weights.down_injection, weights.up}, normalized, workspace,
                block_input, &injection, execution);
}

void hc_mix(const Tensor& normalized, const HcFinalMixWeights& weights, MemoryArena& workspace,
            Tensor& sample_hidden, CudaExecutionView execution) {
    execute_mix({mix::Kind::Final, weights.down, weights.up}, normalized, workspace, sample_hidden,
                nullptr, execution);
}

std::size_t hc_mix_workspace_capacity_bytes(const HcMixWeights& weights, std::int32_t first,
                                            std::int32_t last) {
    return mix::capacity({mix::Kind::Block, weights.down_injection, weights.up}, first, last);
}

std::size_t hc_mix_workspace_capacity_bytes(const HcFinalMixWeights& weights, std::int32_t first,
                                            std::int32_t last) {
    return mix::capacity({mix::Kind::Final, weights.down, weights.up}, first, last);
}
} // namespace ninfer::ops
