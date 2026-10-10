#include "ninfer/ops/linear_split.h"
#include "ops/linear/split/dispatch.h"

#include <stdexcept>

namespace ninfer::ops {
namespace {
namespace split = detail::linear_split;

void execute(const Tensor& x, split::Weights weights, std::span<const Tensor> outputs,
             WorkspaceArena* workspace, cudaStream_t stream) {
    const auto& shape = split::select_split_shape(weights);
    split::validate_operands(x, weights, outputs, shape);
    const auto bytes = shape.capacity ? shape.capacity(weights.policy, x.ne[1], x.ne[1]) : 0;
    if (!bytes) {
        shape.launch(x, weights, outputs, nullptr, stream);
        return;
    }
    if (!workspace) throw std::logic_error("linear_split: A16 route requested workspace");
    auto scope         = workspace->scope();
    const auto backing = workspace->alloc_bytes(bytes, 256);
    split::validate_scratch(x, weights, outputs, backing);
    WorkspaceArena local(backing);
    shape.launch(x, weights, outputs, &local, stream);
}
} // namespace

void linear_split(const Tensor& x, const ProjectionWeights& weights,
                  std::span<const Tensor> outputs, WorkspaceArena& workspace, cudaStream_t stream) {
    execute(x, split::describe_weights(weights), outputs, &workspace, stream);
}

void linear_split(const Tensor& x, const Weight& weight, std::span<const Tensor> outputs,
                  cudaStream_t stream) {
    execute(x, {weight}, outputs, nullptr, stream);
}

std::size_t linear_split_workspace_capacity_bytes(const ProjectionWeights& weights,
                                                  std::span<const std::int32_t> output_rows,
                                                  std::int32_t first, std::int32_t last) {
    const auto native = split::describe_weights(weights);
    const auto& shape = split::select_split_shape(native);
    split::validate_segments(output_rows, shape);
    split::validate_interval(first, last);
    return shape.capacity ? shape.capacity(native.policy, first, last) : 0;
}
} // namespace ninfer::ops
