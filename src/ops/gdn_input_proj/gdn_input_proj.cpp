#include "ninfer/ops/gdn_input_proj.h"
#include "ops/gdn_input_proj/execute.h"
#include "ops/gdn_input_proj/workspace.h"

#include <array>

namespace ninfer::ops {
namespace {
namespace input = detail::gdn_input;

// Scratch is a scoped tail allocation. Live caller tensors may occupy an earlier part of
// the same arena. The zero-scratch path never constructs an empty borrowed DeviceArena.
template <class Validate, class Launch>
void execute_with_scratch(const input::WeightOperands& weights, input::ProjectionPlan plan,
                          std::int32_t columns, bool materialized_snapshot,
                          WorkspaceArena& workspace, Validate&& validate, Launch&& launch) {
    auto scope       = workspace.scope();
    const auto bytes = input::workspace_bytes(weights, plan, columns, materialized_snapshot);
    if (bytes == 0) {
        validate(DeviceSpan{});
        launch(input::Scratch{});
        return;
    }
    const DeviceSpan backing = workspace.alloc_bytes(bytes, 256);
    validate(backing);
    WorkspaceArena local(backing);
    launch(input::allocate_workspace(local, weights, plan, columns, materialized_snapshot));
}
} // namespace

void gdn_input_proj(const Tensor& x, const ProjectionWeights& weights, GdnProjectionOutputs out,
                    WorkspaceArena& workspace, cudaStream_t stream) {
    const auto operands = input::describe_weights(weights);
    input::validate_weights(operands);
    input::validate_projection(x, operands, out);
    const auto plan = input::resolve_projection_plan(operands, x.ne[1]);
    const std::array<const Tensor*, 3> tensors{&x, &out.qkv, &out.z};
    execute_with_scratch(
        operands, plan, x.ne[1], false, workspace,
        [&](DeviceSpan scratch) { input::validate_disjoint(operands, tensors, scratch); },
        [&](const input::Scratch& scratch) {
            input::launch_projection(x, operands, out, plan, scratch, stream);
        });
}

std::size_t gdn_input_proj_workspace_capacity_bytes(const ProjectionWeights& weights,
                                                    std::int32_t first, std::int32_t last) {
    return input::projection_capacity(input::describe_weights(weights), first, last);
}

void gdn_input_proj_conv_snapshot(const Tensor& x, const ProjectionWeights& weights,
                                  const Tensor& conv_weight, GdnConvSnapshotState state,
                                  const Tensor& valid, GdnConvOutputs out,
                                  WorkspaceArena& workspace, cudaStream_t stream) {
    const auto operands = input::describe_weights(weights);
    input::validate_weights(operands);
    const auto batch =
        input::validate_convolution(x, operands, conv_weight, state.history, state.source_slots,
                                    valid, out, input::Publication::Snapshot, state.snapshot_bases);
    const auto plan = input::resolve_conv_plan(operands, batch);
    const std::array<const Tensor*, 10> tensors{
        &x,     &conv_weight, &state.history, &state.source_slots, &state.snapshot_bases,
        &valid, &out.query,   &out.key,       &out.value,          &out.z};
    execute_with_scratch(
        operands, plan.projection, batch.columns(), plan.materialized(), workspace,
        [&](DeviceSpan scratch) {
            input::validate_disjoint(operands, tensors, scratch, &state.source_slots,
                                     &state.snapshot_bases);
        },
        [&](const input::Scratch& scratch) {
            input::launch_snapshot(x, operands, conv_weight, state, valid, out, batch, plan,
                                   scratch, stream);
        });
}

std::size_t gdn_input_proj_conv_snapshot_workspace_capacity_bytes(const ProjectionWeights& weights,
                                                                  std::int32_t batch,
                                                                  std::int32_t first,
                                                                  std::int32_t last) {
    return input::convolution_capacity(input::describe_weights(weights),
                                       input::Publication::Snapshot, batch, first, last);
}

void gdn_input_proj_conv_record(const Tensor& x, const ProjectionWeights& weights,
                                const Tensor& conv_weight, GdnConvSourceState state,
                                const Tensor& valid, GdnConvOutputs out, Tensor& record,
                                WorkspaceArena& workspace, cudaStream_t stream) {
    const auto operands = input::describe_weights(weights);
    input::validate_weights(operands);
    const auto batch =
        input::validate_convolution(x, operands, conv_weight, state.history, state.source_slots,
                                    valid, out, input::Publication::Record, record);
    const auto plan = input::resolve_conv_plan(operands, batch);
    const std::array<const Tensor*, 10> tensors{
        &x,      &conv_weight, &state.history, &state.source_slots, &valid,
        &record, &out.query,   &out.key,       &out.value,          &out.z};
    execute_with_scratch(
        operands, plan.projection, batch.columns(), false, workspace,
        [&](DeviceSpan scratch) { input::validate_disjoint(operands, tensors, scratch); },
        [&](const input::Scratch& scratch) {
            input::launch_record(x, operands, conv_weight, state, valid, out, record, batch, plan,
                                 scratch, stream);
        });
}

std::size_t gdn_input_proj_conv_record_workspace_capacity_bytes(const ProjectionWeights& weights,
                                                                std::int32_t batch,
                                                                std::int32_t first,
                                                                std::int32_t last) {
    return input::convolution_capacity(input::describe_weights(weights), input::Publication::Record,
                                       batch, first, last);
}

} // namespace ninfer::ops
