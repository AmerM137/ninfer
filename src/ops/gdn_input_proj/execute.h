#pragma once

#include "ops/gdn_input_proj/plan.h"

namespace ninfer::ops::detail::gdn_input {

struct Scratch;

void launch_projection(const Tensor& x, const WeightOperands& weights, GdnProjectionOutputs outputs,
                       ProjectionPlan plan, const Scratch& scratch, cudaStream_t stream);
void launch_snapshot(const Tensor& x, const WeightOperands& weights, const Tensor& conv_weight,
                     GdnConvSnapshotState state, const Tensor& valid, GdnConvOutputs outputs,
                     Batch batch, ConvPlan plan, const Scratch& scratch, cudaStream_t stream);
void launch_record(const Tensor& x, const WeightOperands& weights, const Tensor& conv_weight,
                   GdnConvSourceState state, const Tensor& valid, GdnConvOutputs outputs,
                   Tensor& record, Batch batch, ConvPlan plan, const Scratch& scratch,
                   cudaStream_t stream);

} // namespace ninfer::ops::detail::gdn_input
