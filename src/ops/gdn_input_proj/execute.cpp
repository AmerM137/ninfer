#include "ops/gdn_input_proj/execute.h"
#include "ops/gdn_input_proj/fp8/kernels.h"
#include "ops/gdn_input_proj/nvfp4/kernels.h"
#include "ops/gdn_input_proj/projected_conv.h"
#include "ops/gdn_input_proj/q4_q5/kernels.h"
#include "ops/gdn_input_proj/q8/kernels.h"
#include "ops/gdn_input_proj/workspace.h"

#include <stdexcept>

namespace ninfer::ops::detail::gdn_input {

void launch_projection(const Tensor& x, const WeightOperands& weights, GdnProjectionOutputs out,
                       ProjectionPlan plan, const Scratch& scratch, cudaStream_t stream) {
    const auto& w = *weights.first;
    switch (weights.format) {
    case Format::Q4Q5:
        q4_q5_gdn_input_dispatch(x, w, *weights.second, out.qkv, out.z, stream);
        break;
    case Format::Q8:
        q8_gdn_input_dispatch(x, w, out.qkv, out.z, stream);
        break;
    case Format::Fp8:
        if (plan.compute == Compute::A8)
            fp8_gdn_input_a8_launch(x, w, out.qkv, out.z, scratch.fp8, stream);
        else if (x.ne[1] == 1)
            fp8_gdn_input_decode_launch(x, w, out.qkv, out.z, stream);
        else
            fp8_gdn_input_matrix_launch(x, w, out.qkv, out.z, stream);
        break;
    case Format::Nvfp4:
        if (plan.compute == Compute::A4)
            nvfp4_gdn_input_a4_launch(x, w, out.qkv, out.z, scratch.nvfp4, stream);
        else
            nvfp4_gdn_input_a16_launch(x, w, out.qkv, out.z, stream);
        break;
    }
}

void launch_snapshot(const Tensor& x, const WeightOperands& weights, const Tensor& conv,
                     GdnConvSnapshotState state, const Tensor& valid, GdnConvOutputs out,
                     Batch batch, ConvPlan plan, const Scratch& scratch, cudaStream_t stream) {
    const auto& w = *weights.first;
    switch (plan.kernel) {
    case ConvKernel::Q4Q5Fused:
        q4_q5_gdn_input_conv_snapshot_launch(x, w, *weights.second, conv, state.history, valid,
                                             state.source_slots, state.snapshot_bases, out.query,
                                             out.key, out.value, out.z, stream);
        return;
    case ConvKernel::Q8Decode:
        q8_gdn_input_decode_conv_snapshot_launch(x, w, conv, state.history, valid,
                                                 state.source_slots, state.snapshot_bases,
                                                 out.query, out.key, out.value, out.z, stream);
        return;
    case ConvKernel::Q8SlicedK:
        q8_gdn_input_splitk_conv_snapshot_launch(x, w, conv, state.history, valid,
                                                 state.source_slots, state.snapshot_bases,
                                                 out.query, out.key, out.value, out.z, stream);
        return;
    case ConvKernel::Q8Window:
    case ConvKernel::Q8Aggregate:
        q8_gdn_conv_n16384_k2560_snapshot_launch(
            x, w, conv, state.history, valid, state.source_slots, state.snapshot_bases, out.query,
            out.key, out.value, out.z, plan.kernel == ConvKernel::Q8Aggregate, stream);
        return;
    case ConvKernel::Fp8Fused:
        fp8_gdn_snapshot_fused_launch(x, w, conv, state.history, valid, state.source_slots,
                                      state.snapshot_bases, out.query, out.key, out.value, out.z,
                                      stream);
        return;
    case ConvKernel::Nvfp4Decode:
        nvfp4_gdn_snapshot_decode_launch(x, w, conv, state.history, valid, state.source_slots,
                                         state.snapshot_bases, out.query, out.key, out.value, out.z,
                                         stream);
        return;
    case ConvKernel::Nvfp4Window:
        nvfp4_gdn_snapshot_small_t_launch(x, w, conv, state.history, valid, state.source_slots,
                                          state.snapshot_bases, out.query, out.key, out.value,
                                          out.z, stream);
        return;
    case ConvKernel::Materialized:
        break;
    }
    const Tensor flat_x = x.view({weights.geometry.input_rows, batch.columns()});
    Tensor flat_z       = out.z.view({weights.geometry.value_rows, batch.columns()});
    launch_projection(flat_x, weights, {scratch.projected, flat_z}, plan.projection, scratch,
                      stream);
    const Tensor projected =
        scratch.projected.view({weights.geometry.channels(), batch.tokens, batch.rows});
    if (plan.post == ProjectedConvKernel::TokenParallel)
        gdn_projected_conv_parallel_snapshot_launch(projected, conv, state.history, valid,
                                                    state.source_slots, state.snapshot_bases,
                                                    out.query, out.key, out.value, stream);
    else
        gdn_projected_conv_snapshot_launch(projected, conv, state.history, valid,
                                           state.source_slots, state.snapshot_bases, out.query,
                                           out.key, out.value, stream);
}

void launch_record(const Tensor& x, const WeightOperands& weights, const Tensor& conv,
                   GdnConvSourceState state, const Tensor& valid, GdnConvOutputs out,
                   Tensor& record, Batch batch, ConvPlan plan, const Scratch& scratch,
                   cudaStream_t stream) {
    const auto& w = *weights.first;
    switch (plan.kernel) {
    case ConvKernel::Q4Q5Fused:
        q4_q5_gdn_input_conv_record_launch(x, w, *weights.second, conv, state.history, valid,
                                           state.source_slots, record, out.query, out.key,
                                           out.value, out.z, stream);
        return;
    case ConvKernel::Q8SlicedK:
        q8_gdn_input_splitk_conv_record_launch(x, w, conv, state.history, valid, state.source_slots,
                                               record, out.query, out.key, out.value, out.z,
                                               stream);
        return;
    case ConvKernel::Q8Window:
    case ConvKernel::Q8Aggregate:
        q8_gdn_conv_n16384_k2560_record_launch(x, w, conv, state.history, valid, state.source_slots,
                                               record, out.query, out.key, out.value, out.z,
                                               plan.kernel == ConvKernel::Q8Aggregate, stream);
        return;
    case ConvKernel::Fp8Fused:
        fp8_gdn_record_fused_launch(x, w, conv, state.history, valid, state.source_slots, record,
                                    out.query, out.key, out.value, out.z, stream);
        return;
    case ConvKernel::Nvfp4Window:
        nvfp4_gdn_record_small_t_launch(x, w, conv, state.history, valid, state.source_slots,
                                        record, out.query, out.key, out.value, out.z, stream);
        return;
    case ConvKernel::Q8Decode:
    case ConvKernel::Nvfp4Decode:
        throw std::logic_error("GDN record selected a one-token kernel");
    case ConvKernel::Materialized:
        break;
    }
    const Tensor flat_x = x.view({weights.geometry.input_rows, batch.columns()});
    const GdnProjectionOutputs projected{
        record.view({weights.geometry.channels(), batch.columns()}),
        out.z.view({weights.geometry.value_rows, batch.columns()})};
    launch_projection(flat_x, weights, projected, plan.projection, scratch, stream);
    if (plan.post == ProjectedConvKernel::TokenParallel)
        gdn_projected_conv_parallel_record_launch(record, conv, state.history, valid,
                                                  state.source_slots, out.query, out.key, out.value,
                                                  stream);
    else
        gdn_projected_conv_record_launch(record, conv, state.history, valid, state.source_slots,
                                         out.query, out.key, out.value, stream);
}

} // namespace ninfer::ops::detail::gdn_input
