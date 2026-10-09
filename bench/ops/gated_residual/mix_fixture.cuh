#pragma once

#include "options.h"
#include "ninfer/ops/gated_residual.h"

namespace ninfer::bench::hc {
inline Weight bf16_weight(const DeviceBuffer& buffer, int n, int k) {
    Weight w;
    w.payload = w.qdata = buffer.p;
    w.payload_bytes     = buffer.bytes;
    w.qtype             = QType::BF16;
    w.layout            = QuantLayout::Contiguous;
    w.ndim              = 2;
    w.n = w.shape[0] = w.padded_shape[0] = n;
    w.k = w.shape[1] = w.padded_shape[1] = k;
    return w;
}

struct MixFixture {
    MixFixture(const Options& options, int tokens, DeviceExecutionView execution)
        : op(options.op), m(columns(tokens, options.batch)),
          input(make_bf16(std::size_t(kWidth) * kStreams * m, 6201, -1.F, 1.F)),
          down(make_bf16(
              std::size_t(kRank + (op == Op::BlockMix ? kStreams : 0)) * kWidth * kStreams, 6203,
              -std::sqrt(3.F / (kWidth * kStreams)), std::sqrt(3.F / (kWidth * kStreams)))),
          up(make_bf16(std::size_t(kWidth) * kStreams * kRank, 6207, -4.F * std::sqrt(3.F / kRank),
                       4.F * std::sqrt(3.F / kRank))),
          output(make_zeros(std::size_t(kWidth) * m * sizeof(__nv_bfloat16))),
          injection(op == Op::BlockMix ? make_zeros(std::size_t(kStreams) * m * sizeof(float))
                                       : DeviceBuffer{}),
          wd(bf16_weight(down, kRank + (op == Op::BlockMix ? kStreams : 0), kWidth * kStreams)),
          wu(bf16_weight(up, kWidth * kStreams, kRank)),
          z(input.p, DType::BF16, {kWidth, kStreams, m}), x(output.p, DType::BF16, {kWidth, m}),
          j(injection.p, DType::FP32, {kStreams, m}),
          capacity(op == Op::BlockMix ? ops::hc_mix_workspace_capacity_bytes(
                                            ops::HcMixWeights{wd, wu}, m, m, execution)
                                      : ops::hc_mix_workspace_capacity_bytes(
                                            ops::HcFinalMixWeights{wd, wu}, m, m, execution)),
          arena(capacity) {}

    void launch(DeviceExecutionView execution) {
        if (op == Op::BlockMix)
            ops::hc_mix(z, ops::HcMixWeights{wd, wu}, arena, x, j, execution);
        else
            ops::hc_mix(z, ops::HcFinalMixWeights{wd, wu}, arena, x, execution);
    }

    void restore(cudaStream_t) {}

    std::size_t workspace_capacity() const { return capacity; }

    std::size_t workspace_peak() const { return arena.peak_used(); }

    Op op;
    int m;
    DeviceBuffer input, down, up, output, injection;
    Weight wd, wu;
    Tensor z, x, j;
    std::size_t capacity;
    WorkspaceArena arena;
};
} // namespace ninfer::bench::hc
