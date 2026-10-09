#pragma once
#include "options.h"
#include "ninfer/ops/gated_residual.h"

namespace ninfer::bench::hc {
inline constexpr float kEpsilon = 1.e-6F;

struct Fixture {
    Fixture(const Options& options, int tokens)
        : op(options.op), in_place(options.in_place), m(columns(tokens, options.batch)),
          input(make_bf16(std::size_t(kWidth) * m * (op == Op::ExpandNorm ? 1 : kStreams), 6101,
                          -1.F, 1.F)),
          block(combines(op) ? make_bf16(std::size_t(kWidth) * m, 6103, -0.75F, 0.75F)
                             : DeviceBuffer{}),
          weight(op != Op::Combine ? make_bf16(kWidth * kStreams, 6107, -0.25F, 0.25F)
                                   : DeviceBuffer{}),
          injection(combines(op) && op != Op::AddNorm ? make_f32(std::size_t(kStreams) * m, 6109,
                                                                 op == Op::AddThenNorm ? 1.F : 0.F,
                                                                 op == Op::AddThenNorm ? 1.F : 2.F)
                                                      : DeviceBuffer{}),
          backup(in_place ? make_zeros(input.bytes) : DeviceBuffer{}),
          residual(op != Op::GroupedNorm && !in_place
                       ? make_zeros(std::size_t(kWidth) * kStreams * m * sizeof(__nv_bfloat16))
                       : DeviceBuffer{}),
          normalized(op != Op::Combine
                         ? make_zeros(std::size_t(kWidth) * kStreams * m * sizeof(__nv_bfloat16))
                         : DeviceBuffer{}),
          h(input.p, DType::BF16, {kWidth, kStreams, m}), b(block.p, DType::BF16, {kWidth, m}),
          w(weight.p, DType::BF16, {kWidth, kStreams}), j(injection.p, DType::FP32, {kStreams, m}),
          r(in_place ? input.p : residual.p, DType::BF16, {kWidth, kStreams, m}),
          z(normalized.p, DType::BF16, {kWidth, kStreams, m}) {
        if (op == Op::ExpandNorm) h = Tensor(input.p, DType::BF16, {kWidth, m});
        if (in_place)
            CUDA_CHECK(cudaMemcpy(backup.p, input.p, input.bytes, cudaMemcpyDeviceToDevice));
    }

    void restore(cudaStream_t stream) {
        if (in_place)
            CUDA_CHECK(
                cudaMemcpyAsync(input.p, backup.p, input.bytes, cudaMemcpyDeviceToDevice, stream));
    }

    std::size_t workspace_capacity() const { return 0; }

    std::size_t workspace_peak() const { return 0; }

    void launch(DeviceExecutionView e) {
        switch (op) {
        case Op::ExpandNorm:
            ops::hc_expand_norm(h, w, kEpsilon, r, z, e);
            break;
        case Op::GroupedNorm:
            ops::hc_grouped_rmsnorm(h, w, kEpsilon, z, e);
            break;
        case Op::Combine:
            ops::hc_combine(h, b, j, r, e);
            break;
        case Op::CombineNorm:
            ops::hc_combine_norm(h, b, j, w, kEpsilon, r, z, e);
            break;
        case Op::AddNorm:
            ops::hc_combine_norm(h, b, w, kEpsilon, r, z, e);
            break;
        case Op::CombineThenNorm:
        case Op::AddThenNorm:
            ops::hc_combine(h, b, j, r, e);
            ops::hc_grouped_rmsnorm(r, w, kEpsilon, z, e);
            break;
        case Op::BlockMix:
        case Op::FinalMix:
            throw std::logic_error("Mix requires its projection fixture");
        }
    }

    Op op;
    bool in_place;
    int m;
    DeviceBuffer input, block, weight, injection, backup, residual, normalized;
    Tensor h, b, w, j, r, z;
};
} // namespace ninfer::bench::hc
