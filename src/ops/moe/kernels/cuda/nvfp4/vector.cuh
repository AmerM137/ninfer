#pragma once

#include "ops/moe/kernels/cuda/nvfp4/mma.cuh"
#include "ops/moe/kernels/cuda/block_scaled/output.cuh"

namespace ninfer::ops::detail::moe_nvfp4 {
using namespace moe_block_scaled;

__device__ __forceinline__ bool includes(ExpertSubsetKind kind, const int* mask, int expert) {
    return kind == ExpertSubsetKind::All || mask[expert] == (kind == ExpertSubsetKind::MaskOnes);
}

__device__ __forceinline__ void expand_group(Matrix weight, int row, int group, int k,
                                             float* values) {
    const auto decoded =
        nvfp4_dequant_f16x16(weight.codes + std::int64_t(row) * (k / 2) + group * 8,
                             weight.scales[nvfp4_scale_byte_offset(row, group, k)]);
    const int words[8] = {decoded.lo.x, decoded.lo.y, decoded.lo.z, decoded.lo.w,
                          decoded.hi.x, decoded.hi.y, decoded.hi.z, decoded.hi.w};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const auto pair   = __half22float2(*reinterpret_cast<const __half2*>(&words[i]));
        values[2 * i]     = pair.x;
        values[2 * i + 1] = pair.y;
    }
}

__global__ void vector_gate(const __nv_bfloat16* x, ExpertMatrixDeviceView bank, const int* ids,
                            ExpertSubsetKind kind, const int* mask, float* activation) {
    const int assignment = blockIdx.y, expert = ids[assignment];
    if (!includes(kind, mask, expert)) { return; }
    const auto weight = resolve(bank, expert);
    const int row = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const auto* input = x + std::int64_t(assignment / kTopK) * kHidden;
    float gate = 0, up = 0;
#pragma unroll
    for (int group = lane; group < kHidden / 16; group += 32) {
        float g[16], u[16];
        expand_group(weight, row, group, kHidden, g);
        expand_group(weight, row + kIntermediate, group, kHidden, u);
        const auto x0           = load_vec<uint4>(input + group * 16);
        const auto x1           = load_vec<uint4>(input + group * 16 + 8);
        const unsigned words[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
            const float2 v = bf16x2_bits_to_float2(words[pair]);
            gate           = fmaf(g[pair * 2], v.x, gate);
            gate           = fmaf(g[pair * 2 + 1], v.y, gate);
            up             = fmaf(u[pair * 2], v.x, up);
            up             = fmaf(u[pair * 2 + 1], v.y, up);
        }
    }
    gate = warp_reduce_sum(gate) / weight.divisor;
    up   = warp_reduce_sum(up) / weight.divisor;
    if (!lane) { activation[std::int64_t(assignment) * kIntermediate + row] = silu(gate) * up; }
}

template <int Rows, class Output>
__global__ void vector_down(const float* activation, ExpertMatrixDeviceView bank, const int* ids,
                            const float* coefficients, ExpertSubsetKind kind, const int* mask,
                            Output output) {
    __shared__ float values[kTopK][Rows];
    const int token = blockIdx.y, route = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int assignment = token * kTopK + route, expert = ids[assignment];
    float sum[Rows] = {};
    if (includes(kind, mask, expert)) {
        const auto weight = resolve(bank, expert);
        for (int group = lane; group < kIntermediate / 16; group += 32) {
            float x[16];
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                x[k] = activation[std::int64_t(assignment) * kIntermediate + group * 16 + k];
            }
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
                float w[16];
                expand_group(weight, blockIdx.x * Rows + r, group, kIntermediate, w);
#pragma unroll
                for (int k = 0; k < 16; ++k) { sum[r] = fmaf(x[k], w[k], sum[r]); }
            }
        }
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            sum[r] = warp_reduce_sum(sum[r]) * (coefficients[assignment] / weight.divisor);
        }
    }
    if (!lane) {
#pragma unroll
        for (int r = 0; r < Rows; ++r) { values[route][r] = sum[r]; }
    }
    __syncthreads();
    if (threadIdx.x < Rows) {
        float result = 0;
#pragma unroll
        for (int j = 0; j < kTopK; ++j) { result += values[j][threadIdx.x]; }
        const int i = token * kHidden + blockIdx.x * Rows + threadIdx.x;
        output.store(i, output.initial(i) + result);
    }
}

} // namespace ninfer::ops::detail::moe_nvfp4
