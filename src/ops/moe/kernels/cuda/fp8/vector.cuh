#pragma once

#include "ops/moe/kernels/cuda/fp8/mma.cuh"
#include "ops/moe/kernels/cuda/block_scaled/output.cuh"

namespace ninfer::ops::detail::moe_fp8 {

__device__ __forceinline__ bool includes(ExpertSubsetKind kind, const int* mask, int expert) {
    return kind == ExpertSubsetKind::All || mask[expert] == (kind == ExpertSubsetKind::MaskOnes);
}

__device__ __forceinline__ float dot_four(unsigned codes, float4 x) {
    const float2 lo = fp8_code2_to_float2(codes), hi = fp8_code2_to_float2(codes >> 16);
    return fmaf(hi.y, x.w, fmaf(hi.x, x.z, fmaf(lo.y, x.y, lo.x * x.x)));
}

__global__ void vector_gate(const __nv_bfloat16* x, ExpertMatrixDeviceView bank, const int* ids,
                            ExpertSubsetKind kind, const int* mask, float* activation) {
    const int assignment = blockIdx.y, expert = ids[assignment];
    if (!includes(kind, mask, expert)) { return; }
    const auto weight = resolve(bank, expert);
    const int row = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    float gate = 0, up = 0;
    for (int block = 0; block < kHidden / 128; ++block) {
        const int k     = block * 128 + lane * 4;
        const auto bits = load_vec<uint2>(x + std::int64_t(assignment / kTopK) * kHidden + k);
        const float2 a = bf16x2_bits_to_float2(bits.x), b = bf16x2_bits_to_float2(bits.y);
        const float4 input = make_float4(a.x, a.y, b.x, b.y);
        const unsigned g   = load_vec<unsigned>(weight.codes + std::int64_t(row) * kHidden + k);
        const unsigned u =
            load_vec<unsigned>(weight.codes + std::int64_t(row + kIntermediate) * kHidden + k);
        gate = fmaf(dot_four(g, input),
                    __bfloat162float(weight.scales[(row / 128) * (kHidden / 128) + block]), gate);
        up   = fmaf(dot_four(u, input),
                    __bfloat162float(
                      weight.scales[((row + kIntermediate) / 128) * (kHidden / 128) + block]),
                    up);
    }
    gate = warp_reduce_sum(gate);
    up   = warp_reduce_sum(up);
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
        for (int block = 0; block < kIntermediate / 128; ++block) {
            const int k = block * 128 + lane * 4;
            const float4 x =
                load_vec<float4>(activation + std::int64_t(assignment) * kIntermediate + k);
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
                const int row = blockIdx.x * Rows + r;
                const unsigned w =
                    load_vec<unsigned>(weight.codes + std::int64_t(row) * kIntermediate + k);
                sum[r] = fmaf(
                    dot_four(w, x),
                    __bfloat162float(weight.scales[(row / 128) * (kIntermediate / 128) + block]),
                    sum[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            sum[r] = warp_reduce_sum(sum[r]) * coefficients[assignment];
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
} // namespace ninfer::ops::detail::moe_fp8
