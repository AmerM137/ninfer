#pragma once

#include "ops/sparse_moe/plan.h"
#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/numeric/math.cuh"
#include "ops/common/numeric/convert.cuh"
#include "ops/common/cuda/memory.cuh"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/quant/integer.cuh"
#include <stdexcept>

namespace ninfer::ops::moe_detail {
using namespace common;

constexpr int kAdaptiveGateUpBlocks = 5 * kIntermediate;
constexpr int kAdaptiveDownBlocks   = 5 * (kHidden / 4);

struct Q4Codec {
    static constexpr int kGroupK             = 64;
    static constexpr bool kGateUpPackedWord8 = true;
};

struct Q5Codec : Q5SimtDecodeAtom {
    static constexpr int kGroupK       = 64;
    static constexpr bool kPackedWord8 = true;
};

struct Q6Codec : Q6SimtDecodeAtom {
    static constexpr int kGroupK       = 64;
    static constexpr bool kPackedWord8 = true;
};

struct Q8Codec : Q8ScalarDecodeAtom {
    static constexpr bool kGateUpSingleValuePerLane = true;
    static constexpr bool kGateUpPackedWord8        = false;
    static constexpr bool kPackedWord8              = false;
};

template <class Codec, int K>
__device__ __forceinline__ void dot_two_rows(const std::uint8_t* codes, const std::uint8_t* high,
                                             const std::uint8_t* scales, int row0, int row1,
                                             const __nv_bfloat16* x, int k_begin, int k_end,
                                             float& result0, float& result1) {
    constexpr int kGroups = K / Codec::kGroupK;
    const int lane        = static_cast<int>(threadIdx.x) & 31;
    float acc0            = 0.0f;
    float acc1            = 0.0f;
    const int first_group = k_begin / Codec::kGroupK;
    const int last_group  = k_end / Codec::kGroupK;
    if constexpr (Codec::kGateUpPackedWord8) {
        // Four adjacent Q4 groups form one 128-byte warp transaction. Each lane owns eight
        // consecutive K values, so one mantissa decode feeds eight FP32 FMAs instead of issuing
        // four scalar code-pair/decode iterations.
        const int lane_group    = lane >> 3;
        const int lane_in_group = lane & 7;
        for (int group_base = first_group; group_base < last_group; group_base += 4) {
            const int group           = group_base + lane_group;
            const std::int64_t index0 = static_cast<std::int64_t>(row0) * kGroups + group;
            const std::int64_t index1 = static_cast<std::int64_t>(row1) * kGroups + group;
            const std::uint32_t packed0 =
                *reinterpret_cast<const std::uint32_t*>(codes + index0 * 32 + lane_in_group * 4);
            const std::uint32_t packed1 =
                *reinterpret_cast<const std::uint32_t*>(codes + index1 * 32 + lane_in_group * 4);
            const auto scale0 = *reinterpret_cast<const std::uint16_t*>(scales + index0 * 2);
            const auto scale1 = *reinterpret_cast<const std::uint16_t*>(scales + index1 * 2);
            float weights0[8];
            float weights1[8];
            Q4SimtDecodeAtom::decode_eight(packed0, scale0, weights0);
            Q4SimtDecodeAtom::decode_eight(packed1, scale1, weights1);
            const uint4 input     = load_vec<uint4>(x + group * Codec::kGroupK + lane_in_group * 8);
            const float2 x0       = bf16x2_bits_to_float2(input.x);
            const float2 x1       = bf16x2_bits_to_float2(input.y);
            const float2 x2       = bf16x2_bits_to_float2(input.z);
            const float2 x3       = bf16x2_bits_to_float2(input.w);
            const float values[8] = {x0.x, x0.y, x1.x, x1.y, x2.x, x2.y, x3.x, x3.y};
#pragma unroll
            for (int item = 0; item < 8; ++item) {
                acc0 = fmaf(weights0[item], values[item], acc0);
                acc1 = fmaf(weights1[item], values[item], acc1);
            }
        }
    } else if constexpr (Codec::kGateUpSingleValuePerLane) {
        for (int group = first_group; group < last_group; ++group) {
            const std::int64_t index0 = static_cast<std::int64_t>(row0) * kGroups + group;
            const std::int64_t index1 = static_cast<std::int64_t>(row1) * kGroups + group;
            const float w0            = Codec::load_one(codes, scales, index0, lane);
            const float w1            = Codec::load_one(codes, scales, index1, lane);
            const float xv            = __bfloat162float(x[group * Codec::kGroupK + lane]);
            acc0                      = fmaf(w0, xv, acc0);
            acc1                      = fmaf(w1, xv, acc1);
        }
    } else if (lane < Codec::kGroupK / 2) {
        for (int group = first_group; group < last_group; ++group) {
            float w00, w01, w10, w11;
            Codec::load_pair(codes, high, scales, static_cast<std::int64_t>(row0) * kGroups + group,
                             lane, w00, w01);
            Codec::load_pair(codes, high, scales, static_cast<std::int64_t>(row1) * kGroups + group,
                             lane, w10, w11);
            const int k     = group * Codec::kGroupK + lane * 2;
            const float2 xv = __bfloat1622float2(load_vec<__nv_bfloat162>(x + k));
            acc0            = fmaf(w00, xv.x, acc0);
            acc0            = fmaf(w01, xv.y, acc0);
            acc1            = fmaf(w10, xv.x, acc1);
            acc1            = fmaf(w11, xv.y, acc1);
        }
    }
    result0 = warp_reduce_sum(acc0);
    result1 = warp_reduce_sum(acc1);
}

template <class RoutedCodec>
__global__ void gate_up_decode_kernel(
    const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids,
    const std::uint8_t* __restrict__ routed_codes, const std::uint8_t* __restrict__ routed_high,
    const std::uint8_t* __restrict__ routed_scales, const std::uint8_t* __restrict__ shared_codes,
    const std::uint8_t* __restrict__ shared_scales, float* __restrict__ act) {
    __shared__ __align__(16) __nv_bfloat16 x_shared[kHidden];
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (tid == 0) { pdl::trigger_dependents(); }
    if (tid < 256) { store_vec(x_shared + tid * 8, load_vec<uint4>(x + tid * 8)); }
    __syncthreads();

    const int j = static_cast<int>(blockIdx.x);
    float gate  = 0.0f;
    float up    = 0.0f;
    if (warp < kTopK) {
        pdl::wait_for_dependencies();
        const int expert   = ids[warp];
        const int row_base = expert * 1024;
        dot_two_rows<RoutedCodec, kHidden>(routed_codes, routed_high, routed_scales, row_base + j,
                                           row_base + kIntermediate + j, x_shared, 0, kHidden, gate,
                                           up);
    } else {
        dot_two_rows<Q8Codec, kHidden>(shared_codes, nullptr, shared_scales, j, kIntermediate + j,
                                       x_shared, 0, kHidden, gate, up);
    }
    if (lane == 0) { act[static_cast<std::int64_t>(warp) * kIntermediate + j] = silu(gate) * up; }
}

template <class RoutedCodec, int PathsPerBlock, bool Adaptive>
__global__ void gate_up_token_kernel(
    const __nv_bfloat16* __restrict__ x, const int* __restrict__ token_ids,
    const std::uint8_t* __restrict__ routed_codes, const std::uint8_t* __restrict__ routed_high,
    const std::uint8_t* __restrict__ routed_scales, const std::uint8_t* __restrict__ shared_codes,
    const std::uint8_t* __restrict__ shared_scales, float* __restrict__ token_activations,
    int tokens, const int* __restrict__ adaptive_route_jobs) {
    // Three path CTAs per token/output row expose enough blocks for the 170-SM target and keep the
    // heavier shared Q8 path from holding eight completed routed warps resident.
    static_assert(PathsPerBlock > 0 && (kTopK + 1) % PathsPerBlock == 0);
    constexpr int kPathBlocks = (kTopK + 1) / PathsPerBlock;
    __shared__ __align__(16) __nv_bfloat16 x_shared[kHidden];
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if constexpr (Adaptive) {
        if (*adaptive_route_jobs >= 0) { return; }
    }
    const int total_work = tokens * kPathBlocks * kIntermediate;
    const int first_work =
        Adaptive ? static_cast<int>(blockIdx.x)
                 : static_cast<int>(blockIdx.y) * kIntermediate + static_cast<int>(blockIdx.x);
    const int work_stride = Adaptive ? static_cast<int>(gridDim.x) : total_work;
    for (int work = first_work; work < total_work; work += work_stride) {
        const int token_path = work / kIntermediate;
        const int j          = work - token_path * kIntermediate;
        const int token      = token_path / kPathBlocks;
        const int path_block = token_path - token * kPathBlocks;
        const int path       = path_block * PathsPerBlock + warp;
        const auto* input    = x + static_cast<std::int64_t>(token) * kHidden;
        for (int vector = tid; vector < kHidden / 8; vector += static_cast<int>(blockDim.x)) {
            store_vec(x_shared + vector * 8, load_vec<uint4>(input + vector * 8));
        }
        __syncthreads();

        constexpr int kRouterPartitions = 4;
        const int activation_begin      = (token * (kTopK + 1) + path) * kIntermediate;
        // Routed paths need selection's ids. A shared path is independent only when its output lies
        // beyond the partial-score prefix that selection may still read from the lifetime-unioned
        // scratch.
        const bool must_wait_for_selection =
            path < kTopK || activation_begin < tokens * kRouterRows * kRouterPartitions;
        if constexpr (!Adaptive) {
            if (must_wait_for_selection) { pdl::wait_for_dependencies(); }
        }
        float gate = 0.0f;
        float up   = 0.0f;
        if (path < kTopK) {
            const int expert   = token_ids[token * kTopK + path];
            const int row_base = expert * (2 * kIntermediate);
            dot_two_rows<RoutedCodec, kHidden>(routed_codes, routed_high, routed_scales,
                                               row_base + j, row_base + kIntermediate + j, x_shared,
                                               0, kHidden, gate, up);
        } else {
            dot_two_rows<Q8Codec, kHidden>(shared_codes, nullptr, shared_scales, j,
                                           kIntermediate + j, x_shared, 0, kHidden, gate, up);
        }
        if (lane == 0) {
            token_activations[(static_cast<std::int64_t>(token) * (kTopK + 1) + path) *
                                  kIntermediate +
                              j] = silu(gate) * up;
        }
        if constexpr (Adaptive) { __syncthreads(); }
    }
}

template <class Codec, int Rows>
__device__ __forceinline__ void dot_fp32_rows(const std::uint8_t* codes, const std::uint8_t* high,
                                              const std::uint8_t* scales, int row_base,
                                              const float* x, int first_group, int last_group,
                                              float (&result)[Rows]) {
    constexpr int kGroups = kIntermediate / Codec::kGroupK;
    const int lane        = static_cast<int>(threadIdx.x) & 31;
    float acc[Rows];
#pragma unroll
    for (int row = 0; row < Rows; ++row) { acc[row] = 0.0f; }
    if constexpr (Codec::kPackedWord8) {
        // Q5/Q6 use the same eight-value lane ownership as gate/up. The high plane and FP16 scale
        // are decoded exactly from their registered row-split codec before FP32 accumulation.
        const int lane_group    = lane >> 3;
        const int lane_in_group = lane & 7;
        for (int group_base = first_group; group_base < last_group; group_base += 4) {
            const int group = group_base + lane_group;
            const float4 x0 = load_vec<float4>(x + group * Codec::kGroupK + lane_in_group * 8);
            const float4 x1 = load_vec<float4>(x + group * Codec::kGroupK + lane_in_group * 8 + 4);
            const float values[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
            for (int row = 0; row < Rows; ++row) {
                const std::int64_t group_index =
                    static_cast<std::int64_t>(row_base + row) * kGroups + group;
                float weights[8];
                Codec::load_eight(codes, high, scales, group_index, lane_in_group, weights);
#pragma unroll
                for (int item = 0; item < 8; ++item) {
                    acc[row] = fmaf(weights[item], values[item], acc[row]);
                }
            }
        }
    } else if (lane < Codec::kGroupK / 2) {
        for (int group = first_group; group < last_group; ++group) {
            const int k     = group * Codec::kGroupK + lane * 2;
            const float2 xv = load_vec<float2>(x + k);
#pragma unroll
            for (int row = 0; row < Rows; ++row) {
                float w0, w1;
                Codec::load_pair(codes, high, scales,
                                 static_cast<std::int64_t>(row_base + row) * kGroups + group, lane,
                                 w0, w1);
                acc[row] = fmaf(w0, xv.x, acc[row]);
                acc[row] = fmaf(w1, xv.y, acc[row]);
            }
        }
    }
#pragma unroll
    for (int row = 0; row < Rows; ++row) { result[row] = warp_reduce_sum(acc[row]); }
}

template <class RoutedCodec, int Rows>
__global__ void down_decode_kernel(
    const int* __restrict__ ids, const float* __restrict__ alpha,
    const float* __restrict__ shared_scale, const float* __restrict__ act,
    const std::uint8_t* __restrict__ routed_codes, const std::uint8_t* __restrict__ routed_high,
    const std::uint8_t* __restrict__ routed_scales, const std::uint8_t* __restrict__ shared_codes,
    const std::uint8_t* __restrict__ shared_scales, __nv_bfloat16* __restrict__ destination,
    const char* __restrict__ prefetch_data, unsigned long long prefetch_bytes) {
    __shared__ float paths[kTopK + 1][Rows];
    pdl::wait_for_dependencies();
    const int warp     = static_cast<int>(threadIdx.x) >> 5;
    const int lane     = static_cast<int>(threadIdx.x) & 31;
    const int row_base = static_cast<int>(blockIdx.x) * Rows;
    if (warp < kTopK) {
        const int expert = ids[warp];
        float dot[Rows];
        dot_fp32_rows<RoutedCodec, Rows>(routed_codes, routed_high, routed_scales,
                                         expert * kHidden + row_base,
                                         act + static_cast<std::int64_t>(warp) * kIntermediate, 0,
                                         kIntermediate / RoutedCodec::kGroupK, dot);
        if (lane == 0) {
#pragma unroll
            for (int row = 0; row < Rows; ++row) { paths[warp][row] = alpha[warp] * dot[row]; }
        }
    } else {
        float dot[Rows];
        dot_fp32_rows<Q8Codec, Rows>(shared_codes, nullptr, shared_scales, row_base,
                                     act + static_cast<std::int64_t>(kTopK) * kIntermediate, 0,
                                     kIntermediate / Q8Codec::kGroupK, dot);
        if (lane == 0) {
#pragma unroll
            for (int row = 0; row < Rows; ++row) { paths[kTopK][row] = *shared_scale * dot[row]; }
        }
    }
    if (prefetch_data != nullptr) {
        // Fire-and-forget L2 warmup of the next consumer's weight codes,
        // issued BEFORE the block barrier so it hides behind the slowest warp instead
        // of extending the grid tail. Pure cache hint; no values, no addition order.
        const unsigned long long offset =
            (static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x) * 128ull;
        if (offset < prefetch_bytes) {
            asm volatile("prefetch.global.L2 [%0];" ::"l"(prefetch_data + offset));
        }
    }
    __syncthreads();
    if (warp == 0 && lane < Rows) {
        float value = __bfloat162float(destination[row_base + lane]);
#pragma unroll
        for (int path = 0; path < kTopK + 1; ++path) { value += paths[path][lane]; }
        destination[row_base + lane] = __float2bfloat16_rn(value);
    }
}

template <class RoutedCodec, int Rows, bool Adaptive>
__global__ void down_token_kernel(
    const int* __restrict__ token_ids, const float* __restrict__ token_alpha,
    const float* __restrict__ shared_scale, const float* __restrict__ token_activations,
    const std::uint8_t* __restrict__ routed_codes, const std::uint8_t* __restrict__ routed_high,
    const std::uint8_t* __restrict__ routed_scales, const std::uint8_t* __restrict__ shared_codes,
    const std::uint8_t* __restrict__ shared_scales, __nv_bfloat16* __restrict__ destination,
    int tokens, const int* __restrict__ adaptive_route_jobs) {
    // Token is a grid dimension rather than an in-CTA serial loop. Rows lets one routed-weight
    // stream serve adjacent outputs while retaining the deterministic rank-order FP32 epilogue.
    __shared__ float paths[kTopK + 1][Rows];
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    if constexpr (Adaptive) {
        if (*adaptive_route_jobs >= 0) { return; }
    }
    constexpr int kRowBlocks = kHidden / Rows;
    const int total_work     = tokens * kRowBlocks;
    const int first_work =
        Adaptive ? static_cast<int>(blockIdx.x)
                 : static_cast<int>(blockIdx.y) * kRowBlocks + static_cast<int>(blockIdx.x);
    const int work_stride = Adaptive ? static_cast<int>(gridDim.x) : total_work;
    for (int work = first_work; work < total_work; work += work_stride) {
        const int token     = work / kRowBlocks;
        const int row_block = work - token * kRowBlocks;
        const int row_base  = row_block * Rows;
        const float* act =
            token_activations + static_cast<std::int64_t>(token) * (kTopK + 1) * kIntermediate;
        if (warp < kTopK) {
            const int expert = token_ids[token * kTopK + warp];
            float dot[Rows];
            dot_fp32_rows<RoutedCodec, Rows>(routed_codes, routed_high, routed_scales,
                                             expert * kHidden + row_base,
                                             act + static_cast<std::int64_t>(warp) * kIntermediate,
                                             0, kIntermediate / RoutedCodec::kGroupK, dot);
            if (lane == 0) {
#pragma unroll
                for (int row = 0; row < Rows; ++row) {
                    paths[warp][row] = token_alpha[token * kTopK + warp] * dot[row];
                }
            }
        } else {
            float dot[Rows];
            dot_fp32_rows<Q8Codec, Rows>(shared_codes, nullptr, shared_scales, row_base,
                                         act + static_cast<std::int64_t>(kTopK) * kIntermediate, 0,
                                         kIntermediate / Q8Codec::kGroupK, dot);
            if (lane == 0) {
#pragma unroll
                for (int row = 0; row < Rows; ++row) {
                    paths[kTopK][row] = shared_scale[token] * dot[row];
                }
            }
        }
        __syncthreads();
        if (warp == 0 && lane < Rows) {
            __nv_bfloat16* output =
                destination + static_cast<std::int64_t>(token) * kHidden + row_base + lane;
            float value = __bfloat162float(*output);
#pragma unroll
            for (int path = 0; path < kTopK + 1; ++path) { value += paths[path][lane]; }
            *output = __float2bfloat16_rn(value);
        }
        if constexpr (Adaptive) { __syncthreads(); }
    }
}

} // namespace ninfer::ops::moe_detail
