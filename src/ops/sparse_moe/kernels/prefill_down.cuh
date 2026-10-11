#pragma once

#include "ops/sparse_moe/prefill/schedule.h"
#include "ops/common/numeric/math.cuh"
#include "ops/common/numeric/convert.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/cuda/mma.cuh"
#include "ops/common/quant/integer.cuh"

namespace ninfer::ops::moe_detail {
using namespace common;

namespace {

struct Q5DownMma {
    static constexpr int kCodeBytes    = Q5RowSplitStorage::kCodeBytesPerGroup;
    static constexpr int kHighBytes    = Q5RowSplitStorage::kHighBytesPerGroup;
    static constexpr int kHighPerChunk = Q5RowSplitStorage::kHighBytesPerChunk;

    __device__ static __forceinline__ void
    decode_eight(unsigned word, const std::uint8_t* high_chunk, float scale, unsigned (&out)[4]) {
        Q5MmaDecodeAtom::decode_eight(word, high_chunk, scale, out);
    }
};

struct Q6DownMma {
    static constexpr int kCodeBytes    = Q6RowSplitStorage::kCodeBytesPerGroup;
    static constexpr int kHighBytes    = Q6RowSplitStorage::kHighBytesPerGroup;
    static constexpr int kHighPerChunk = Q6RowSplitStorage::kHighBytesPerChunk;

    __device__ static __forceinline__ void
    decode_eight(unsigned word, const std::uint8_t* high_chunk, float scale, unsigned (&out)[4]) {
        Q6MmaDecodeAtom::decode_eight(word, high_chunk, scale, out);
    }
};

template <class Codec, int ExpertWarps, int ExpertBN>
__global__ __launch_bounds__(ExpertWarps * 32, 3) void sparse_moe_prefill_qx_down_kernel(
    const __nv_bfloat16* __restrict__ activation, const int* __restrict__ expert_offsets,
    const int* __restrict__ route_job_experts, const int* __restrict__ route_job_columns,
    const int* __restrict__ route_job_count, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high, const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ output) {
    constexpr int ExpertThreads = ExpertWarps * 32;
    constexpr int GroupsPerRow  = kIntermediate / 64;
    __shared__ __align__(16) __nv_bfloat16 As[kExpertBM * kExpertBK];
    __shared__ __align__(16) __nv_bfloat16 Bs[kExpertStages][ExpertBN * kExpertBK];
    __shared__ __align__(16) std::uint8_t Cr[kExpertStages][kExpertBM * 32];
    __shared__ __align__(16) std::uint8_t Hr[kExpertStages][kExpertBM * Codec::kHighBytes];
    __shared__ __align__(16) std::uint8_t Sr[kExpertBM * GroupsPerRow * 2];

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int a_mat          = lane >> 3;
    const int a_rin          = lane & 7;
    const int a_rowoff       = a_rin + ((a_mat & 1) << 3);
    const int a_coloff       = (a_mat >> 1) << 3;
    const int b_rin          = lane & 7;
    const int b_koff         = ((lane >> 3) & 1) << 3;
    const int gid            = lane >> 2;
    const int lid            = lane & 3;
    constexpr int row_blocks = kHidden / kExpertBM;
    const int total_work     = *route_job_count * row_blocks;
    for (int work = static_cast<int>(blockIdx.x); work < total_work;
         work += static_cast<int>(gridDim.x)) {
        const int route_job   = work / row_blocks;
        const int row_block   = work - route_job * row_blocks;
        const int expert      = route_job_experts[route_job];
        const int row0        = row_block * kExpertBM;
        const int begin       = expert_offsets[expert];
        const int count       = expert_offsets[expert + 1] - begin;
        const int column_base = route_job_columns[route_job];
        const int cols        = count - column_base < ExpertBN ? count - column_base : ExpertBN;
        float acc[4][4]       = {};

        auto stage_scales = [&] {
            for (int row = tid; row < kExpertBM; row += ExpertThreads) {
                const int global_row  = expert * kHidden + row0 + row;
                const std::int64_t gi = static_cast<std::int64_t>(global_row) * GroupsPerRow;
                cp_async<16, Cache::cg>(&Sr[row * GroupsPerRow * 2], &scales[gi * 2]);
            }
        };

        auto stage_inputs = [&](int stage, int kt) {
            const int k0 = kt * kExpertBK;
            for (int item = tid; item < ExpertBN * (kExpertBK / 8); item += ExpertThreads) {
                const int col        = item / (kExpertBK / 8);
                const int k8         = item - col * (kExpertBK / 8);
                const int packed_col = begin + column_base + col;
                auto* dst = &Bs[stage][col * kExpertBK + common::swizzle_xor_64(col, k8 * 8)];
                const auto* src =
                    activation +
                    static_cast<std::int64_t>(col < cols ? packed_col : begin) * kIntermediate +
                    k0 + k8 * 8;
                cp_async_zfill<16, Cache::cg>(dst, src, col < cols ? 16 : 0);
            }

            for (int item = tid; item < kExpertBM * 2; item += ExpertThreads) {
                const int row         = item >> 1;
                const int half        = item & 1;
                const int global_row  = expert * kHidden + row0 + row;
                const std::int64_t gi = static_cast<std::int64_t>(global_row) * GroupsPerRow + kt;
                cp_async<16, Cache::cg>(&Cr[stage][row * 32 + half * 16],
                                        &codes[gi * 32 + half * 16]);
            }
            for (int row = tid; row < kExpertBM; row += ExpertThreads) {
                const int global_row  = expert * kHidden + row0 + row;
                const std::int64_t gi = static_cast<std::int64_t>(global_row) * GroupsPerRow + kt;
                if constexpr (Codec::kHighBytes == 8) {
                    cp_async<8>(&Hr[stage][row * Codec::kHighBytes], &high[gi * Codec::kHighBytes]);
                } else {
                    cp_async<16, Cache::cg>(&Hr[stage][row * Codec::kHighBytes],
                                            &high[gi * Codec::kHighBytes]);
                }
            }
        };

        auto decode_weight = [&](int stage, int kt) {
            constexpr int CodeChunksPerRow = Codec::kCodeBytes / 4;
            constexpr int HighPerChunk     = Codec::kHighPerChunk;
            static_assert(CodeChunksPerRow * 8 == kExpertBK,
                          "a row of codes must decode to exactly the tile's k width");
            for (int item = tid; item < kExpertBM * CodeChunksPerRow; item += ExpertThreads) {
                const int row     = item / CodeChunksPerRow;
                const int chunk   = item - row * CodeChunksPerRow;
                const float scale = __half2float(__ushort_as_half(
                    *reinterpret_cast<const std::uint16_t*>(&Sr[(row * GroupsPerRow + kt) * 2])));
                unsigned decoded[4];
                Codec::decode_eight(
                    *reinterpret_cast<const unsigned*>(&Cr[stage][row * 32 + chunk * 4]),
                    &Hr[stage][row * Codec::kHighBytes + chunk * HighPerChunk], scale, decoded);
                store_vec(&As[row * kExpertBK + common::swizzle_xor_64(row, chunk * 8)],
                          make_int4(static_cast<int>(decoded[0]), static_cast<int>(decoded[1]),
                                    static_cast<int>(decoded[2]), static_cast<int>(decoded[3])));
            }
        };

        stage_scales();
        cp_commit();
#pragma unroll
        for (int stage = 0; stage < kExpertStages; ++stage) {
            stage_inputs(stage, stage);
            cp_commit();
        }

#pragma unroll
        for (int kt = 0; kt < kIntermediate / kExpertBK; ++kt) {
            const int stage = kt & 1;
            cp_wait<kExpertStages - 1>();
            __syncthreads();
            decode_weight(stage, kt);
            __syncthreads();

            if (warp * 8 < cols) {
#pragma unroll
                for (int ki = 0; ki < kExpertBK / 16; ++ki) {
                    unsigned af[4][4];
                    unsigned bf[2];
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        const int row = mi * 16 + a_rowoff;
                        const int col = ki * 16 + a_coloff;
                        ldmatrix_x4(
                            af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                            smem_addr(&As[row * kExpertBK + common::swizzle_xor_64(row, col)]));
                    }
                    const int brow = warp * 8 + b_rin;
                    const int bcol = ki * 16 + b_koff;
                    ldmatrix_x2(
                        bf[0], bf[1],
                        smem_addr(
                            &Bs[stage][brow * kExpertBK + common::swizzle_xor_64(brow, bcol)]));
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        mma_bf16(acc[mi][0], acc[mi][1], acc[mi][2], acc[mi][3], af[mi][0],
                                 af[mi][1], af[mi][2], af[mi][3], bf[0], bf[1]);
                    }
                }
            }

            __syncthreads();
            const int next = kt + kExpertStages;
            if (next < kIntermediate / kExpertBK) { stage_inputs(stage, next); }
            cp_commit();
        }
        cp_wait<0>();
        __syncthreads();

        if (warp * 8 < cols) {
#pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                const int output_row0 = row0 + mi * 16 + gid;
                const int output_row1 = output_row0 + 8;
                const int col0        = begin + column_base + warp * 8 + 2 * lid;
                const int col1        = col0 + 1;
                const int local_col0  = warp * 8 + 2 * lid;
                const int local_col1  = local_col0 + 1;
                if (local_col0 < cols) {
                    output[static_cast<std::int64_t>(col0) * kHidden + output_row0] =
                        __float2bfloat16_rn(acc[mi][0]);
                    output[static_cast<std::int64_t>(col0) * kHidden + output_row1] =
                        __float2bfloat16_rn(acc[mi][2]);
                }
                if (local_col1 < cols) {
                    output[static_cast<std::int64_t>(col1) * kHidden + output_row0] =
                        __float2bfloat16_rn(acc[mi][1]);
                    output[static_cast<std::int64_t>(col1) * kHidden + output_row1] =
                        __float2bfloat16_rn(acc[mi][3]);
                }
            }
        }
        __syncthreads();
    }
}

template <bool Routed, bool Adaptive = false>
__global__ __launch_bounds__(kExpertThreads, 1) void sparse_moe_prefill_q8_down_kernel(
    const __nv_bfloat16* __restrict__ input, const int* __restrict__ expert_offsets,
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ grouped_output, const float* __restrict__ routed_sum,
    const float* __restrict__ shared_scale, __nv_bfloat16* __restrict__ destination, int tokens,
    const int* __restrict__ route_job_count) {
    __shared__ __align__(16) __nv_bfloat16 As[kExpertBM * kExpertBK];
    __shared__ __align__(16) __nv_bfloat16 Bs[kExpertStages][kExpertBN * kExpertBK];
    __shared__ __align__(16) std::uint8_t Cr[kExpertBM * kExpertBK];
    __shared__ __align__(16) std::uint8_t Sr[kExpertBM * 16];

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if constexpr (Adaptive) {
        if (*route_job_count < 0) { return; }
    }
    const int expert = Routed ? static_cast<int>(blockIdx.y) : 0;
    const int row0   = static_cast<int>(blockIdx.x) * kExpertBM;
    const int begin  = Routed ? expert_offsets[expert] : static_cast<int>(blockIdx.y) * kExpertBN;
    const int count  = Routed ? expert_offsets[expert + 1] - begin
                              : (tokens - begin < kExpertBN ? tokens - begin : kExpertBN);
    if (count <= 0) { return; }

    const int a_mat              = lane >> 3;
    const int a_rin              = lane & 7;
    const int a_rowoff           = a_rin + ((a_mat & 1) << 3);
    const int a_coloff           = (a_mat >> 1) << 3;
    const int b_rin              = lane & 7;
    const int b_koff             = ((lane >> 3) & 1) << 3;
    const int gid                = lane >> 2;
    const int lid                = lane & 3;
    constexpr int groups_per_row = kIntermediate / 32;

    const int column_iterations = Routed ? (count + kExpertBN - 1) / kExpertBN : 1;
    for (int iteration = 0; iteration < column_iterations; ++iteration) {
        const int column_base = iteration * kExpertBN;
        const int cols        = count - column_base < kExpertBN ? count - column_base : kExpertBN;
        float acc[4][4]       = {};

        auto stage_x = [&](int stage, int kt) {
            const int k0 = kt * kExpertBK;
            for (int item = tid; item < kExpertBN * (kExpertBK / 8); item += kExpertThreads) {
                const int col        = item / (kExpertBK / 8);
                const int k8         = item - col * (kExpertBK / 8);
                const int source_col = begin + column_base + col;
                auto* dst = &Bs[stage][col * kExpertBK + common::swizzle_xor_64(col, k8 * 8)];
                const auto* src =
                    input +
                    static_cast<std::int64_t>(col < cols ? source_col : begin) * kIntermediate +
                    k0 + k8 * 8;
                cp_async_zfill<16, Cache::cg>(dst, src, col < cols ? 16 : 0);
            }
        };

        auto stage_weight = [&](int kt) {
            constexpr int groups_per_tile   = kExpertBK / 32;
            constexpr int scale_cache_tiles = 8 / groups_per_tile;
            const int group0                = kt * groups_per_tile;
            for (int item = tid; item < kExpertBM * (kExpertBK / 16); item += kExpertThreads) {
                const int row        = item / (kExpertBK / 16);
                const int chunk      = item - row * (kExpertBK / 16);
                const int global_row = (Routed ? expert * kHidden : 0) + row0 + row;
                const std::int64_t gi =
                    static_cast<std::int64_t>(global_row) * groups_per_row + group0;
                cp_async<16, Cache::cg>(&Cr[row * kExpertBK + chunk * 16],
                                        &codes[gi * 32 + chunk * 16]);
            }
            if ((kt % scale_cache_tiles) == 0) {
                for (int row = tid; row < kExpertBM; row += kExpertThreads) {
                    const int global_row = (Routed ? expert * kHidden : 0) + row0 + row;
                    const std::int64_t gi =
                        static_cast<std::int64_t>(global_row) * groups_per_row + group0;
                    cp_async<16, Cache::cg>(&Sr[row * 16], &scales[gi * 2]);
                }
            }
        };

        auto decode_weight = [&](int kt) {
            constexpr int groups_per_tile   = kExpertBK / 32;
            constexpr int scale_cache_tiles = 8 / groups_per_tile;
            const int scale_offset          = (kt % scale_cache_tiles) * groups_per_tile * 2;
            const int half                  = lane >> 4;
            const int half_lane             = lane & 15;
            for (int row_pair = warp * 2; row_pair < kExpertBM; row_pair += kExpertWarps * 2) {
                const int row = row_pair + half;
                unsigned scale_pair =
                    half_lane == 0
                        ? *reinterpret_cast<const std::uint32_t*>(&Sr[row * 16 + scale_offset])
                        : 0;
                scale_pair = __shfl_sync(0xffffffffu, scale_pair, half * 16);
#pragma unroll
                for (int group = 0; group < groups_per_tile; ++group) {
                    const float scale = __half2float(__ushort_as_half(
                        static_cast<std::uint16_t>((scale_pair >> (group * 16)) & 0xffffu)));
                    const int col     = group * 32 + half_lane * 2;
                    const std::uint16_t packed =
                        *reinterpret_cast<const std::uint16_t*>(&Cr[row * kExpertBK + col]);
                    const int q0 = static_cast<int>(static_cast<std::int8_t>(packed & 0xffu));
                    const int q1 = static_cast<int>(static_cast<std::int8_t>(packed >> 8));
                    const __nv_bfloat162 value = __floats2bfloat162_rn(
                        static_cast<float>(q0) * scale, static_cast<float>(q1) * scale);
                    store_vec(&As[row * kExpertBK + common::swizzle_xor_64(row, col)], value);
                }
            }
        };

        stage_x(0, 0);
        stage_weight(0);
        cp_commit();

#pragma unroll
        for (int kt = 0; kt < kIntermediate / kExpertBK; ++kt) {
            const int stage = kt & 1;
            cp_wait<0>();
            __syncthreads();
            decode_weight(kt);
            __syncthreads();

            const int next = kt + 1;
            if (next < kIntermediate / kExpertBK) {
                stage_x(next & 1, next);
                stage_weight(next);
                cp_commit();
            }

            if (warp * 8 < cols) {
                unsigned af[2][4][4];
                unsigned bf[2][2];
                auto load_fragments = [&](int slot, int ki) {
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        const int row = mi * 16 + a_rowoff;
                        const int col = ki * 16 + a_coloff;
                        ldmatrix_x4(
                            af[slot][mi][0], af[slot][mi][1], af[slot][mi][2], af[slot][mi][3],
                            smem_addr(&As[row * kExpertBK + common::swizzle_xor_64(row, col)]));
                    }
                    const int brow = warp * 8 + b_rin;
                    const int bcol = ki * 16 + b_koff;
                    ldmatrix_x2(
                        bf[slot][0], bf[slot][1],
                        smem_addr(
                            &Bs[stage][brow * kExpertBK + common::swizzle_xor_64(brow, bcol)]));
                };
                load_fragments(0, 0);
#pragma unroll
                for (int ki = 0; ki < kExpertBK / 16; ++ki) {
                    const int slot = ki & 1;
                    if (ki + 1 < kExpertBK / 16) { load_fragments(slot ^ 1, ki + 1); }
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        mma_bf16(acc[mi][0], acc[mi][1], acc[mi][2], acc[mi][3], af[slot][mi][0],
                                 af[slot][mi][1], af[slot][mi][2], af[slot][mi][3], bf[slot][0],
                                 bf[slot][1]);
                    }
                }
            }
        }
        cp_wait<0>();
        __syncthreads();

        if (warp * 8 < cols) {
#pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                const int output_row0 = row0 + mi * 16 + gid;
                const int output_row1 = output_row0 + 8;
                const int col0        = begin + column_base + warp * 8 + 2 * lid;
                const int col1        = col0 + 1;
                const int local_col0  = warp * 8 + 2 * lid;
                const int local_col1  = local_col0 + 1;
                auto store            = [&](int col, int row, float value) {
                    if constexpr (Routed) {
                        grouped_output[static_cast<std::int64_t>(col) * kHidden + row] =
                            __float2bfloat16_rn(value);
                    } else {
                        const float merged =
                            __bfloat162float(
                                destination[static_cast<std::int64_t>(col) * kHidden + row]) +
                            routed_sum[static_cast<std::int64_t>(col) * kHidden + row] +
                            shared_scale[col] * value;
                        destination[static_cast<std::int64_t>(col) * kHidden + row] =
                            __float2bfloat16_rn(merged);
                    }
                };
                if (local_col0 < cols) {
                    store(col0, output_row0, acc[mi][0]);
                    store(col0, output_row1, acc[mi][2]);
                }
                if (local_col1 < cols) {
                    store(col1, output_row0, acc[mi][1]);
                    store(col1, output_row1, acc[mi][3]);
                }
            }
        }
        __syncthreads();
    }
}

template <bool Adaptive>
__global__ void sparse_moe_prefill_reduce_kernel(const __nv_bfloat16* __restrict__ grouped_output,
                                                 const int* __restrict__ packed_index,
                                                 const float* __restrict__ alpha,
                                                 float* __restrict__ routed_sum,
                                                 const int* __restrict__ route_job_count) {
    if constexpr (Adaptive) {
        if (*route_job_count < 0) { return; }
    }
    __shared__ int columns[kTopK];
    __shared__ float weights[kTopK];
    const int token = static_cast<int>(blockIdx.x);
    const int tid   = static_cast<int>(threadIdx.x);
    if (tid < kTopK) {
        columns[tid] = packed_index[token * kTopK + tid];
        weights[tid] = alpha[token * kTopK + tid];
    }
    __syncthreads();

    float values[8] = {};
#pragma unroll
    for (int route = 0; route < kTopK; ++route) {
        Bf16x8Bits packed;
        packed.raw = load_vec<uint4>(grouped_output +
                                     static_cast<std::int64_t>(columns[route]) * kHidden + tid * 8);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const float2 decoded = __bfloat1622float2(packed.pair[pair]);
            values[2 * pair] += weights[route] * decoded.x;
            values[2 * pair + 1] += weights[route] * decoded.y;
        }
    }
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        routed_sum[static_cast<std::int64_t>(token) * kHidden + tid * 8 + item] = values[item];
    }
}

} // namespace

} // namespace ninfer::ops::moe_detail
