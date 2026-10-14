#pragma once

#include "ops/moe/kernels/cuda/nvfp4/launch.h"
#include "ops/common/cuda/mma.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/numeric/math.cuh"
#include "ops/common/quant/nvfp4.cuh"

namespace ninfer::ops::detail::moe_nvfp4 {
using namespace moe_block_scaled;
using namespace common;

template <int Columns, int Stages, bool Gate>
struct Schedule {
    static constexpr int columns = Columns, stages = Stages;
    static constexpr int threads      = Columns >= 64 ? 256 : 128;
    static constexpr int warp_columns = Columns >= 64 ? Columns / 2 : Columns;
    static constexpr int mma_columns  = warp_columns / 8;
    static constexpr int rows         = Gate ? 64 : 128;
    static constexpr int k            = Gate ? kHidden : kIntermediate;
    static constexpr int min_blocks   = Columns == 128 ? 2 : Columns == 64 ? 3 : 1;
};

template <class S>
struct TileStorage {
    alignas(16) std::uint8_t weight[S::stages][128 * 64];
    alignas(16) std::uint8_t input[S::stages][S::columns * 64];
    alignas(16) unsigned weight_scales[S::stages][128 * 2];
    alignas(16) unsigned input_scales[S::stages][S::columns * 2];
    int sources[S::columns];
};

__device__ __forceinline__ int swizzle(int row, int byte) { return byte ^ ((row & 3) << 4); }

struct Matrix {
    const std::uint8_t* codes;
    const std::uint8_t* scales;
    float divisor;
};

__device__ __forceinline__ Matrix resolve(const ExpertMatrixDeviceView& matrix, int expert) {
    const auto offset = matrix.offsets ? matrix.offsets[expert] : 0;
    return {reinterpret_cast<const std::uint8_t*>(matrix.codes) +
                (matrix.offsets ? offset : expert * matrix.code_stride),
            reinterpret_cast<const std::uint8_t*>(matrix.scales) +
                (matrix.offsets ? offset : expert * matrix.scale_stride),
            matrix.divisors[expert * matrix.divisor_stride]};
}

template <class S, bool Gate>
__device__ __forceinline__ void stage_inputs(TileStorage<S>& tile, int stage, int kt, int row0,
                                             Matrix weight, const std::uint8_t* x,
                                             const std::uint8_t* scales) {
    for (int task = threadIdx.x; task < 128 * 4; task += S::threads) {
        const int row = task / 4, segment = task % 4;
        const int wr = row0 + (Gate ? (row < 64 ? row : row - 64 + kIntermediate) : row);
        cp_async<16>(tile.weight[stage] + row * 64 + swizzle(row, segment * 16),
                     weight.codes + std::int64_t(wr) * (S::k / 2) + kt * 64 + segment * 16);
    }
    constexpr int parts = Gate ? 2 : 1;
    constexpr int bytes = Gate ? 8 : 16;
    for (int task = threadIdx.x; task < 2 * parts * 32; task += S::threads) {
        const int row = task % 32, part = (task / 32) % parts, k64 = task / (32 * parts);
        const int wr      = row0 + row + (Gate ? part * kIntermediate : 0);
        auto* destination = reinterpret_cast<std::uint8_t*>(tile.weight_scales[stage]) + k64 * 512 +
                            row * 16 + part * bytes;
        cp_async<bytes>(destination,
                        weight.scales + nvfp4_scale_byte_offset(wr, kt * 8 + k64 * 4, S::k));
    }
    for (int task = threadIdx.x; task < S::columns * 4; task += S::threads) {
        const int row = task / 4, segment = task % 4;
        const int source = tile.sources[row];
        cp_async_zfill<16>(tile.input[stage] + row * 64 + swizzle(row, segment * 16),
                           x + std::int64_t(source >= 0 ? source : 0) * (S::k / 2) + kt * 64 +
                               segment * 16,
                           source >= 0 ? 16 : 0);
    }
    for (int row = threadIdx.x; row < S::columns; row += S::threads) {
        const int source = tile.sources[row];
        cp_async_zfill<8>(&tile.input_scales[stage][row * 2],
                          scales + std::int64_t(source >= 0 ? source : 0) * (S::k / 16) + kt * 8,
                          source >= 0 ? 8 : 0);
    }
    cp_commit();
}

__device__ __forceinline__ void store_product(float lo, float hi, float divisor,
                                              std::uint8_t* codes, std::uint8_t* scales,
                                              int assignment, int feature) {
    const int lane = threadIdx.x & 31, r = lane >> 2;
    float maximum = fmaxf(fabsf(lo), fabsf(hi));
#pragma unroll
    for (int delta = 4; delta <= 16; delta *= 2) {
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, delta));
    }
    const auto scale      = nvfp4_activation_scale(maximum, divisor);
    const unsigned packed = quantize_nvfp4_activation_pair(make_float2(lo, hi), divisor, scale);
    unsigned low = (packed & 15) << (r * 4), high = (packed >> 4) << (r * 4);
#pragma unroll
    for (int delta = 4; delta <= 16; delta *= 2) {
        low |= __shfl_xor_sync(0xffffffff, low, delta);
        high |= __shfl_xor_sync(0xffffffff, high, delta);
    }
    if (!r && assignment >= 0) {
        *reinterpret_cast<uint2*>(codes + std::int64_t(assignment) * (kIntermediate / 2) +
                                  feature / 2) = make_uint2(low, high);
        scales[std::int64_t(assignment) * (kIntermediate / 16) + feature / 16] = scale;
    }
}

template <int Columns, int Stages, bool Gate, bool Atomic = false>
__global__ __launch_bounds__(
    Schedule<Columns, Stages, Gate>::threads,
    Schedule<Columns, Stages, Gate>::min_blocks) void grouped_mma(const std::uint8_t* input,
                                                                  const std::uint8_t* input_scales,
                                                                  ExpertMatrixDeviceView bank,
                                                                  float activation_divisor,
                                                                  float product_divisor,
                                                                  Groups groups,
                                                                  std::uint8_t* product,
                                                                  std::uint8_t* product_scales,
                                                                  float* contributions,
                                                                  const float* coefficients,
                                                                  float* routed_sum, int subtiles) {
    using S = Schedule<Columns, Stages, Gate>;
    __shared__ TileStorage<S> tile;
    constexpr int row_tiles = (Gate ? kIntermediate : kHidden) / S::rows;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int warp_m = warp % 4, warp_n = warp / 4;
    const int count = *groups.jobs * row_tiles * subtiles;
    for (int task = blockIdx.x; task < count; task += gridDim.x) {
        const int job = task / (row_tiles * subtiles), row0 = (task % row_tiles) * S::rows;
        const int subtile = (task / row_tiles) % subtiles;
        const int expert  = groups.job_experts[job];
        const int begin   = groups.offsets[expert] + groups.job_columns[job] + subtile * Columns;
        const int end     = groups.offsets[expert + 1];
        if (begin >= end) { continue; }
        const auto weight = resolve(bank, expert);
        if (threadIdx.x < Columns) {
            const int packed = begin + threadIdx.x;
            tile.sources[threadIdx.x] =
                packed < end ? (Gate ? groups.assignments[packed] / kTopK : packed) : -1;
        }
        __syncthreads();
#pragma unroll
        for (int stage = 0; stage < Stages; ++stage) {
            stage_inputs<S, Gate>(tile, stage, stage, row0, weight, input, input_scales);
        }
        float acc[2][S::mma_columns][4] = {};
#pragma unroll 1
        for (int kt = 0; kt < S::k / 128; ++kt) {
            const int stage = kt % Stages;
            cp_wait<Stages - 1>();
            __syncthreads();
#pragma unroll
            for (int ki = 0; ki < 2; ++ki) {
                unsigned a[2][4], as[2];
#pragma unroll
                for (int m = 0; m < 2; ++m) {
                    const int row_base = Gate ? warp_m * 16 + m * 64 : warp_m * 32 + m * 16;
                    const int row      = row_base + (lane & 7) + ((lane >> 3 & 1) * 8);
                    const int col      = ki * 32 + (lane >> 4) * 16;
                    ldmatrix_x4(a[m][0], a[m][1], a[m][2], a[m][3],
                                smem_addr(tile.weight[stage] + row * 64 + swizzle(row, col)));
                    const int sr = row_base + ((lane & 1) << 3) + (lane >> 2);
                    as[m]        = tile.weight_scales[stage][ki * 128 + (sr % 32) * 4 + sr / 32];
                }
#pragma unroll
                for (int n = 0; n < S::mma_columns; ++n) {
                    const int column = warp_n * S::warp_columns + n * 8;
                    if (begin + column < end) {
                        const int row = column + (lane & 7);
                        const int col = ki * 32 + ((lane >> 3) & 1) * 16;
                        unsigned b[2];
                        ldmatrix_x2(b[0], b[1],
                                    smem_addr(tile.input[stage] + row * 64 + swizzle(row, col)));
                        const auto bs = tile.input_scales[stage][(column + (lane >> 2)) * 2 + ki];
#pragma unroll
                        for (int m = 0; m < 2; ++m) {
                            mma_nvfp4_e4m3(acc[m][n][0], acc[m][n][1], acc[m][n][2], acc[m][n][3],
                                           a[m][0], a[m][1], a[m][2], a[m][3], b[0], b[1], as[m],
                                           bs);
                        }
                    }
                }
            }
            __syncthreads();
            const int next = kt + Stages;
            if (next < S::k / 128) {
                stage_inputs<S, Gate>(tile, stage, next, row0, weight, input, input_scales);
            } else {
                // Empty groups drain the fixed-distance pipeline without reusing a live stage.
                cp_commit();
            }
        }
        cp_wait<0>();
        const float factor = 1.0F / (activation_divisor * weight.divisor);
#pragma unroll
        for (int n = 0; n < S::mma_columns; ++n) {
            if (begin + warp_n * S::warp_columns + n * 8 >= end) { continue; }
            const int column = begin + warp_n * S::warp_columns + n * 8 + 2 * (lane & 3);
            if constexpr (Gate) {
#pragma unroll
                for (int t = 0; t < 2; ++t) {
                    const float g0 = acc[0][n][t] * factor, g1 = acc[0][n][t + 2] * factor;
                    const float h0 = silu(g0) * (acc[1][n][t] * factor);
                    const float h1 = silu(g1) * (acc[1][n][t + 2] * factor);
                    store_product(h0, h1, product_divisor, product, product_scales,
                                  column + t < end ? column + t : -1, row0 + warp_m * 16);
                }
            } else {
#pragma unroll
                for (int m = 0; m < 2; ++m) {
#pragma unroll
                    for (int v = 0; v < 4; ++v) {
                        const int c = column + (v & 1);
                        if (c < end) {
                            const int row = row0 + warp_m * 32 + m * 16 + (lane >> 2) + (v / 2) * 8;
                            if constexpr (Atomic) {
                                const int assignment = groups.assignments[c];
                                atomicAdd(routed_sum + std::int64_t(assignment / kTopK) * kHidden +
                                              row,
                                          acc[m][n][v] * factor * coefficients[assignment]);
                            } else {
                                contributions[std::int64_t(c) * kHidden + row] =
                                    acc[m][n][v] * factor;
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
}

} // namespace ninfer::ops::detail::moe_nvfp4
