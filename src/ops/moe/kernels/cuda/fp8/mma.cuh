#pragma once

#include "ops/moe/kernels/cuda/fp8/launch.h"
#include "ops/common/cuda/mma.cuh"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/numeric/math.cuh"
#include "ops/common/quant/fp8.cuh"

namespace ninfer::ops::detail::moe_fp8 {
using namespace common;
using namespace moe_block_scaled;

struct Matrix {
    const std::uint8_t* codes;
    const __nv_bfloat16* scales;
};

__device__ __forceinline__ Matrix resolve(const ExpertMatrixDeviceView& matrix, int expert) {
    const auto offset = matrix.offsets ? matrix.offsets[expert] : 0;
    return {reinterpret_cast<const std::uint8_t*>(matrix.codes) +
                (matrix.offsets ? offset : expert * matrix.code_stride),
            reinterpret_cast<const __nv_bfloat16*>(
                matrix.scales + (matrix.offsets ? offset : expert * matrix.scale_stride))};
}

template <int Columns, bool Gate>
struct Schedule {
    static constexpr int columns = Columns, bk = 128;
    static constexpr int warp_rows    = Gate ? 8 : 4;
    static constexpr int warp_groups  = Columns >= 64 ? 2 : 1;
    static constexpr int threads      = warp_rows * warp_groups * 32;
    static constexpr int warp_columns = Columns / warp_groups;
    static constexpr int mma_columns  = warp_columns / 8;
    static constexpr int k            = Gate ? kHidden : kIntermediate;
    static constexpr int weight_rows  = Gate ? 256 : 128;
};

template <class S>
struct TileStorage {
    union {
        struct {
            alignas(16) std::uint8_t weights[2][S::weight_rows * S::bk];
            alignas(16) std::uint8_t input[2][S::columns * S::bk];
        } stage;

        float product[S::weight_rows == 256 ? S::columns : 1][128];
    };

    int sources[S::columns];
};

template <class S>
__device__ __forceinline__ int swizzle(int row, int byte) {
    return byte ^ ((row & (S::bk / 16 - 1)) << 4);
}

template <class S, bool Gate>
__device__ __forceinline__ void stage_inputs(TileStorage<S>& tile, int stage, int kt, int row0,
                                             Matrix weight, const std::uint8_t* input) {
    for (int task = threadIdx.x; task < S::weight_rows * (S::bk / 16); task += S::threads) {
        const int row = task / (S::bk / 16), col = task % (S::bk / 16) * 16;
        const int wr = row0 + (Gate && row >= 128 ? row - 128 + kIntermediate : row);
        cp_async<16, Cache::cg>(tile.stage.weights[stage] + row * S::bk + swizzle<S>(row, col),
                                weight.codes + std::int64_t(wr) * S::k + kt * S::bk + col);
    }
    for (int task = threadIdx.x; task < S::columns * (S::bk / 16); task += S::threads) {
        const int row = task / (S::bk / 16), col = task % (S::bk / 16) * 16;
        const int source = tile.sources[row];
        cp_async_zfill<16, Cache::cg>(tile.stage.input[stage] + row * S::bk + swizzle<S>(row, col),
                                      input + std::int64_t(source >= 0 ? source : 0) * S::k +
                                          kt * S::bk + col,
                                      source >= 0 ? 16 : 0);
    }
    cp_commit();
}

template <int Columns, bool Gate, bool Atomic = false>
__global__ __launch_bounds__(Schedule<Columns, Gate>::threads) void grouped_mma(
    const std::uint8_t* input, const float* input_scales, ExpertMatrixDeviceView bank,
    Groups groups, std::uint8_t* product, float* product_scales, float* contributions,
    const float* coefficients, float* routed_sum, int subtiles) {
    using S = Schedule<Columns, Gate>;
    extern __shared__ __align__(16) unsigned char shared_bytes[];
    auto& tile              = *reinterpret_cast<TileStorage<S>*>(shared_bytes);
    constexpr int row_tiles = (Gate ? kIntermediate : kHidden) / 128;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int wm = warp % S::warp_rows, wn = warp / S::warp_rows;
    const int tasks = *groups.jobs * row_tiles * subtiles;
    for (int task = blockIdx.x; task < tasks; task += gridDim.x) {
        const int job = task / (row_tiles * subtiles), row0 = (task % row_tiles) * 128;
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
        stage_inputs<S, Gate>(tile, 0, 0, row0, weight, input);
        stage_inputs<S, Gate>(tile, 1, 1, row0, weight, input);
        float acc[2][S::mma_columns][4] = {};
        for (int kt = 0; kt < S::k / S::bk; ++kt) {
            const int stage = kt % 2;
            cp_wait<1>();
            __syncthreads();
            const float sg = __bfloat162float(weight.scales[(row0 / 128) * (S::k / 128) + kt]);
            const float su =
                Gate ? __bfloat162float(
                           weight.scales[((row0 + kIntermediate) / 128) * (S::k / 128) + kt])
                     : sg;
#pragma unroll
            for (int n = 0; n < S::mma_columns; ++n) {
                const int column = wn * S::warp_columns + n * 8;
                if (begin + column < end) {
                    float p[2][4] = {};
#pragma unroll
                    for (int ki = 0; ki < 4; ++ki) {
                        unsigned a[2][4], b[2];
#pragma unroll
                        for (int m = 0; m < 2; ++m) {
                            const int row_base = Gate ? wm * 16 + m * 128 : wm * 32 + m * 16;
                            const int row      = row_base + (lane & 7) + ((lane >> 3 & 1) * 8);
                            const int col      = ki * 32 + (lane >> 4) * 16;
                            ldmatrix_x4(a[m][0], a[m][1], a[m][2], a[m][3],
                                        smem_addr(tile.stage.weights[stage] + row * S::bk +
                                                  swizzle<S>(row, col)));
                        }
                        const int row = column + (lane & 7), col = ki * 32 + ((lane >> 3) & 1) * 16;
                        ldmatrix_x2(b[0], b[1],
                                    smem_addr(tile.stage.input[stage] + row * S::bk +
                                              swizzle<S>(row, col)));
#pragma unroll
                        for (int m = 0; m < 2; ++m) {
                            mma_fp8_e4m3(p[m][0], p[m][1], p[m][2], p[m][3], a[m][0], a[m][1],
                                         a[m][2], a[m][3], b[0], b[1]);
                        }
                    }
#pragma unroll
                    for (int v = 0; v < 4; ++v) {
                        const int source = tile.sources[column + (lane & 3) * 2 + (v & 1)];
                        const float sx =
                            source >= 0 ? input_scales[std::int64_t(source) * (S::k / 128) + kt]
                                        : 0;
                        acc[0][n][v] = fmaf(p[0][v], sg * sx, acc[0][n][v]);
                        acc[1][n][v] = fmaf(p[1][v], su * sx, acc[1][n][v]);
                    }
                }
            }
            __syncthreads();
            if (kt + 2 < S::k / S::bk) {
                stage_inputs<S, Gate>(tile, stage, kt + 2, row0, weight, input);
            } else {
                cp_commit();
            }
        }
        cp_wait<0>();
        __syncthreads();
#pragma unroll
        for (int n = 0; n < S::mma_columns; ++n) {
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                const int column = wn * S::warp_columns + n * 8 + (lane & 3) * 2 + (v & 1);
                if constexpr (Gate) {
                    const int row             = wm * 16 + (lane >> 2) + (v / 2) * 8;
                    tile.product[column][row] = silu(acc[0][n][v]) * acc[1][n][v];
                } else {
                    if (begin + column < end) {
#pragma unroll
                        for (int m = 0; m < 2; ++m) {
                            const int row = row0 + wm * 32 + m * 16 + (lane >> 2) + (v / 2) * 8;
                            if constexpr (Atomic) {
                                const int assignment = groups.assignments[begin + column];
                                atomicAdd(routed_sum + std::int64_t(assignment / kTopK) * kHidden +
                                              row,
                                          acc[m][n][v] * coefficients[assignment]);
                            } else {
                                contributions[std::int64_t(begin + column) * kHidden + row] =
                                    acc[m][n][v];
                            }
                        }
                    }
                }
            }
        }
        if constexpr (Gate) {
            __syncthreads();
            for (int column = warp; column < Columns; column += S::threads / 32) {
                if (begin + column < end) {
                    const float4 value =
                        *reinterpret_cast<const float4*>(&tile.product[column][lane * 4]);
                    float maximum     = fmaxf(fmaxf(fabsf(value.x), fabsf(value.y)),
                                              fmaxf(fabsf(value.z), fabsf(value.w)));
                    maximum           = warp_max(maximum);
                    const float scale = fp8_fp32_scale(maximum);
                    const unsigned packed =
                        fp8_quant_fp32_pair(make_float2(value.x, value.y), scale) |
                        (unsigned(fp8_quant_fp32_pair(make_float2(value.z, value.w), scale)) << 16);
                    *reinterpret_cast<unsigned*>(
                        product + std::int64_t(begin + column) * kIntermediate + row0 + lane * 4) =
                        packed;
                    if (!lane) {
                        product_scales[std::int64_t(begin + column) * (kIntermediate / 128) +
                                       row0 / 128] = scale;
                    }
                }
            }
        }
        __syncthreads();
    }
}
} // namespace ninfer::ops::detail::moe_fp8
