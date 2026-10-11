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

__global__ __launch_bounds__(kRouterThreads, 2) void sparse_moe_prefill_router_mma_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ weight,
    float* __restrict__ scores, int tokens) {
    __shared__ __align__(16) __nv_bfloat16 Ws[kRouterStages][kRouterBM * kRouterBK];
    __shared__ __align__(16) __nv_bfloat16 Xs[kRouterStages][kRouterBN * kRouterBK];

    const int tid    = static_cast<int>(threadIdx.x);
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int row0   = static_cast<int>(blockIdx.x) * kRouterBM;
    const int token0 = static_cast<int>(blockIdx.y) * kRouterBN;

    float acc[4] = {};

    auto stage_inputs = [&](int stage, int kt) {
        const int k0 = kt * kRouterBK;
        for (int item = tid; item < kRouterBM * (kRouterBK / 8); item += kRouterThreads) {
            const int row   = item / (kRouterBK / 8);
            const int k8    = item - row * (kRouterBK / 8);
            const int rr    = row0 + row;
            auto* dst       = &Ws[stage][row * kRouterBK + common::swizzle_xor_64(row, k8 * 8)];
            const auto* src = weight +
                              static_cast<std::int64_t>(rr < kRouterRows ? rr : 0) * kHidden + k0 +
                              k8 * 8;
            cp_async_zfill<16, Cache::cg>(dst, src, rr < kRouterRows ? 16 : 0);
        }
        for (int item = tid; item < kRouterBN * (kRouterBK / 8); item += kRouterThreads) {
            const int col = item / (kRouterBK / 8);
            const int k8  = item - col * (kRouterBK / 8);
            const int tt  = token0 + col;
            auto* dst     = &Xs[stage][col * kRouterBK + common::swizzle_xor_64(col, k8 * 8)];
            const auto* src =
                x + static_cast<std::int64_t>(tt < tokens ? tt : 0) * kHidden + k0 + k8 * 8;
            cp_async_zfill<16, Cache::cg>(dst, src, tt < tokens ? 16 : 0);
        }
    };

#pragma unroll
    for (int stage = 0; stage < kRouterStages; ++stage) {
        stage_inputs(stage, stage);
        cp_commit();
    }

    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

#pragma unroll 1
    for (int kt = 0; kt < kHidden / kRouterBK; ++kt) {
        const int stage = kt & 1;
        cp_wait<kRouterStages - 1>();
        __syncthreads();

#pragma unroll
        for (int ki = 0; ki < kRouterBK / 16; ++ki) {
            unsigned af[4];
            unsigned bf[2];
            const int arow = a_rowoff;
            const int acol = ki * 16 + a_coloff;
            const int brow = warp * 8 + b_rin;
            const int bcol = ki * 16 + b_koff;
            ldmatrix_x4(
                af[0], af[1], af[2], af[3],
                smem_addr(&Ws[stage][arow * kRouterBK + common::swizzle_xor_64(arow, acol)]));
            ldmatrix_x2(
                bf[0], bf[1],
                smem_addr(&Xs[stage][brow * kRouterBK + common::swizzle_xor_64(brow, bcol)]));
            mma_bf16(acc[0], acc[1], acc[2], acc[3], af[0], af[1], af[2], af[3], bf[0], bf[1]);
        }

        __syncthreads();
        const int next = kt + kRouterStages;
        if (next < kHidden / kRouterBK) { stage_inputs(stage, next); }
        cp_commit();
    }
    cp_wait<0>();

    const int gid = lane >> 2;
    const int lid = lane & 3;
    const int r0  = row0 + gid;
    const int r1  = r0 + 8;
    const int c0  = token0 + warp * 8 + 2 * lid;
    const int c1  = c0 + 1;
    if (r0 < kRouterRows && c0 < tokens) {
        scores[static_cast<std::int64_t>(c0) * kRouterScoreRows + r0] = acc[0];
    }
    if (r0 < kRouterRows && c1 < tokens) {
        scores[static_cast<std::int64_t>(c1) * kRouterScoreRows + r0] = acc[1];
    }
    if (r1 < kRouterRows && c0 < tokens) {
        scores[static_cast<std::int64_t>(c0) * kRouterScoreRows + r1] = acc[2];
    }
    if (r1 < kRouterRows && c1 < tokens) {
        scores[static_cast<std::int64_t>(c1) * kRouterScoreRows + r1] = acc[3];
    }
}

} // namespace

} // namespace ninfer::ops::moe_detail
