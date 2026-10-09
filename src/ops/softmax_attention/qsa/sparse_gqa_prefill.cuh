#pragma once
#include "ops/softmax_attention/qsa/sparse_gqa_kernel.cuh"

namespace ninfer::ops::detail::qsa {

template <int N, int Queries, bool Vector>
__device__ __forceinline__ void
sparse_gqa_all_select(SparseGqaArguments a, SparseGqaStorage<N, Queries>& s, std::int64_t first,
                      int head, std::uint16_t* qs) {
    constexpr int Nt = N / 8, Ks = N / 16;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
    const auto total   = std::int64_t(a.tokens) * a.batch;
    const int b        = first / a.tokens;
    const auto last    = min(first + Queries, std::int64_t(b) * a.tokens + a.valid[b]);
    const int end      = last > first ? a.positions[last - 1] + 1 : 0;
    const auto q       = first + warp;
    const bool live    = q < last;
    const int position = live ? a.positions[q] : -1;
    auto* ks           = s.kv;
    auto* vs           = s.kv + N * 256;
    for (int i = tid; i < Queries * 16 * 32; i += Queries * 32) {
        const int row = i / 32, d = i % 32 * 8, local = row / 16, h = row % 16;
        auto* dst = qs + row * 256 + causal_swizzle(row, d);
        if (first + local < last && h < 12)
            cp_async<16>(dst, a.query + ((first + local) * 24 + head * 12 + h) * 256 + d);
        else
            store_vec(dst, make_int4(0, 0, 0, 0));
    }
    cp_commit();
    cp_wait<0>();
    __syncthreads();
    unsigned qf[4];
    const int ar     = warp * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
    float acc[32][4] = {};
    Bf16KvSoftmaxRow state[2];
    const int gid = lane >> 2, lid = lane & 3;
    constexpr float scale = kLog2E / 16;
    const SparseGqaWork work{end / 4, end / 4, end, 0, 1};
    if (end) {
        sparse_gqa_addresses(a, s, work, last - 1, head, 0, end);
        sparse_gqa_stage<N, Queries, Vector, true>(a, s, 0, end);
        cp_commit();
    }
    for (int start = 0; start < end; start += N) {
        cp_wait<0>();
        __syncthreads();
        sparse_gqa_stage<N, Queries, Vector, false>(a, s, start, end);
        cp_commit();
        float score[Nt][4] = {};
#pragma unroll
        for (int n = 0; n < Nt; ++n) {
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                unsigned bf[2];
                const int r = n * 8 + (lane & 7), d = k * 16 + ((lane >> 3) & 1) * 8;
                ldmatrix_x2(bf[0], bf[1], smem_addr(ks + r * 256 + causal_swizzle(r, d)));
                const int qd = k * 16 + (lane >> 4) * 8;
                ldmatrix_x4(qf[0], qf[1], qf[2], qf[3],
                            smem_addr(qs + ar * 256 + causal_swizzle(ar, qd)));
                mma_bf16(score[n][0], score[n][1], score[n][2], score[n][3], qf[0], qf[1], qf[2],
                         qf[3], bf[0], bf[1]);
            }
        }
        cp_wait<0>();
        __syncthreads();
        if (start + N < end) {
            sparse_gqa_addresses(a, s, work, last - 1, head, start + N, end);
            sparse_gqa_stage<N, Queries, Vector, true>(a, s, start + N, end);
            cp_commit();
        }
        float maximum[2] = {-CUDART_INF_F, -CUDART_INF_F};
#pragma unroll
        for (int n = 0; n < Nt; ++n)
            for (int j = 0; j < 4; ++j) {
                if (start + n * 8 + lid * 2 + (j & 1) > position) score[n][j] = -CUDART_INF_F;
                maximum[j / 2] = fmaxf(maximum[j / 2], score[n][j]);
            }
        const float alpha[2] = {state[0].update(warp_max<4>(maximum[0], 0xffffffffu), scale),
                                state[1].update(warp_max<4>(maximum[1], 0xffffffffu), scale)};
        unsigned pf[Ks][4];
        float sum[2] = {};
#pragma unroll
        for (int n = 0; n < Nt; ++n) {
            const float p0 = state[0].probability(score[n][0], scale),
                        p1 = state[0].probability(score[n][1], scale),
                        p2 = state[1].probability(score[n][2], scale),
                        p3 = state[1].probability(score[n][3], scale);
            sum[0] += p0 + p1;
            sum[1] += p2 + p3;
            pf[n / 2][n % 2 * 2]     = pack_f16x2(p0, p1);
            pf[n / 2][n % 2 * 2 + 1] = pack_f16x2(p2, p3);
        }
        state[0].accumulate(alpha[0], sum[0]);
        state[1].accumulate(alpha[1], sum[1]);
#pragma unroll
        for (int n = 0; n < 32; ++n) {
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[n][j] *= alpha[j / 2];
#pragma unroll
            for (int k = 0; k < Ks; ++k) {
                unsigned vf[2];
                const int r = k * 16 + ((lane >> 3) & 1) * 8 + (lane & 7), d = n * 8;
                ldmatrix_x2_t(vf[0], vf[1], smem_addr(vs + r * 256 + causal_swizzle(r, d)));
                mma_f16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[k][0], pf[k][1], pf[k][2],
                        pf[k][3], vf[0], vf[1]);
            }
        }
        __syncthreads();
    }
    state[0].finish();
    state[1].finish();
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int h = gid + j * 8;
        if (q >= total || h >= 12) continue;
        const float inv = live ? __frcp_rn(state[j].sum) : 0;
#pragma unroll
        for (int n = 0; n < 32; ++n) {
            const int d = n * 8 + lid * 2;
            if (live)
                sparse_gqa_store(a, q, head * 12 + h, d, acc[n][j * 2] * inv,
                                 acc[n][j * 2 + 1] * inv);
            else
                *reinterpret_cast<unsigned*>(a.out + (q * 24 + head * 12 + h) * 256 + d) = 0;
        }
    }
}

template <int N, int Queries, bool Vector>
__launch_bounds__(Queries * 32) __global__ void sparse_gqa_all_select_kernel(SparseGqaArguments a) {
    extern __shared__ __align__(16) unsigned char smem[];
    auto& s          = *reinterpret_cast<SparseGqaStorage<N, Queries>*>(smem);
    const auto first = a.begin + std::int64_t(blockIdx.x / 2) * Queries,
               total = std::int64_t(a.tokens) * a.batch;
    const int head   = blockIdx.x % 2;
    auto* qs = reinterpret_cast<std::uint16_t*>(smem + (sizeof(SparseGqaStorage<N, Queries>) + 15) /
                                                           16 * 16);
    if (sparse_gqa_all_selected<Queries>(a, first)) {
        sparse_gqa_all_select<N, Queries, Vector>(a, s, first, head, qs);
    } else {
#pragma unroll 1
        for (int i = 0; i < Queries && first + i < total; ++i) {
            sparse_gqa_mma<N, Queries, Vector>(a, s, first + i, head, 0);
            __syncthreads();
        }
    }
}
} // namespace ninfer::ops::detail::qsa
