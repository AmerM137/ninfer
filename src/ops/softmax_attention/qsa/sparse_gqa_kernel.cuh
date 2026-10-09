#pragma once

#include "ops/softmax_attention/qsa/sparse_gqa_arguments.cuh"
#include "ops/softmax_attention/common/causal_tile_io.cuh"
#include "ops/softmax_attention/common/bf16_softmax.cuh"

namespace ninfer::ops::detail::qsa {

template <int N, int Warps>
struct SparseGqaStorage {
    union {
        std::uint16_t kv[2 * N * 256];
        float partial[Warps * 16 * 256];
    };

    // Address staging ends before the synchronized warp reduction. Sharing these lifetimes
    // also keeps the 32-key schedule below the three-CTA shared-memory allocation boundary.
    union {
        struct {
            std::int64_t key[N / 4], value[N / 4];
        } addresses;

        struct {
            float maximum[Warps * 16], sum[Warps * 16];
        } statistics;
    };
};

template <int N, int Warps>
__device__ void sparse_gqa_addresses(const SparseGqaArguments& a, SparseGqaStorage<N, Warps>& s,
                                     SparseGqaWork work, std::int64_t q, int head, int start,
                                     int end) {
    const int i = threadIdx.x;
    if (i < N / 4 && start + i * 4 < end) {
        const int ordinal = start / 4 + i;
        // Distinct ids covering every complete block denote the full prefix in any order.
        const int g          = ordinal < work.count
                                   ? (work.count == work.complete ? ordinal : a.ids[q * 512 + ordinal])
                                   : work.complete;
        const int table_row  = a.table_rows[q / a.tokens];
        const int page       = a.tables[std::int64_t(table_row) * a.table_stride + g / 16];
        const int row        = 4 * (g % 16);
        s.addresses.key[i]   = page * a.key.page + head * a.key.head + row * a.key.row;
        s.addresses.value[i] = page * a.value.page + head * a.value.head + row * a.value.row;
    }
    __syncthreads();
}

template <int N, int Warps, bool VectorCache, bool Key>
__device__ void sparse_gqa_stage(const SparseGqaArguments& a, SparseGqaStorage<N, Warps>& s,
                                 int start, int end) {
    const auto plane      = Key ? a.key : a.value;
    auto* dst             = s.kv + (Key ? 0 : N * 256);
    const auto* addresses = Key ? s.addresses.key : s.addresses.value;
#pragma unroll 1
    for (int i = threadIdx.x; i < N * 32; i += Warps * 32) {
        const int row = i / 32, d = (i % 32) * 8;
        auto* target = dst + row * 256 + causal_swizzle(row, d);
        if (start + row >= end) {
            store_vec(target, make_int4(0, 0, 0, 0));
        } else {
            const auto* source = plane.data + addresses[row / 4] + (row % 4) * plane.row + d;
            if constexpr (VectorCache) {
                cp_async<16>(target, source);
            } else {
#pragma unroll
                for (int j = 0; j < 8; ++j) target[j] = source[j];
            }
        }
    }
}

__device__ __forceinline__ void sparse_gqa_store(const SparseGqaArguments& a, std::int64_t q, int h,
                                                 int d, float x, float y) {
    const auto offset = (q * 24 + h) * 256 + d;
    const float2 gate =
        __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(a.gate + offset));
    x *= __frcp_rn(1.0f + exp2_approx(-gate.x * kLog2E));
    y *= __frcp_rn(1.0f + exp2_approx(-gate.y * kLog2E));
    *reinterpret_cast<unsigned*>(a.out + offset) = pack_bf16x2(x, y);
}

template <int N, int Warps, bool VectorCache>
__device__ __forceinline__ void sparse_gqa_mma(SparseGqaArguments a, SparseGqaStorage<N, Warps>& s,
                                               std::int64_t q, int head, int split) {
    constexpr int NK = N / Warps, QKNt = NK / 8, PVKs = NK / 16;
    static_assert(NK >= 16 && NK % 16 == 0);
    auto* ks      = s.kv;
    auto* vs      = s.kv + N * 256;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
    const auto work = a.work(q);
    if (work.size == 0) {
        if (a.numerator == nullptr)
            for (int i = tid; i < 12 * 128; i += Warps * 32)
                *reinterpret_cast<unsigned*>(a.out + (q * 24 + head * 12 + i / 128) * 256 +
                                             (i % 128) * 2) = 0;
        return;
    }
    if (split >= work.splits) return;
    const int start = (split * work.tiles / work.splits) * N;
    const int end   = min(work.size, ((split + 1) * work.tiles / work.splits) * N);
    // Q occupies K's shared storage only during the prologue, then remains in registers.
    for (int i = tid; i < 16 * 32; i += Warps * 32) {
        const int h = i / 32, d = i % 32 * 8;
        auto* dst = ks + h * 256 + causal_swizzle(h, d);
        if (h < 12)
            cp_async<16>(dst, a.query + (q * 24 + head * 12 + h) * 256 + d);
        else
            store_vec(dst, make_int4(0, 0, 0, 0));
    }
    cp_commit();
    cp_wait<0>();
    __syncthreads();
    unsigned qf[16][4];
    const int ar = (lane & 7) + ((lane >> 3) & 1) * 8;

#pragma unroll
    for (int k = 0; k < 16; ++k) {
        const int d = k * 16 + (lane >> 4) * 8;
        ldmatrix_x4(qf[k][0], qf[k][1], qf[k][2], qf[k][3],
                    smem_addr(ks + ar * 256 + causal_swizzle(ar, d)));
    }

    __syncthreads();
    float acc[32][4] = {};
    Bf16KvSoftmaxRow state[2];
    constexpr float scale = kLog2E / 16.0f;
    const int gid = lane >> 2, lid = lane & 3;
    sparse_gqa_addresses(a, s, work, q, head, start, end);
    sparse_gqa_stage<N, Warps, VectorCache, true>(a, s, start, end);

    cp_commit();
    for (int k0 = start; k0 < end; k0 += N) {
        cp_wait<0>();
        __syncthreads();
        sparse_gqa_stage<N, Warps, VectorCache, false>(a, s, k0, end);
        cp_commit();
        float score[QKNt][4] = {};
#pragma unroll
        for (int n = 0; n < QKNt; ++n) {
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                unsigned bf[2];
                const int row = warp * NK + n * 8 + (lane & 7);
                const int d   = k * 16 + ((lane >> 3) & 1) * 8;
                ldmatrix_x2(bf[0], bf[1], smem_addr(ks + row * 256 + causal_swizzle(row, d)));
                mma_bf16(score[n][0], score[n][1], score[n][2], score[n][3], qf[k][0], qf[k][1],
                         qf[k][2], qf[k][3], bf[0], bf[1]);
            }
        }
        cp_wait<0>();
        __syncthreads();
        if (k0 + N < end) {
            sparse_gqa_addresses(a, s, work, q, head, k0 + N, end);
            sparse_gqa_stage<N, Warps, VectorCache, true>(a, s, k0 + N, end);
            cp_commit();
        }
        float maximum[2] = {-CUDART_INF_F, -CUDART_INF_F};
#pragma unroll
        for (int n = 0; n < QKNt; ++n) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int ordinal = k0 + warp * NK + n * 8 + 2 * lid + (j & 1);
                if (ordinal >= end) score[n][j] = -CUDART_INF_F;
                maximum[j / 2] = fmaxf(maximum[j / 2], score[n][j]);
            }
        }
        const float alpha[2] = {state[0].update(warp_max<4>(maximum[0], 0xffffffffu), scale),
                                state[1].update(warp_max<4>(maximum[1], 0xffffffffu), scale)};
        unsigned pf[PVKs][4];
        float tile_sum[2] = {};
#pragma unroll
        for (int n = 0; n < QKNt; ++n) {
            const float p0 = state[0].probability(score[n][0], scale);
            const float p1 = state[0].probability(score[n][1], scale);
            const float p2 = state[1].probability(score[n][2], scale);
            const float p3 = state[1].probability(score[n][3], scale);
            tile_sum[0] += p0 + p1;
            tile_sum[1] += p2 + p3;
            pf[n / 2][(n % 2) * 2]     = pack_f16x2(p0, p1);
            pf[n / 2][(n % 2) * 2 + 1] = pack_f16x2(p2, p3);
        }
        state[0].accumulate(alpha[0], tile_sum[0]);
        state[1].accumulate(alpha[1], tile_sum[1]);
#pragma unroll
        for (int n = 0; n < 32; ++n) {
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[n][j] *= alpha[j / 2];
#pragma unroll
            for (int k = 0; k < PVKs; ++k) {
                unsigned vf[2];
                const int row = warp * NK + k * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
                const int d   = n * 8;
                ldmatrix_x2_t(vf[0], vf[1], smem_addr(vs + row * 256 + causal_swizzle(row, d)));
                mma_f16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[k][0], pf[k][1], pf[k][2],
                        pf[k][3], vf[0], vf[1]);
            }
        }
        __syncthreads();
    }
    state[0].finish();
    state[1].finish();
    float maximum[2] = {state[0].maximum * scale, state[1].maximum * scale};
    float sum[2]     = {state[0].sum, state[1].sum};
    if constexpr (Warps > 1) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int row = warp * 16 + gid + j * 8;
            if (lid == 0) {
                s.statistics.maximum[row] = maximum[j];
                s.statistics.sum[row]     = sum[j];
            }
#pragma unroll
            for (int n = 0; n < 32; ++n)
                *reinterpret_cast<float2*>(s.partial + row * 256 + n * 8 + lid * 2) =
                    make_float2(acc[n][2 * j], acc[n][2 * j + 1]);
        }
        __syncthreads();
        if (warp != 0) return;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int row = gid + j * 8;
            maximum[j]    = -CUDART_INF_F;
            sum[j]        = 0;
#pragma unroll
            for (int w = 0; w < Warps; ++w)
                maximum[j] = fmaxf(maximum[j], s.statistics.maximum[w * 16 + row]);
            float weights[Warps];
#pragma unroll
            for (int w = 0; w < Warps; ++w) {
                weights[w] = bf16_kv_state_weight(s.statistics.maximum[w * 16 + row],
                                                  s.statistics.sum[w * 16 + row], maximum[j]);
                sum[j] += weights[w] * s.statistics.sum[w * 16 + row];
            }
#pragma unroll
            for (int n = 0; n < 32; ++n) {
                float x = 0, y = 0;
#pragma unroll
                for (int w = 0; w < Warps; ++w) {
                    const auto a = *reinterpret_cast<const float2*>(
                        s.partial + (w * 16 + row) * 256 + n * 8 + lid * 2);
                    x += weights[w] * a.x;
                    y += weights[w] * a.y;
                }
                acc[n][2 * j]     = x;
                acc[n][2 * j + 1] = y;
            }
        }
    }
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int h = gid + j * 8;
        if (h >= 12) continue;
        const auto row    = a.partial_row(q, split, head * 12 + h);
        const bool direct = a.numerator == nullptr;
        const float inv   = direct ? __frcp_rn(sum[j]) : 1.0f;
#pragma unroll
        for (int n = 0; n < 32; ++n) {
            const int d = n * 8 + lid * 2;
            if (direct)
                sparse_gqa_store(a, q, head * 12 + h, d, acc[n][2 * j] * inv,
                                 acc[n][2 * j + 1] * inv);
            else
                *reinterpret_cast<float2*>(a.numerator + row * 256 + d) =
                    make_float2(acc[n][2 * j], acc[n][2 * j + 1]);
        }
        if (!direct && lid == 0) {
            a.maximum[row] = maximum[j];
            a.sum[row]     = sum[j];
        }
    }
}

template <int Queries>
__device__ __forceinline__ bool sparse_gqa_all_selected(const SparseGqaArguments& a,
                                                        std::int64_t first) {
    const auto total = std::int64_t(a.tokens) * a.batch;
    const auto last  = min(first + Queries, total);
    if (first / a.tokens != (last - 1) / a.tokens) return false;
    for (auto q = first; q < last; ++q) {
        const auto work = a.work(q);
        if (work.count != work.complete) return false;
    }
    return true;
}

template <int N, int Warps, bool VectorCache>
__launch_bounds__(Warps * 32) __global__ void sparse_gqa_mma_kernel(SparseGqaArguments a) {
    extern __shared__ __align__(16) unsigned char smem[];
    auto& s = *reinterpret_cast<SparseGqaStorage<N, Warps>*>(smem);
    sparse_gqa_mma<N, Warps, VectorCache>(a, s, a.begin + blockIdx.x / 2, blockIdx.x % 2,
                                          blockIdx.y);
}

static __global__ void sparse_gqa_merge_kernel(SparseGqaArguments a) {
    const auto q = a.begin + blockIdx.x / 24;
    const int h = blockIdx.x % 24, d = threadIdx.x * 2;
    const auto work = a.work(q);
    if (!work.size) {
        *reinterpret_cast<unsigned*>(a.out + (q * 24 + h) * 256 + d) = 0;
        return;
    }
    __shared__ float weights[64], denominator;
    if (threadIdx.x < 32) {
        const int lane = threadIdx.x;
        float m        = -CUDART_INF_F;
        for (int s = lane; s < work.splits; s += 32)
            m = fmaxf(m, a.maximum[a.partial_row(q, s, h)]);
        m         = warp_max(m);
        float sum = 0;
        for (int s = lane; s < work.splits; s += 32) {
            const auto i  = a.partial_row(q, s, h);
            const float w = bf16_kv_state_weight(a.maximum[i], a.sum[i], m);
            weights[s]    = w;
            sum += w * a.sum[i];
        }
        sum = warp_sum(sum);
        if (lane == 0) denominator = sum;
    }
    __syncthreads();
    float x = 0, y = 0;
    for (int s = 0; s < work.splits; ++s) {
        const auto v =
            *reinterpret_cast<const float2*>(a.numerator + a.partial_row(q, s, h) * 256 + d);
        x += weights[s] * v.x;
        y += weights[s] * v.y;
    }
    const float inv = __frcp_rn(denominator);
    sparse_gqa_store(a, q, h, d, x * inv, y * inv);
}

} // namespace ninfer::ops::detail::qsa
