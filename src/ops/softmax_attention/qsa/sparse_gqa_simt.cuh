#pragma once
#include "ops/softmax_attention/qsa/sparse_gqa_kernel.cuh"

namespace ninfer::ops::detail::qsa {
template <int Warps, bool Vector>
__launch_bounds__(Warps * 32) __global__ void sparse_gqa_simt_kernel(SparseGqaArguments a) {
    const auto q   = a.begin + blockIdx.x / 24;
    const int head = blockIdx.x % 24, kvh = head / 12, split = blockIdx.y;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
    const auto work = a.work(q);
    if (!work.size) {
        if (a.numerator == nullptr)
            for (int d = tid; d < 256; d += Warps * 32)
                a.out[(q * 24 + head) * 256 + d] = __float2bfloat16(0.f);
        return;
    }
    if (split >= work.splits) return;
    const int start = split * work.tiles / work.splits * a.key_rows;
    const int end   = min(work.size, (split + 1) * work.tiles / work.splits * a.key_rows);
    float query[8], acc[8] = {}, maximum = -CUDART_INF_F, sum = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i)
        query[i] = __bfloat162float(a.query[(q * 24 + head) * 256 + lane * 8 + i]);
    const auto table = a.tables + std::int64_t(a.table_rows[q / a.tokens]) * a.table_stride;
    for (int j = start + warp; j < end; j += Warps) {
        const int ordinal  = j / 4;
        const int g        = ordinal < work.count
                                 ? (work.count == work.complete ? ordinal : a.ids[q * 512 + ordinal])
                                 : work.complete;
        const int physical = table[g / 16], row = 4 * (g % 16) + j % 4;
        const auto ko = physical * a.key.page + kvh * a.key.head + row * a.key.row + lane * 8;
        const auto vo = physical * a.value.page + kvh * a.value.head + row * a.value.row + lane * 8;

        union {
            int4 vector;
            std::uint16_t word[8];
        } k, v;

        if constexpr (Vector) {
            k.vector = load_vec<int4>(a.key.data + ko);
            v.vector = load_vec<int4>(a.value.data + vo);
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                k.word[i] = a.key.data[ko + i];
                v.word[i] = a.value.data[vo + i];
            }
        }
        float dot = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            dot = fmaf(query[i], __bfloat162float(__ushort_as_bfloat16(k.word[i])), dot);
        dot              = warp_sum(dot) * (kLog2E / 16.f);
        const float next = fmaxf(maximum, dot), alpha = exp2_approx(maximum - next),
                    p = exp2_approx(dot - next);
        sum           = fmaf(alpha, sum, p);
        maximum       = next;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            acc[i] = fmaf(alpha, acc[i], p * __half2float(__ushort_as_half(v.word[i])));
    }
    __shared__ float output[Warps][256], maxima[Warps], sums[Warps];
    if (lane == 0) {
        maxima[warp] = maximum;
        sums[warp]   = sum;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) output[warp][lane * 8 + i] = acc[i];
    __syncthreads();
    if (warp != 0) return;
    maximum = -CUDART_INF_F;
    sum     = 0;
#pragma unroll
    for (int w = 0; w < Warps; ++w) maximum = fmaxf(maximum, maxima[w]);
    float weights[Warps];
#pragma unroll
    for (int w = 0; w < Warps; ++w) {
        weights[w] = bf16_kv_state_weight(maxima[w], sums[w], maximum);
        sum += weights[w] * sums[w];
    }
    const bool direct  = a.numerator == nullptr;
    const float inv    = direct ? __frcp_rn(sum) : 1.f;
    const auto partial = a.partial_row(q, split, head);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int d = lane * 8 + i * 2;
        float x = 0, y = 0;
#pragma unroll
        for (int w = 0; w < Warps; ++w) {
            x += weights[w] * output[w][d];
            y += weights[w] * output[w][d + 1];
        }
        if (direct)
            sparse_gqa_store(a, q, head, d, x * inv, y * inv);
        else
            *reinterpret_cast<float2*>(a.numerator + partial * 256 + d) = make_float2(x, y);
    }
    if (!direct && lane == 0) {
        a.maximum[partial] = maximum;
        a.sum[partial]     = sum;
    }
}
} // namespace ninfer::ops::detail::qsa
