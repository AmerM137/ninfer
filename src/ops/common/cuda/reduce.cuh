#pragma once

#include <cuda_runtime.h>

#include <climits>
#include <cstdint>

namespace ninfer::ops::common {

inline constexpr int kWarpSize          = 32;
inline constexpr unsigned kFullWarpMask = 0xffffffffu;

template <int Width = kWarpSize, class T>
__device__ __forceinline__ T warp_sum(T x, unsigned mask = kFullWarpMask) {
    static_assert(Width > 0 && Width <= kWarpSize && (Width & (Width - 1)) == 0);
#pragma unroll
    for (int offset = Width / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(mask, x, offset, Width);
    }
    return x;
}

template <int Width = kWarpSize, class T>
__device__ __forceinline__ T warp_reduce_sum(T x, unsigned mask = kFullWarpMask) {
    static_assert(Width > 0 && Width <= kWarpSize && (Width & (Width - 1)) == 0);
#pragma unroll
    for (int offset = Width / 2; offset > 0; offset >>= 1) {
        x += __shfl_down_sync(mask, x, offset, Width);
    }
    return x;
}

template <int Width = kWarpSize>
__device__ __forceinline__ float warp_max(float x, unsigned mask = kFullWarpMask) {
    static_assert(Width > 0 && Width <= kWarpSize && (Width & (Width - 1)) == 0);
#pragma unroll
    for (int offset = Width / 2; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(mask, x, offset, Width));
    }
    return x;
}

template <int BlockSize>
__device__ __forceinline__ float block_reduce_sum(float x, float* sums) {
    static_assert(BlockSize >= kWarpSize && BlockSize <= 1024);
    static_assert((BlockSize & (BlockSize - 1)) == 0);
    constexpr int Warps = BlockSize / kWarpSize;

    x = warp_reduce_sum(x);
    if constexpr (Warps == 1) { return x; }

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    if (lane == 0) { sums[warp] = x; }
    __syncthreads();

    x = threadIdx.x < Warps ? sums[lane] : 0.0f;
    if (warp == 0) { x = warp_reduce_sum<Warps>(x); }
    return x;
}

// Finite score/id total order: larger scores win, ties prefer lower ids; signed
// zero is canonicalized. Key zero is reserved below every valid score/id pair.
struct ScoreIdOrderGreater {
    __device__ __forceinline__ bool operator()(std::uint64_t lhs, std::uint64_t rhs) const {
        return lhs > rhs;
    }
};

__device__ __forceinline__ std::uint32_t ordered_score_bits(float value) {
    if (value == 0.0f) { value = 0.0f; }
    const std::uint32_t bits = __float_as_uint(value);
    return (bits & 0x80000000u) != 0 ? ~bits : (bits ^ 0x80000000u);
}

__device__ __forceinline__ std::uint64_t score_id_order_key(float value, std::int32_t id) {
    if (id == INT_MAX) { return 0; }
    return (static_cast<std::uint64_t>(ordered_score_bits(value)) << 32) |
           static_cast<std::uint32_t>(0xffffffffu - static_cast<std::uint32_t>(id));
}

__device__ __forceinline__ float score_from_order_key(std::uint64_t key) {
    const std::uint32_t ordered = static_cast<std::uint32_t>(key >> 32);
    const std::uint32_t bits    = (ordered & 0x80000000u) != 0 ? (ordered ^ 0x80000000u) : ~ordered;
    return __uint_as_float(bits);
}

__device__ __forceinline__ std::int32_t id_from_order_key(std::uint64_t key) {
    if (key == 0) { return INT_MAX; }
    return static_cast<std::int32_t>(0xffffffffu - static_cast<std::uint32_t>(key));
}

} // namespace ninfer::ops::common
