#pragma once

#include "core/cyclic_kv_cache.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Host launch-resource promise for device-selected prefix append. It bounds every device count
 * during capture/replay but neither selects nor publishes a committed frontier.
 */
struct KVCacheAppendPrefixExecutionEnvelope {
    std::uint32_t min_count = 0;
    std::uint32_t max_count = 0;
};

/**
 * Append every K/V row to single-sequence paged growing-cache storage.
 *
 * k/v are contiguous BF16 [256,4|2,T] and positions is contiguous sequential device I32 [T].
 * BF16 cache rows are copied bit-for-bit. INT8-G64 cache rows use one scale for each contiguous
 * 64-value group. For represented BF16 source values x, their exact observable encoding is
 *
 *   a          = max_i abs(FP32(x[i]))
 *   scale_bits = FP16_RNE(a / 127)
 *   s          = FP32(scale_bits)
 *   inv        = s == 0 ? 0 : FP32(1 / s)
 *   code[i]    = s == 0 ? 0 : I8(clamp(RNE_even(FP32(x[i]) * inv), -127, 127))
 *   decode[i]  = FP32(code[i]) * s.
 *
 * The fused append performed by causal_softmax_attention produces identical code and scale bits.
 * Every addressed code/value and scale is overwritten, and no unrelated cache row is read or
 * written. Inputs and every cache plane/table are pairwise non-overlapping. The Op owns no
 * persistent allocation, frontier, request identity, or commit authority.
 */
void kv_cache_append(const Tensor& k, const Tensor& v, const Tensor& positions,
                     PagedKVLayerView cache, cudaStream_t stream);

/**
 * Append device-selected exact BF16 prefixes to batched paged growing-cache storage.
 *
 * k/v are contiguous BF16 [128,8,T,B], positions is contiguous device I32 [T,B], and counts and
 * table_rows are contiguous device I32 [B]. For row b and i in [0,counts[b]), k/v[:, :, i, b]
 * are copied bit-for-bit to logical position positions[i,b] through table row table_rows[b]. The
 * paged planes use head-major order [128,64,Nphysical,8]. No byte belonging only to the rejected
 * physical tail [counts[b],T) is written. Inputs are unchanged, and the Op neither decides nor
 * publishes a committed frontier.
 *
 * The caller guarantees T>0, B=1..8, envelope.min_count <= counts[b] <= envelope.max_count <= T,
 * sequential nonnegative live positions, materialized table entries for every position allowed by
 * the envelope, and pairwise non-aliasing of inputs and cache storage.
 */
void kv_cache_append_prefix(const Tensor& k, const Tensor& v, const Tensor& positions,
                            const Tensor& counts, const Tensor& table_rows,
                            KVCacheAppendPrefixExecutionEnvelope envelope,
                            PagedKVBatchLayerView cache, cudaStream_t stream);

/**
 * Append device-selected exact BF16 prefixes to lane-owned cyclic storage.
 *
 * k/v, positions, counts, and their exact-copy and mutation contracts match the paged overload;
 * lanes[b] selects the destination lane. The fixed geometry is D=128, Hkv=8, capacity=4096, and
 * absolute position p maps to slot p mod 4096. The caller guarantees that each row's existing live
 * interval ends immediately before positions[0,b], advancing it by counts[b] makes every
 * overwritten old slot dead, and one row commits at most the ring capacity. Consequently, no two
 * live writes race for one physical slot. The Op does not own or publish the lane frontier.
 */
void kv_cache_append_prefix(const Tensor& k, const Tensor& v, const Tensor& positions,
                            const Tensor& counts, const Tensor& lanes,
                            KVCacheAppendPrefixExecutionEnvelope envelope,
                            CyclicKVCacheLayerView cache, cudaStream_t stream);

} // namespace ninfer::ops
