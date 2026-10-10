#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace ninfer::ops {

/**
 * destination[:,indices[i]] = source[:,i].
 *
 * Source [D,V] and destination [D,T] are contiguous BF16; indices [V] is contiguous I32.
 * Indices lie in [0,T), with no duplicates when deterministic overwrite order is required.
 * Copies BF16 bits exactly and preserves unselected columns. Inputs and destination do not
 * overlap. An empty source does no work. No workspace or other state is used.
 */
void scatter(const Tensor& source, const Tensor& indices, Tensor& destination, cudaStream_t stream);

/**
 * destination[:,j,lanes[b]] = source[:,j,b], for j < valid_columns[b].
 *
 * Source is contiguous BF16 [D,W,B]; lanes and valid_columns are contiguous I32 [B].
 * Destination is a BF16 [D,W,C] view with contiguous features and explicit column/lane strides.
 * A feature slice retains its parent's strides. D is divisible by eight and every BF16 column
 * starts at a 16-byte-aligned address. Inputs and destination do not overlap.
 *
 * The caller supplies distinct lanes in [0,C), counts in [0,W], and disjoint destination
 * columns/lanes. Copies BF16 bits exactly; invalid tails, unselected lanes and features outside
 * the destination slice are preserved. Zero counts read no source values. No workspace is used.
 */
void scatter_bf16_batch(const Tensor& source, const Tensor& lanes, const Tensor& valid_columns,
                        Tensor& destination, cudaStream_t stream);

/**
 * destination[d,t] = source[feature_offset+d,t].
 *
 * Source [D,T] and destination [F,T] are non-overlapping contiguous BF16 matrices with positive
 * extents. The feature interval [feature_offset,feature_offset+F) lies within [0,D). Copies BF16
 * bits exactly, writes all of destination and preserves source. No workspace is used.
 */
void extract_bf16_features(const Tensor& source, std::int32_t feature_offset, Tensor& destination,
                           cudaStream_t stream);

} // namespace ninfer::ops
