#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>
#include <span>

namespace ninfer::ops {

/**
 * Y_s[r,t] = sum_k decode(W)[begin_s+r,k] * X[k,t],
 * where begin_s is the sum of preceding output row counts.
 *
 * X is BF16 [K,T], W represents [N,K], and outputs are ordered BF16 [N_s,T] matrices
 * with positive row counts summing to N. Every activation/output is contiguous, non-null
 * and 16-byte aligned; T is positive. Outputs are mutually disjoint and overlap neither
 * X nor any weight plane. Read-only inputs are preserved.
 *
 * Supports registered parent geometries and segment layouts for contiguous BF16 and
 * RowSplit Q8-G32 with FP16 scales. Logical and padded weight dimensions must agree.
 * Uses A16 computation and writes the final segments directly, without a packed parent
 * output, transient workspace, or hidden allocation. Capture/replay is supported.
 */
void linear_split(const Tensor& x, const Weight& weight, std::span<Tensor> outputs,
                  cudaStream_t stream);

} // namespace ninfer::ops
