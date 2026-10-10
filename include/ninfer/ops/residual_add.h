#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * x[i] = x[i] + y[i].
 *
 * x/y are same-shaped, contiguous, non-overlapping BF16 tensors.
 * Updates x in place and preserves y. Empty tensors perform no device work.
 */
void residual_add(const Tensor& y, Tensor& x, cudaStream_t stream);

} // namespace ninfer::ops
