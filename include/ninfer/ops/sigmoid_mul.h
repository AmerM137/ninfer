#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Elementwise sigmoid gate:
 *
 *   ideal[i] = x[i] * (1 / (1 + exp(-gate[i]))).
 *
 * gate/x are same-shaped, contiguous, non-overlapping BF16 tensors.
 * Updates x in place and preserves gate. Empty tensors perform no device work.
 */
void sigmoid_mul(const Tensor& gate, Tensor& x, cudaStream_t stream);

} // namespace ninfer::ops
