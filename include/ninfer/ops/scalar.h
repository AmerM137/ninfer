#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace ninfer::ops {

// Every tensor is one contiguous, non-null element of its named type. Add/increment callers
// supply values whose exact result is representable. Each call writes only its destination;
// assign/add require destination storage distinct from inputs. No workspace is used.

// destination = value.
void set_i32_scalar(Tensor& destination, std::int32_t value, cudaStream_t stream);
// destination = source.
void assign_i32_scalar(const Tensor& source, Tensor& destination, cudaStream_t stream);
// destination = lhs+rhs.
void add_i32_scalars(const Tensor& lhs, const Tensor& rhs, Tensor& destination,
                     cudaStream_t stream);
// scalar += 1.
void increment_i32_scalar(Tensor& scalar, cudaStream_t stream);
void increment_i64_scalar(Tensor& scalar, cudaStream_t stream);

} // namespace ninfer::ops
