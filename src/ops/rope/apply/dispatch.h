#pragma once

#include "core/device.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void rope_dispatch(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
                   DeviceExecutionView execution);

void rope_dispatch(const Tensor& positions, int rotary_dim, float theta, Tensor& x,
                   DeviceExecutionView execution);

} // namespace ninfer::ops::detail
