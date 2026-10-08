#pragma once

#include "ninfer/ops/ple.h"

namespace ninfer::ops::detail::ple {

void launch_conv(const Tensor& residual, const Tensor& gated, const Tensor& kernel,
                 const PleBatchView& batch, const PleConvStateView& before,
                 const PleConvRecordWindowView& current, Tensor& updated, const Tensor* norm_weight,
                 float epsilon, Tensor* normalized, DeviceExecutionView execution);
void launch_extract(const PleConvStateView& before, const PleConvRecordReadView& records,
                    const Tensor& accepted, PleConvStateView after, DeviceExecutionView execution);

} // namespace ninfer::ops::detail::ple
