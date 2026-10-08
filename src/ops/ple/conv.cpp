#include "ninfer/ops/ple.h"
#include "ops/ple/launch.h"
#include "ops/ple/validation.h"

namespace ninfer::ops {

void ple_conv_add(const Tensor& residual, const Tensor& gated_value, const Tensor& kernel,
                  const PleBatchView& batch, const PleConvStateView& before,
                  const PleConvRecordWindowView& current, Tensor& updated_residual,
                  DeviceExecutionView execution) {
    detail::ple::validate_conv(residual, gated_value, kernel, batch, before, current,
                               updated_residual, nullptr, 0, nullptr, execution);
    detail::ple::launch_conv(residual, gated_value, kernel, batch, before, current,
                             updated_residual, nullptr, 0, nullptr, execution);
}

void ple_conv_add_norm(const Tensor& residual, const Tensor& gated_value, const Tensor& kernel,
                       const PleBatchView& batch, const PleConvStateView& before,
                       const PleConvRecordWindowView& current, const Tensor& next_hc_norm_weight,
                       float next_hc_epsilon, Tensor& updated_residual, Tensor& normalized,
                       DeviceExecutionView execution) {
    detail::ple::validate_conv(residual, gated_value, kernel, batch, before, current,
                               updated_residual, &next_hc_norm_weight, next_hc_epsilon, &normalized,
                               execution);
    detail::ple::launch_conv(residual, gated_value, kernel, batch, before, current,
                             updated_residual, &next_hc_norm_weight, next_hc_epsilon, &normalized,
                             execution);
}

} // namespace ninfer::ops
