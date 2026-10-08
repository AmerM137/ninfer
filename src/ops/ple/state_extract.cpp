#include "ninfer/ops/ple.h"
#include "ops/ple/launch.h"
#include "ops/ple/validation.h"

namespace ninfer::ops {

void ple_state_extract(const PleConvStateView& before, const PleConvRecordReadView& records,
                       const Tensor& accepted_columns, PleConvStateView after,
                       DeviceExecutionView execution) {
    detail::ple::validate_extract(before, records, accepted_columns, after, execution);
    detail::ple::launch_extract(before, records, accepted_columns, after, execution);
}

} // namespace ninfer::ops
