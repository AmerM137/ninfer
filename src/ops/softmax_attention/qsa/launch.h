#pragma once

#include "ninfer/ops/qsa.h"

namespace ninfer::ops::detail::qsa {

void launch_prepare(const QsaProjectionView& projected, const QsaPrepareParameters& parameters,
                    const QsaBatchView& batch, const Tensor& rope_positions,
                    const QsaTailStateView& before, QsaCacheView cache,
                    QsaRawRecordWriteView records, Tensor& query, Tensor& index_query,
                    DeviceExecutionView execution);

void launch_tail_extract(const QsaTailStateView& before, const QsaRawRecordReadView& records,
                         const Tensor& accepted, QsaTailStateView after,
                         DeviceExecutionView execution);

} // namespace ninfer::ops::detail::qsa
