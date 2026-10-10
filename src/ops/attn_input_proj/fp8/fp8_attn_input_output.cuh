#pragma once

#include "ops/linear/common/output.cuh"

namespace ninfer::ops::detail {

using Fp8AttentionInputOutput = LinearBf16SegmentedOutput<6144, 1024, 6144, 1024>;

} // namespace ninfer::ops::detail
