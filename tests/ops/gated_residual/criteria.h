#pragma once
#include "ops/op_check.h"

namespace ninfer::test::hc {
// BF16 RNE has unit roundoff 2^-8. Coherent storage error can approach that
// bound even on constant, unit-scale inputs; random-input RMS limits are not
// universal bounds. Leave the small remaining margin for FP32 FMA/reduction/rsqrt.
// Nonfinite results fail. These budgets cover all forms/routes and all cases.
inline constexpr PointwiseCriterion kResidual{1.0e-7, 4.0e-3};
inline constexpr ReductionCriterion kNormalized{4.0e-3, 1.0e-6, 4.0e-3};
} // namespace ninfer::test::hc
