#pragma once

#include "core/weight.h"
#include "ninfer/ops/linear.h"

#include <variant>

namespace ninfer::ops {

// Prepared, non-owning operands. The consuming Op defines the logical row order.
struct SingleProjectionWeight {
    Weight weight;
    LinearPolicy policy = LinearPolicy::A16Only;
};

// The registered paired forms use A16 for both parents.
struct PairedProjectionWeights {
    Weight first, second;
};

using ProjectionWeights = std::variant<SingleProjectionWeight, PairedProjectionWeights>;

} // namespace ninfer::ops
