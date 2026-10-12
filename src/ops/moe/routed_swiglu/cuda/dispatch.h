#pragma once

#include "ops/moe/kernels/cuda/launch.h"

#include <span>

namespace ninfer::ops::detail::moe_routed {

struct Plan {
    TokenPlan token;
    GroupedPlan group;
    bool grouped = false;
};

struct ShapePolicy {
    Plan (*select)(int tokens, bool shared);
    std::span<const int> allocation_boundaries;
};

const ShapePolicy& resolve_shape(const RoutedSwiGLUProfile& profile);
const ShapePolicy& q4_q5_h2048_i512_e256_k8();
const ShapePolicy& q4_q6_h2048_i512_e256_k8();
const ShapePolicy& q8_q8_h2048_i512_e256_k8();
SharedPlan select_shared(int tokens);

} // namespace ninfer::ops::detail::moe_routed
