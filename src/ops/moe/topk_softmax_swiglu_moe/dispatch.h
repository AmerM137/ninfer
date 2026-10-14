#pragma once

#include "ops/moe/kernels/cuda/launch.h"
#include "ops/moe/kernels/cuda/block_scaled/launch.h"

#include <span>

namespace ninfer::ops::detail::moe_complete {

struct Plan {
    RouterPlan router;
    TokenPlan token;
    GroupedPlan group;
    bool grouped = false;
    moe_block_scaled::Plan block_scaled;
};

struct ShapePolicy {
    Plan (*select)(int tokens);
    std::span<const int> allocation_boundaries;
};

const ShapePolicy& resolve_shape(const MoeProfile& profile);
const ShapePolicy& q4_q5_h2048_i512_e256_k8();
const ShapePolicy& q4_q6_h2048_i512_e256_k8();
const ShapePolicy& q8_q8_h2048_i512_e256_k8();
const ShapePolicy& nvfp4_h2560_i640_e512_k10();
const ShapePolicy& fp8_h2560_i640_e512_k10();

} // namespace ninfer::ops::detail::moe_complete
