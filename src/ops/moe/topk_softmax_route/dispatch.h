#pragma once

#include "ops/moe/kernels/cuda/launch.h"

#include <span>

namespace ninfer::ops::detail::moe_route {
RouterPlan select_h2048_e256_k8(int tokens);
std::span<const int> allocation_boundaries();
RouterPlan select_h2560_e512_k10(int tokens);
} // namespace ninfer::ops::detail::moe_route
