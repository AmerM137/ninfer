#pragma once

#include "ops/common/host/operands.h"
#include "ops/linear_attention/gated_delta_net/operands.h"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail::gated_delta_net::replay {
constexpr std::int32_t kStateDim    = ::ninfer::ops::detail::gated_delta_net::kStateDim;
constexpr std::int32_t kMaximumRows = 8;

inline common::operand::Range layer_range(const Tensor& layer0, std::int64_t stride_bytes,
                                          std::int32_t layer, const char* label) {
    const auto base    = reinterpret_cast<std::uintptr_t>(layer0.data);
    const auto stride  = static_cast<std::uint64_t>(stride_bytes);
    const auto layer_u = static_cast<std::uint64_t>(layer);
    if (layer_u != 0 && stride > std::numeric_limits<std::uintptr_t>::max() / layer_u) {
        throw std::overflow_error(std::string(label) + " layer offset overflows");
    }
    const auto offset = static_cast<std::uintptr_t>(stride * layer_u);
    if (offset > std::numeric_limits<std::uintptr_t>::max() - base) {
        throw std::overflow_error(std::string(label) + " layer address overflows");
    }
    return common::operand::range(reinterpret_cast<const void*>(base + offset), layer0.bytes(),
                                  label);
}

} // namespace ninfer::ops::detail::gated_delta_net::replay
