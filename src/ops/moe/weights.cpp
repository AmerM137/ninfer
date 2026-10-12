#include "ninfer/ops/moe.h"

#include <limits>
#include <stdexcept>

namespace ninfer::ops {
namespace {
std::int32_t dimension(std::uint64_t value) {
    if (!value || value > std::uint64_t(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("SwiGLU bank dimension is outside the Tensor domain");
    }
    return static_cast<std::int32_t>(value);
}

QuantLayout matrix_layout(QuantLayout layout) {
    return layout == QuantLayout::BatchedBlockScaleK16M128x4 ? QuantLayout::BlockScaleK16M128x4
                                                             : layout;
}
} // namespace

SwiGLUProfile swiglu_profile(const SwiGLUWeights& weights) {
    const auto& gu   = weights.gate_up.weight;
    const auto& down = weights.down.weight;
    if (gu.n <= 0 || gu.n % 2 || gu.k <= 0 || down.n != gu.k || down.k != gu.n / 2 ||
        !valid_linear_policy(weights.gate_up.policy) || !valid_linear_policy(weights.down.policy)) {
        throw std::invalid_argument("SwiGLU matrices have inconsistent geometry or policy");
    }
    return {gu.k,
            gu.n / 2,
            gu.qtype,
            down.qtype,
            gu.layout,
            down.layout,
            weights.gate_up.policy,
            weights.down.policy};
}

RoutedSwiGLUProfile routed_swiglu_profile(const WeightBankView& gate_up, const WeightBankView& down,
                                          std::int32_t top_k, LinearPolicy gate_up_policy,
                                          LinearPolicy down_policy) {
    const auto& gu = gate_up.layout;
    const auto& dn = down.layout;
    if (!gate_up.parent || !down.parent || gu.region.rows % 2 ||
        gu.region.count != dn.region.count || gu.columns != dn.region.rows ||
        gu.region.rows / 2 != dn.columns || top_k <= 0 || std::uint32_t(top_k) > gu.region.count ||
        !valid_linear_policy(gate_up_policy) || !valid_linear_policy(down_policy)) {
        throw std::invalid_argument("routed SwiGLU banks have inconsistent geometry or policy");
    }
    return {{dimension(gu.columns), dimension(dn.columns), gu.format, dn.format,
             matrix_layout(gu.layout), matrix_layout(dn.layout), gate_up_policy, down_policy},
            dimension(gu.region.count),
            top_k};
}
} // namespace ninfer::ops
