#include "ops/moe/topk_softmax_swiglu_moe/dispatch.h"
#include <array>

namespace ninfer::ops::detail::moe_complete {
namespace {
constexpr int kGroupedFirst = 20;
constexpr int kWideFirst    = 384;
constexpr std::array kAllocationBoundaries{2, kGroupedFirst, kWideFirst};

Plan select(int tokens) {
    Plan plan;
    plan.grouped = tokens >= kGroupedFirst;
    plan.router  = {tokens == 1    ? RouterAlgorithm::Vector
                    : plan.grouped ? RouterAlgorithm::Mma
                                   : RouterAlgorithm::TokenTile,
                   !plan.grouped, plan.grouped};
    plan.token = {tokens == 1   ? 9
                  : tokens <= 5 ? 1
                                : 9,
                  tokens == 1   ? 1
                  : tokens <= 8 ? 1
                                : 4,
                  tokens == 1, true};
    plan.group = {tokens >= kWideFirst ? 64 : 32, false, 7, 4, 32, 64};
    return plan;
}
} // namespace

const ShapePolicy& q8_q8_h2048_i512_e256_k8() {
    static const ShapePolicy policy{select, kAllocationBoundaries};
    return policy;
}
} // namespace ninfer::ops::detail::moe_complete
