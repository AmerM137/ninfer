#include "ops/moe/topk_softmax_swiglu_moe/dispatch.h"
#include <array>

namespace ninfer::ops::detail::moe_complete {
namespace {
constexpr int kGroupedFirst = 47;
constexpr std::array kAllocationBoundaries{2, kGroupedFirst, 768};

Plan select(int tokens) {
    Plan plan;
    plan.grouped = tokens >= kGroupedFirst;
    plan.router  = {tokens == 1    ? RouterAlgorithm::Vector
                    : plan.grouped ? RouterAlgorithm::Mma
                                   : RouterAlgorithm::TokenTile,
                   !plan.grouped, plan.grouped};
    plan.token = {tokens == 1 ? 9 : 3,
                  tokens == 1    ? 1
                  : tokens <= 2  ? 1
                  : tokens <= 17 ? 2
                                 : 4,
                  tokens == 1, true};
    plan.group          = {tokens >= 768 ? 64 : 32, false, 7, 4, 32, 64};
    plan.group.adaptive = plan.grouped && tokens <= 51;
    if (plan.group.adaptive) { plan.token = {3, 4, false, false}; }
    return plan;
}
} // namespace

const ShapePolicy& q4_q5_h2048_i512_e256_k8() {
    static const ShapePolicy policy{select, kAllocationBoundaries};
    return policy;
}
} // namespace ninfer::ops::detail::moe_complete
