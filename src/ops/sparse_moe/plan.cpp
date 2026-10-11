#include "ops/sparse_moe/plan.h"
#include "ops/sparse_moe/workspace.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::moe_detail {

int prefill_first(QType gate_up, QType down) {
    if (gate_up == QType::Q4_G64_FP16 && (down == QType::Q5_G64_FP16 || down == QType::Q6_G64_FP16))
        return 47;
    if (gate_up == QType::Q8_G32_FP16 && down == QType::Q8_G32_FP16) return 20;
    throw std::invalid_argument("sparse_moe: unsupported routed codec profile");
}

Plan select_plan(int tokens, QType gate_up, QType down, const SparseMoeHints& hints) {
    const int grouped_first = prefill_first(gate_up, down);
    if (tokens <= 0) throw std::invalid_argument("sparse_moe: T must be positive");
    WorkspaceLayoutBuilder layout;
    if (tokens == 1) {
        (void)allocate_decode_workspace(layout);
        DecodePlan plan{layout.peak_bytes(1)};
        if (hints.next_weight_prefetch) {
            plan.next_weight_prefetch = hints.next_weight_prefetch;
            plan.next_weight_prefetch_bytes =
                std::min(hints.next_weight_prefetch_bytes, std::size_t{8} << 20);
        }
        return plan;
    }
    if (tokens >= grouped_first) {
        const int slice_tokens = std::min(tokens, kPrefillSliceMax);
        (void)allocate_prefill_workspace(layout, slice_tokens);
        return PrefillPlan{tokens, slice_tokens, layout.peak_bytes(1)};
    }
    (void)allocate_small_workspace(layout, tokens);
    SmallPlan plan{tokens, layout.peak_bytes(1)};
    if (gate_up == QType::Q8_G32_FP16) {
        plan.gate_up = tokens <= 5 ? GateUpSchedule::Paths1 : GateUpSchedule::Paths9;
        plan.down    = tokens <= 8 ? DownSchedule::Rows1 : DownSchedule::Rows4;
    } else {
        plan.gate_up            = GateUpSchedule::Paths3;
        const int two_rows_last = down == QType::Q5_G64_FP16 ? 17 : 11;
        plan.down               = tokens <= 2               ? DownSchedule::Rows1
                                  : tokens <= two_rows_last ? DownSchedule::Rows2
                                                            : DownSchedule::Rows4;
    }
    return plan;
}

std::size_t workspace_bytes(const Plan& plan) {
    return std::visit([](const auto& selected) { return selected.workspace_bytes; }, plan);
}

} // namespace ninfer::ops::moe_detail
