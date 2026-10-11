#include "ninfer/ops/sparse_moe.h"
#include "ops/sparse_moe/operands.h"
#include "ops/sparse_moe/workspace.h"
#include "core/nvtx.h"

#include <algorithm>
#include <array>
#include <type_traits>

namespace ninfer::ops {

std::size_t sparse_moe_workspace_capacity_bytes(QType gate_up, QType down, std::int32_t min_tokens,
                                                std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens)
        throw std::invalid_argument("sparse_moe workspace: invalid token interval");
    const int grouped_first = moe_detail::prefill_first(gate_up, down);
    // Each recipe is monotone inside its route; also inspect both sides of route boundaries.
    const std::array candidates{min_tokens, max_tokens, 1, grouped_first - 1, grouped_first};
    std::size_t capacity = 0;
    for (int tokens : candidates) {
        if (tokens < min_tokens || tokens > max_tokens) continue;
        capacity = std::max(
            capacity, moe_detail::workspace_bytes(moe_detail::select_plan(tokens, gate_up, down)));
    }
    return capacity;
}

void sparse_moe(const Tensor& x, const SparseMoeWeights& weights, SparseMoeEpilogue epilogue,
                Tensor& destination, WorkspaceArena& workspace, DeviceExecutionView execution) {
    sparse_moe(x, weights, epilogue, destination, SparseMoeHints{}, workspace, execution);
}

void sparse_moe(const Tensor& x, const SparseMoeWeights& weights, SparseMoeEpilogue epilogue,
                Tensor& destination, const SparseMoeHints& hints, WorkspaceArena& workspace,
                DeviceExecutionView execution) {
    using namespace moe_detail;
    constexpr const char* op = "sparse_moe";
    if (execution.multiprocessor_count <= 0)
        throw std::invalid_argument("sparse_moe: physical SM count must be positive");
    if (epilogue != SparseMoeEpilogue::AddResidual)
        throw std::invalid_argument("sparse_moe: unsupported epilogue");
    const auto tokens = require_tensor(x, "x");
    if (require_tensor(destination, "destination") != tokens)
        throw std::invalid_argument("sparse_moe: x and destination token counts must match");

    std::vector<common::operand::Range> ranges;
    ranges.reserve(16);
    ranges.push_back(common::operand::range(x, op));
    ranges.push_back(common::operand::range(destination, op));
    validate_weights(weights, ranges);
    const auto plan =
        select_plan(tokens, weights.routed_gate_up.qtype, weights.routed_down.qtype, hints);
    const auto required = workspace_bytes(plan);
    if (!workspace.base() || workspace.capacity() < required ||
        workspace.used() > workspace.capacity() - required)
        throw std::invalid_argument("sparse_moe: insufficient workspace capacity");
    ranges.push_back(common::operand::range(workspace.base(), workspace.capacity(), op));
    common::operand::disjoint(ranges, op);

    auto scope = workspace.scope();
    std::visit(
        [&](const auto& selected) {
            using Selected = std::decay_t<decltype(selected)>;
            if constexpr (std::is_same_v<Selected, DecodePlan>) {
                nvtx::ScopedRange range(nvtx::Name::SparseMoeDecode, nvtx::Category::Moe, tokens);
                const auto scratch = allocate_decode_workspace(workspace);
                launch_decode(x, weights, destination, selected, scratch, execution.stream);
            } else if constexpr (std::is_same_v<Selected, SmallPlan>) {
                nvtx::ScopedRange range(nvtx::Name::SparseMoeSmallT, nvtx::Category::Moe, tokens);
                const auto scratch = allocate_small_workspace(workspace, tokens);
                launch_small(x, weights, destination, selected, scratch, execution.stream);
            } else {
                nvtx::ScopedRange range(nvtx::Name::SparseMoePrefill, nvtx::Category::Moe, tokens);
                const auto scratch = allocate_prefill_workspace(workspace, selected.slice_tokens);
                launch_prefill(x, weights, destination, selected, scratch, execution);
            }
        },
        plan);
}

} // namespace ninfer::ops
