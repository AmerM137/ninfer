#pragma once

#include "core/expert_cache.h"
#include "models/qwen3_5/execution/moe.h"
#include "ninfer/types.h"

#include <array>
#include <span>

namespace ninfer::models::qwen3_5 {
class LoadPlan;
struct TextConfig;

namespace execution {
struct ModelProfile;
}

namespace detail {

// Empty when the model geometry has no calibrated default.
[[nodiscard]] std::span<const std::uint32_t> default_expert_priority(const TextConfig&,
                                                                     std::size_t layer);

struct MoeLayerPlan {
    ExpertBankId bank;
    std::int32_t experts           = 0;
    std::int32_t hybrid_max_tokens = 0;
    std::vector<std::uint32_t> priority;
    std::uint32_t fixed_experts = 0;
    std::uint32_t cache_experts = 0;
    std::vector<ExpertPlacement> stream_layout;

    [[nodiscard]] bool requires_full_bank(std::int32_t tokens) const noexcept {
        return tokens > hybrid_max_tokens;
    }
};

struct MoePlan {
    std::vector<MoeLayerPlan> layers;
    ExpertCacheRequirements cache;
    execution::MoeExecutionRequirements bridge;
    CpuTeamPlan team;
    CpuAffinity caller_cpus;
    std::vector<ExpertPlacement> fixed_layout, extra_layout;
    std::array<LayoutRegion, 2> windows;
    std::uint32_t window_count = 0;
    std::size_t device_bytes   = 0;
    std::size_t target_bytes   = 0;
};

// Resolve fixed per-layer shares and streaming capacity together; then fill the reusable span
// with additional small-T experts. Returns false when even one streaming window cannot fit.
[[nodiscard]] bool plan_moe_cache(MoePlan&);

[[nodiscard]] MoePlan describe_moe_plan(const LoadPlan&, const execution::ModelProfile&,
                                        const MoeOptions& = {});
[[nodiscard]] std::optional<MoePlan> resolve_moe_plan(const LoadPlan&,
                                                      const execution::ModelProfile&,
                                                      std::size_t device_bytes, const MoeOptions&,
                                                      std::int32_t max_tokens, const CpuTopology&);

} // namespace detail
} // namespace ninfer::models::qwen3_5
