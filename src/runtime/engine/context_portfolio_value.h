#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::runtime {

struct ContextPortfolioOwnerPolicy {
    std::uint32_t ordinal              = 0;
    std::uint32_t private_prior_weight = 0;
    bool explicit_shared_credit        = false;
};

struct ContextPortfolioCheckpointValue {
    std::uint32_t owner_ordinal        = 0;
    std::uint32_t demand_mask          = 0;
    std::uint64_t rebuild_ns           = 0;
    std::uint64_t baseline_recovery_ns = 0;
    std::uint64_t target_recovery_ns   = 0;
};

struct ContextPortfolioValueResult {
    std::uint64_t baseline = 0;
    std::uint64_t target   = 0;
    std::uint64_t loss     = 0;
    bool saturated         = false;
};

// Folds one complete inactive checkpoint portfolio. A demand record receives only the best
// matching checkpoint saving, so nested prefixes cannot count the same future request twice.
class ContextPortfolioValue {
public:
    ContextPortfolioValue() { owner_scratch_.reserve(32U); }

    [[nodiscard]] ContextPortfolioValueResult
    fold(std::span<const ContextPortfolioOwnerPolicy> owners,
         std::span<const ContextPortfolioCheckpointValue> checkpoints) {
        std::array<std::uint64_t, 32> baseline_demand{};
        std::array<std::uint64_t, 32> target_demand{};
        owner_scratch_.clear();
        for (const ContextPortfolioOwnerPolicy& policy : owners) {
            if (std::find_if(owner_scratch_.begin(), owner_scratch_.end(), [&](const auto& item) {
                    return item.ordinal == policy.ordinal;
                }) != owner_scratch_.end()) {
                throw std::logic_error("portfolio owner policy ordinal is duplicated");
            }
            owner_scratch_.push_back(
                OwnerValue{.ordinal                = policy.ordinal,
                           .private_prior_weight   = policy.private_prior_weight,
                           .explicit_shared_credit = policy.explicit_shared_credit});
        }

        for (const ContextPortfolioCheckpointValue& checkpoint : checkpoints) {
            const auto owner = std::find_if(
                owner_scratch_.begin(), owner_scratch_.end(),
                [&](const OwnerValue& item) { return item.ordinal == checkpoint.owner_ordinal; });
            if (owner == owner_scratch_.end()) {
                throw std::logic_error("portfolio checkpoint has no owner policy");
            }
            const std::uint64_t baseline_saving =
                checkpoint.rebuild_ns > checkpoint.baseline_recovery_ns
                    ? checkpoint.rebuild_ns - checkpoint.baseline_recovery_ns
                    : 0;
            const std::uint64_t target_saving =
                checkpoint.rebuild_ns > checkpoint.target_recovery_ns
                    ? checkpoint.rebuild_ns - checkpoint.target_recovery_ns
                    : 0;
            owner->baseline_best = std::max(owner->baseline_best, baseline_saving);
            owner->target_best   = std::max(owner->target_best, target_saving);
            for (std::uint32_t bit = 0; bit < 32U; ++bit) {
                if ((checkpoint.demand_mask & (1U << bit)) == 0) { continue; }
                baseline_demand[bit] = std::max(baseline_demand[bit], baseline_saving);
                target_demand[bit]   = std::max(target_demand[bit], target_saving);
            }
        }

        ContextPortfolioValueResult result;
        for (std::size_t bit = 0; bit < baseline_demand.size(); ++bit) {
            add(result.baseline, baseline_demand[bit], result.saturated);
            add(result.target, target_demand[bit], result.saturated);
        }
        for (const OwnerValue& owner : owner_scratch_) {
            if (owner.private_prior_weight != 0) {
                add_weighted(result.baseline, owner.baseline_best, owner.private_prior_weight,
                             result.saturated);
                add_weighted(result.target, owner.target_best, owner.private_prior_weight,
                             result.saturated);
            }
            if (owner.explicit_shared_credit) {
                add(result.baseline, owner.baseline_best, result.saturated);
                add(result.target, owner.target_best, result.saturated);
            }
        }
        result.loss = result.baseline > result.target ? result.baseline - result.target : 0;
        return result;
    }

private:
    struct OwnerValue {
        std::uint32_t ordinal              = 0;
        std::uint32_t private_prior_weight = 0;
        bool explicit_shared_credit        = false;
        std::uint64_t baseline_best        = 0;
        std::uint64_t target_best          = 0;
    };

    static void add(std::uint64_t& value, std::uint64_t increment, bool& saturated) noexcept {
        if (increment > std::numeric_limits<std::uint64_t>::max() - value) {
            value     = std::numeric_limits<std::uint64_t>::max();
            saturated = true;
        } else {
            value += increment;
        }
    }

    static void add_weighted(std::uint64_t& value, std::uint64_t increment, std::uint32_t weight,
                             bool& saturated) noexcept {
        if (weight != 0 && increment > std::numeric_limits<std::uint64_t>::max() / weight) {
            add(value, std::numeric_limits<std::uint64_t>::max(), saturated);
            saturated = true;
            return;
        }
        add(value, increment * weight, saturated);
    }

    std::vector<OwnerValue> owner_scratch_;
};

} // namespace ninfer::runtime
