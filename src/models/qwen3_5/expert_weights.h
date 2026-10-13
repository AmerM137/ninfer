#pragma once

#include "models/common/moe/weights.h"
#include "models/qwen3_5/weights.h"

namespace ninfer::models::qwen3_5 {
namespace loading {
struct PendingWeight;
}

[[nodiscard]] std::vector<moe::BankDescription>
describe_expert_weights(const artifact::Reader&, ModelWeights&,
                        std::span<const loading::PendingWeight>);

} // namespace ninfer::models::qwen3_5
