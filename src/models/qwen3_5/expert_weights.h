#pragma once

#include "artifact/materializer.h"
#include "models/qwen3_5/weights.h"

namespace ninfer::models::qwen3_5 {

namespace loading {
struct PendingWeight;
}

struct ExpertMatrixSource {
    artifact::ObjectHandle object;
    WeightBankLayout layout;
};

struct ExpertBankDescription {
    ExpertMatrixSource gate_up, down;
    bool mtp = false;
};

[[nodiscard]] std::vector<ExpertBankDescription>
describe_expert_weights(const artifact::Reader& reader, ModelWeights& weights,
                        std::span<const loading::PendingWeight> pending);
[[nodiscard]] std::vector<ExpertBankSource>
bind_expert_weights(std::span<const ExpertBankDescription> banks,
                    const artifact::MaterializedArtifact& backing, std::span<const Device> devices);

} // namespace ninfer::models::qwen3_5
