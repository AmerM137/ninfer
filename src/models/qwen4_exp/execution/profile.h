#pragma once

#include "models/qwen4_exp/config.h"
#include "ninfer/ops/moe.h"
#include "core/expert_cache.h"

#include <memory>
#include <variant>

namespace ninfer::models::qwen4_exp {
class LoadPlan;
class Model;

namespace execution {
struct HcProfile {
    ops::LinearProfile down, up;
};

struct GdnProfile {
    ops::ProjectionProfile input, control;
    ops::LinearProfile output;
};

struct MainQsaProjectionProfile {
    ops::LinearProfile query_gate, kv_index;
};

struct QsaProfile {
    std::variant<MainQsaProjectionProfile, ops::LinearProfile> input;
    ops::LinearProfile output;
};

struct MoeProfile {
    ops::MoeProfile math;
    ExpertBankId bank;
    ops::ExpertActivationDivisors activation_divisors;
};

struct BlockProfile {
    HcProfile attention_hc, moe_hc;
    std::variant<GdnProfile, QsaProfile> mixer;
    MoeProfile moe;
};

struct TextProfile {
    std::vector<BlockProfile> layers;
    HcProfile final_hc;
    ops::LinearProfile ple, head;
};

struct MtpProfile {
    ops::LinearProfile fc_embedding, fc_hidden, head;
    BlockProfile layer;
    HcProfile final_hc;
};

// Immutable, address-free descriptions of the selected actual representations.
struct ModelProfile {
    Config config;
    LoadOptions options;
    std::uint32_t public_tokens;
    TextProfile text;
    std::optional<MtpProfile> mtp;
    std::optional<qwen::VisionProfile> vision;
    std::vector<ExpertBankCacheSpec> banks;
};

[[nodiscard]] std::shared_ptr<const ModelProfile> describe_model(const LoadPlan&);
[[nodiscard]] std::shared_ptr<const ModelProfile> describe_model(const Model&);
[[nodiscard]] std::size_t hc_workspace_bytes(const HcProfile&, std::int32_t first,
                                             std::int32_t last);
[[nodiscard]] std::size_t qsa_projection_workspace_bytes(const QsaProfile&, std::int32_t first,
                                                         std::int32_t last);
} // namespace execution
} // namespace ninfer::models::qwen4_exp
