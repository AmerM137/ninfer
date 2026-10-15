#pragma once

#include "models/common/qwen/vision/parameters.h"
#include "models/load_options.h"

#include <optional>
#include <vector>

namespace ninfer::artifact {
struct Directory;
}

namespace ninfer::models::qwen4_exp {

enum class MixerKind { Gdn, Qsa };

struct LayerConfig {
    MixerKind mixer;
    std::uint32_t state_index;
};

struct HcConfig {
    std::uint32_t streams, lowrank;
};

struct GdnConfig {
    std::uint32_t key_heads, value_heads, key_dim, value_dim, conv_width;

    [[nodiscard]] std::uint32_t key_width() const { return key_heads * key_dim; }

    [[nodiscard]] std::uint32_t value_width() const { return value_heads * value_dim; }

    [[nodiscard]] std::uint32_t conv_channels() const { return 2 * key_width() + value_width(); }
};

struct QsaConfig {
    std::uint32_t query_heads, kv_heads, head_dim;
    std::uint32_t index_heads, index_dim, compression, block_budget;
    float rope_theta;

    [[nodiscard]] std::uint32_t query_width() const { return query_heads * head_dim; }

    [[nodiscard]] std::uint32_t kv_width() const { return kv_heads * head_dim; }
};

struct MoeConfig {
    std::uint32_t experts, top_k, intermediate, shared_intermediate;
};

struct PleConfig {
    std::uint32_t layer, shards, rows_per_shard;
    std::int32_t eos;
};

struct TextConfig {
    std::uint32_t hidden, vocabulary, max_context;
    float norm_epsilon;
    HcConfig hc;
    GdnConfig gdn;
    QsaConfig qsa;
    MoeConfig moe;
    PleConfig ple;
    std::vector<LayerConfig> layers;
    std::uint32_t gdn_layers = 0, qsa_layers = 0;

    [[nodiscard]] std::uint32_t residual_width() const { return hidden * hc.streams; }
};

struct Config {
    TextConfig text;
    std::optional<qwen::VisionConfig> vision;
    bool mtp = false;
};

[[nodiscard]] Config parse_config(const artifact::Directory&, const LoadOptions&);

} // namespace ninfer::models::qwen4_exp
