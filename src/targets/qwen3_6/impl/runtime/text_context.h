#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.

#include "targets/qwen3_6/impl/runtime/linear_state_slots.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/gqa_attention.h"
#include <ninfer/targets/qwen3_6/decoder_state.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>
#include <ninfer/targets/qwen3_6/round_state.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

using Phase = qwen3_6::TextPhase;

enum class GdnStateAction : std::uint8_t {
    UpdateInPlace,
    RecordForReplay,
};

struct NullTap {
    static constexpr bool enabled = false;
};

struct PrefillChunkResult {
    std::uint32_t processed_tokens = 0;
    bool finalized                 = false;
};

struct DFlashFeatureSink {
    static constexpr bool enabled = true;
    using PrefillConsumer         = std::function<void(const Tensor&, const Tensor&, bool)>;

    Tensor* features                  = nullptr;
    Tensor* positions                 = nullptr;
    Tensor* batch_features            = nullptr;
    const Tensor* batch_lanes         = nullptr;
    const Tensor* batch_valid_columns = nullptr;
    std::int32_t batch_width          = 0;
    std::int32_t batch_size           = 0;
    std::span<const int> layers;
    PrefillConsumer consume_prefill;
    std::uint32_t captured_mask = 0;
    std::int32_t active_tokens  = 0;

    void begin(const Tensor& value);
    void capture_layer(int layer, const Tensor& value, cudaStream_t stream);
    void capture_positions(const Tensor& source, cudaStream_t stream);
    void consume_prefill_chunk(std::int32_t tokens, bool rewrite_checkpoint);
};

class VisionPrefillSession;

class TextContext {
public:
    TextContext(DeviceContext& device, const LoadedModelData& model, WorkspaceArena& workspace,
                qwen3_6::PagedKVCacheView text_kv, LinearAttentionStatePool& linear_state,
                qwen3_6::RoundState& round, Tensor& prefill_hidden,
                std::uint32_t prefill_chunk_size, std::uint32_t text_kv_base,
                qwen3_6::PagedKVCacheView mtp_kv           = qwen3_6::PagedKVCacheView(),
                const qwen3_6::PagedKVCache* batch_text_kv = nullptr,
                const qwen3_6::PagedKVCache* batch_mtp_kv  = nullptr);
    TextContext(const TextContext&)            = delete;
    TextContext& operator=(const TextContext&) = delete;

    void set_proposal_head(const Weight* weight, const std::int32_t* ids, int count) noexcept {
        proposal_head_weight = weight;
        proposal_token_ids   = ids;
        proposal_token_count = count;
    }

    void set_sampling(const ops::SamplingConfig* config) noexcept { sampling = config; }

    void set_prefill_rewrite_checkpoint_frontier(std::int64_t position) noexcept {
        rewrite_checkpoint_frontier = position;
    }

    void set_rewrite_checkpoint_hidden_output(Tensor* output) noexcept {
        rewrite_checkpoint_hidden = output;
    }

    void set_mtp_proposal_extent(std::uint32_t extent) noexcept { mtp_proposal_extent = extent; }

    void set_linear_state_slots(std::int32_t current_slot, std::int32_t rewrite_checkpoint_slot);
    void set_gdn_state_action(GdnStateAction action, const GdnReplayRecords* replay_records);

    [[nodiscard]] const Weight* proposal_head() const noexcept { return proposal_head_weight; }

    [[nodiscard]] const std::int32_t* proposal_head_ids() const noexcept {
        return proposal_token_ids;
    }

    [[nodiscard]] int proposal_head_count() const noexcept { return proposal_token_count; }

    [[nodiscard]] PrefillChunkResult prefill_chunk(std::span<const int> full_ids,
                                                   std::uint32_t begin,
                                                   std::uint32_t nominal_length,
                                                   bool finalize_at_end);
    [[nodiscard]] PrefillChunkResult prefill_chunk(std::span<const int> full_ids,
                                                   std::uint32_t begin,
                                                   std::uint32_t nominal_length,
                                                   bool finalize_at_end, DFlashFeatureSink& sink);
    [[nodiscard]] PrefillChunkResult
    prefill_chunk(const qwen3_6::PreparedPromptData& input, std::uint32_t begin,
                  std::uint32_t nominal_length, VisionPrefillSession& vision, bool finalize_at_end);
    void ordinary_decode_batch(const Tensor& ids, const Tensor& cache_positions,
                               const Tensor& rope_positions, const Tensor& kv_table_rows,
                               const Tensor& linear_state_slots, ops::GqaExecutionEnvelope envelope,
                               Tensor& hidden, Tensor& logits);
    void target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                             const Tensor& rope_positions, const Tensor& valid_columns,
                             const Tensor& kv_table_rows, const Tensor& linear_state_slots,
                             ops::GqaExecutionEnvelope envelope, Tensor& hidden, Tensor& logits,
                             Tensor& target_tokens);
    void target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                             const Tensor& rope_positions, const Tensor& valid_columns,
                             const Tensor& kv_table_rows, const Tensor& linear_state_slots,
                             ops::GqaExecutionEnvelope envelope, Tensor& hidden, Tensor& logits,
                             Tensor& target_tokens, DFlashFeatureSink& sink);
    void mtp_forward_decode_batch(const Tensor& ids, const Tensor& hidden,
                                  const Tensor& cache_positions, const Tensor& rope_positions,
                                  const Tensor& valid_columns, const Tensor& kv_table_rows,
                                  ops::GqaExecutionEnvelope envelope, Tensor& mtp_hidden);
    void mtp_propose_batch(const Tensor& hidden, Tensor& logits, Tensor& draft_tokens);
    void mtp_forward_batch(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                           ops::GqaExecutionEnvelope envelope, Tensor& mtp_hidden,
                           int logits_column, Tensor* logits, Tensor* draft_token,
                           const Tensor* explicit_rope_positions = nullptr,
                           const Tensor* input_embeddings        = nullptr);
    void mtp_forward_ar_step(const Tensor& token, const Tensor& previous_hidden,
                             const Tensor& position, ops::GqaExecutionEnvelope envelope,
                             Tensor& mtp_hidden, Tensor& logits, Tensor& draft_token);
private:
    [[nodiscard]] bool mtp_enabled() const noexcept {
        return mtp_kv.valid() || batch_mtp_kv != nullptr;
    }

    [[nodiscard]] const MtpWeights& mtp_weights() const;
    void attn_mix(const FullAttentionWeights& weights, Tensor& x, int index, Phase phase);
    void gdn_mix(const GdnWeights& weights, Tensor& x, int index, Phase phase);
    void mlp_tail(const Tensor& post_norm, const MlpWeights& weights, Tensor& x, Phase phase);
    void run_layers(Tensor& x, Phase phase);
    template <class Tap>
    void run_layers(Tensor& x, Phase phase, Tap& tap);
    template <class Tap>
    void target_verify_batch_impl(const Tensor& ids, const Tensor& cache_positions,
                                  const Tensor& rope_positions, const Tensor& valid_columns,
                                  const Tensor& kv_table_rows, const Tensor& linear_state_slots,
                                  ops::GqaExecutionEnvelope envelope, Tensor& hidden,
                                  Tensor& logits, Tensor& target_tokens, Tap& tap);

    void mtp_forward_stem(const Tensor& ids, const Tensor& hidden, const Tensor* input_embeddings,
                          Tensor& x, Tensor& ah);
    void mtp_forward_tail(Tensor& x, const Tensor& ah, const Tensor& positions,
                          const Tensor& rope_positions, ops::GqaExecutionEnvelope envelope,
                          Tensor& mtp_hidden);
    void mtp_forward_core(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                          const Tensor& rope_positions, ops::GqaExecutionEnvelope envelope,
                          Tensor& mtp_hidden, const Tensor* input_embeddings);
    void mtp_prefill_chunk(const Tensor& ids, const Tensor& hidden, const Tensor* input_embeddings,
                           const Tensor& positions, const Tensor& rope_positions,
                           ops::GqaExecutionEnvelope envelope, bool final_chunk,
                           Tensor* final_hidden, Tensor* logits, Tensor* draft_token);
    void proposal_argmax(const Tensor& hidden, Tensor& logits, Tensor& proposal_tokens);

    struct MultimodalPrefill {
        std::span<const int> token_ids;
        std::span<const std::int32_t> positions;
        VisionPrefillSession* vision = nullptr;
        std::uint32_t begin          = 0;
        std::int32_t rope_delta      = 0;
    };

    struct TextPrefill {
        std::span<const int> token_ids;
        std::uint32_t begin = 0;
    };

    template <class Tap>
    [[nodiscard]] PrefillChunkResult
    prefill_impl(std::span<const int> ids, const TextPrefill* text_prefill,
                 const MultimodalPrefill* multimodal, Tap& tap, bool finalize_at_end);

    // Stable state borrowed for this schedule execution.
    DeviceContext& device;
    const LoadedModelData& model;
    WorkspaceArena& workspace;
    qwen3_6::PagedKVCacheView text_kv;
    qwen3_6::PagedKVCacheView mtp_kv;
    const qwen3_6::PagedKVCache* batch_text_kv = nullptr;
    const qwen3_6::PagedKVCache* batch_mtp_kv  = nullptr;
    LinearAttentionStatePool& linear_state;
    qwen3_6::RoundState& round;
    Tensor& prefill_hidden;
    std::uint32_t prefill_chunk_size;
    std::uint32_t text_kv_base;

    // Temporary bindings installed while executing a decode or verification batch.
    const Tensor* active_cache_positions                 = nullptr;
    const Tensor* active_rope_positions                  = nullptr;
    const Tensor* active_kv_table_rows                   = nullptr;
    const Tensor* active_linear_state_slots              = nullptr;
    const Tensor* active_valid_columns                   = nullptr;
    const Tensor* active_backend_kv_table_rows           = nullptr;
    const ops::GqaExecutionEnvelope* active_gqa_envelope = nullptr;
    std::int32_t active_sequence_batch                   = 0;
    std::int32_t active_sequence_width                   = 0;

    // Per-request execution policy and mutable schedule state.
    std::int32_t rope_delta                    = 0;
    std::int32_t current_state_slot            = 0;
    std::int32_t rewrite_checkpoint_state_slot = 0;
    GdnStateAction gdn_state_action            = GdnStateAction::UpdateInPlace;
    const GdnReplayRecords* gdn_replay_records = nullptr;
    std::int64_t rewrite_checkpoint_frontier   = -1;
    Tensor* rewrite_checkpoint_hidden          = nullptr;
    std::uint32_t mtp_proposal_extent          = 0;

    // Output projection selected for the current request.
    const Weight* proposal_head_weight     = nullptr;
    const std::int32_t* proposal_token_ids = nullptr;
    int proposal_token_count               = 0;
    const ops::SamplingConfig* sampling    = nullptr;
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
