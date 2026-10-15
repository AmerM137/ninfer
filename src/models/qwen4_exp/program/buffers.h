#pragma once

#include "core/gdn_replay_records.h"
#include "models/qwen4_exp/config.h"

namespace ninfer::models::qwen4_exp {
struct QsaRecordLayout {
    TensorRegion keys, coordinates;
    std::uint32_t layers, width, batch;
};

struct BatchLayout {
    TensorRegion ids, positions, rope_positions;
    std::optional<TensorRegion> visual_indices;
    TensorRegion valid_columns, source_slots, destination_slots, main_tables, backend_tables;
    TensorRegion base_positions, record_offsets, recorded_columns, accepted_columns;
    std::uint32_t max_positions, max_batch;
};

struct BufferLayout {
    BatchLayout batch;
    TensorRegion prefill_sample;
    std::optional<TensorRegion> prefill_conditioning;
    TensorRegion pending_sample;
    std::optional<TensorRegion> pending_conditioning;
    QsaRecordLayout main_qsa;
    TensorRegion ple_records;
    std::optional<GdnReplayRecordLayout> gdn_records;
    std::optional<QsaRecordLayout> mtp_qsa;
    std::optional<TensorRegion> mtp_hidden, mtp_sample, draft_ids;
    std::optional<TensorRegion> logits;
    std::optional<TensorRegion> sampled_ids, output_counts, next_anchors, accepted_drafts;
    std::optional<TensorRegion> token_counts, sampling, grammar_masks;
};

[[nodiscard]] BufferLayout plan_buffers(LayoutBuilder&, const Config&, const EngineOptions&,
                                        std::uint32_t chunk, std::uint32_t public_tokens);
} // namespace ninfer::models::qwen4_exp
