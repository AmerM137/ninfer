#include "models/qwen4_exp/program/buffers.h"
#include "ninfer/ops/sampling.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp {
BufferLayout plan_buffers(LayoutBuilder& b, const Config& c, const EngineOptions& o,
                          std::uint32_t chunk, std::uint32_t public_tokens) {
    const auto batch     = static_cast<std::int32_t>(o.max_concurrency);
    const auto width     = static_cast<std::int32_t>(c.mtp ? o.speculative.draft_tokens + 1 : 1);
    const auto positions = std::max(static_cast<std::int32_t>(chunk), batch * width);
    const auto tensor    = [&](DType dtype, std::initializer_list<std::int32_t> shape,
                            const char* label) { return b.add_tensor(dtype, shape, 256, label); };
    const auto rows      = [&] { return tensor(DType::I32, {batch}, "batch metadata"); };
    BufferLayout out;
    out.batch.ids               = tensor(DType::I32, {positions}, "input IDs");
    out.batch.positions         = tensor(DType::I32, {positions}, "input ordinals");
    out.batch.rope_positions    = tensor(DType::I32, {positions, 3}, "input RoPE coordinates");
    out.batch.valid_columns     = rows();
    out.batch.source_slots      = rows();
    out.batch.destination_slots = rows();
    out.batch.main_tables       = rows();
    if (c.mtp) out.batch.backend_tables = rows();
    out.batch.base_positions   = rows();
    out.batch.record_offsets   = rows();
    out.batch.recorded_columns = rows();
    out.batch.accepted_columns = rows();
    out.batch.max_positions    = positions;
    out.batch.max_batch        = o.max_concurrency;
    out.prefill_sample =
        tensor(DType::BF16, {2560, static_cast<std::int32_t>(chunk)}, "prefill sample");
    out.pending_sample = tensor(DType::BF16, {2560, width, batch}, "pending sample");
    const auto records = [&](std::uint32_t layers, std::int32_t columns) {
        return QsaRecordLayout{
            tensor(DType::BF16, {128, columns, batch, static_cast<std::int32_t>(layers)},
                   "QSA raw records"),
            tensor(DType::I32, {3, columns, batch, static_cast<std::int32_t>(layers)},
                   "QSA coordinate records"),
            layers, static_cast<std::uint32_t>(columns), o.max_concurrency};
    };
    out.main_qsa    = records(c.text.qsa_layers, width);
    out.ple_records = tensor(DType::BF16, {10240, width, batch}, "PLE pending records");
    if (c.mtp) {
        out.prefill_conditioning =
            tensor(DType::BF16, {10240, static_cast<std::int32_t>(chunk)}, "prefill conditioning");
        out.pending_conditioning =
            tensor(DType::BF16, {10240, width, batch}, "pending conditioning");
        out.gdn_records = plan_gdn_replay_records(b, {static_cast<std::int32_t>(c.text.gdn_layers),
                                                      batch, width, 10240, 16, 48, 128, 128});
        out.mtp_qsa     = records(1, width + 1);
        out.mtp_hidden  = tensor(DType::BF16, {10240, 2, batch}, "MTP working residuals");
        out.mtp_sample  = tensor(DType::BF16, {2560, batch}, "MTP sample");
        out.draft_ids   = tensor(DType::I32, {width - 1, batch}, "MTP draft IDs");
    }
    const bool scoring = o.purpose == EnginePurpose::CausalScoring;
    if (!scoring) {
        out.logits =
            tensor(DType::BF16, {static_cast<std::int32_t>(c.text.vocabulary), width * batch},
                   "head logits");
        out.sampled_ids   = tensor(DType::I32, {width, batch}, "generated IDs");
        out.output_counts = rows();
        if (c.mtp) {
            out.next_anchors    = rows();
            out.accepted_drafts = rows();
        }
        out.token_counts =
            tensor(DType::I32, {static_cast<std::int32_t>(public_tokens), batch}, "token counts");
        out.sampling =
            tensor(DType::I32, {static_cast<std::int32_t>(sizeof(ops::SamplingConfig) / 4), batch},
                   "sampling config");
        out.grammar_masks =
            tensor(DType::I32, {static_cast<std::int32_t>((public_tokens + 31) / 32), width, batch},
                   "grammar masks");
    }
    return out;
}
} // namespace ninfer::models::qwen4_exp
