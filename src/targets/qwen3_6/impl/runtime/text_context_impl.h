#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/text_context.h"
#include "targets/qwen3_6/impl/runtime/workspace_recipe.h"

#include "core/nvtx.h"
#include "targets/qwen3_6/impl/runtime/visual_scatter.h"
#include "targets/qwen3_6/impl/runtime/vision_context.h"
#include <ninfer/targets/qwen3_6/vision_control.h>
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gdn_gating.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/gqa_attention.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/mtp_pack.h"
#include "ninfer/ops/position.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ninfer/ops/silu_mul.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
namespace {

void copy_i32(const std::int32_t* source, Tensor& destination, cudaStream_t stream) {
    if (source == nullptr || destination.dtype != DType::I32 || !destination.is_contiguous() ||
        destination.data == nullptr) {
        throw std::invalid_argument("copy_i32: invalid host source or I32 destination");
    }
    CUDA_CHECK(cudaMemcpyAsync(destination.data, source, destination.bytes(),
                               cudaMemcpyHostToDevice, stream));
}

void require_tensor_shape(const Tensor& t, DType dtype, std::initializer_list<std::int32_t> shape,
                          const char* label) {
    if (t.dtype != dtype) { throw std::invalid_argument(std::string(label) + " dtype mismatch"); }
    int i = 0;
    for (const std::int32_t dim : shape) {
        if (t.ne[i] != dim) { throw std::invalid_argument(std::string(label) + " shape mismatch"); }
        ++i;
    }
    for (; i < 4; ++i) {
        if (t.ne[i] != 1) { throw std::invalid_argument(std::string(label) + " shape mismatch"); }
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(label) + " must be contiguous");
    }
    if (t.data == nullptr) { throw std::invalid_argument(std::string(label) + " data is null"); }
}

void require_tensor_window(const Tensor& t, DType dtype, std::int32_t rows, std::int32_t cols,
                           const char* label) {
    if (cols <= 0) { throw std::invalid_argument(std::string(label) + " cols must be positive"); }
    if (t.dtype != dtype) { throw std::invalid_argument(std::string(label) + " dtype mismatch"); }
    if (t.ne[0] != rows || t.ne[1] < cols || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument(std::string(label) + " shape mismatch");
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(label) + " must be contiguous");
    }
    if (t.data == nullptr) { throw std::invalid_argument(std::string(label) + " data is null"); }
}

Tensor matrix_window(Tensor& t, std::int32_t cols) {
    if (cols <= 0) { throw std::invalid_argument("matrix_window cols must be positive"); }
    if (t.ne[1] < cols || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument("matrix_window shape mismatch");
    }
    return t.slice(1, 0, cols);
}

template <class T>
class ScopedValue {
public:
    ScopedValue(T& target, T value) : slot(target), previous(target) { slot = value; }

    ScopedValue(const ScopedValue&)            = delete;
    ScopedValue& operator=(const ScopedValue&) = delete;

    ~ScopedValue() { slot = previous; }

private:
    T& slot;
    T previous;
};

} // namespace

void DFlashFeatureSink::begin(const Tensor& value) {
    const bool prefill = features != nullptr && positions != nullptr && batch_features == nullptr;
    const bool batch   = batch_features != nullptr && batch_lanes != nullptr &&
                         batch_valid_columns != nullptr && batch_width > 0 && batch_size > 0;
    if ((!prefill && !batch) || layers.empty()) {
        throw std::logic_error("DFlash feature sink is incomplete");
    }
    captured_mask = 0;
    active_tokens = batch ? batch_width * batch_size : value.ne[1];
    if (value.ne[1] != active_tokens) {
        throw std::logic_error("DFlash batch feature source has an invalid width");
    }
}

void DFlashFeatureSink::capture_layer(int layer, const Tensor& value, cudaStream_t stream) {
    const auto it = std::find(layers.begin(), layers.end(), layer);
    if (it == layers.end()) { return; }
    const std::size_t index = static_cast<std::size_t>(it - layers.begin());
    Tensor* destination     = batch_features != nullptr ? batch_features : features;
    if (layers.size() > 32 || active_tokens <= 0 || value.dtype != DType::BF16 ||
        destination == nullptr ||
        value.ne[0] * static_cast<std::int32_t>(layers.size()) != destination->ne[0] ||
        value.ne[1] != active_tokens) {
        throw std::logic_error("DFlash feature capture shape is invalid");
    }
    if (batch_features != nullptr) {
        Tensor source = value.view({value.ne[0], batch_width, batch_size});
        Tensor target =
            batch_features->slice(0, static_cast<std::int32_t>(index) * value.ne[0], value.ne[0]);
        ops::scatter_bf16_batch(source, *batch_lanes, *batch_valid_columns, target, stream);
        captured_mask |= 1U << index;
        return;
    }
    if (active_tokens > features->ne[1]) {
        throw std::logic_error("DFlash prefill feature capture exceeds its buffer");
    }
    const std::size_t element_bytes = dtype_size(DType::BF16);
    const std::size_t width_bytes   = static_cast<std::size_t>(value.ne[0]) * element_bytes;
    const std::size_t source_pitch  = static_cast<std::size_t>(value.nb[1]);
    const std::size_t target_pitch  = static_cast<std::size_t>(features->nb[1]);
    auto* target                    = static_cast<std::byte*>(features->data) + index * width_bytes;
    CUDA_CHECK(cudaMemcpy2DAsync(target, target_pitch, value.data, source_pitch, width_bytes,
                                 static_cast<std::size_t>(active_tokens), cudaMemcpyDeviceToDevice,
                                 stream));
    captured_mask |= 1U << index;
}

void DFlashFeatureSink::capture_positions(const Tensor& source, cudaStream_t stream) {
    const std::uint32_t complete_mask = layers.size() == 32 ? ~0U : ((1U << layers.size()) - 1U);
    if (captured_mask != complete_mask) {
        throw std::logic_error("DFlash target call did not publish every feature layer");
    }
    if (batch_features != nullptr) {
        if (source.dtype != DType::I32 || source.ne[0] != batch_width ||
            source.ne[1] != batch_size) {
            throw std::logic_error("DFlash batch feature positions are invalid");
        }
        return;
    }
    if (active_tokens <= 0 || source.dtype != DType::I32 || source.ne[0] != active_tokens ||
        positions == nullptr || active_tokens > positions->ne[0]) {
        throw std::logic_error("DFlash feature positions are invalid");
    }
    CUDA_CHECK(cudaMemcpyAsync(positions->data, source.data,
                               static_cast<std::size_t>(active_tokens) * sizeof(std::int32_t),
                               cudaMemcpyDeviceToDevice, stream));
}

void DFlashFeatureSink::consume_prefill_chunk(std::int32_t tokens, bool rewrite_checkpoint) {
    if (!consume_prefill || tokens != active_tokens) {
        throw std::logic_error("DFlash prefill feature consumer is unavailable");
    }
    Tensor feature_window  = features->slice(1, 0, tokens);
    Tensor position_window = positions->slice(0, 0, tokens);
    consume_prefill(feature_window, position_window, rewrite_checkpoint);
}

TextContext::TextContext(DeviceContext& device, const LoadedModelData& model,
                         WorkspaceArena& workspace, qwen3_6::PagedKVCacheView text_kv,
                         LinearAttentionStatePool& linear_state, qwen3_6::RoundState& round,
                         Tensor& prefill_hidden, std::uint32_t prefill_chunk_size,
                         std::uint32_t text_kv_base, qwen3_6::PagedKVCacheView mtp_kv,
                         const qwen3_6::PagedKVCache* batch_text_kv,
                         const qwen3_6::PagedKVCache* batch_mtp_kv)
    : device(device), model(model), workspace(workspace), text_kv(text_kv), mtp_kv(mtp_kv),
      batch_text_kv(batch_text_kv), batch_mtp_kv(batch_mtp_kv), linear_state(linear_state),
      round(round), prefill_hidden(prefill_hidden), prefill_chunk_size(prefill_chunk_size),
      text_kv_base(text_kv_base) {
    if (prefill_chunk_size == 0 ||
        prefill_chunk_size > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("TextContext effective prefill chunk must fit positive int32");
    }
    if (mtp_enabled() && !round.mtp_decode && !round.mtp) {
        throw std::invalid_argument("MTP TextContext requires MTP round state");
    }
    if (mtp_enabled() && !model.mtp) {
        throw std::invalid_argument("MTP state was enabled without materialized MTP weights");
    }
    set_linear_state_slots(0, linear_state.slot_count() > 1 ? 1 : 0);
    if (model.optimized_proposal) {
        const auto& proposal = *model.optimized_proposal;
        set_proposal_head(&proposal.head, static_cast<const std::int32_t*>(proposal.token_ids.data),
                          proposal.head.n);
    }
}

void TextContext::set_linear_state_slots(std::int32_t current_slot,
                                         std::int32_t rewrite_checkpoint_slot) {
    if (current_slot < 0 || current_slot >= linear_state.slot_count() ||
        rewrite_checkpoint_slot < 0 || rewrite_checkpoint_slot >= linear_state.slot_count() ||
        current_slot == rewrite_checkpoint_slot) {
        throw std::invalid_argument("TextContext Linear Attention slots are invalid");
    }
    current_state_slot            = current_slot;
    rewrite_checkpoint_state_slot = rewrite_checkpoint_slot;
}

void TextContext::set_gdn_state_action(GdnStateAction action,
                                       const GdnReplayRecords* replay_records) {
    if ((action == GdnStateAction::RecordForReplay) != (replay_records != nullptr)) {
        throw std::invalid_argument("TextContext GDN state action has inconsistent records");
    }
    gdn_state_action   = action;
    gdn_replay_records = replay_records;
}

const MtpWeights& TextContext::mtp_weights() const {
    if (!mtp_enabled() || !model.mtp) {
        throw std::runtime_error("MTP draft weights are not enabled");
    }
    return *model.mtp;
}

void TextContext::mtp_forward_stem(const Tensor& ids, const Tensor& hidden,
                                   const Tensor* input_embeddings, Tensor& x, Tensor& ah) {
    const MtpWeights& mtp = mtp_weights();
    cudaStream_t s        = device.stream;
    const int T           = ids.ne[0] * ids.ne[1];
    Tensor flat_ids       = ids.view({T});
    Tensor flat_hidden    = hidden.view({TextConfig::hidden, T});

    auto roots = workspace_recipe::mtp_stem<TextConfig>(workspace, T, input_embeddings == nullptr);
    Tensor emb;
    if (input_embeddings != nullptr) {
        if (input_embeddings->dtype != DType::BF16 ||
            input_embeddings->ne[0] != TextConfig::hidden ||
            input_embeddings->numel() != static_cast<std::int64_t>(TextConfig::hidden) * T ||
            !input_embeddings->is_contiguous() || input_embeddings->data == nullptr) {
            throw std::invalid_argument("MTP input embeddings shape mismatch");
        }
        emb = input_embeddings->view({TextConfig::hidden, T});
    } else {
        emb = roots.embedding;
        ops::embedding(flat_ids, model.token_embedding, emb, s);
    }

    Tensor e = roots.normalized_embedding;
    Tensor h = roots.normalized_hidden;
    ops::rmsnorm(emb, mtp.embedding_norm, TextConfig::rms_epsilon, true, e, s);
    ops::rmsnorm(flat_hidden, mtp.hidden_norm, TextConfig::rms_epsilon, true, h, s);

    Tensor fc_in = roots.packed_input;
    ops::mtp_pack_fc_input(e, h, fc_in, s);

    x = roots.residual;
    ops::linear(fc_in, mtp.input_projection, x, s);

    ah = roots.attention_hidden;
    ops::rmsnorm(x, mtp.input_norm, TextConfig::rms_epsilon, true, ah, s);
}

void TextContext::mtp_forward_tail(Tensor& x, const Tensor& ah, const Tensor& positions,
                                   const Tensor& rope_positions, ops::GqaExecutionEnvelope envelope,
                                   Tensor& mtp_hidden) {
    const MtpWeights& mtp = mtp_weights();
    cudaStream_t s        = device.stream;
    const int T           = x.ne[1];

    const auto projection = workspace_recipe::mtp_attention_projection<TextConfig>(workspace, T);
    Tensor q         = projection.query.view({TextConfig::head_dim, TextConfig::query_heads, T});
    Tensor k         = projection.key.view({TextConfig::head_dim, TextConfig::kv_heads, T});
    Tensor gate      = projection.gate.view({TextConfig::head_dim, TextConfig::query_heads, T});
    Tensor v         = projection.value.view({TextConfig::head_dim, TextConfig::kv_heads, T});
    Tensor q_flat    = q.view({TextConfig::query_size, T});
    Tensor gate_flat = gate.view({TextConfig::query_size, T});
    Tensor k_flat    = k.view({TextConfig::kv_size, T});
    Tensor v_flat    = v.view({TextConfig::kv_size, T});
    Variant::mtp_attention_projection(ah, mtp.attention, q_flat, gate_flat, k_flat, v_flat,
                                      workspace, s);

    const auto results = workspace_recipe::mtp_attention_results<TextConfig>(workspace, T);
    Tensor qn = results.normalized_query.view({TextConfig::head_dim, TextConfig::query_heads, T});
    Tensor kn = results.normalized_key.view({TextConfig::head_dim, TextConfig::kv_heads, T});
    ops::rmsnorm(q, mtp.query_norm, TextConfig::rms_epsilon, true, qn, s);
    ops::rmsnorm(k, mtp.key_norm, TextConfig::rms_epsilon, true, kn, s);
    Tensor rope_for_op = active_sequence_batch != 0 ? rope_positions.view({T}) : rope_positions;
    ops::rope(rope_for_op, TextConfig::rotary_dim, TextConfig::rope_theta, qn, kn, s);

    Tensor a = results.attention.view({TextConfig::head_dim, TextConfig::query_heads, T});
    if (active_sequence_batch != 0) {
        const std::int32_t width = active_sequence_width;
        if (width <= 0 || width * active_sequence_batch != T ||
            active_backend_kv_table_rows == nullptr || active_valid_columns == nullptr) {
            throw std::logic_error("MTP sequence batch binding is incomplete");
        }
        Tensor q_batch =
            qn.view({TextConfig::head_dim, TextConfig::query_heads, width, active_sequence_batch});
        Tensor k_batch =
            kn.view({TextConfig::head_dim, TextConfig::kv_heads, width, active_sequence_batch});
        Tensor v_batch =
            v.view({TextConfig::head_dim, TextConfig::kv_heads, width, active_sequence_batch});
        Tensor a_batch =
            a.view({TextConfig::head_dim, TextConfig::query_heads, width, active_sequence_batch});
        Tensor position_batch = positions.view({width, active_sequence_batch});
        ops::gqa_attention(q_batch, k_batch, v_batch, position_batch, *active_valid_columns,
                           *active_backend_kv_table_rows, kAttentionScale,
                           batch_mtp_kv->batch_layer_view(0), envelope, workspace, a_batch, s);
    } else {
        ops::gqa_attention(qn, kn, v, positions, Tensor{}, round.backend_kv_table_row,
                           kAttentionScale, batch_mtp_kv->batch_layer_view(0), envelope, workspace,
                           a, s);
    }
    ops::sigmoid_mul(gate, a, s);

    const auto post = workspace_recipe::mtp_post_attention<TextConfig>(workspace, T);
    Tensor o        = post.output;
    ops::linear(a.view({TextConfig::query_size, T}), mtp.output, o, s);
    ops::residual_add(o, x, s);

    Tensor mh = post.post_mixer_hidden;
    ops::rmsnorm(x, mtp.post_attention_norm, TextConfig::rms_epsilon, true, mh, s);

    {
        auto post_mixer_scope = workspace.scope();
        Variant::mtp_post_mixer(mh, mtp.post_mixer, x, workspace, s);
    }

    Tensor flat_mtp_hidden = mtp_hidden.view({TextConfig::hidden, T});
    ops::rmsnorm(x, mtp.final_norm, TextConfig::rms_epsilon, true, flat_mtp_hidden, s);
}

void TextContext::mtp_forward_core(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                                   const Tensor& rope_positions, ops::GqaExecutionEnvelope envelope,
                                   Tensor& mtp_hidden, const Tensor* input_embeddings) {
    if (batch_mtp_kv == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    auto scratch_scope = workspace.scope();
    Tensor x;
    Tensor ah;
    mtp_forward_stem(ids, hidden, input_embeddings, x, ah);
    mtp_forward_tail(x, ah, positions, rope_positions, envelope, mtp_hidden);
}

void TextContext::mtp_prefill_chunk(const Tensor& ids, const Tensor& hidden,
                                    const Tensor* input_embeddings, const Tensor& positions,
                                    const Tensor& rope_positions,
                                    ops::GqaExecutionEnvelope envelope, bool final_chunk,
                                    Tensor* final_hidden, Tensor* logits, Tensor* draft_token) {
    if (!mtp_kv.valid()) { throw std::runtime_error("MTP prefill is not enabled"); }
    const MtpWeights& mtp = mtp_weights();
    const int T           = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_size) {
        throw std::invalid_argument("MTP prefill chunk T must be in [1,prefill_chunk]");
    }
    nvtx::ScopedRange mtp_prefill_range(nvtx::Name::PrefillMtpChunk, nvtx::Category::Mtp,
                                        static_cast<std::uint64_t>(T));
    require_tensor_shape(ids, DType::I32, {T}, "MTP prefill ids");
    require_tensor_shape(hidden, DType::BF16, {TextConfig::hidden, T}, "MTP prefill hidden");
    require_tensor_shape(positions, DType::I32, {T}, "MTP prefill positions");
    if (rope_positions.dtype != DType::I32 || rope_positions.ne[0] != T ||
        (rope_positions.ne[1] != 1 && rope_positions.ne[1] != 3) || rope_positions.ne[2] != 1 ||
        rope_positions.ne[3] != 1 || !rope_positions.is_contiguous() ||
        rope_positions.data == nullptr) {
        throw std::invalid_argument("MTP prefill rope positions must be [T] or [T,3]");
    }
    if (final_chunk) {
        if (final_hidden == nullptr || logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("MTP final prefill outputs are required");
        }
        require_tensor_shape(*final_hidden, DType::BF16, {TextConfig::hidden, 1},
                             "MTP final prefill hidden");
        require_tensor_shape(*logits, DType::BF16, {TextConfig::output_rows, 1},
                             "MTP final prefill logits");
        require_tensor_shape(*draft_token, DType::I32, {1}, "MTP final prefill draft token");
    }

    cudaStream_t s     = device.stream;
    auto scratch_scope = workspace.scope();
    Tensor x_last;
    Tensor ah_last;
    if (final_chunk) {
        x_last  = workspace.alloc(DType::BF16, {TextConfig::hidden, 1});
        ah_last = workspace.alloc(DType::BF16, {TextConfig::hidden, 1});
    }

    {
        auto bulk_scope = workspace.scope();
        Tensor x;
        Tensor ah;
        mtp_forward_stem(ids, hidden, input_embeddings, x, ah);

        Tensor k_flat = workspace.alloc(DType::BF16, {TextConfig::kv_size, T});
        Tensor v_flat = workspace.alloc(DType::BF16, {TextConfig::kv_size, T});
        Variant::mtp_kv_projection(ah, mtp.attention, k_flat, v_flat, workspace, s);
        Tensor k  = k_flat.view({TextConfig::head_dim, TextConfig::kv_heads, T});
        Tensor v  = v_flat.view({TextConfig::head_dim, TextConfig::kv_heads, T});
        Tensor kn = workspace.alloc(DType::BF16, {TextConfig::head_dim, TextConfig::kv_heads, T});
        ops::rmsnorm(k, mtp.key_norm, TextConfig::rms_epsilon, true, kn, s);
        ops::rope(rope_positions, TextConfig::rotary_dim, TextConfig::rope_theta, kn, s);
        ops::gqa_kv_append(kn, v, positions, mtp_kv.layer_view(0), s);

        if (final_chunk) {
            const std::size_t column_bytes =
                static_cast<std::size_t>(TextConfig::hidden) * dtype_size(DType::BF16);
            const auto* x_src  = static_cast<const unsigned char*>(x.data) +
                                 static_cast<std::size_t>(T - 1) * column_bytes;
            const auto* ah_src = static_cast<const unsigned char*>(ah.data) +
                                 static_cast<std::size_t>(T - 1) * column_bytes;
            CUDA_CHECK(
                cudaMemcpyAsync(x_last.data, x_src, column_bytes, cudaMemcpyDeviceToDevice, s));
            CUDA_CHECK(
                cudaMemcpyAsync(ah_last.data, ah_src, column_bytes, cudaMemcpyDeviceToDevice, s));
        }
    }

    if (final_chunk) {
        Tensor q_flat    = workspace.alloc(DType::BF16, {TextConfig::query_size, 1});
        Tensor gate_flat = workspace.alloc(DType::BF16, {TextConfig::query_size, 1});
        Variant::mtp_q_gate_projection(ah_last, mtp.attention, q_flat, gate_flat, workspace, s);
        Tensor q    = q_flat.view({TextConfig::head_dim, TextConfig::query_heads, 1});
        Tensor gate = gate_flat.view({TextConfig::head_dim, TextConfig::query_heads, 1});
        Tensor qn =
            workspace.alloc(DType::BF16, {TextConfig::head_dim, TextConfig::query_heads, 1});
        ops::rmsnorm(q, mtp.query_norm, TextConfig::rms_epsilon, true, qn, s);
        Tensor last_position = positions.slice(0, T - 1, 1);
        Tensor last_rope_position;
        if (rope_positions.ne[1] == 1) {
            last_rope_position = rope_positions.slice(0, T - 1, 1);
        } else {
            last_rope_position = workspace.alloc(DType::I32, {1, 3});
            for (int axis = 0; axis < 3; ++axis) {
                const auto* src = static_cast<const std::int32_t*>(rope_positions.data) +
                                  static_cast<std::size_t>(axis) * T + (T - 1);
                auto* dst       = static_cast<std::int32_t*>(last_rope_position.data) + axis;
                CUDA_CHECK(
                    cudaMemcpyAsync(dst, src, sizeof(std::int32_t), cudaMemcpyDeviceToDevice, s));
            }
        }
        ops::rope(last_rope_position, TextConfig::rotary_dim, TextConfig::rope_theta, qn, s);

        Tensor a = workspace.alloc(DType::BF16, {TextConfig::head_dim, TextConfig::query_heads, 1});
        ops::gqa_attention_cached(qn, last_position, kAttentionScale, mtp_kv.layer_view(0),
                                  envelope, workspace, a, s);
        ops::sigmoid_mul(gate, a, s);

        Tensor o = workspace.alloc(DType::BF16, {TextConfig::hidden, 1});
        ops::linear(a.view({TextConfig::query_size, 1}), mtp.output, o, s);
        ops::residual_add(o, x_last, s);

        Tensor mh = workspace.alloc(DType::BF16, {TextConfig::hidden, 1});
        ops::rmsnorm(x_last, mtp.post_attention_norm, TextConfig::rms_epsilon, true, mh, s);
        {
            auto post_mixer_scope = workspace.scope();
            Variant::mtp_post_mixer(mh, mtp.post_mixer, x_last, workspace, s);
        }
        ops::rmsnorm(x_last, mtp.final_norm, TextConfig::rms_epsilon, true, *final_hidden, s);
        proposal_argmax(*final_hidden, *logits, *draft_token);
    }
}

void TextContext::proposal_argmax(const Tensor& hidden, Tensor& logits, Tensor& proposal_tokens) {
    const int T = hidden.ne[1];
    require_tensor_shape(hidden, DType::BF16, {TextConfig::hidden, T}, "proposal hidden");
    require_tensor_shape(proposal_tokens, DType::I32, {T}, "proposal tokens");
    require_tensor_window(logits, DType::BF16, TextConfig::output_rows, T, "proposal logits");
    if (proposal_head_weight != nullptr) {
        Tensor proposal_logits = workspace.alloc(DType::BF16, {proposal_token_count, T});
        ops::linear(hidden, *proposal_head_weight, proposal_logits, device.stream);
        ops::argmax(proposal_logits, proposal_tokens, proposal_token_count, device.stream);
        ops::proposal_remap_token_ids(proposal_tokens, proposal_token_ids, proposal_token_count,
                                      device.stream);
    } else {
        Tensor output_logits = matrix_window(logits, T);
        ops::linear(hidden, model.output_head, output_logits, device.stream);
        ops::argmax(output_logits, proposal_tokens, TextConfig::token_domain, device.stream);
    }
}

void TextContext::mtp_forward_batch(const Tensor& ids, const Tensor& hidden,
                                    const Tensor& positions, ops::GqaExecutionEnvelope envelope,
                                    Tensor& mtp_hidden, int logits_column, Tensor* logits,
                                    Tensor* draft_token, const Tensor* explicit_rope_positions,
                                    const Tensor* input_embeddings) {
    if (batch_mtp_kv == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_size) {
        throw std::invalid_argument("MTP batch T must be in [1,prefill_chunk]");
    }
    require_tensor_shape(ids, DType::I32, {T}, "MTP ids");
    require_tensor_shape(positions, DType::I32, {T}, "MTP positions");
    require_tensor_shape(hidden, DType::BF16, {TextConfig::hidden, T}, "MTP hidden");
    require_tensor_shape(mtp_hidden, DType::BF16, {TextConfig::hidden, T}, "MTP output hidden");
    if (logits_column >= T) { throw std::invalid_argument("MTP logits column out of range"); }
    if (logits_column >= 0) {
        if (logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("MTP logits and draft_token outputs are required");
        }
        require_tensor_shape(*logits, DType::BF16, {TextConfig::output_rows, 1}, "MTP logits");
        require_tensor_shape(*draft_token, DType::I32, {1}, "MTP draft token");
    }

    auto position_scope = workspace.scope();
    Tensor generated_rope_positions;
    const Tensor* rope_positions = explicit_rope_positions;
    if (rope_positions == nullptr) {
        generated_rope_positions = workspace.alloc(DType::I32, {T});
        ops::offset_i32_positions(positions, round.rope_delta, generated_rope_positions,
                                  device.stream);
        rope_positions = &generated_rope_positions;
    } else if (rope_positions->dtype != DType::I32 || rope_positions->ne[0] != T ||
               (rope_positions->ne[1] != 1 && rope_positions->ne[1] != 3) ||
               rope_positions->ne[2] != 1 || rope_positions->ne[3] != 1 ||
               !rope_positions->is_contiguous() || rope_positions->data == nullptr) {
        throw std::invalid_argument("MTP explicit rope positions must be [T] or [T,3]");
    }
    mtp_forward_core(ids, hidden, positions, *rope_positions, envelope, mtp_hidden,
                     input_embeddings);

    if (logits_column >= 0) {
        auto logits_scope = workspace.scope();
        Tensor col        = mtp_hidden.slice(1, logits_column, 1);
        proposal_argmax(col, *logits, *draft_token);
    }
}

void TextContext::mtp_forward_ar_step(const Tensor& token, const Tensor& previous_hidden,
                                      const Tensor& position, ops::GqaExecutionEnvelope envelope,
                                      Tensor& mtp_hidden, Tensor& logits, Tensor& draft_token) {
    if (batch_mtp_kv == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    require_tensor_shape(token, DType::I32, {1}, "MTP AR token");
    require_tensor_shape(position, DType::I32, {1}, "MTP AR position");
    require_tensor_shape(previous_hidden, DType::BF16, {TextConfig::hidden, 1},
                         "MTP AR previous hidden");
    require_tensor_shape(mtp_hidden, DType::BF16, {TextConfig::hidden, 1}, "MTP AR output hidden");
    require_tensor_shape(logits, DType::BF16, {TextConfig::output_rows, 1}, "MTP AR logits");
    require_tensor_shape(draft_token, DType::I32, {1}, "MTP AR draft token");

    auto position_scope  = workspace.scope();
    Tensor rope_position = workspace.alloc(DType::I32, {1});
    ops::offset_i32_positions(position, round.rope_delta, rope_position, device.stream);
    mtp_forward_core(token, previous_hidden, position, rope_position, envelope, mtp_hidden,
                     nullptr);
    auto logits_scope = workspace.scope();
    proposal_argmax(mtp_hidden, logits, draft_token);
}

void TextContext::ordinary_decode_batch(const Tensor& ids, const Tensor& cache_positions,
                                        const Tensor& rope_positions, const Tensor& kv_table_rows,
                                        const Tensor& linear_state_slots,
                                        ops::GqaExecutionEnvelope envelope, Tensor& hidden,
                                        Tensor& logits) {
    const std::int32_t batch = ids.ne[0];
    if (batch <= 0 || batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("ordinary decode batch size must be in [1,8]");
    }
    require_tensor_shape(ids, DType::I32, {batch}, "ordinary decode ids");
    require_tensor_shape(cache_positions, DType::I32, {batch}, "ordinary decode cache positions");
    require_tensor_shape(rope_positions, DType::I32, {batch}, "ordinary decode RoPE positions");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "ordinary decode KV rows");
    require_tensor_shape(linear_state_slots, DType::I32, {batch},
                         "ordinary decode Linear Attention slots");
    require_tensor_shape(hidden, DType::BF16, {TextConfig::hidden, batch},
                         "ordinary decode hidden");
    require_tensor_shape(logits, DType::BF16, {TextConfig::output_rows, batch},
                         "ordinary decode logits");

    cudaStream_t stream = device.stream;
    workspace.reset();
    {
        ScopedValue<const Tensor*> cache_binding(active_cache_positions, &cache_positions);
        ScopedValue<const Tensor*> rope_binding(active_rope_positions, &rope_positions);
        ScopedValue<const ops::GqaExecutionEnvelope*> envelope_binding(active_gqa_envelope,
                                                                       &envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows, &kv_table_rows);
        ScopedValue<const Tensor*> state_binding(active_linear_state_slots, &linear_state_slots);
        ScopedValue<std::int32_t> batch_binding(active_sequence_batch, batch);
        ScopedValue<std::int32_t> width_binding(active_sequence_width, 1);

        Tensor x = workspace.alloc(DType::BF16, {TextConfig::hidden, batch});
        ops::embedding(ids, model.token_embedding, x, stream);
        NullTap tap;
        run_layers(x, Phase::Verify, tap);
        ops::rmsnorm(x, model.final_norm, TextConfig::rms_epsilon, true, hidden, stream);
        ops::linear(hidden, model.output_head, logits, stream);
    }
    workspace.reset();
}

template <class Tap>
void TextContext::target_verify_batch_impl(const Tensor& ids, const Tensor& cache_positions,
                                           const Tensor& rope_positions,
                                           const Tensor& valid_columns, const Tensor& kv_table_rows,
                                           const Tensor& linear_state_slots,
                                           ops::GqaExecutionEnvelope envelope, Tensor& hidden,
                                           Tensor& logits, Tensor& target_tokens, Tap& tap) {
    const std::int32_t width = ids.ne[0];
    const std::int32_t batch = ids.ne[1];
    if (width <= 0 || width > static_cast<std::int32_t>(kDFlashDecodeMaximumWidth) || batch <= 0 ||
        batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("target verify batch shape is outside the supported domain");
    }
    const std::int32_t columns = width * batch;
    require_tensor_shape(ids, DType::I32, {width, batch}, "target verify batch ids");
    require_tensor_shape(cache_positions, DType::I32, {width, batch},
                         "target verify batch cache positions");
    require_tensor_shape(rope_positions, DType::I32, {width, batch},
                         "target verify batch RoPE positions");
    require_tensor_shape(valid_columns, DType::I32, {batch}, "target verify batch valid columns");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "target verify batch KV rows");
    require_tensor_shape(linear_state_slots, DType::I32, {batch},
                         "target verify batch Linear Attention slots");
    require_tensor_shape(hidden, DType::BF16, {TextConfig::hidden, width, batch},
                         "target verify batch hidden");
    require_tensor_shape(logits, DType::BF16, {TextConfig::output_rows, width, batch},
                         "target verify batch logits");
    require_tensor_shape(target_tokens, DType::I32, {width, batch}, "target verify batch tokens");

    cudaStream_t stream = device.stream;
    workspace.reset();
    {
        ScopedValue<const Tensor*> cache_binding(active_cache_positions, &cache_positions);
        ScopedValue<const Tensor*> rope_binding(active_rope_positions, &rope_positions);
        ScopedValue<const ops::GqaExecutionEnvelope*> envelope_binding(active_gqa_envelope,
                                                                       &envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows, &kv_table_rows);
        ScopedValue<const Tensor*> state_binding(active_linear_state_slots, &linear_state_slots);
        ScopedValue<const Tensor*> valid_binding(active_valid_columns, &valid_columns);
        ScopedValue<std::int32_t> batch_binding(active_sequence_batch, batch);
        ScopedValue<std::int32_t> width_binding(active_sequence_width, width);

        Tensor x        = workspace.alloc(DType::BF16, {TextConfig::hidden, columns});
        Tensor flat_ids = ids.view({columns});
        ops::embedding(flat_ids, model.token_embedding, x, stream);
        if constexpr (Tap::enabled) { tap.begin(x); }
        run_layers(x, Phase::Verify, tap);
        if constexpr (requires { tap.capture_positions(cache_positions, stream); }) {
            tap.capture_positions(cache_positions, stream);
        }
        Tensor flat_hidden = hidden.view({TextConfig::hidden, columns});
        Tensor flat_logits = logits.view({TextConfig::output_rows, columns});
        Tensor flat_tokens = target_tokens.view({columns});
        ops::rmsnorm(x, model.final_norm, TextConfig::rms_epsilon, true, flat_hidden, stream);
        ops::linear(flat_hidden, model.output_head, flat_logits, stream);
        ops::argmax(flat_logits, flat_tokens, TextConfig::token_domain, stream);
    }
    workspace.reset();
}

void TextContext::target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                                      const Tensor& rope_positions, const Tensor& valid_columns,
                                      const Tensor& kv_table_rows, const Tensor& linear_state_slots,
                                      ops::GqaExecutionEnvelope envelope, Tensor& hidden,
                                      Tensor& logits, Tensor& target_tokens) {
    NullTap tap;
    target_verify_batch_impl(ids, cache_positions, rope_positions, valid_columns, kv_table_rows,
                             linear_state_slots, envelope, hidden, logits, target_tokens, tap);
}

void TextContext::target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                                      const Tensor& rope_positions, const Tensor& valid_columns,
                                      const Tensor& kv_table_rows, const Tensor& linear_state_slots,
                                      ops::GqaExecutionEnvelope envelope, Tensor& hidden,
                                      Tensor& logits, Tensor& target_tokens,
                                      DFlashFeatureSink& sink) {
    target_verify_batch_impl(ids, cache_positions, rope_positions, valid_columns, kv_table_rows,
                             linear_state_slots, envelope, hidden, logits, target_tokens, sink);
}

void TextContext::mtp_forward_decode_batch(const Tensor& ids, const Tensor& hidden,
                                           const Tensor& cache_positions,
                                           const Tensor& rope_positions,
                                           const Tensor& valid_columns, const Tensor& kv_table_rows,
                                           ops::GqaExecutionEnvelope envelope, Tensor& mtp_hidden) {
    if (batch_mtp_kv == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    const std::int32_t width = ids.ne[0];
    const std::int32_t batch = ids.ne[1];
    if (width <= 0 || width > static_cast<std::int32_t>(kMaximumMtpDraftTokens + 1) || batch <= 0 ||
        batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("MTP decode batch shape is outside the supported domain");
    }
    require_tensor_shape(ids, DType::I32, {width, batch}, "MTP decode batch ids");
    require_tensor_shape(hidden, DType::BF16, {TextConfig::hidden, width, batch},
                         "MTP decode batch target hidden");
    require_tensor_shape(cache_positions, DType::I32, {width, batch},
                         "MTP decode batch cache positions");
    require_tensor_shape(rope_positions, DType::I32, {width, batch},
                         "MTP decode batch RoPE positions");
    require_tensor_shape(valid_columns, DType::I32, {batch}, "MTP decode batch valid columns");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "MTP decode batch KV rows");
    require_tensor_shape(mtp_hidden, DType::BF16, {TextConfig::hidden, width, batch},
                         "MTP decode batch hidden");

    ScopedValue<const Tensor*> backend_binding(active_backend_kv_table_rows, &kv_table_rows);
    ScopedValue<const Tensor*> valid_binding(active_valid_columns, &valid_columns);
    ScopedValue<std::int32_t> batch_binding(active_sequence_batch, batch);
    ScopedValue<std::int32_t> width_binding(active_sequence_width, width);
    mtp_forward_core(ids, hidden, cache_positions, rope_positions, envelope, mtp_hidden, nullptr);
}

void TextContext::mtp_propose_batch(const Tensor& hidden, Tensor& logits, Tensor& draft_tokens) {
    const std::int32_t batch = hidden.ne[1];
    require_tensor_shape(hidden, DType::BF16, {TextConfig::hidden, batch},
                         "MTP proposal batch hidden");
    require_tensor_shape(logits, DType::BF16, {TextConfig::output_rows, batch},
                         "MTP proposal batch logits");
    require_tensor_shape(draft_tokens, DType::I32, {batch}, "MTP proposal batch tokens");
    proposal_argmax(hidden, logits, draft_tokens);
}

void TextContext::attn_mix(const FullAttentionWeights& weights, Tensor& x, int index, Phase phase) {
    cudaStream_t s = device.stream;
    const int T    = x.ne[1];
    if (active_gqa_envelope == nullptr) {
        throw std::logic_error("Text GQA execution envelope is not set");
    }

    const auto projection = workspace_recipe::text_attention_projection<TextConfig>(workspace, T);
    Tensor h              = projection.hidden;
    ops::rmsnorm(x, weights.input_norm, TextConfig::rms_epsilon, true, h, s);

    Tensor q         = projection.query.view({TextConfig::head_dim, TextConfig::query_heads, T});
    Tensor gate      = projection.gate.view({TextConfig::head_dim, TextConfig::query_heads, T});
    Tensor k         = projection.key.view({TextConfig::head_dim, TextConfig::kv_heads, T});
    Tensor v         = projection.value.view({TextConfig::head_dim, TextConfig::kv_heads, T});
    Tensor q_flat    = q.view({TextConfig::query_size, T});
    Tensor gate_flat = gate.view({TextConfig::query_size, T});
    Tensor k_flat    = k.view({TextConfig::kv_size, T});
    Tensor v_flat    = v.view({TextConfig::kv_size, T});
    Variant::attention_projection(h, weights.projection, q_flat, gate_flat, k_flat, v_flat, phase,
                                  workspace, s);

    const auto results = workspace_recipe::text_attention_results<TextConfig>(workspace, T);
    Tensor qn = results.normalized_query.view({TextConfig::head_dim, TextConfig::query_heads, T});
    Tensor kn = results.normalized_key.view({TextConfig::head_dim, TextConfig::kv_heads, T});
    ops::rmsnorm(q, weights.query_norm, TextConfig::rms_epsilon, true, qn, s);
    ops::rmsnorm(k, weights.key_norm, TextConfig::rms_epsilon, true, kn, s);
    const Tensor& cache_positions =
        active_cache_positions != nullptr ? *active_cache_positions : round.pos;
    const Tensor& rope_positions =
        active_rope_positions != nullptr ? *active_rope_positions : round.rope_pos;
    Tensor rope_for_op = active_sequence_batch != 0 ? rope_positions.view({T}) : rope_positions;
    ops::rope(rope_for_op, TextConfig::rotary_dim, TextConfig::rope_theta, qn, kn, s);

    Tensor a = results.attention.view({TextConfig::head_dim, TextConfig::query_heads, T});
    const Tensor& kv_table_rows =
        active_kv_table_rows != nullptr ? *active_kv_table_rows : round.text_kv_table_row;
    if (active_sequence_batch != 0) {
        const std::int32_t width = active_sequence_width;
        if (width <= 0 || width * active_sequence_batch != T) {
            throw std::logic_error("Text sequence batch binding does not match aggregate columns");
        }
        Tensor q_batch =
            qn.view({TextConfig::head_dim, TextConfig::query_heads, width, active_sequence_batch});
        Tensor k_batch =
            kn.view({TextConfig::head_dim, TextConfig::kv_heads, width, active_sequence_batch});
        Tensor v_batch =
            v.view({TextConfig::head_dim, TextConfig::kv_heads, width, active_sequence_batch});
        Tensor a_batch =
            a.view({TextConfig::head_dim, TextConfig::query_heads, width, active_sequence_batch});
        Tensor position_batch = cache_positions.view({width, active_sequence_batch});
        const Tensor valid    = active_valid_columns != nullptr ? *active_valid_columns : Tensor{};
        ops::gqa_attention(q_batch, k_batch, v_batch, position_batch, valid, kv_table_rows,
                           kAttentionScale, batch_text_kv->batch_layer_view(index),
                           *active_gqa_envelope, workspace, a_batch, s);
    } else {
        ops::gqa_attention(qn, kn, v, cache_positions, Tensor{}, kv_table_rows, kAttentionScale,
                           batch_text_kv->batch_layer_view(index), *active_gqa_envelope, workspace,
                           a, s);
    }
    ops::sigmoid_mul(gate, a, s);

    Variant::attention_output_projection(a.view({TextConfig::query_size, T}), weights.output, x,
                                         phase, workspace, s);
}

void TextContext::gdn_mix(const GdnWeights& weights, Tensor& x, int index, Phase phase) {
    cudaStream_t s = device.stream;
    const int T    = x.ne[1];

    const auto control = workspace_recipe::gdn_control<TextConfig>(workspace, T);
    Tensor h           = control.hidden;
    Tensor g           = control.g;
    Tensor beta        = control.beta;
    Variant::gdn_norm_control_projection(x, weights.input_norm, TextConfig::rms_epsilon,
                                         weights.projection, h, g, beta, workspace, s);

    const auto projection = workspace_recipe::gdn_projection<TextConfig>(workspace, T);
    Tensor z              = projection.output_gate.view(
        {TextConfig::gdn_value_head_dim, TextConfig::gdn_value_heads, T});
    Tensor qc = projection.query;
    Tensor kc = projection.key;
    Tensor vc = projection.value;
    if (phase == Phase::Verify) {
        if (active_sequence_batch == 0 || active_linear_state_slots == nullptr) {
            throw std::logic_error(
                "Verify GDN requires an explicit sequence batch and state slots");
        }
        const std::int32_t width = active_sequence_width;
        if (width <= 0 || width * active_sequence_batch != T) {
            throw std::logic_error("GDN sequence batch binding does not match aggregate columns");
        }
        Tensor projection_input = h.view({TextConfig::hidden, width, active_sequence_batch});
        Tensor query_output     = qc.view({TextConfig::key_dim, width, active_sequence_batch});
        Tensor key_output       = kc.view({TextConfig::key_dim, width, active_sequence_batch});
        Tensor value_output     = vc.view({TextConfig::value_dim, width, active_sequence_batch});
        Tensor gate_output      = z.view({TextConfig::value_dim, width, active_sequence_batch});
        Tensor& conv_states     = linear_state.conv.at(static_cast<std::size_t>(index));
        const Tensor valid = active_valid_columns != nullptr ? *active_valid_columns : Tensor{};
        if (gdn_state_action == GdnStateAction::RecordForReplay) {
            if (gdn_replay_records == nullptr) {
                throw std::logic_error("Replay-record GDN has no record storage");
            }
            GdnReplayRecordLayer records = gdn_replay_records->layer(index, active_sequence_batch);
            Variant::gdn_input_projection_record(
                projection_input, weights.projection, weights.convolution, conv_states, valid,
                *active_linear_state_slots, records.conv, query_output, key_output, value_output,
                gate_output, phase, workspace, s);
        } else {
            Variant::gdn_input_projection_snapshot(
                projection_input, weights.projection, weights.convolution, conv_states, valid,
                *active_linear_state_slots, *active_linear_state_slots, query_output, key_output,
                value_output, gate_output, phase, workspace, s);
        }
    } else {
        const auto conv = workspace_recipe::gdn_prefill_conv<TextConfig>(workspace, T);
        Tensor qkv      = conv.projected;
        Variant::gdn_input_projection(h, weights.projection, qkv, z, phase, workspace, s);
        Tensor qkv_c = conv.convolved;
        Tensor conv_state =
            linear_state.conv_slot(static_cast<std::uint32_t>(index), current_state_slot);
        ops::causal_conv1d_silu(qkv, weights.convolution, conv_state, conv_state, qkv_c, s);
        ops::extract_bf16_columns(qkv_c, 0, qc, s);
        ops::extract_bf16_columns(qkv_c, TextConfig::key_dim, kc, s);
        ops::extract_bf16_columns(qkv_c, 2 * TextConfig::key_dim, vc, s);
    }

    Tensor q_recurrent = qc.view({TextConfig::gdn_key_head_dim, TextConfig::gdn_key_heads, T});
    Tensor k_recurrent = kc.view({TextConfig::gdn_key_head_dim, TextConfig::gdn_key_heads, T});

    Tensor vv = vc.view({TextConfig::gdn_value_head_dim, TextConfig::gdn_value_heads, T});
    Tensor o  = workspace_recipe::gdn_recurrent_output<TextConfig>(workspace, T)
                    .view({TextConfig::gdn_value_head_dim, TextConfig::gdn_value_heads, T});
    if (phase == Phase::Verify) {
        Tensor& recurrent_states = linear_state.recurrent.at(static_cast<std::size_t>(index));
        const std::int32_t width = active_sequence_width;
        Tensor q_batch = q_recurrent.view({TextConfig::gdn_key_head_dim, TextConfig::gdn_key_heads,
                                           width, active_sequence_batch});
        Tensor k_batch = k_recurrent.view({TextConfig::gdn_key_head_dim, TextConfig::gdn_key_heads,
                                           width, active_sequence_batch});
        Tensor v_batch = vv.view({TextConfig::gdn_value_head_dim, TextConfig::gdn_value_heads,
                                  width, active_sequence_batch});
        Tensor g_batch = g.view({TextConfig::gdn_value_heads, width, active_sequence_batch});
        Tensor beta_batch  = beta.view({TextConfig::gdn_value_heads, width, active_sequence_batch});
        Tensor out_batch   = o.view({TextConfig::gdn_value_head_dim, TextConfig::gdn_value_heads,
                                     width, active_sequence_batch});
        const Tensor valid = active_valid_columns != nullptr ? *active_valid_columns : Tensor{};
        if (gdn_state_action == GdnStateAction::RecordForReplay) {
            GdnReplayRecordLayer records = gdn_replay_records->layer(index, active_sequence_batch);
            ops::gated_delta_net_replay_record(
                q_batch, k_batch, v_batch, g_batch, beta_batch, kGdnScale, recurrent_states, valid,
                *active_linear_state_slots, records.key, records.value, records.gate, out_batch, s);
        } else {
            ops::gated_delta_net_snapshot(q_batch, k_batch, v_batch, g_batch, beta_batch, kGdnScale,
                                          /*normalize_qk=*/true, recurrent_states, valid,
                                          *active_linear_state_slots, *active_linear_state_slots,
                                          out_batch, s);
        }
    } else {
        Tensor recurrent_state =
            linear_state.recurrent_slot(static_cast<std::uint32_t>(index), current_state_slot);
        ops::gated_delta_net(q_recurrent, k_recurrent, vv, g, beta, kGdnScale,
                             /*normalize_qk=*/true, workspace, recurrent_state, o, s);
    }

    Tensor on = workspace_recipe::gdn_normalized_output<TextConfig>(workspace, T)
                    .view({TextConfig::gdn_value_head_dim, TextConfig::gdn_value_heads, T});
    ops::gated_rmsnorm(o, weights.norm, z, TextConfig::rms_epsilon, on, s);

    Variant::gdn_output_projection(on.view({TextConfig::value_dim, T}), weights.output, x, phase,
                                   workspace, s);
}

void TextContext::mlp_tail(const Tensor& post_norm, const MlpWeights& weights, Tensor& x,
                           Phase phase) {
    cudaStream_t s = device.stream;
    const int T    = x.ne[1];
    Tensor h       = workspace_recipe::post_mixer_hidden<TextConfig>(workspace, T);
    ops::rmsnorm(x, post_norm, TextConfig::rms_epsilon, true, h, s);

    Variant::post_mixer(h, weights, x, phase, workspace, s);
}

template <class Tap>
void TextContext::run_layers(Tensor& x, Phase phase, Tap& tap) {
    const bool prefill = phase == Phase::Prefill;
    for (int layer = 0; layer < TextConfig::layers; ++layer) {
        if (TextConfig::is_full_attention(layer)) {
            const int full_index = TextConfig::full_attention_index(layer);
            const FullAttentionWeights& full_weights =
                model.full_layers.at(static_cast<std::size_t>(full_index));
            nvtx::ScopedRange layer_range(
                prefill ? nvtx::Name::PrefillLayerFull : nvtx::Name::VerifyLayerFull,
                nvtx::Category::Attention, static_cast<std::uint64_t>(layer));
            {
                nvtx::ScopedRange mixer_range(
                    prefill ? nvtx::Name::PrefillAttention : nvtx::Name::VerifyAttention,
                    nvtx::Category::Attention, static_cast<std::uint64_t>(layer));
                auto mixer_scope = workspace.scope();
                attn_mix(full_weights, x, full_index, phase);
            }
            {
                nvtx::ScopedRange post_mixer_range(
                    prefill ? nvtx::Name::PrefillPostMixer : nvtx::Name::VerifyPostMixer,
                    nvtx::Category::PostMixer, static_cast<std::uint64_t>(layer));
                auto mlp_scope = workspace.scope();
                mlp_tail(full_weights.post_attention_norm, full_weights.post_mixer, x, phase);
                if constexpr (Tap::enabled) { tap.capture_layer(layer, x, device.stream); }
            }
        } else {
            const int gdn_index = TextConfig::gdn_index(layer);
            const GdnWeights& gdn_weights =
                model.gdn_layers.at(static_cast<std::size_t>(gdn_index));
            nvtx::ScopedRange layer_range(prefill ? nvtx::Name::PrefillLayerGdn
                                                  : nvtx::Name::VerifyLayerGdn,
                                          nvtx::Category::Gdn, static_cast<std::uint64_t>(layer));
            {
                nvtx::ScopedRange mixer_range(
                    prefill ? nvtx::Name::PrefillGdn : nvtx::Name::VerifyGdn, nvtx::Category::Gdn,
                    static_cast<std::uint64_t>(layer));
                auto mixer_scope = workspace.scope();
                gdn_mix(gdn_weights, x, gdn_index, phase);
            }
            {
                nvtx::ScopedRange post_mixer_range(
                    prefill ? nvtx::Name::PrefillPostMixer : nvtx::Name::VerifyPostMixer,
                    nvtx::Category::PostMixer, static_cast<std::uint64_t>(layer));
                auto mlp_scope = workspace.scope();
                mlp_tail(gdn_weights.post_attention_norm, gdn_weights.post_mixer, x, phase);
                if constexpr (Tap::enabled) { tap.capture_layer(layer, x, device.stream); }
            }
        }
    }
}

void TextContext::run_layers(Tensor& x, Phase phase) {
    NullTap tap;
    run_layers(x, phase, tap);
}

template <class Tap>
PrefillChunkResult
TextContext::prefill_impl(std::span<const int> ids, const TextPrefill* text_prefill,
                          const MultimodalPrefill* multimodal, Tap& tap, bool finalize_at_end) {
    if (ids.empty()) { throw std::invalid_argument("TextContext::prefill requires tokens"); }
    if (ids.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("TextContext::prefill token count exceeds int32");
    }
    cudaStream_t s           = device.stream;
    const int T              = static_cast<int>(ids.size());
    const int chunk          = static_cast<int>(prefill_chunk_size);
    const std::uint32_t base = text_kv_base;

    if (text_prefill != nullptr) {
        if (multimodal != nullptr || base != text_prefill->begin ||
            text_prefill->token_ids.size() < static_cast<std::size_t>(base) + ids.size()) {
            throw std::invalid_argument("text prefill chunk does not match its full prompt");
        }
    }
    if (multimodal != nullptr) {
        if (base != multimodal->begin ||
            multimodal->token_ids.size() < static_cast<std::size_t>(base) + ids.size()) {
            throw std::invalid_argument("multimodal prefill suffix does not match its cache base");
        }
        if (multimodal->positions.size() != 3 * multimodal->token_ids.size()) {
            throw std::invalid_argument("multimodal positions must have shape [3,T]");
        }
        if (multimodal->vision == nullptr) {
            throw std::invalid_argument("multimodal prefill requires a Vision session");
        }
        rope_delta = multimodal->rope_delta;
    } else if (text_kv_base == 0) {
        rope_delta = 0;
    }
    ops::set_i32_scalar(round.rope_delta, rope_delta, s);

    // Prefix-append prefill continues an existing cache: positions are absolute (start at the
    // resident length) and KV/GDN state is not reset. For a reset prefill base == 0.
    if (static_cast<std::uint64_t>(base) + static_cast<std::uint64_t>(T) >
        static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("TextContext::prefill absolute position exceeds int32");
    }
    const int base_i = static_cast<int>(base);

    const std::int64_t base64         = static_cast<std::int64_t>(base);
    const std::int64_t checkpoint_abs = rewrite_checkpoint_frontier;
    const bool has_rewrite_checkpoint =
        checkpoint_abs > base64 && checkpoint_abs <= base64 + static_cast<std::int64_t>(T);
    const int checkpoint_rel =
        has_rewrite_checkpoint ? static_cast<int>(checkpoint_abs - base64) : -1;
    const std::int32_t rewrite_checkpoint_slot = rewrite_checkpoint_state_slot;

    const bool prepare_mtp_prompt = mtp_enabled() && round.mtp.has_value();
    if (prepare_mtp_prompt &&
        mtp_proposal_extent > static_cast<std::uint32_t>(round.mtp->draft_tokens.ne[0])) {
        throw std::logic_error("MTP proposal extent exceeds the configured draft window");
    }
    int t0 = 0;
    for (; t0 < T;) {
        int len = std::min(chunk, T - t0);
        if (checkpoint_rel > 0 && t0 < checkpoint_rel && t0 + len > checkpoint_rel) {
            len = checkpoint_rel - t0;
        }
        workspace.reset();

        VisionChunk vision_chunk;
        const std::uint32_t prompt_t0 = base + static_cast<std::uint32_t>(t0);
        if (multimodal != nullptr) {
            if (multimodal->vision == nullptr) {
                throw std::logic_error("multimodal prefill has no Vision session");
            }
            vision_chunk =
                multimodal->vision->prepare_chunk(prompt_t0, static_cast<std::uint32_t>(len));
            len = vision_chunk.length;
        }
        const bool is_last = finalize_at_end && (t0 + len == T);
        nvtx::ScopedRange chunk_range(nvtx::Name::PrefillChunk, nvtx::Category::Prefill,
                                      static_cast<std::uint64_t>(len));

        {
            std::vector<std::int32_t> local_scatter_indices;
            std::int32_t visual_begin = 0;
            if (vision_chunk.control != nullptr) {
                const auto scatter =
                    std::span<const std::int32_t>(vision_chunk.control->scatter_indices);
                const auto chunk_begin = static_cast<std::int32_t>(prompt_t0);
                const auto chunk_end =
                    static_cast<std::int32_t>(prompt_t0 + static_cast<std::uint32_t>(len));
                const auto begin = std::lower_bound(scatter.begin(), scatter.end(), chunk_begin);
                const auto end   = std::lower_bound(begin, scatter.end(), chunk_end);
                const auto count = static_cast<std::int32_t>(end - begin);
                visual_begin     = static_cast<std::int32_t>(begin - scatter.begin());
                local_scatter_indices.resize(static_cast<std::size_t>(count));
                for (std::int32_t i = 0; i < count; ++i) {
                    local_scatter_indices[static_cast<std::size_t>(i)] =
                        begin[i] - static_cast<std::int32_t>(prompt_t0);
                }
            }

            const std::int32_t rope_axes = multimodal != nullptr ? 3 : (rope_delta != 0 ? 1 : 0);
            const auto roots             = workspace_recipe::text_prefill_roots<TextConfig>(
                workspace, len, rope_axes, static_cast<std::int32_t>(local_scatter_indices.size()));
            Tensor ids_device = roots.ids;
            copy_i32(ids.data() + t0, ids_device, s);

            Tensor positions = roots.positions;
            ops::fill_i32_positions(positions, base_i + t0, s);

            Tensor rope_positions = positions;
            std::vector<std::int32_t> rope_positions_host;
            if (multimodal != nullptr) {
                rope_positions = roots.rope_positions;
                rope_positions_host.resize(static_cast<std::size_t>(3) * len);
                const std::size_t prompt_tokens = multimodal->token_ids.size();
                for (int axis = 0; axis < 3; ++axis) {
                    const auto* src = multimodal->positions.data() +
                                      static_cast<std::size_t>(axis) * prompt_tokens + prompt_t0;
                    std::copy_n(src, len,
                                rope_positions_host.data() + static_cast<std::size_t>(axis) * len);
                }
                copy_i32(rope_positions_host.data(), rope_positions, s);
            } else if (rope_delta != 0) {
                rope_positions = roots.rope_positions;
                ops::offset_i32_positions(positions, round.rope_delta, rope_positions, s);
            }
            ScopedValue<const Tensor*> scoped_cache(active_cache_positions, &positions);
            ScopedValue<const Tensor*> scoped_rope(active_rope_positions, &rope_positions);
            const auto visible = static_cast<std::uint32_t>(base_i + t0 + len);
            const ops::GqaExecutionEnvelope chunk_envelope{visible, visible};
            ScopedValue<const ops::GqaExecutionEnvelope*> scoped_envelope(active_gqa_envelope,
                                                                          &chunk_envelope);

            Tensor x = roots.residual;
            ops::embedding(ids_device, model.token_embedding, x, s);
            if (!local_scatter_indices.empty()) {
                Tensor indices_device = roots.scatter_indices;
                copy_i32(local_scatter_indices.data(), indices_device, s);
                Tensor embeddings = vision_chunk.embeddings.slice(
                    1, visual_begin, static_cast<std::int32_t>(local_scatter_indices.size()));
                ops::scatter(embeddings, indices_device, x, s);
            }
            if constexpr (Tap::enabled) { tap.begin(x); }
            run_layers(x, Phase::Prefill, tap);
            if constexpr (requires { tap.capture_positions(positions, s); }) {
                tap.capture_positions(positions, s);
            }

            Tensor xf = prefill_hidden.data != nullptr
                            ? matrix_window(prefill_hidden, len)
                            : workspace.alloc(DType::BF16, {TextConfig::hidden, len});
            ops::rmsnorm(x, model.final_norm, TextConfig::rms_epsilon, true, xf, s);

            if (is_last) {
                Tensor last_xf = xf.slice(1, len - 1, 1);
                Tensor logits  = matrix_window(round.logits, 1);
                ops::linear(last_xf, model.output_head, logits, s);
                // Set round.pos to the bonus token's absolute position (base + T) before picking so
                // the sampler RNG is keyed by it (prefill purpose keeps it distinct from the first
                // decode step, which reuses the same round.pos).
                ops::set_i32_scalar(round.pos, base_i + T, s);
                ops::set_i32_scalar(round.rope_pos, base_i + T + rope_delta, s);
                if (sampling != nullptr) {
                    ops::sample(logits, round.token, TextConfig::token_domain, sampling, round.pos,
                                ops::kSamplePurposePrefill, workspace, s);
                } else {
                    ops::argmax(logits, round.token, TextConfig::token_domain, s);
                }
            }

            if (prepare_mtp_prompt) {
                const std::uint32_t alignment_tokens =
                    multimodal != nullptr ? static_cast<std::uint32_t>(multimodal->token_ids.size())
                    : text_prefill != nullptr
                        ? static_cast<std::uint32_t>(text_prefill->token_ids.size())
                        : static_cast<std::uint32_t>(T);
                const std::uint32_t alignment_begin =
                    multimodal != nullptr || text_prefill != nullptr
                        ? prompt_t0
                        : static_cast<std::uint32_t>(t0);
                const qwen3_6::MtpAlignmentWindow mtp_window = qwen3_6::plan_mtp_alignment_window(
                    alignment_tokens, alignment_begin, static_cast<std::uint32_t>(len));
                const std::span<const int> alignment_ids =
                    multimodal != nullptr     ? multimodal->token_ids
                    : text_prefill != nullptr ? text_prefill->token_ids
                                              : ids;
                std::vector<int> mtp_ids_host(static_cast<std::size_t>(len));
                const int prompt_columns =
                    len - static_cast<int>(mtp_window.final_column_uses_generated_token);
                for (int j = 0; j < prompt_columns; ++j) {
                    mtp_ids_host[static_cast<std::size_t>(j)] =
                        alignment_ids[static_cast<std::size_t>(mtp_window.shifted_embedding_begin) +
                                      static_cast<std::size_t>(j)];
                }
                if (mtp_window.final_column_uses_generated_token) {
                    int next_token = 0;
                    CUDA_CHECK(cudaStreamSynchronize(s));
                    CUDA_CHECK(cudaMemcpy(&next_token, round.token.data, sizeof(next_token),
                                          cudaMemcpyDeviceToHost));
                    mtp_ids_host[static_cast<std::size_t>(len - 1)] = next_token;
                }

                Tensor mtp_ids = workspace.alloc(DType::I32, {len});
                copy_i32(mtp_ids_host.data(), mtp_ids, s);
                Tensor mtp_input_embeddings;
                const Tensor* mtp_input_embeddings_ptr = nullptr;
                if (multimodal != nullptr) {
                    mtp_input_embeddings = workspace.alloc(DType::BF16, {TextConfig::hidden, len});
                    ops::embedding(mtp_ids, model.token_embedding, mtp_input_embeddings, s);
                    if (vision_chunk.control != nullptr) {
                        const qwen3_6::MtpVisualOverlap overlap = qwen3_6::shifted_visual_overlap(
                            vision_chunk.control->scatter_indices, alignment_tokens, mtp_window);
                        if (!overlap.empty()) {
                            Tensor shifted_indices = workspace_recipe::visual_scatter_indices(
                                workspace, static_cast<std::int32_t>(overlap.size()));
                            qwen3_6::detail::scatter_shifted_visual_embeddings(
                                mtp_input_embeddings, vision_chunk.embeddings, overlap,
                                shifted_indices, s);
                        }
                    }
                    mtp_input_embeddings_ptr = &mtp_input_embeddings;
                }
                if (is_last && mtp_proposal_extent != 0) {
                    Tensor logits = matrix_window(round.logits, 1);
                    Tensor draft0 = round.mtp->draft_tokens.slice(0, 0, 1);
                    mtp_prefill_chunk(mtp_ids, xf, mtp_input_embeddings_ptr, positions,
                                      rope_positions, chunk_envelope, true, &round.mtp->ar_hidden,
                                      &logits, &draft0);

                    Tensor ar_position = round.mtp->position.slice(0, 0, 1);
                    ops::set_i32_scalar(ar_position, base_i + T, s);
                    for (int i = 1; i < static_cast<int>(mtp_proposal_extent); ++i) {
                        Tensor prev_token  = round.mtp->draft_tokens.slice(0, i - 1, 1);
                        Tensor next_token  = round.mtp->draft_tokens.slice(0, i, 1);
                        Tensor next_hidden = workspace.alloc(DType::BF16, {TextConfig::hidden, 1});
                        const auto ar_visible = static_cast<std::uint32_t>(base_i + T + i);
                        const ops::GqaExecutionEnvelope ar_envelope{ar_visible, ar_visible};
                        mtp_forward_ar_step(prev_token, round.mtp->ar_hidden, ar_position,
                                            ar_envelope, next_hidden, logits, next_token);
                        CUDA_CHECK(cudaMemcpyAsync(round.mtp->ar_hidden.data, next_hidden.data,
                                                   round.mtp->ar_hidden.bytes(),
                                                   cudaMemcpyDeviceToDevice, s));
                        ops::increment_i32_scalar(ar_position, s);
                    }
                } else {
                    mtp_prefill_chunk(mtp_ids, xf, mtp_input_embeddings_ptr, positions,
                                      rope_positions, chunk_envelope, false, nullptr, nullptr,
                                      nullptr);
                }
            }

            if (checkpoint_rel > 0 && t0 + len == checkpoint_rel &&
                rewrite_checkpoint_hidden != nullptr) {
                require_tensor_shape(*rewrite_checkpoint_hidden, DType::BF16,
                                     {TextConfig::hidden, 1}, "rewrite checkpoint hidden output");
                const Tensor checkpoint_hidden = xf.slice(1, len - 1, 1);
                CUDA_CHECK(cudaMemcpyAsync(rewrite_checkpoint_hidden->data, checkpoint_hidden.data,
                                           checkpoint_hidden.bytes(), cudaMemcpyDeviceToDevice, s));
            }
        }

        if constexpr (requires { tap.consume_prefill_chunk(len, false); }) {
            workspace.reset();
            tap.consume_prefill_chunk(len, checkpoint_rel > 0 && t0 + len == checkpoint_rel);
        }

        if (checkpoint_rel > 0 && t0 + len == checkpoint_rel) {
            linear_state.copy_slot(current_state_slot, rewrite_checkpoint_slot, s);
        }

        t0 += len;
        break;
    }

    rewrite_checkpoint_frontier = -1;

    device.synchronize();
    workspace.reset();
    return PrefillChunkResult{.processed_tokens = static_cast<std::uint32_t>(t0),
                              .finalized        = finalize_at_end && t0 == T};
}

PrefillChunkResult TextContext::prefill_chunk(std::span<const int> full_ids, std::uint32_t begin,
                                              std::uint32_t nominal_length, bool finalize_at_end) {
    if (begin >= full_ids.size() || nominal_length == 0 ||
        nominal_length > full_ids.size() - begin) {
        throw std::invalid_argument("text prefill chunk is outside the prompt");
    }
    const TextPrefill text_prefill{full_ids, begin};
    NullTap tap;
    return prefill_impl(full_ids.subspan(begin, nominal_length), &text_prefill, nullptr, tap,
                        finalize_at_end);
}

PrefillChunkResult TextContext::prefill_chunk(std::span<const int> full_ids, std::uint32_t begin,
                                              std::uint32_t nominal_length, bool finalize_at_end,
                                              DFlashFeatureSink& sink) {
    if (begin >= full_ids.size() || nominal_length == 0 ||
        nominal_length > full_ids.size() - begin) {
        throw std::invalid_argument("text prefill chunk is outside the prompt");
    }
    const TextPrefill text_prefill{full_ids, begin};
    return prefill_impl(full_ids.subspan(begin, nominal_length), &text_prefill, nullptr, sink,
                        finalize_at_end);
}

PrefillChunkResult TextContext::prefill_chunk(const qwen3_6::PreparedPromptData& input,
                                              std::uint32_t begin, std::uint32_t nominal_length,
                                              VisionPrefillSession& vision, bool finalize_at_end) {
    if (begin >= input.token_ids.size() || nominal_length == 0 ||
        nominal_length > input.token_ids.size() - begin) {
        throw std::invalid_argument("multimodal prefill chunk is outside the prompt");
    }
    const std::span<const int> tokens(input.token_ids);
    const MultimodalPrefill multimodal{tokens, input.positions, &vision, begin, input.rope_delta};
    NullTap tap;
    return prefill_impl(tokens.subspan(begin, nominal_length), nullptr, &multimodal, tap,
                        finalize_at_end);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
