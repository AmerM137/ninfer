#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/vision_context.h"

#include "core/device.h"
#include "core/layout.h"
#include <ninfer/targets/qwen3_6/vision_control.h>
#include "ninfer/ops/add_bias.h"
#include "ninfer/ops/gelu.h"
#include "ninfer/ops/layer_norm.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/vision_attention.h"
#include "ninfer/ops/vision_pos_embed.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
namespace {

std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(std::string("Vision ") + label + " overflows size_t");
    }
    return a * b;
}

constexpr std::size_t kWorkspaceAlignment = 256;

struct VisionWorkspaceLayout {
    TensorRegion position_ids;
    TensorRegion cu_seqlens;
    TensorRegion pos_indices;
    TensorRegion pos_weights;
    TensorRegion x;
    TensorRegion patch_bf16;
    TensorRegion attended;
    TensorRegion qkv;
    TensorRegion attention_norm;
    std::optional<LayoutRegion> attention_workspace;
    TensorRegion projected;
    TensorRegion mlp_down;
    TensorRegion mlp_up;
    TensorRegion mlp_norm;
    TensorRegion normalized;
    TensorRegion merger_hidden;
    std::size_t bytes = 0;
};

VisionWorkspaceLayout build_workspace_layout(std::size_t patches64, std::size_t tokens64,
                                             std::size_t segment_count) {
    if (patches64 == 0 || tokens64 == 0 ||
        patches64 != checked_mul(tokens64, VisionConfig::merge_unit, "patch/token relation")) {
        throw std::invalid_argument("Vision workspace requires P=4V>0");
    }
    if (patches64 > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
        tokens64 > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
        segment_count == 0 ||
        segment_count >= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("Vision request dimensions exceed int32");
    }
    const auto patches = static_cast<std::int32_t>(patches64);
    const auto tokens  = static_cast<std::int32_t>(tokens64);

    LayoutBuilder builder;
    VisionWorkspaceLayout out;
    const auto add = [&](DType dtype, std::initializer_list<std::int32_t> shape,
                         const char* label) {
        return builder.add_tensor(dtype, shape, kWorkspaceAlignment, label);
    };
    out.position_ids = add(DType::I32, {patches, 2}, "vision position ids");
    out.cu_seqlens =
        add(DType::I32, {static_cast<std::int32_t>(segment_count + 1)}, "vision segment bounds");
    out.pos_indices = add(DType::I32, {4, patches}, "vision position indices");
    out.pos_weights = add(DType::FP32, {4, patches}, "vision position weights");
    out.x           = add(DType::BF16, {VisionConfig::hidden, patches}, "vision residual");
    out.patch_bf16  = add(DType::BF16, {VisionConfig::patch_dim, patches}, "vision BF16 patches");
    {
        auto attention_scope = builder.scope();
        out.attended         = add(DType::BF16, {VisionConfig::hidden, patches}, "vision attended");
        {
            auto qkv_scope = builder.scope();
            out.qkv        = add(DType::BF16, {3 * VisionConfig::hidden, patches}, "vision QKV");
            {
                auto norm_scope = builder.scope();
                out.attention_norm =
                    add(DType::BF16, {VisionConfig::hidden, patches}, "vision attention norm");
            }
            const std::size_t attention_bytes = ops::vision_attention_workspace_capacity_bytes(
                patches, patches, static_cast<std::int32_t>(segment_count),
                static_cast<std::int32_t>(segment_count));
            if (attention_bytes != 0) {
                out.attention_workspace =
                    builder.add(attention_bytes, kWorkspaceAlignment, "vision attention workspace");
            }
        }
        out.projected = add(DType::BF16, {VisionConfig::hidden, patches}, "vision projected");
    }
    {
        auto mlp_scope = builder.scope();
        out.mlp_up     = add(DType::BF16, {VisionConfig::intermediate, patches}, "vision MLP up");
        {
            auto norm_scope = builder.scope();
            out.mlp_norm    = add(DType::BF16, {VisionConfig::hidden, patches}, "vision MLP norm");
        }
        out.mlp_down = add(DType::BF16, {VisionConfig::hidden, patches}, "vision MLP down");
    }
    out.normalized = add(DType::BF16, {VisionConfig::hidden, patches}, "vision merger norm");
    out.merger_hidden =
        add(DType::BF16, {VisionConfig::merger_hidden, tokens}, "vision merger hidden");
    out.bytes = builder.finish(1, "vision workspace");
    return out;
}

void copy_host(const void* src, Tensor& dst, cudaStream_t stream) {
    if (dst.bytes() == 0) { return; }
    CUDA_CHECK(cudaMemcpyAsync(dst.data, src, dst.bytes(), cudaMemcpyHostToDevice, stream));
}

const VisionWeights& require_vision_weights(const LoadedModelData& model) {
    if (!model.vision) {
        throw std::invalid_argument("Vision execution was requested without materialized weights");
    }
    return *model.vision;
}

} // namespace

VisionContext::VisionContext(DeviceContext& device, const LoadedModelData& model)
    : device(device), vision(require_vision_weights(model)) {}

std::size_t VisionContext::workspace_bytes(const qwen3_6::VisionItemControl& item) {
    return build_workspace_layout(item.patch_count, item.merged_count,
                                  static_cast<std::size_t>(item.segment_count))
        .bytes;
}

std::size_t VisionContext::output_transient_bytes(std::size_t merged_tokens) {
    if (merged_tokens == 0 ||
        merged_tokens > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("Vision output transient extent must fit positive int32");
    }
    LayoutBuilder layout;
    (void)layout.add_tensor(DType::BF16,
                            {VisionConfig::output_hidden, static_cast<std::int32_t>(merged_tokens)},
                            kWorkspaceAlignment, "Vision item output transient");
    return layout.finish(kWorkspaceAlignment, "Vision item output transient layout");
}

std::size_t VisionContext::workspace_capacity_bytes(std::uint32_t max_merged_tokens,
                                                    std::uint32_t max_segments) {
    if (max_merged_tokens == 0 || max_segments == 0) {
        throw std::invalid_argument("Vision workspace capacity bounds must be positive");
    }
    const std::uint32_t segments = std::min(max_merged_tokens, max_segments);
    return build_workspace_layout(
               checked_mul(max_merged_tokens, VisionConfig::merge_unit, "capacity patch count"),
               max_merged_tokens, segments)
        .bytes;
}

void VisionContext::encode(const VisionItemView& item, Tensor& output,
                           WorkspaceArena& workspace) const {
    if (item.control == nullptr) { throw std::invalid_argument("Vision item control is null"); }
    const qwen3_6::VisionItemControl& control = *item.control;
    const auto patches64                      = control.patch_count;
    const auto tokens64                       = control.merged_count;
    if (item.patches.size() != checked_mul(patches64, VisionConfig::patch_dim, "patch elements")) {
        throw std::invalid_argument("Vision processor patch buffer has invalid shape");
    }
    if (output.dtype != DType::BF16 || output.ne[0] != VisionConfig::output_hidden ||
        output.ne[1] != static_cast<std::int32_t>(tokens64) || output.ne[2] != 1 ||
        output.ne[3] != 1 || !output.is_contiguous() || output.data == nullptr) {
        throw std::invalid_argument("Vision output must be contiguous BF16 [H,V]");
    }
    const VisionWorkspaceLayout layout = build_workspace_layout(
        patches64, tokens64, static_cast<std::size_t>(control.segment_count));
    if (workspace.capacity() < layout.bytes) {
        throw std::invalid_argument("Vision workspace capacity is too small for request");
    }
    const auto patches  = static_cast<std::int32_t>(patches64);
    const auto tokens   = static_cast<std::int32_t>(tokens64);
    cudaStream_t stream = device.stream;
    workspace.reset();
    const DeviceSpan backing = workspace.alloc_bytes(layout.bytes, kWorkspaceAlignment);

    Tensor position_ids = layout.position_ids.bind(backing);
    Tensor cu_seqlens   = layout.cu_seqlens.bind(backing);
    Tensor pos_indices  = layout.pos_indices.bind(backing);
    Tensor pos_weights  = layout.pos_weights.bind(backing);
    copy_host(control.position_ids.data(), position_ids, stream);
    copy_host(control.cu_seqlens.data(), cu_seqlens, stream);
    copy_host(control.position_table_indices.data(), pos_indices, stream);
    copy_host(control.position_table_weights.data(), pos_weights, stream);

    Tensor x          = layout.x.bind(backing);
    Tensor patch_bf16 = layout.patch_bf16.bind(backing);
    copy_host(item.patches.data(), patch_bf16, stream);
    ops::linear(patch_bf16, vision.common.patch_embedding, x, stream);
    ops::add_bias(vision.common.patch_embedding_bias, x, stream);
    // The artifact records the source table shape [rows,hidden], while Tensor's
    // contiguous matrix convention is [inner,columns]. The payload is already
    // row-major, so this is a zero-copy [hidden,rows] view, not a transpose.
    Tensor position_table = vision.common.position_embedding.reshape(
        {VisionConfig::hidden, VisionConfig::position_embeddings});
    ops::vision_pos_embed_add(position_table, pos_indices, pos_weights, x, stream);
    for (const qwen3_6::VisionLayerWeights& block : vision.common.layers) {
        {
            Tensor attended = layout.attended.bind(backing);
            {
                Tensor qkv = layout.qkv.bind(backing);
                {
                    Tensor h = layout.attention_norm.bind(backing);
                    ops::layer_norm(x, block.norm1_weight, block.norm1_bias,
                                    VisionConfig::norm_epsilon, h, stream);
                    ops::linear(h, block.qkv, qkv, stream);
                }
                ops::add_bias(block.qkv_bias, qkv, stream);
                const std::int32_t plane      = VisionConfig::hidden;
                const std::size_t plane_bytes = static_cast<std::size_t>(plane) * 2;
                Tensor q(qkv.data, DType::BF16,
                         {VisionConfig::head_dim, VisionConfig::heads, patches});
                Tensor k(static_cast<unsigned char*>(qkv.data) + plane_bytes, DType::BF16,
                         {VisionConfig::head_dim, VisionConfig::heads, patches});
                Tensor v(static_cast<unsigned char*>(qkv.data) + 2 * plane_bytes, DType::BF16,
                         {VisionConfig::head_dim, VisionConfig::heads, patches});
                q.nb[2] = qkv.nb[1];
                k.nb[2] = qkv.nb[1];
                v.nb[2] = qkv.nb[1];
                ops::rope(position_ids, VisionConfig::rotary_dim, VisionConfig::rope_theta, q, k,
                          stream);
                Tensor attended_heads =
                    attended.view({VisionConfig::head_dim, VisionConfig::heads, patches});
                const DeviceSpan attention_backing = layout.attention_workspace
                                                         ? layout.attention_workspace->bind(backing)
                                                         : backing;
                WorkspaceArena attention_workspace(attention_backing);
                ops::vision_attention(q, k, v, cu_seqlens, attention_workspace, attended_heads,
                                      stream);
            }
            Tensor projected = layout.projected.bind(backing);
            ops::linear(attended, block.output, projected, stream);
            ops::add_bias(block.output_bias, projected, stream);
            ops::residual_add(projected, x, stream);
        }
        {
            Tensor down = layout.mlp_down.bind(backing);
            Tensor up   = layout.mlp_up.bind(backing);
            {
                Tensor h = layout.mlp_norm.bind(backing);
                ops::layer_norm(x, block.norm2_weight, block.norm2_bias, VisionConfig::norm_epsilon,
                                h, stream);
                ops::linear(h, block.fc1, up, stream);
            }
            ops::add_bias(block.fc1_bias, up, stream);
            ops::gelu(up, ops::GeluMode::Tanh, stream);
            ops::linear(up, block.fc2, down, stream);
            ops::add_bias(block.fc2_bias, down, stream);
            ops::residual_add(down, x, stream);
        }
    }

    Tensor normalized = layout.normalized.bind(backing);
    ops::layer_norm(x, vision.common.merger_norm_weight, vision.common.merger_norm_bias,
                    VisionConfig::norm_epsilon, normalized, stream);
    Tensor merged = normalized.view({VisionConfig::merger_hidden, tokens});
    Tensor hidden = layout.merger_hidden.bind(backing);
    ops::linear(merged, vision.common.merger_fc1, hidden, stream);
    ops::add_bias(vision.common.merger_fc1_bias, hidden, stream);
    ops::gelu(hidden, ops::GeluMode::Exact, stream);
    ops::linear(hidden, vision.merger_fc2, output, stream);
    ops::add_bias(vision.merger_fc2_bias, output, stream);
}

VisionPrefillSession::VisionPrefillSession(DeviceContext& device, const LoadedModelData& model,
                                           WorkspaceArena& workspace,
                                           qwen3_6::PreparedPromptData& prompt,
                                           const VisionPrefillPlan& plan,
                                           runtime::TransientRegion transient)
    : device(device), workspace(workspace), prompt(prompt), plan(plan), transient(transient),
      context(device, model) {
    if (plan.control == nullptr || plan.control->items.empty() || plan.uses.empty()) {
        throw std::invalid_argument("Vision prefill plan has no suffix item spans");
    }
    if (transient.data == nullptr || transient.alignment < kWorkspaceAlignment) {
        throw std::invalid_argument("Vision item output transient is missing or misaligned");
    }
    encoded_payloads_pending_release.reserve(plan.uses.size());
    timers.reserve(plan.uses.size());
}

VisionChunk VisionPrefillSession::prepare_chunk(std::uint32_t begin, std::uint32_t nominal_length) {
    if (nominal_length == 0 || begin >= prompt.token_ids.size()) {
        throw std::invalid_argument("Vision chunk range is empty or outside the prompt");
    }
    const std::uint64_t nominal_end64 =
        static_cast<std::uint64_t>(begin) + static_cast<std::uint64_t>(nominal_length);
    std::uint32_t end =
        static_cast<std::uint32_t>(std::min<std::uint64_t>(nominal_end64, prompt.token_ids.size()));

    const VisionUseSpan* active = nullptr;
    for (const VisionUseSpan& use : plan.uses) {
        if (use.end <= begin) { continue; }
        if (use.begin >= end) { break; }
        if (active == nullptr) {
            active = &use;
        } else {
            end = std::min(end, use.begin);
            break;
        }
    }
    if (end <= begin) { throw std::logic_error("Vision chunk cap made no forward progress"); }
    if (active == nullptr) {
        return VisionChunk{static_cast<std::int32_t>(end - begin), nullptr, {}};
    }
    if (active->item_index >= plan.control->items.size() ||
        active->item_index >= prompt.vision_items.size() ||
        active->item_index >= prompt.media_payloads.size()) {
        throw std::logic_error("Vision prefill item index is out of range");
    }
    const qwen3_6::VisionItemControl& control = plan.control->items[active->item_index];
    const qwen3_6::VisionItem& source         = prompt.vision_items[active->item_index];
    if (source.modality != control.modality || source.grid.temporal != control.grid.temporal ||
        source.grid.height != control.grid.height || source.grid.width != control.grid.width ||
        source.patch_begin != control.patch_begin || source.patch_count != control.patch_count) {
        throw std::invalid_argument("Vision prefill plan does not describe the prepared item");
    }
    if (control.merged_count > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("Vision item output columns exceed int32");
    }
    const std::size_t output_bytes =
        checked_mul(checked_mul(static_cast<std::size_t>(VisionConfig::output_hidden),
                                control.merged_count, "item output elements"),
                    dtype_size(DType::BF16), "item output bytes");
    if (output_bytes > transient.size) {
        throw std::invalid_argument("Vision item output transient is too small");
    }
    Tensor output(transient.data, DType::BF16,
                  {VisionConfig::output_hidden, static_cast<std::int32_t>(control.merged_count)});

    if (!active_item || *active_item != active->item_index) {
        if (active_item && active->item_index <= *active_item) {
            throw std::logic_error("Vision items are not consumed in strictly increasing order");
        }
        const std::size_t patch_elements =
            checked_mul(control.patch_count, static_cast<std::size_t>(VisionConfig::patch_dim),
                        "item patch elements");
        const auto& payload = prompt.media_payloads[active->item_index];
        if (!payload || payload->patch_elements != patch_elements) {
            throw std::invalid_argument("Vision item patch payload has an invalid shape");
        }
        timers.emplace_back(device);
        timers.back().start();
        context.encode(VisionItemView{payload->span(), &control}, output, workspace);
        timers.back().record_stop();
        workspace.reset();
        active_item = active->item_index;
        encoded_payloads_pending_release.push_back(active->item_index);
    }
    return VisionChunk{static_cast<std::int32_t>(end - begin), &control, output};
}

void VisionPrefillSession::release_encoded_media_payloads() noexcept {
    for (const std::uint32_t item_index : encoded_payloads_pending_release) {
        if (item_index >= prompt.media_payloads.size()) { std::terminate(); }
        prompt.media_payloads[item_index].reset();
    }
    encoded_payloads_pending_release.clear();
}

double VisionPrefillSession::elapsed_seconds() const {
    double milliseconds = 0.0;
    for (const CudaEventTimer& timer : timers) { milliseconds += timer.elapsed_ms(); }
    return milliseconds / 1000.0;
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
