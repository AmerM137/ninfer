#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include <ninfer/targets/qwen3_6/vision_control.h>
#include "runtime/contract/transient_region.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

struct VisionItemView {
    std::span<const std::uint16_t> patches;
    const qwen3_6::VisionItemControl* control = nullptr;
};

class VisionContext {
public:
    VisionContext(DeviceContext& device, const LoadedModelData& model);

    [[nodiscard]] static std::size_t output_transient_bytes(std::size_t merged_tokens);
    [[nodiscard]] static std::size_t workspace_bytes(const qwen3_6::VisionItemControl& item);
    [[nodiscard]] static std::size_t workspace_capacity_bytes(std::uint32_t max_merged_tokens,
                                                              std::uint32_t max_segments);
    void encode(const VisionItemView& item, Tensor& output, WorkspaceArena& workspace) const;

private:
    DeviceContext& device;
    const VisionWeights& vision;
};

struct VisionChunk {
    std::int32_t length                       = 0;
    const qwen3_6::VisionItemControl* control = nullptr;
    Tensor embeddings;
};

class VisionPrefillSession {
public:
    VisionPrefillSession(DeviceContext& device, const LoadedModelData& model,
                         WorkspaceArena& workspace, qwen3_6::PreparedPromptData& prompt,
                         const VisionPrefillPlan& plan, runtime::TransientRegion transient);

    [[nodiscard]] VisionChunk prepare_chunk(std::uint32_t begin, std::uint32_t nominal_length);
    void release_encoded_media_payloads() noexcept;
    [[nodiscard]] double elapsed_seconds() const;

private:
    DeviceContext& device;
    WorkspaceArena& workspace;
    qwen3_6::PreparedPromptData& prompt;
    const VisionPrefillPlan& plan;
    runtime::TransientRegion transient;
    VisionContext context;
    std::optional<std::uint32_t> active_item;
    std::vector<std::uint32_t> encoded_payloads_pending_release;
    std::vector<CudaEventTimer> timers;
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
