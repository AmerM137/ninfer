#pragma once

// Identity-free Qwen3.5 family runtime helper.

#include "core/device.h"
#include "core/tensor.h"
#include "models/qwen3_5/program/speculative/mtp_alignment.h"

namespace ninfer::models::qwen3_5::detail {

// Composes the generic scatter Op from the family-provided shifted-window interpretation.
void scatter_shifted_visual_embeddings(Tensor& input_embeddings, const Tensor& visual_embeddings,
                                       const qwen3_5::MtpVisualOverlap& overlap,
                                       Tensor& destination_indices, CudaExecutionView execution);

} // namespace ninfer::models::qwen3_5::detail
