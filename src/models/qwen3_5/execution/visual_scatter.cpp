#include "models/qwen3_5/execution/visual_scatter.h"

#include "core/device.h"
#include "ninfer/ops/indexing.h"

#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {

void scatter_shifted_visual_embeddings(Tensor& input_embeddings, const Tensor& visual_embeddings,
                                       const qwen3_5::MtpVisualOverlap& overlap,
                                       Tensor& destination_indices, CudaExecutionView execution) {
    if (overlap.empty() || destination_indices.dtype != DType::I32 ||
        destination_indices.ne[0] <= 0 ||
        static_cast<std::size_t>(destination_indices.ne[0]) != overlap.size() ||
        destination_indices.ne[1] != 1 || destination_indices.ne[2] != 1 ||
        destination_indices.ne[3] != 1 || !destination_indices.is_contiguous() ||
        destination_indices.data == nullptr || destination_indices.device != execution.device()) {
        throw std::invalid_argument("shifted visual scatter has invalid destination indices");
    }
    const auto count = static_cast<std::int32_t>(overlap.size());
    Tensor embeddings =
        visual_embeddings.slice(1, static_cast<std::int32_t>(overlap.source_begin), count);
    CUDA_CHECK(cudaMemcpyAsync(destination_indices.data, overlap.destination_columns.data(),
                               overlap.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                               execution.stream()));
    ops::scatter(embeddings, destination_indices, input_embeddings, execution);
}

} // namespace ninfer::models::qwen3_5::detail
