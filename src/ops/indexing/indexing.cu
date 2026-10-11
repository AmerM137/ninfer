#include "ninfer/ops/indexing.h"
#include "core/device.h"
#include "ops/common/host/operands.h"
#include <cuda_bf16.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {
__global__ void scatter_bf16x8_kernel(const uint4* src, const std::int32_t* indices, uint4* dst,
                                      std::int32_t vectors_per_column) {
    const std::int32_t src_col  = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t dst_col  = indices[src_col];
    const std::int64_t src_base = static_cast<std::int64_t>(src_col) * vectors_per_column;
    const std::int64_t dst_base = static_cast<std::int64_t>(dst_col) * vectors_per_column;
    for (std::int32_t vector = static_cast<std::int32_t>(threadIdx.x); vector < vectors_per_column;
         vector += static_cast<std::int32_t>(blockDim.x)) {
        dst[dst_base + vector] = src[src_base + vector];
    }
}

__global__ void scatter_bf16x2_kernel(const __nv_bfloat162* src, const std::int32_t* indices,
                                      __nv_bfloat162* dst, std::int32_t pairs) {
    const std::int32_t src_col = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t dst_col = indices[src_col];

    const std::int64_t src_base = static_cast<std::int64_t>(src_col) * pairs;
    const std::int64_t dst_base = static_cast<std::int64_t>(dst_col) * pairs;
    for (std::int32_t pair = static_cast<std::int32_t>(threadIdx.x); pair < pairs;
         pair += static_cast<std::int32_t>(blockDim.x)) {
        dst[dst_base + pair] = src[src_base + pair];
    }
}

__global__ void scatter_scalar_kernel(const __nv_bfloat16* src, const std::int32_t* indices,
                                      __nv_bfloat16* dst, std::int32_t d) {
    const std::int32_t src_col = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t dst_col = indices[src_col];

    const std::int64_t src_base = static_cast<std::int64_t>(src_col) * d;
    const std::int64_t dst_base = static_cast<std::int64_t>(dst_col) * d;
    for (std::int32_t row = static_cast<std::int32_t>(threadIdx.x); row < d;
         row += static_cast<std::int32_t>(blockDim.x)) {
        dst[dst_base + row] = src[src_base + row];
    }
}

inline constexpr int kScatterBatchThreads = 128;

// A 2 KiB feature tile exposes parallelism even when the block has few columns.
__launch_bounds__(kScatterBatchThreads) __global__
    void scatter_bf16_batch_kernel(const uint4* __restrict__ source,
                                   const std::int32_t* __restrict__ lanes,
                                   const std::int32_t* __restrict__ valid_columns,
                                   uint4* __restrict__ destination, std::int32_t vectors_per_column,
                                   std::int32_t width, std::int64_t destination_column_stride,
                                   std::int64_t destination_lane_stride) {
    const std::int32_t column = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t batch  = static_cast<std::int32_t>(blockIdx.y);
    if (column >= valid_columns[batch]) return;
    const std::int64_t source_base =
        (static_cast<std::int64_t>(batch) * width + column) * vectors_per_column;
    const std::int64_t destination_base =
        static_cast<std::int64_t>(lanes[batch]) * destination_lane_stride +
        static_cast<std::int64_t>(column) * destination_column_stride;
    // Keep the tile loop CTA-uniform; only the store has a per-thread tail.
    for (std::int64_t base = static_cast<std::int64_t>(blockIdx.z) * kScatterBatchThreads;
         base < vectors_per_column;
         base += static_cast<std::int64_t>(gridDim.z) * kScatterBatchThreads) {
#pragma unroll
        for (int offset = threadIdx.x; offset < kScatterBatchThreads;
             offset += kScatterBatchThreads) {
            const std::int64_t vector = base + offset;
            if (vector < vectors_per_column)
                destination[destination_base + vector] = source[source_base + vector];
        }
    }
}

using ScatterLaunch = void (*)(const Tensor&, const Tensor&, Tensor&, cudaStream_t);

template <int Width>
void launch_scatter(const Tensor& source, const Tensor& indices, Tensor& destination,
                    cudaStream_t stream) {
    const int columns = source.ne[1];
    const int items   = source.ne[0] / Width;
    const auto* map   = static_cast<const std::int32_t*>(indices.data);
    if constexpr (Width == 8)
        scatter_bf16x8_kernel<<<columns, 128, 0, stream>>>(
            static_cast<const uint4*>(source.data), map, static_cast<uint4*>(destination.data),
            items);
    else if constexpr (Width == 2)
        scatter_bf16x2_kernel<<<columns, 256, 0, stream>>>(
            static_cast<const __nv_bfloat162*>(source.data), map,
            static_cast<__nv_bfloat162*>(destination.data), items);
    else
        scatter_scalar_kernel<<<columns, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(source.data), map,
            static_cast<__nv_bfloat16*>(destination.data), items);
}

ScatterLaunch select_scatter(const Tensor& source, const Tensor& destination) {
    if (source.ne[0] % 8 == 0 && common::operand::aligned(source.data, 16) &&
        common::operand::aligned(destination.data, 16))
        return launch_scatter<8>;
    if (source.ne[0] % 2 == 0 && common::operand::aligned(source.data, 4) &&
        common::operand::aligned(destination.data, 4))
        return launch_scatter<2>;
    return launch_scatter<1>;
}

void launch_scatter_bf16_batch(const Tensor& source, const Tensor& lanes,
                               const Tensor& valid_columns, Tensor& destination,
                               cudaStream_t stream) {
    constexpr int block = kScatterBatchThreads;
    // Keep a grid-stride feature loop beyond CUDA's z-dimension limit.
    const int feature_tiles = std::min((source.ne[0] / 8 + block - 1) / block, 65535);
    const dim3 grid(source.ne[1], source.ne[2], feature_tiles);
    scatter_bf16_batch_kernel<<<grid, block, 0, stream>>>(
        static_cast<const uint4*>(source.data), static_cast<const std::int32_t*>(lanes.data),
        static_cast<const std::int32_t*>(valid_columns.data), static_cast<uint4*>(destination.data),
        source.ne[0] / 8, source.ne[1],
        destination.nb[1] / static_cast<std::int64_t>(sizeof(uint4)),
        destination.nb[2] / static_cast<std::int64_t>(sizeof(uint4)));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void scatter(const Tensor& src, const Tensor& indices, Tensor& dst, CudaExecutionView execution) {
    const auto stream = execution.stream();
    if (src.dtype != DType::BF16 || dst.dtype != DType::BF16 || indices.dtype != DType::I32) {
        throw std::invalid_argument("scatter: src/dst must be BF16 and indices must be I32");
    }
    if (src.ne[2] != 1 || src.ne[3] != 1 || dst.ne[2] != 1 || dst.ne[3] != 1 ||
        indices.ne[1] != 1 || indices.ne[2] != 1 || indices.ne[3] != 1 || src.ne[0] != dst.ne[0] ||
        indices.ne[0] != src.ne[1]) {
        throw std::invalid_argument("scatter: expected src [D,V], indices [V], dst [D,T]");
    }
    if (src.ne[0] < 0 || src.ne[1] < 0 || dst.ne[1] < 0) {
        throw std::invalid_argument("scatter: negative dimension");
    }
    if (src.ne[0] == 0 || src.ne[1] == 0) { return; }
    if (src.device != execution.device() || indices.device != execution.device() ||
        dst.device != execution.device()) {
        throw std::invalid_argument("scatter: operands must belong to the execution device");
    }
    if (!src.is_contiguous() || !indices.is_contiguous() || !dst.is_contiguous()) {
        throw std::invalid_argument("scatter: tensors must be contiguous");
    }
    if (src.data == nullptr || indices.data == nullptr || dst.data == nullptr) {
        throw std::invalid_argument("scatter: tensor data must be non-null");
    }
    select_scatter(src, dst)(src, indices, dst, stream);
    CUDA_CHECK(cudaGetLastError());
}

void scatter_bf16_batch(const Tensor& source, const Tensor& lanes, const Tensor& valid_columns,
                        Tensor& destination, CudaExecutionView execution) {
    if (source.device != execution.device() || lanes.device != execution.device() ||
        valid_columns.device != execution.device() || destination.device != execution.device()) {
        throw std::invalid_argument(
            "scatter_bf16_batch: operands must belong to the execution device");
    }
    const auto stream        = execution.stream();
    const std::int32_t width = source.ne[1];
    const std::int32_t batch = source.ne[2];
    if (source.dtype != DType::BF16 || destination.dtype != DType::BF16 ||
        lanes.dtype != DType::I32 || valid_columns.dtype != DType::I32 || source.ne[0] <= 0 ||
        (source.ne[0] % 8) != 0 || width <= 0 || batch <= 0 || source.ne[3] != 1 ||
        destination.ne[0] != source.ne[0] || destination.ne[1] != width || destination.ne[2] <= 0 ||
        destination.ne[3] != 1 || lanes.ne[0] != batch || valid_columns.ne[0] != batch ||
        lanes.ne[1] != 1 || valid_columns.ne[1] != 1 || lanes.ne[2] != 1 ||
        valid_columns.ne[2] != 1 || lanes.ne[3] != 1 || valid_columns.ne[3] != 1) {
        throw std::invalid_argument("scatter_bf16_batch: invalid [D,W,B] to [D,W,C] geometry");
    }
    if (!source.is_contiguous() || !lanes.is_contiguous() || !valid_columns.is_contiguous() ||
        source.data == nullptr || lanes.data == nullptr || valid_columns.data == nullptr ||
        destination.data == nullptr || destination.nb[0] != 2 ||
        (destination.nb[1] % static_cast<std::int64_t>(sizeof(uint4))) != 0 ||
        (destination.nb[2] % static_cast<std::int64_t>(sizeof(uint4))) != 0 ||
        ((reinterpret_cast<std::uintptr_t>(source.data) |
          reinterpret_cast<std::uintptr_t>(destination.data)) &
         15U) != 0U) {
        throw std::invalid_argument(
            "scatter_bf16_batch: tensors must have aligned dense-column storage");
    }
    launch_scatter_bf16_batch(source, lanes, valid_columns, destination, stream);
}

void extract_bf16_features(const Tensor& source, std::int32_t feature_offset, Tensor& destination,
                           CudaExecutionView execution) {
    if (source.device != execution.device() || destination.device != execution.device()) {
        throw std::invalid_argument(
            "extract_bf16_features: operands must belong to the execution device");
    }
    const auto stream = execution.stream();
    if (source.dtype != DType::BF16 || destination.dtype != DType::BF16) {
        throw std::invalid_argument("extract_bf16_features: tensors must be BF16");
    }
    if (source.ne[0] <= 0 || source.ne[1] <= 0 || destination.ne[0] <= 0 ||
        destination.ne[1] != source.ne[1] || source.ne[2] != 1 || source.ne[3] != 1 ||
        destination.ne[2] != 1 || destination.ne[3] != 1 || feature_offset < 0 ||
        feature_offset > source.ne[0] - destination.ne[0]) {
        throw std::invalid_argument("extract_bf16_features: invalid rank-2 slice geometry");
    }
    if (!source.is_contiguous() || !destination.is_contiguous() || source.data == nullptr ||
        destination.data == nullptr) {
        throw std::invalid_argument(
            "extract_bf16_features: tensors must be contiguous and non-null");
    }
    if (source.data == destination.data) {
        throw std::invalid_argument("extract_bf16_features: source and destination must not alias");
    }
    const std::size_t element_bytes = dtype_size(DType::BF16);
    const std::size_t width         = static_cast<std::size_t>(destination.ne[0]) * element_bytes;
    const std::size_t source_pitch  = static_cast<std::size_t>(source.ne[0]) * element_bytes;
    const std::size_t destination_pitch =
        static_cast<std::size_t>(destination.ne[0]) * element_bytes;
    const auto* source_ptr = static_cast<const unsigned char*>(source.data) +
                             static_cast<std::size_t>(feature_offset) * element_bytes;
    CUDA_CHECK(cudaMemcpy2DAsync(destination.data, destination_pitch, source_ptr, source_pitch,
                                 width, static_cast<std::size_t>(destination.ne[1]),
                                 cudaMemcpyDeviceToDevice, stream));
}

} // namespace ninfer::ops
