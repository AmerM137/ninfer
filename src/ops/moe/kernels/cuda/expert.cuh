#pragma once

#include "ops/moe/kernels/cuda/launch.h"
#include "ops/common/numeric/math.cuh"
#include "ops/common/numeric/convert.cuh"
#include "ops/common/cuda/memory.cuh"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/quant/integer.cuh"
#include "core/pdl.cuh"

namespace ninfer::ops::detail {

struct ExpertPlanes {
    const std::uint8_t* codes;
    const std::uint8_t* high;
    const std::uint8_t* scales;
    int row;
};

struct PackedExpertMatrix {
    const std::uint8_t* codes;
    const std::uint8_t* high;
    const std::uint8_t* scales;

    template <int Rows>
    __device__ __forceinline__ ExpertPlanes resolve(int expert) const {
        return {codes, high, scales, expert * Rows};
    }
};

struct StridedExpertMatrix {
    PackedExpertMatrix planes;
    std::uint64_t code_stride, high_stride, scale_stride;

    template <int Rows>
    __device__ __forceinline__ ExpertPlanes resolve(int expert) const {
        return {planes.codes + expert * code_stride,
                planes.high ? planes.high + expert * high_stride : nullptr,
                planes.scales + expert * scale_stride, 0};
    }
};

struct IndexedExpertMatrix {
    PackedExpertMatrix planes;
    const std::uint64_t* offsets;

    template <int Rows>
    __device__ __forceinline__ ExpertPlanes resolve(int expert) const {
        const auto offset = offsets[expert];
        return {planes.codes + offset, planes.high ? planes.high + offset : nullptr,
                planes.scales + offset, 0};
    }
};

inline PackedExpertMatrix matrix(const ExpertMatrixDeviceView& view) {
    return {reinterpret_cast<const std::uint8_t*>(view.codes),
            reinterpret_cast<const std::uint8_t*>(view.high),
            reinterpret_cast<const std::uint8_t*>(view.scales)};
}

inline PackedExpertMatrix matrix(const Weight& weight) {
    return {static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.qhigh),
            static_cast<const std::uint8_t*>(weight.scales)};
}

template <class Function>
void with_expert_addresses(const CudaSwiGLUExperts& experts, const RoutedSwiGLUProfile& profile,
                           Function&& function) {
    const auto& gu   = experts.bank.gate_up;
    const auto& down = experts.bank.down;
    if (gu.offsets) {
        function(IndexedExpertMatrix{matrix(gu), gu.offsets},
                 IndexedExpertMatrix{matrix(down), down.offsets});
        return;
    }
    const auto packed = [](const ExpertMatrixDeviceView& view, QType type, int n, int k) {
        const auto elements = std::uint64_t(n) * k;
        const bool q8       = type == QType::Q8_G32_FP16;
        const auto high     = type == QType::Q5_G64_FP16   ? elements / 8
                              : type == QType::Q6_G64_FP16 ? elements / 4
                                                           : 0;
        return view.code_stride == (q8 ? elements : elements / 2) && view.high_stride == high &&
               view.scale_stride == elements / (q8 ? 16 : 32);
    };
    const auto& p = profile.ffn;
    if (packed(gu, p.gate_up_type, 2 * p.intermediate, p.hidden) &&
        packed(down, p.down_type, p.hidden, p.intermediate)) {
        function(matrix(gu), matrix(down));
        return;
    }
    function(
        StridedExpertMatrix{matrix(gu), gu.code_stride, gu.high_stride, gu.scale_stride},
        StridedExpertMatrix{matrix(down), down.code_stride, down.high_stride, down.scale_stride});
}

struct FloatContribution {
    float* data;

    __device__ __forceinline__ float initial(std::int64_t) const { return 0; }

    __device__ __forceinline__ void store(std::int64_t index, float value) const {
        data[index] = value;
    }
};

struct BfloatResult {
    __nv_bfloat16* data;
    bool add_residual;

    __device__ __forceinline__ float initial(std::int64_t index) const {
        return add_residual ? __bfloat162float(data[index]) : 0.0F;
    }

    __device__ __forceinline__ void store(std::int64_t index, float value) const {
        data[index] = __float2bfloat16_rn(value);
    }
};

template <class Function>
void with_output(Tensor& output, MoeEpilogue epilogue, Function&& function) {
    if (output.dtype == DType::FP32) {
        function(FloatContribution{static_cast<float*>(output.data)});
    } else {
        function(BfloatResult{static_cast<__nv_bfloat16*>(output.data),
                              epilogue == MoeEpilogue::AddResidual});
    }
}

__device__ __forceinline__ void read_ahead(const MoeExecutionHints& hints) {
    const auto offset = (std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) * 128;
    if (hints.read_ahead && offset < hints.read_ahead_bytes) {
        const auto* address = static_cast<const char*>(hints.read_ahead) + offset;
        asm volatile("prefetch.global.L2 [%0];" ::"l"(address));
    }
}

} // namespace ninfer::ops::detail
