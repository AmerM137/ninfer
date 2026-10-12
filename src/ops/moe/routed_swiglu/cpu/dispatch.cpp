#include "ops/moe/routed_swiglu/cpu/dispatch.h"

#include <stdexcept>

namespace ninfer::ops::detail::moe_routed_cpu {

Isa select_isa(CpuFeatures features) {
    const auto bit = [](CpuFeature feature) { return static_cast<std::uint32_t>(feature); };
    const CpuFeatures avx2{bit(CpuFeature::Avx2) | bit(CpuFeature::Fma) | bit(CpuFeature::F16c)};
    if (!features.contains(avx2)) {
        throw std::invalid_argument("routed_swiglu CPU requires AVX2, FMA and F16C workers");
    }
#ifndef NINFER_MOE_CPU_AVX2_ONLY
    const CpuFeatures avx512{avx2.bits | bit(CpuFeature::Avx512F) | bit(CpuFeature::Avx512Bw) |
                             bit(CpuFeature::Avx512Dq) | bit(CpuFeature::Avx512Vl)};
    if (features.contains(avx512)) {
#    ifndef NINFER_MOE_CPU_FMA_ONLY
        if (features.has(CpuFeature::Avx512Bf16)) { return Isa::Avx512Bf16; }
#    endif
        return Isa::Avx512;
    }
#endif
    return Isa::Avx2;
}

const ShapePolicy& resolve_shape(const RoutedSwiGLUProfile& p) {
    const auto& f = p.ffn;
    if (f.hidden != 2048 || f.intermediate != 512 || p.experts != 256 || p.top_k != 8 ||
        f.gate_up_type != QType::Q4_G64_FP16 || f.gate_up_layout != QuantLayout::RowSplit ||
        f.down_layout != QuantLayout::RowSplit || !valid_linear_policy(f.gate_up_policy) ||
        !valid_linear_policy(f.down_policy)) {
        throw std::invalid_argument("routed_swiglu CPU: unsupported profile");
    }
    switch (f.down_type) {
    case QType::Q5_G64_FP16:
        return q4_q5_h2048_i512_e256_k8();
    case QType::Q6_G64_FP16:
        return q4_q6_h2048_i512_e256_k8();
    default:
        throw std::invalid_argument("routed_swiglu CPU: unsupported down format");
    }
}

void validate_capacity(ExpertSubsetKind subset, int first, int last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("routed_swiglu CPU: invalid token interval");
    }
    if (subset != ExpertSubsetKind::All && subset != ExpertSubsetKind::None &&
        subset != ExpertSubsetKind::MaskOnes && subset != ExpertSubsetKind::MaskZeros) {
        throw std::invalid_argument("routed_swiglu CPU: invalid expert subset");
    }
}

} // namespace ninfer::ops::detail::moe_routed_cpu
