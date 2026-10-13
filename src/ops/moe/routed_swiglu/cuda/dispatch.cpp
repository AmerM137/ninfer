#include "ops/moe/routed_swiglu/cuda/dispatch.h"
#include "ops/moe/operands.h"

namespace ninfer::ops::detail::moe_routed {

const ShapePolicy& resolve_shape(const RoutedSwiGLUProfile& profile) {
    validate_expert_profile(profile);
    switch (profile.ffn.down_type) {
    case QType::NVFP4:
        return nvfp4_h2560_i640_e512_k10();
    case QType::Q5_G64_FP16:
        return q4_q5_h2048_i512_e256_k8();
    case QType::Q6_G64_FP16:
        return q4_q6_h2048_i512_e256_k8();
    default:
        return q8_q8_h2048_i512_e256_k8();
    }
}

SharedPlan select_shared(int tokens) {
    return {{1, tokens == 1 ? 1 : 4, tokens == 1, false}, tokens >= 16 ? 64 : 0};
}

} // namespace ninfer::ops::detail::moe_routed
