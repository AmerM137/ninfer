#include "ops/moe/topk_softmax_swiglu_moe/dispatch.h"
#include "ops/moe/operands.h"

namespace ninfer::ops::detail::moe_complete {

const ShapePolicy& resolve_shape(const MoeProfile& profile) {
    validate_moe_profile(profile);
    switch (profile.routed.ffn.down_type) {
    case QType::Q5_G64_FP16:
        return q4_q5_h2048_i512_e256_k8();
    case QType::Q6_G64_FP16:
        return q4_q6_h2048_i512_e256_k8();
    default:
        return q8_q8_h2048_i512_e256_k8();
    }
}
} // namespace ninfer::ops::detail::moe_complete
