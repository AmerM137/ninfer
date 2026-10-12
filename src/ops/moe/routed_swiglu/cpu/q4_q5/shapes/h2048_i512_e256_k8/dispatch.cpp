#include "ops/moe/routed_swiglu/cpu/dispatch.h"

namespace ninfer::ops::detail::moe_routed_cpu {
namespace {
Plan select(CpuFeatures features) {
    switch (select_isa(features)) {
    case Isa::Avx512Bf16:
        return {moe_cpu::run_avx512_bf16, 4, true};
    case Isa::Avx512:
        return {moe_cpu::run_avx512, 4, true};
    case Isa::Avx2:
        return {moe_cpu::run_avx2, 2, false};
    }
    __builtin_unreachable();
}
} // namespace

const ShapePolicy& q4_q5_h2048_i512_e256_k8() {
    static const ShapePolicy policy{8, 5, select};
    return policy;
}
} // namespace ninfer::ops::detail::moe_routed_cpu
