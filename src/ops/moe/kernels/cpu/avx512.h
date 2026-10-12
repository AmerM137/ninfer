#pragma once

#include "ops/common/cpu/avx512.h"

namespace ninfer::ops::detail::moe_cpu {
struct Avx512 {
    using Float                = __m512;
    static constexpr int lanes = 16, rows = 4, max_columns = 4;
    static constexpr bool bf16_gate = false;

    static Float zero() noexcept { return _mm512_setzero_ps(); }

    static Float load(const float* p) noexcept { return _mm512_loadu_ps(p); }

    static void store(float* p, Float v) noexcept { _mm512_storeu_ps(p, v); }

    static Float bf16(const std::uint16_t* p) noexcept {
        return ::ninfer::ops::common::cpu::avx512::load_bf16(p);
    }

    static Float fma(Float a, Float b, Float c) noexcept { return _mm512_fmadd_ps(a, b, c); }

    static Float scale(std::uint16_t bits) noexcept { return _mm512_set1_ps(_cvtsh_ss(bits)); }

    static float sum(Float v) noexcept { return ::ninfer::ops::common::cpu::avx512::reduce_sum(v); }

    template <int Bits>
    static Float decode(const std::uint8_t* code, const std::uint8_t* high) noexcept {
        return common::cpu::avx512::decode_signed_q<Bits>(code, high);
    }
};
} // namespace ninfer::ops::detail::moe_cpu
