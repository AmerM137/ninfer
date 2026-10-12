#pragma once

#include "ops/common/cpu/avx2.h"

#include <immintrin.h>
#include <cstdint>
#include <cstring>

namespace ninfer::ops::common::cpu::avx512 {

// Private primitives for the isolated avx512f/bw/dq/vl + avx2/fma/f16c profile.
// Unaligned full-vector pointers are supported.
inline __m512 load_bf16(const std::uint16_t* source) noexcept {
    const auto words = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(source));
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(words), 16));
}

inline float reduce_sum(__m512 values) noexcept { return _mm512_reduce_add_ps(values); }

// 16 packed bytes -> 32 exact BF16 signed Q4 codes, without group scales.
inline __m512i decode_signed_q4_bf16(const std::uint8_t* source) noexcept {
    const auto unpacked                                  = avx2::unpack_u4(source);
    const auto indexes                                   = _mm512_cvtepu8_epi16(unpacked);
    alignas(64) static constexpr std::uint16_t table[32] = {
        0,      0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0, 0x40e0, 0xc100, 0xc0e0, 0xc0c0,
        0xc0a0, 0xc080, 0xc040, 0xc000, 0xbf80, 0,      0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0,
        0x40c0, 0x40e0, 0xc100, 0xc0e0, 0xc0c0, 0xc0a0, 0xc080, 0xc040, 0xc000, 0xbf80};
    return _mm512_permutexvar_epi16(indexes, _mm512_load_si512(table));
}

template <int Bits>
inline __m512 decode_signed_q(const std::uint8_t* code, const std::uint8_t* high) noexcept {
    static_assert(Bits >= 4 && Bits <= 6);
    std::uint64_t packed;
    std::memcpy(&packed, code, sizeof(packed));
    const auto words = _mm_cvtepu8_epi16(_mm_cvtsi64_si128(packed));
    const auto bytes =
        _mm_or_si128(_mm_and_si128(words, _mm_set1_epi16(15)),
                     _mm_and_si128(_mm_slli_epi16(words, 4), _mm_set1_epi16(0x0f00)));
    auto values = _mm512_cvtepu8_epi32(bytes);
    if constexpr (Bits == 6) {
        std::uint32_t packed_high;
        std::memcpy(&packed_high, high, sizeof(packed_high));
        auto upper = _mm512_srlv_epi32(
            _mm512_set1_epi32(static_cast<int>(packed_high)),
            _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30));
        // The high two bits form a signed two-bit integer: 0, 1, -2, -1.
        // Sign-extend it and multiply by 16, then add the unsigned low nibble.
        upper  = _mm512_srai_epi32(_mm512_slli_epi32(upper, 30), 26);
        values = _mm512_add_epi32(values, upper);
    } else if constexpr (Bits == 5) {
        std::uint16_t bits;
        std::memcpy(&bits, high, sizeof(bits));
        values = _mm512_mask_sub_epi32(values, bits, values, _mm512_set1_epi32(16));
    } else {
        values =
            _mm512_sub_epi32(_mm512_xor_si512(values, _mm512_set1_epi32(8)), _mm512_set1_epi32(8));
    }
    return _mm512_cvtepi32_ps(values);
}

} // namespace ninfer::ops::common::cpu::avx512
