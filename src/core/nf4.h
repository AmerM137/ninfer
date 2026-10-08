#pragma once

#include <array>
#include <bit>
#include <cstdint>

namespace ninfer {

inline constexpr std::array<std::uint32_t, 16> kNf4CodebookWords{
    0xbf800000, 0xbf3239b1, 0xbf066b30, 0xbeca32a0, 0xbe91a24d, 0xbe3d353f, 0xbdba7871, 0x00000000,
    0x3da2faff, 0x3e24cae3, 0x3e7c04dd, 0x3ead033a, 0x3ee1a4b8, 0x3f1007ab, 0x3f3913b3, 0x3f800000};

inline constexpr float nf4_code_value(std::uint8_t code) {
    return std::bit_cast<float>(kNf4CodebookWords[code & 15]);
}

} // namespace ninfer
