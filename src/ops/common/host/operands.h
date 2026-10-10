#pragma once

#include "core/weight.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>

namespace ninfer::ops::common::operand {

struct Range {
    std::uintptr_t begin, end;
};

inline bool aligned(const void* pointer, std::size_t alignment) noexcept {
    return pointer && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

inline std::size_t bytes(std::uint64_t count, std::size_t width, const char* op) {
    if (count > std::numeric_limits<std::size_t>::max() / width)
        throw std::overflow_error(std::string(op) + ": byte count overflows");
    return static_cast<std::size_t>(count) * width;
}

inline Range range(const void* pointer, std::size_t count, const char* op) {
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (count > std::numeric_limits<std::uintptr_t>::max() - begin)
        throw std::overflow_error(std::string(op) + ": address range overflows");
    return {begin, begin + count};
}

inline bool overlaps(Range a, Range b) noexcept { return a.begin < b.end && b.begin < a.end; }

inline void disjoint(Range a, Range b, const char* op) {
    if (overlaps(a, b))
        throw std::invalid_argument(std::string(op) + ": operands must not overlap");
}

inline void disjoint(std::span<const Range> ranges, const char* op) {
    for (std::size_t i = 0; i < ranges.size(); ++i)
        for (std::size_t j = 0; j < i; ++j) disjoint(ranges[i], ranges[j], op);
}

} // namespace ninfer::ops::common::operand

namespace ninfer::ops::common {

struct Fp8WeightGeometry {
    std::uint64_t code_plane_bytes;
    std::uint64_t scale_plane_offset;
    std::uint64_t scale_plane_bytes;
    std::uint64_t required_payload_bytes;
};

Fp8WeightGeometry validate_fp8_weight(const Weight& weight, const char* operation);

struct Nvfp4WeightGeometry {
    std::uint64_t code_plane_bytes;
    std::uint64_t scale_plane_offset;
    std::uint64_t scale_plane_bytes;
    std::uint64_t required_payload_bytes;
};

Nvfp4WeightGeometry validate_nvfp4_weight(const Weight& weight, const char* operation);

} // namespace ninfer::ops::common
