#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <string>

namespace ninfer::ops::common::operand {

struct Range {
    std::uintptr_t begin, end;
};

inline void dtype(const Tensor& tensor, DType expected, const char* op, const char* name) {
    if (tensor.dtype != expected) throw std::invalid_argument(std::string(op) + ": " + name);
}

inline void shape(const Tensor& tensor, std::array<std::int32_t, 4> expected, const char* op,
                  const char* name) {
    for (int d = 0; d < 4; ++d)
        if (tensor.ne[d] != expected[d])
            throw std::invalid_argument(std::string(op) + ": invalid shape for " + name);
}

inline void contiguous_nonnull(const Tensor& tensor, const char* op, const char* name) {
    if (!tensor.is_contiguous())
        throw std::invalid_argument(std::string(op) + ": " + name + " must be contiguous");
    if (!tensor.data)
        throw std::invalid_argument(std::string(op) + ": " + name + " data must be non-null");
}

inline std::int64_t numel(const Tensor& tensor, const char* op) {
    bool empty = false;
    for (const auto extent : tensor.ne) {
        if (extent < 0)
            throw std::invalid_argument(std::string(op) + ": negative tensor dimension");
        empty |= extent == 0;
    }
    if (empty) return 0;
    std::int64_t count = 1;
    for (const auto extent : tensor.ne) {
        if (count > std::numeric_limits<std::int64_t>::max() / extent)
            throw std::overflow_error(std::string(op) + ": tensor size overflows int64");
        count *= extent;
    }
    return count;
}

inline bool aligned(const void* pointer, std::size_t alignment) noexcept {
    return pointer && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

inline void tensor(const Tensor& value, DType dtype, std::initializer_list<std::int32_t> dimensions,
                   std::size_t alignment, const char* op, const char* name) {
    const Tensor expected(nullptr, dtype, dimensions);
    if (value.dtype != dtype || !value.is_contiguous() || !aligned(value.data, alignment))
        throw std::invalid_argument(std::string(op) + ": invalid " + name);
    for (int axis = 0; axis < 4; ++axis)
        if (value.ne[axis] != expected.ne[axis])
            throw std::invalid_argument(std::string(op) + ": invalid " + name);
    if (value.bytes() != expected.bytes())
        throw std::invalid_argument(std::string(op) + ": invalid " + name);
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

inline Range range(const Tensor& tensor, const char* op) {
    return range(tensor.data, tensor.bytes(), op);
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
