#pragma once

#include "core/tensor.h"
#include "ops/linear_attention/gated_delta_net/common.h"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail::gated_delta_net::replay {
constexpr std::int32_t kStateDim    = ::ninfer::ops::detail::gated_delta_net::kStateDim;
constexpr std::int32_t kMaximumRows = 8;

inline bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

inline void require_tensor(const Tensor& tensor, DType dtype,
                           std::initializer_list<std::int32_t> shape, std::uintptr_t alignment,
                           const char* op, const char* label) {
    const Tensor expected(nullptr, dtype, shape);
    if (tensor.data == nullptr || tensor.dtype != dtype || !tensor.is_contiguous() ||
        tensor.bytes() != expected.bytes() || !aligned_to(tensor.data, alignment)) {
        throw std::invalid_argument(std::string(op) + ": invalid " + label);
    }
    for (int dim = 0; dim < 4; ++dim) {
        if (tensor.ne[dim] != expected.ne[dim]) {
            throw std::invalid_argument(std::string(op) + ": invalid " + label);
        }
    }
}

struct MemoryRange {
    std::uintptr_t begin;
    std::uintptr_t end;
};

inline MemoryRange make_range(const void* pointer, std::size_t bytes, const char* label) {
    if (pointer == nullptr || bytes == 0) {
        throw std::invalid_argument(std::string(label) + " has an empty memory range");
    }
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
        throw std::overflow_error(std::string(label) + " address range overflows");
    }
    return {begin, begin + bytes};
}

inline MemoryRange tensor_range(const Tensor& tensor, const char* label) {
    return make_range(tensor.data, tensor.bytes(), label);
}

inline MemoryRange layer_range(const Tensor& layer0, std::int64_t stride_bytes, std::int32_t layer,
                               const char* label) {
    const auto base    = reinterpret_cast<std::uintptr_t>(layer0.data);
    const auto stride  = static_cast<std::uint64_t>(stride_bytes);
    const auto layer_u = static_cast<std::uint64_t>(layer);
    if (layer_u != 0 && stride > std::numeric_limits<std::uintptr_t>::max() / layer_u) {
        throw std::overflow_error(std::string(label) + " layer offset overflows");
    }
    const auto offset = static_cast<std::uintptr_t>(stride * layer_u);
    if (offset > std::numeric_limits<std::uintptr_t>::max() - base) {
        throw std::overflow_error(std::string(label) + " layer address overflows");
    }
    return make_range(reinterpret_cast<const void*>(base + offset), layer0.bytes(), label);
}

inline bool overlaps(MemoryRange lhs, MemoryRange rhs) {
    return lhs.begin < rhs.end && rhs.begin < lhs.end;
}

template <class Ranges>
inline void require_pairwise_disjoint(const Ranges& ranges, const char* message) {
    for (std::size_t lhs = 0; lhs < ranges.size(); ++lhs) {
        for (std::size_t rhs = lhs + 1; rhs < ranges.size(); ++rhs) {
            if (overlaps(ranges[lhs], ranges[rhs])) { throw std::invalid_argument(message); }
        }
    }
}

inline void require_scale(float scale, const char* op) {
    const float expected = 1.0f / std::sqrt(static_cast<float>(kStateDim));
    if (!std::isfinite(scale) || scale <= 0.0f || std::abs(scale - expected) > 1.0e-6f) {
        throw std::invalid_argument(std::string(op) + ": scale must be 1/sqrt(128)");
    }
}

} // namespace ninfer::ops::detail::gated_delta_net::replay
