#pragma once

#include "ninfer/ops/linear_split.h"
#include <array>

namespace ninfer::ops::detail::linear_split {

using SplitLaunch = void (*)(const Tensor&, const Weight&, std::span<Tensor>, cudaStream_t);

struct SplitShape {
    int rows, input_rows;
    std::span<const int> segments;
    SplitLaunch launch;
};

template <int... Rows>
inline constexpr std::array kSplitRows{Rows...};

extern const SplitShape kQ8SplitN12288K2560;
extern const SplitShape kQ8SplitN16384K2560;
extern const SplitShape kBf16SplitN1664K2560;
extern const SplitShape kBf16SplitN13952K2560;

const SplitShape& select_split_shape(const Weight& weight, std::span<const Tensor> outputs);

} // namespace ninfer::ops::detail::linear_split
