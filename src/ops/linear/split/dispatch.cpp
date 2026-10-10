#include "ops/linear/split/dispatch.h"

#include <stdexcept>

namespace ninfer::ops::detail::linear_split {
namespace {

const SplitShape& match(std::span<const SplitShape* const> shapes, const Weight& weight,
                        std::span<const Tensor> outputs) {
    for (const auto* shape : shapes) {
        if (shape->rows != weight.n || shape->input_rows != weight.k ||
            shape->segments.size() != outputs.size())
            continue;
        bool matches = true;
        for (std::size_t i = 0; i < outputs.size(); ++i)
            matches = matches && shape->segments[i] == outputs[i].ne[0];
        if (matches) return *shape;
    }
    throw std::invalid_argument("linear_split: unsupported weight geometry or output segments");
}

} // namespace

const SplitShape& select_split_shape(const Weight& weight, std::span<const Tensor> outputs) {
    switch (weight.qtype) {
    case QType::Q8_G32_FP16: {
        const std::array shapes{&kQ8SplitN12288K2560, &kQ8SplitN16384K2560};
        return match(shapes, weight, outputs);
    }
    case QType::BF16: {
        const std::array shapes{&kBf16SplitN1664K2560, &kBf16SplitN13952K2560};
        return match(shapes, weight, outputs);
    }
    default:
        throw std::invalid_argument("linear_split: unsupported weight format");
    }
}

} // namespace ninfer::ops::detail::linear_split
