#pragma once

#include "options.h"
#include "direct_bf16_weight.cuh"
#include "quantized_weight.cuh"

#include <numeric>

namespace ninfer::bench::qsa {

inline constexpr int kInputRows = 2560;

struct Weights {
    std::vector<int> rows;
    int parent_rows;
    DeviceBuffer storage;
    Weight weight;

    explicit Weights(Projection projection)
        : rows(segments(projection)), parent_rows(std::accumulate(rows.begin(), rows.end(), 0)) {
        if (projection == Projection::QueryGate) {
            auto packed = make_row_split_weight(QType::Q8_G32_FP16, parent_rows, kInputRows,
                                                kInputRows, 701U);
            weight      = packed.weight;
            storage     = std::move(packed.storage);
        } else {
            auto packed = make_direct_bf16_weight(parent_rows, kInputRows, 701U);
            weight      = packed.weight;
            storage     = std::move(packed.storage);
        }
    }
};

struct Fixture {
    DeviceBuffer input, output;
    Tensor x, packed;
    std::vector<Tensor> split;

    Fixture(const Weights& weights, int tokens)
        : input(make_bf16(std::size_t(kInputRows) * tokens, 101U)),
          output(make_zeros(std::size_t(weights.parent_rows) * tokens * 2)),
          x(input.p, DType::BF16, {kInputRows, tokens}),
          packed(output.p, DType::BF16, {weights.parent_rows, tokens}) {
        auto* next = static_cast<std::uint16_t*>(output.p);
        for (int rows : weights.rows) {
            split.emplace_back(next, DType::BF16,
                               std::initializer_list<std::int32_t>{rows, tokens});
            next += std::size_t(rows) * tokens;
        }
    }
};

} // namespace ninfer::bench::qsa
