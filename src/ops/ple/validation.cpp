#include "ops/ple/validation.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail::ple {
namespace {

void require(bool condition, const char* label) {
    if (!condition) throw std::invalid_argument(std::string("PLE: invalid ") + label);
}

bool aligned(const void* p, std::size_t bytes) {
    return p && reinterpret_cast<std::uintptr_t>(p) % bytes == 0;
}

void contiguous(const Tensor& t, DType dtype, std::initializer_list<std::int32_t> shape,
                std::size_t alignment, const char* label) {
    const Tensor expected(nullptr, dtype, shape);
    require(t.dtype == dtype && t.is_contiguous() && aligned(t.data, alignment), label);
    for (int d = 0; d < 4; ++d) require(t.ne[d] == expected.ne[d], label);
}

struct Range {
    std::uintptr_t begin, end;
};

Range range(const Tensor& t, std::uint64_t bytes) {
    const auto begin = reinterpret_cast<std::uintptr_t>(t.data);
    require(bytes <= std::numeric_limits<std::uintptr_t>::max() - begin, "address range");
    return {begin, begin + bytes};
}

bool overlap(Range a, Range b) { return a.begin < b.end && b.begin < a.end; }

// A column is contiguous. The two outer axes can be padded or interleaved. One has
// at most nine entries (history columns or batch rows), so validation never scans R/S.
struct Columns {
    Range bounds;
    std::uint64_t small_count, small_stride, large_count, large_stride;
};

Columns columns(const Tensor& t, int count, int outer, const char* label) {
    require(count > 0 && outer > 0 && t.dtype == DType::BF16 && t.ne[0] == kChannels &&
                t.ne[1] == count && t.ne[2] == outer && t.ne[3] == 1 && t.nb[0] == 2 &&
                aligned(t.data, 2),
            label);
    constexpr std::uint64_t width = kChannels * 2;
    std::uint64_t bytes           = width;
    for (int d = 1; d <= 2; ++d) {
        if (t.ne[d] == 1) continue;
        require(t.nb[d] >= static_cast<std::int64_t>(width) && t.nb[d] % 2 == 0, label);
        const auto step = static_cast<std::uint64_t>(t.nb[d]);
        const auto n    = static_cast<std::uint64_t>(t.ne[d] - 1);
        require(step <= (std::numeric_limits<std::int64_t>::max() - bytes) / n, label);
        bytes += step * n;
    }
    const int small = count <= outer ? 1 : 2, large = 3 - small;
    Columns result{range(t, bytes), static_cast<std::uint64_t>(t.ne[small]),
                   t.ne[small] == 1 ? width : static_cast<std::uint64_t>(t.nb[small]),
                   static_cast<std::uint64_t>(t.ne[large]),
                   t.ne[large] == 1 ? width : static_cast<std::uint64_t>(t.nb[large])};
    // Columns from different small-axis positions must not overlap any column of
    // the other progression. Only the two nearest large-axis offsets can overlap.
    for (std::uint64_t delta = 1; delta < result.small_count; ++delta) {
        const auto distance = delta * result.small_stride;
        const auto q        = distance / result.large_stride;
        if (q < result.large_count) require(distance - q * result.large_stride >= width, label);
        if (q + 1 < result.large_count)
            require((q + 1) * result.large_stride - distance >= width, label);
    }
    return result;
}

bool overlap(Range a, const Columns& b) {
    if (!overlap(a, b.bounds)) return false;
    constexpr std::uint64_t width = kChannels * 2;
    for (std::uint64_t i = 0; i < b.small_count; ++i) {
        const auto first = b.bounds.begin + i * b.small_stride;
        if (a.end <= first) continue;
        const auto j = a.begin < first + width ? 0 : (a.begin - first - width) / b.large_stride + 1;
        if (j < b.large_count && first + j * b.large_stride < a.end) return true;
    }
    return false;
}

void metadata(const Tensor& t, int batch, const char* label) {
    contiguous(t, DType::I32, {batch}, 4, label);
}

void execution_domain(int batch, DeviceExecutionView execution) {
    require(batch >= 1 && batch <= 8, "batch extent");
    require(execution.multiprocessor_count > 0, "SM count");
}

} // namespace

void validate_conv(const Tensor& residual, const Tensor& gated, const Tensor& kernel,
                   const PleBatchView& batch, const PleConvStateView& before,
                   const PleConvRecordWindowView& current, const Tensor& updated,
                   const Tensor* norm_weight, float epsilon, const Tensor* normalized,
                   DeviceExecutionView execution) {
    const int tokens = residual.ne[2], rows = residual.ne[3];
    execution_domain(rows, execution);
    require(tokens > 0, "token extent");
    for (const Tensor* t : {&residual, &gated, &updated})
        contiguous(*t, DType::BF16, {kFeatures, kStreams, tokens, rows}, 16, "activation");
    if (updated.data == residual.data)
        for (int d = 0; d < 4; ++d)
            require(updated.nb[d] == residual.nb[d], "in-place residual strides");
    contiguous(kernel, DType::BF16, {kChannels, 4}, 2, "convolution kernel");
    metadata(batch.valid_columns, rows, "valid columns");
    metadata(before.slots, rows, "source slots");
    metadata(current.offsets, rows, "record offsets");
    const auto history = columns(before.history, kHistory, before.history.ne[2], "history");
    const auto records = columns(current.values, current.values.ne[1], rows, "records");
    if (normalized) {
        contiguous(*normalized, DType::BF16, {kFeatures, kStreams, tokens, rows}, 16, "normalized");
        contiguous(*norm_weight, DType::BF16, {kFeatures, kStreams}, 2, "norm weight");
        require(std::isfinite(epsilon) && epsilon > 0, "norm epsilon");
    }
    for (const Tensor* output : {&updated, normalized}) {
        if (!output) continue;
        const auto out = range(*output, output->bytes());
        for (const Tensor* input : {&residual, &gated, &kernel, &batch.valid_columns, &before.slots,
                                    &current.offsets, norm_weight}) {
            if (!input || (output == &updated && input == &residual && output->data == input->data))
                continue;
            require(!overlap(out, range(*input, input->bytes())), "output/input overlap");
        }
        require(!overlap(out, history) && !overlap(out, records),
                "output/history or record overlap");
    }
    if (normalized)
        require(!overlap(range(updated, updated.bytes()), range(*normalized, normalized->bytes())),
                "output overlap");
}

void validate_extract(const PleConvStateView& before, const PleConvRecordReadView& records,
                      const Tensor& accepted, const PleConvStateView& after,
                      DeviceExecutionView execution) {
    const int batch = accepted.ne[0];
    execution_domain(batch, execution);
    metadata(accepted, batch, "accepted columns");
    metadata(records.valid_columns, batch, "record valid columns");
    metadata(before.slots, batch, "source slots");
    metadata(after.slots, batch, "destination slots");
    (void)columns(before.history, kHistory, before.history.ne[2], "source history");
    (void)columns(after.history, kHistory, after.history.ne[2], "destination history");
    (void)columns(records.values, records.values.ne[1], batch, "records");
    // Only selected slots are read/written. Their cross-row and in-place relationships
    // are Device-index promises; overlapping pool descriptors alone are not an error.
}

} // namespace ninfer::ops::detail::ple
