#include "ninfer/ops/linear_split.h"
#include "ops/linear/split/dispatch.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require(bool condition, const char* label) {
    if (!condition) throw std::invalid_argument(std::string("linear_split: invalid ") + label);
}

bool aligned(const void* p) { return p && reinterpret_cast<std::uintptr_t>(p) % 16 == 0; }

void matrix(const Tensor& t, int rows, int columns) {
    require(t.dtype == DType::BF16 && t.ne[0] == rows && t.ne[1] == columns && t.ne[2] == 1 &&
                t.ne[3] == 1 && t.is_contiguous() && aligned(t.data),
            "activation matrix");
}

struct Range {
    std::uintptr_t begin, end;
};

Range range(const void* data, std::uint64_t bytes) {
    const auto begin = reinterpret_cast<std::uintptr_t>(data);
    require(bytes <= std::numeric_limits<std::uintptr_t>::max() - begin, "address range");
    return {begin, begin + bytes};
}

bool overlap(Range a, Range b) { return a.begin < b.end && b.begin < a.end; }

void validate(const Tensor& x, const Weight& w, std::span<const Tensor> outputs,
              const detail::linear_split::SplitShape& shape) {
    const int columns = x.ne[1];
    require(columns > 0, "column extent");
    matrix(x, shape.input_rows, columns);
    const auto elements  = std::uint64_t(shape.rows) * shape.input_rows;
    const bool quantized = w.qtype == QType::Q8_G32_FP16;
    require(w.ndim == 2 && w.shape[0] == shape.rows && w.shape[1] == shape.input_rows &&
                w.padded_shape[0] == shape.rows && w.padded_shape[1] == shape.input_rows &&
                aligned(w.qdata) && w.qhigh == nullptr && w.high_plane_bytes == 0,
            "weight geometry/data");
    if (quantized)
        require(w.layout == QuantLayout::RowSplit && w.group == 32 && w.group_size == 32 &&
                    w.scale_dtype == DType::FP16 && aligned(w.scales),
                "Q8 G32 FP16 weight");
    else
        require(w.layout == QuantLayout::Contiguous && w.group == 0 && w.group_size == 0 &&
                    w.scales == nullptr && w.payload_bytes >= elements * 2,
                "BF16 weight");
    const auto input  = range(x.data, x.bytes());
    const auto data   = range(w.qdata, elements * (quantized ? 1 : 2));
    const auto scales = quantized ? range(w.scales, elements / 32 * 2) : Range{};
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        matrix(outputs[i], shape.segments[i], columns);
        const auto out = range(outputs[i].data, outputs[i].bytes());
        require(!overlap(out, input) && !overlap(out, data) && !overlap(out, scales),
                "output/input overlap");
        for (std::size_t j = 0; j < i; ++j)
            require(!overlap(out, range(outputs[j].data, outputs[j].bytes())), "output overlap");
    }
}

} // namespace

void linear_split(const Tensor& x, const Weight& weight, std::span<Tensor> outputs,
                  cudaStream_t stream) {
    const auto& shape = detail::linear_split::select_split_shape(weight, outputs);
    validate(x, weight, outputs, shape);
    shape.launch(x, weight, outputs, stream);
}

} // namespace ninfer::ops
