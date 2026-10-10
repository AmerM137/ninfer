#include "ninfer/ops/linear_split.h"
#include "ops/common/host/operands.h"
#include "ops/linear/split/dispatch.h"

#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require(bool condition, const char* label) {
    if (!condition) throw std::invalid_argument(std::string("linear_split: invalid ") + label);
}

void matrix(const Tensor& t, int rows, int columns) {
    require(t.dtype == DType::BF16 && t.ne[0] == rows && t.ne[1] == columns && t.ne[2] == 1 &&
                t.ne[3] == 1 && t.is_contiguous() && common::operand::aligned(t.data, 16),
            "activation matrix");
}

void validate(const Tensor& x, const Weight& w, std::span<const Tensor> outputs,
              const detail::linear_split::SplitShape& shape) {
    const int columns = x.ne[1];
    require(columns > 0, "column extent");
    matrix(x, shape.input_rows, columns);
    const auto elements  = std::uint64_t(shape.rows) * shape.input_rows;
    const bool quantized = w.qtype == QType::Q8_G32_FP16;
    require(w.ndim == 2 && w.shape[0] == shape.rows && w.shape[1] == shape.input_rows &&
                w.padded_shape[0] == shape.rows && w.padded_shape[1] == shape.input_rows &&
                common::operand::aligned(w.qdata, 16) && w.qhigh == nullptr &&
                w.high_plane_bytes == 0,
            "weight geometry/data");
    if (quantized)
        require(w.layout == QuantLayout::RowSplit && w.group == 32 && w.group_size == 32 &&
                    w.scale_dtype == DType::FP16 && common::operand::aligned(w.scales, 16),
                "Q8 G32 FP16 weight");
    else
        require(w.layout == QuantLayout::Contiguous && w.group == 0 && w.group_size == 0 &&
                    w.scales == nullptr && w.payload_bytes >= elements * 2,
                "BF16 weight");
    const auto input = common::operand::range(x.data, x.bytes(), "linear_split");
    const auto data =
        common::operand::range(w.qdata, elements * (quantized ? 1 : 2), "linear_split");
    const auto scales = quantized
                            ? common::operand::range(w.scales, elements / 32 * 2, "linear_split")
                            : common::operand::Range{};
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        matrix(outputs[i], shape.segments[i], columns);
        const auto out =
            common::operand::range(outputs[i].data, outputs[i].bytes(), "linear_split");
        require(!common::operand::overlaps(out, input) && !common::operand::overlaps(out, data) &&
                    !common::operand::overlaps(out, scales),
                "output/input overlap");
        for (std::size_t j = 0; j < i; ++j)
            require(!common::operand::overlaps(out, common::operand::range(outputs[j].data,
                                                                           outputs[j].bytes(),
                                                                           "linear_split")),
                    "output overlap");
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
