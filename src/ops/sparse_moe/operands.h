#pragma once

#include "ops/sparse_moe/plan.h"
#include "ops/common/host/operands.h"
#include <vector>

namespace ninfer::ops::moe_detail {

constexpr std::int32_t kExpertRows     = 1024;
constexpr std::int32_t kRoutedGateRows = kExperts * kExpertRows;
constexpr std::int32_t kRoutedDownRows = kExperts * kHidden;
constexpr std::int32_t kSharedGateRows = 2 * kIntermediate;

inline std::int32_t require_tensor(const Tensor& tensor, const char* name, Device device) {
    common::operand::device(tensor.device, device, "sparse_moe", name);
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != kHidden || tensor.ne[1] < 1 ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        tensor.data == nullptr || !common::operand::aligned(tensor.data, 16)) {
        throw std::invalid_argument(std::string("sparse_moe: invalid ") + name);
    }
    return tensor.ne[1];
}

inline void require_matrix_metadata(const Weight& weight, std::int32_t n, std::int32_t k,
                                    const char* name) {
    if (weight.ndim != 2 || weight.n != n || weight.k != k || weight.shape[0] != n ||
        weight.shape[1] != k || weight.shape[2] != 1 || weight.shape[3] != 1 ||
        weight.padded_shape[0] != n || weight.padded_shape[1] != k || weight.padded_shape[2] != 1 ||
        weight.padded_shape[3] != 1) {
        throw std::invalid_argument(std::string("sparse_moe: invalid shape for ") + name);
    }
}

inline void require_router(const Weight& weight, std::vector<common::operand::Range>& ranges) {
    require_matrix_metadata(weight, kRouterRows, kHidden, "router_shared_gate");
    const std::size_t bytes = static_cast<std::size_t>(kRouterRows) * kHidden * 2;
    if (weight.qtype != QType::BF16 || weight.layout != QuantLayout::Contiguous ||
        weight.qdata == nullptr || weight.qhigh != nullptr || weight.scales != nullptr ||
        weight.payload_bytes < bytes || !common::operand::aligned(weight.qdata, 16)) {
        throw std::invalid_argument(
            "sparse_moe: router_shared_gate must be aligned contiguous BF16");
    }
    ranges.push_back(common::operand::range(weight.qdata, bytes, "sparse_moe"));
}

struct QuantGeometry {
    std::int32_t group_size;
    std::size_t code_bytes_per_group;
    std::size_t high_bytes_per_group;
};

inline QuantGeometry quant_geometry(QType qtype) {
    switch (qtype) {
    case QType::Q4_G64_FP16:
        return {64, 32, 0};
    case QType::Q5_G64_FP16:
        return {64, 32, 8};
    case QType::Q6_G64_FP16:
        return {64, 32, 16};
    case QType::Q8_G32_FP16:
        return {32, 32, 0};
    default:
        throw std::invalid_argument("sparse_moe: unsupported quantized weight format");
    }
}

inline void require_quantized(const Weight& weight, std::int32_t n, std::int32_t k,
                              const char* name, std::vector<common::operand::Range>& ranges) {
    require_matrix_metadata(weight, n, k, name);
    const QuantGeometry geometry       = quant_geometry(weight.qtype);
    const std::size_t groups           = static_cast<std::size_t>(n) * k / geometry.group_size;
    const std::size_t code_bytes       = groups * geometry.code_bytes_per_group;
    const std::size_t high_bytes       = groups * geometry.high_bytes_per_group;
    const std::size_t scale_bytes      = groups * 2;
    const std::size_t required_payload = code_bytes + high_bytes + scale_bytes;
    if (weight.layout != QuantLayout::RowSplit || weight.scale_dtype != DType::FP16 ||
        weight.group_size != static_cast<std::uint32_t>(geometry.group_size) ||
        weight.group != geometry.group_size || weight.qdata == nullptr ||
        weight.scales == nullptr || weight.payload_bytes < required_payload ||
        weight.high_plane_bytes < high_bytes || !common::operand::aligned(weight.qdata, 16) ||
        !common::operand::aligned(weight.scales, 16)) {
        throw std::invalid_argument(std::string("sparse_moe: invalid row-split ") + name);
    }
    if ((high_bytes == 0 && weight.qhigh != nullptr) ||
        (high_bytes != 0 &&
         (weight.qhigh == nullptr || !common::operand::aligned(weight.qhigh, 16)))) {
        throw std::invalid_argument(std::string("sparse_moe: invalid high plane for ") + name);
    }
    ranges.push_back(common::operand::range(weight.qdata, code_bytes, "sparse_moe"));
    if (high_bytes != 0) {
        ranges.push_back(common::operand::range(weight.qhigh, high_bytes, "sparse_moe"));
    }
    ranges.push_back(common::operand::range(weight.scales, scale_bytes, "sparse_moe"));
}

inline void validate_weights(const SparseMoeWeights& weights,
                             std::vector<common::operand::Range>& ranges, Device device) {
    for (const Weight* weight :
         {&weights.router_shared_gate, &weights.routed_gate_up, &weights.routed_down,
          &weights.shared_gate_up, &weights.shared_down}) {
        common::operand::device(weight->device, device, "sparse_moe", "weight");
    }
    require_router(weights.router_shared_gate, ranges);
    if (weights.routed_gate_up.qtype != QType::Q4_G64_FP16 &&
        weights.routed_gate_up.qtype != QType::Q8_G32_FP16) {
        throw std::invalid_argument("sparse_moe: routed_gate_up must be Q4 or Q8");
    }
    if (weights.routed_down.qtype != QType::Q5_G64_FP16 &&
        weights.routed_down.qtype != QType::Q6_G64_FP16 &&
        weights.routed_down.qtype != QType::Q8_G32_FP16) {
        throw std::invalid_argument("sparse_moe: routed_down must be Q5, Q6, or Q8");
    }
    if (weights.shared_gate_up.qtype != QType::Q8_G32_FP16 ||
        weights.shared_down.qtype != QType::Q8_G32_FP16) {
        throw std::invalid_argument("sparse_moe: shared weights must be Q8");
    }
    require_quantized(weights.routed_gate_up, kRoutedGateRows, kHidden, "routed_gate_up", ranges);
    require_quantized(weights.routed_down, kRoutedDownRows, kIntermediate, "routed_down", ranges);
    require_quantized(weights.shared_gate_up, kSharedGateRows, kHidden, "shared_gate_up", ranges);
    require_quantized(weights.shared_down, kHidden, kIntermediate, "shared_down", ranges);
}

} // namespace ninfer::ops::moe_detail
