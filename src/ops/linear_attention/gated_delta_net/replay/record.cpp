#include "ninfer/ops/gated_delta_net.h"
#include "ops/linear_attention/gated_delta_net/replay/validation.h"
#include "ops/linear_attention/gated_delta_net/replay/launch.h"
#include <array>
#include <span>

namespace ninfer::ops {
namespace {
using namespace detail::gated_delta_net::replay;

void validate_replay_record(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                            const Tensor& beta, float scale, const Tensor& states,
                            const Tensor& valid_columns, const Tensor& initial_slots,
                            const Tensor& key_record, const Tensor& value_record,
                            const Tensor& gate_record, const Tensor& out) {
    constexpr const char* kOp      = "gated_delta_net_replay_record";
    const std::int32_t qk_heads    = q.ne[1];
    const std::int32_t value_heads = v.ne[1];
    const std::int32_t width       = q.ne[2];
    const std::int32_t rows        = q.ne[3];
    const bool registered_heads    = qk_heads == 16 && (value_heads == 48 || value_heads == 32);
    if (!registered_heads || width < 2 || width > 16 || rows <= 0 || rows > kMaximumRows) {
        throw std::invalid_argument(std::string(kOp) + ": unsupported geometry");
    }
    if (states.ne[3] <= 0) {
        throw std::invalid_argument(std::string(kOp) + ": state slot count must be positive");
    }

    require_tensor(q, DType::BF16, {kStateDim, qk_heads, width, rows}, 8, kOp, "q");
    require_tensor(k, DType::BF16, {kStateDim, qk_heads, width, rows}, 8, kOp, "k");
    require_tensor(v, DType::BF16, {kStateDim, value_heads, width, rows}, 2, kOp, "v");
    require_tensor(g, DType::FP32, {value_heads, width, rows}, 4, kOp, "g");
    require_tensor(beta, DType::FP32, {value_heads, width, rows}, 4, kOp, "beta");
    require_tensor(states, DType::FP32, {kStateDim, kStateDim, value_heads, states.ne[3]}, 16, kOp,
                   "states");
    require_tensor(initial_slots, DType::I32, {rows}, 4, kOp, "initial state slots");
    if (valid_columns.data != nullptr) {
        require_tensor(valid_columns, DType::I32, {rows}, 4, kOp, "valid columns");
    }
    require_tensor(key_record, DType::BF16, {kStateDim, qk_heads, width, rows}, 8, kOp,
                   "key record");
    require_tensor(value_record, DType::BF16, {kStateDim, value_heads, width, rows}, 2, kOp,
                   "value record");
    require_tensor(gate_record, DType::FP32, {2, value_heads, width, rows}, 8, kOp, "gate record");
    require_tensor(out, DType::BF16, {kStateDim, value_heads, width, rows}, 2, kOp, "out");
    require_scale(scale, kOp);

    const std::array<const Tensor*, 12> tensors{&q,
                                                &k,
                                                &v,
                                                &g,
                                                &beta,
                                                &states,
                                                &valid_columns,
                                                &initial_slots,
                                                &key_record,
                                                &value_record,
                                                &gate_record,
                                                &out};
    std::array<MemoryRange, 12> ranges{};
    std::size_t count = 0;
    for (const Tensor* tensor : tensors) {
        if (tensor->data != nullptr) ranges[count++] = tensor_range(*tensor, kOp);
    }
    require_pairwise_disjoint(std::span(ranges.data(), count),
                              "gated_delta_net_replay_record: tensors must not overlap");
}

} // namespace

void gated_delta_net_replay_record(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   const Tensor& ssm_states, const Tensor& valid_columns,
                                   const Tensor& initial_state_slots, GdnRecurrenceRecords records,
                                   Tensor& out, cudaStream_t stream) {
    validate_replay_record(q, k, v, g, beta, scale, ssm_states, valid_columns, initial_state_slots,
                           records.key, records.value, records.gate, out);
    detail::gated_delta_net::launch_recurrent_record(
        q, k, v, g, beta, scale, ssm_states, valid_columns, initial_state_slots, records.key,
        records.value, records.gate, out, stream);
}

} // namespace ninfer::ops
