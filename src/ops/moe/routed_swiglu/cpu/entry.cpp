#include "ninfer/ops/moe.h"
#include "ops/common/host/operands.h"
#include "ops/moe/routed_swiglu/cpu/dispatch.h"

#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace ninfer::ops {
namespace {
namespace impl    = detail::moe_routed_cpu;
namespace kernel  = detail::moe_cpu;
namespace operand = common::operand;
constexpr auto op = "routed_swiglu CPU";

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(message); }
}

class Operands {
public:
    Operands(const Tensor& x, Tensor& out, MemoryArena& workspace, std::size_t capacity) {
        operand::tensor(x, DType::BF16, {kernel::hidden, x.ne[1]}, 2, op, "x", Device::cpu());
        operand::tensor(out, DType::FP32, {kernel::hidden, x.ne[1]}, 4, op, "destination",
                        Device::cpu());
        operand::device(workspace.device(), Device::cpu(), op, "workspace");
        require(capacity <= workspace.capacity() - workspace.used(),
                "routed_swiglu CPU: insufficient workspace");
        output_ = operand::range(out, op);
        if (capacity) {
            const auto* begin =
                static_cast<const std::byte*>(workspace.storage().data) + workspace.used();
            require(operand::aligned(begin, 256),
                    "routed_swiglu CPU: workspace cursor must be 256-byte aligned");
            scratch_ = operand::range(begin, capacity, op);
            operand::disjoint(output_, scratch_, op);
        }
        read(x.data, x.bytes());
    }

    void tensor(const Tensor& value, DType dtype, std::initializer_list<int> shape,
                const char* name) const {
        operand::tensor(value, dtype, shape, dtype_size(dtype), op, name, Device::cpu());
        read(value.data, value.bytes());
    }

    kernel::MatrixBank bank(const WeightBankView& bank, QType type, int rows, int columns,
                            const std::array<bool, kernel::experts>& selected) const {
        require(bank.parent && bank.parent->data, "routed_swiglu CPU: missing weight bank");
        operand::device(bank.parent->device, Device::cpu(), op, "weight bank");
        const auto& layout    = bank.layout;
        const auto& geometry  = bank.parent->geometry;
        const auto elements   = std::size_t(rows) * columns;
        const auto high_bytes = type == QType::Q4_G64_FP16   ? 0
                                : type == QType::Q5_G64_FP16 ? elements / 8
                                                             : elements / 4;
        require(layout.format == type && layout.layout == QuantLayout::RowSplit &&
                    layout.region.count == kernel::experts && layout.region.rows == rows &&
                    layout.columns == columns && geometry.group_size == 64 &&
                    geometry.padded_columns == columns && geometry.scale_block_rows == 1 &&
                    !layout.has_divisors && layout.codes.expert_bytes == elements / 2 &&
                    layout.high.expert_bytes == high_bytes &&
                    layout.scales.expert_bytes == elements / 32 &&
                    layout == describe_weight_bank(geometry, layout.region),
                "routed_swiglu CPU: bank differs from its native profile");
        for (int expert = 0; expert < kernel::experts; ++expert) {
            if (!selected[expert]) { continue; }
            for (const auto& plane : {layout.codes, layout.high, layout.scales}) {
                if (!plane.expert_bytes) { continue; }
                const auto memory = bank.plane(plane, expert);
                require(operand::aligned(memory.data, 16),
                        "routed_swiglu CPU: weight plane must be 16-byte aligned");
                read(memory.data, memory.bytes);
            }
        }
        return {{static_cast<const std::uint8_t*>(bank.plane(layout.codes, 0).data),
                 static_cast<const std::uint8_t*>(bank.plane(layout.high, 0).data),
                 static_cast<const std::uint16_t*>(bank.plane(layout.scales, 0).data)},
                layout.codes.expert_stride,
                layout.high.expert_stride,
                layout.scales.expert_stride};
    }

private:
    void read(const void* pointer, std::size_t bytes) const {
        if (!bytes) { return; }
        const auto range = operand::range(pointer, bytes, op);
        operand::disjoint(output_, range, op);
        if (scratch_.begin != scratch_.end) { operand::disjoint(scratch_, range, op); }
    }

    operand::Range output_{}, scratch_{};
};

} // namespace

std::size_t routed_swiglu_workspace_capacity_bytes(const RoutedSwiGLUProfile& profile,
                                                   ExpertSubsetKind subset, std::int32_t first,
                                                   std::int32_t last) {
    const auto& shape = impl::resolve_shape(profile);
    impl::validate_capacity(subset, first, last);
    return impl::capacity(shape, subset, last);
}

void routed_swiglu(const Tensor& x, const RoutedSwiGLUProfile& profile,
                   const CpuSwiGLUExperts& experts, const ExpertRoutes& routes, ExpertSubset subset,
                   Tensor& destination, MemoryArena& workspace, CpuExecutionView execution) {
    const auto& shape = impl::resolve_shape(profile);
    const int tokens  = x.ne[1];
    impl::validate_capacity(subset.kind, tokens, tokens);
    const auto capacity = impl::capacity(shape, subset.kind, tokens);
    const Operands operands(x, destination, workspace, capacity);
    if (subset.kind == ExpertSubsetKind::None) {
        std::memset(destination.data, 0, destination.bytes());
        return;
    }

    operands.tensor(routes.ids, DType::I32, {profile.top_k, tokens}, "route IDs");
    operands.tensor(routes.coefficients, DType::FP32, {profile.top_k, tokens}, "coefficients");
    const std::int32_t* mask = nullptr;
    if (subset.kind == ExpertSubsetKind::MaskOnes || subset.kind == ExpertSubsetKind::MaskZeros) {
        operands.tensor(subset.mask, DType::I32, {profile.experts}, "mask");
        mask = static_cast<const std::int32_t*>(subset.mask.data);
        for (int e = 0; e < profile.experts; ++e) {
            require(mask[e] == 0 || mask[e] == 1, "routed_swiglu CPU: mask must be binary");
        }
    }
    const auto* ids          = static_cast<const std::int32_t*>(routes.ids.data);
    const auto* coefficients = static_cast<const float*>(routes.coefficients.data);
    std::array<bool, kernel::experts> selected{};
    bool has_work = false;
    for (int t = 0; t < tokens; ++t) {
        const auto offset = std::size_t(t) * profile.top_k;
        for (int j = 0; j < profile.top_k; ++j) {
            const int expert = ids[offset + j];
            require(expert >= 0 && expert < profile.experts &&
                        std::isfinite(coefficients[offset + j]),
                    "routed_swiglu CPU: invalid route ID or coefficient");
            for (int previous = 0; previous < j; ++previous) {
                require(expert != ids[offset + previous], "routed_swiglu CPU: duplicate route ID");
            }
            if (kernel::participates(subset.kind, mask, expert)) {
                selected[expert] = true;
                has_work         = true;
            }
        }
    }
    if (!has_work) {
        std::memset(destination.data, 0, destination.bytes());
        return;
    }
    const auto gate_up = operands.bank(experts.gate_up, profile.ffn.gate_up_type,
                                       2 * profile.ffn.intermediate, profile.ffn.hidden, selected);
    const auto down    = operands.bank(experts.down, profile.ffn.down_type, profile.ffn.hidden,
                                       profile.ffn.intermediate, selected);
    const auto plan    = shape.select(execution.features());
    auto scope         = workspace.scope();
    kernel::Work work{static_cast<const std::uint16_t*>(x.data),
                      ids,
                      coefficients,
                      gate_up,
                      down,
                      subset.kind,
                      mask,
                      static_cast<float*>(destination.data),
                      tokens,
                      shape.token_block,
                      plan.max_columns,
                      shape.down_bits,
                      impl::allocate(workspace, shape, tokens)};
    if (plan.prepare_first) { kernel::prepare_chunk(work, 0, std::min(tokens, shape.token_block)); }
    execution.run(plan.run, &work);
}

} // namespace ninfer::ops
