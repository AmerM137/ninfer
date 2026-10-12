#pragma once

#include "core/memory.h"
#include "core/weight_view.h"

namespace ninfer {

struct ExpertBankId {
    std::uint32_t index = UINT32_MAX;

    friend bool operator==(ExpertBankId, ExpertBankId) = default;
};

struct ExpertKey {
    ExpertBankId bank;
    std::uint32_t expert = 0;

    friend bool operator==(ExpertKey, ExpertKey) = default;
};

// Rows are measured in the flattened parent, preserving its last (K) axis.
struct WeightBankRegion {
    std::uint64_t row_begin  = 0;
    std::uint32_t count      = 0;
    std::uint64_t rows       = 0;
    std::uint64_t row_stride = 0;

    friend bool operator==(const WeightBankRegion&, const WeightBankRegion&) = default;
};

struct WeightBankPlane {
    std::uint64_t offset        = 0;
    std::uint64_t expert_bytes  = 0;
    std::uint64_t expert_stride = 0;

    friend bool operator==(const WeightBankPlane&, const WeightBankPlane&) = default;
};

// Pointer-free description. Plane offsets refer to the original encoded parent.
struct WeightBankLayout {
    WeightBankRegion region;
    QType format          = QType::BF16;
    QuantLayout layout    = QuantLayout::Contiguous;
    std::uint64_t columns = 0;
    WeightBankPlane codes, high, scales;
    std::uint64_t divisor_begin  = 0;
    std::uint64_t divisor_stride = 0;
    bool has_divisors            = false;

    friend bool operator==(const WeightBankLayout&, const WeightBankLayout&) = default;
};

// Borrows immutable backing and Host divisor metadata from the model.
struct WeightBankView {
    const WeightParent* parent = nullptr;
    WeightBankLayout layout;

    [[nodiscard]] ConstMemorySpan plane(const WeightBankPlane& plane, std::uint32_t expert) const;
    // Returns the multiplicative identity when the representation has no matrix divisor.
    [[nodiscard]] float divisor(std::uint32_t expert) const;
};

struct ExpertBankSource {
    WeightBankView gate_up;
    WeightBankView down;
    // Optional for CPU-only math; GPU replicas require registered views of both parents.
    CudaHostView gate_up_upload;
    CudaHostView down_upload;
};

[[nodiscard]] WeightBankLayout describe_weight_bank(const WeightGeometry& geometry,
                                                    const WeightBankRegion& region);
[[nodiscard]] WeightBankView bind_weight_bank(const WeightParent& parent,
                                              const WeightBankLayout& layout);
[[nodiscard]] Weight matrix_weight(const WeightBankView& bank, std::uint32_t expert,
                                   float input_divisor = 0.0F);

} // namespace ninfer
