#pragma once

#include "models/qwen3_5/model.h"

#include <string>

namespace ninfer::models::qwen3_5 {

namespace detail {
struct MoePlan;
}

// Measurement applicability, independent of checkpoint names, object IDs, payload values and
// device addresses. This key never selects or admits an execution implementation.
[[nodiscard]] std::string prefill_signature(const Model& model,
                                            const detail::MoePlan* moe = nullptr,
                                            std::uint32_t chunk        = 0);

} // namespace ninfer::models::qwen3_5
