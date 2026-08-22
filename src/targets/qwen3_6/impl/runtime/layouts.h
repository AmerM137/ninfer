#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.

#include <cstdint>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

[[nodiscard]] SequencePlanner build_sequence_planner(DeviceContext& device,
                                                     const EngineOptions& options,
                                                     WeightsProfile weights_profile);
[[nodiscard]] SequencePlan finalize_sequence_plan(SequencePlanner&& planner,
                                                  std::uint32_t main_page_groups);

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
