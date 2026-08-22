#include "targets/qwen3_6/impl/runtime/instance.h"

#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include "targets/qwen3_6/impl/runtime/layouts.h"
#include "targets/qwen3_6/impl/runtime/program.h"

#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3_6 {

using detail::NINFER_QWEN36_RUNTIME_NS::Variant;

template <>
SequencePlan<Variant> SequencePlanner<Variant>::finalize(std::uint32_t main_page_groups) && {
    return detail::NINFER_QWEN36_RUNTIME_NS::finalize_sequence_plan(std::move(*this),
                                                                    main_page_groups);
}

template <>
Program<Variant>::Program(std::unique_ptr<detail::ProgramImpl<Variant>> implementation) noexcept
    : impl(std::move(implementation)) {}

template <>
Program<Variant>::~Program() noexcept = default;

template <>
RequestBasePlan<Variant>
Program<Variant>::plan_request_base(const PreparedPrompt& prompt,
                                    const runtime::ResolvedExecutionOptions& options) {
    return impl->plan_request_base(prompt.view(), options);
}

template <>
RequestPlan<Variant> Program<Variant>::plan_request_for_lane(std::uint32_t lane,
                                                             const PreparedPrompt& prompt,
                                                             const RequestBasePlan<Variant>& base) {
    return impl->plan_request_for_lane(lane, prompt.view(), base);
}

template <>
bool Program<Variant>::can_admit_lane(std::uint32_t lane,
                                      const RequestPlan<Variant>& plan) const noexcept {
    return impl->can_admit_lane(lane, plan);
}

template <>
bool Program<Variant>::can_admit_lane_after_retained_eviction(
    std::uint32_t lane, const RequestPlan<Variant>& plan) const noexcept {
    return impl->can_admit_lane_after_retained_eviction(lane, plan);
}

template <>
runtime::AdmissionResources Program<Variant>::admission_capacity() const noexcept {
    return impl->admission_capacity();
}

template <>
runtime::PrefillStepResult
Program<Variant>::start_prefill_lane(std::uint32_t lane, PreparedPrompt&& prompt,
                                     RequestPlan<Variant>&& plan,
                                     runtime::TransientRegion transient) {
    return impl->start_prefill_lane(lane, std::move(prompt).take(), std::move(plan), transient);
}

template <>
runtime::PrefillStepResult Program<Variant>::advance_prefill_lane(std::uint32_t lane) {
    return impl->advance_prefill_lane(lane);
}

template <>
runtime::BatchedGeneratedRound
Program<Variant>::decode_batch(std::span<const std::uint32_t> lanes,
                               std::span<const runtime::RoundBudget> budgets) {
    return impl->decode_batch(lanes, budgets);
}

template <>
void Program<Variant>::resolve_pending_batch(std::span<const std::uint32_t> lanes,
                                             std::span<const std::uint32_t> accepted_tokens,
                                             std::span<const std::uint8_t> terminal,
                                             std::span<const std::uint8_t> cancelled) {
    impl->resolve_pending_batch(lanes, accepted_tokens, terminal, cancelled);
}

template <>
void Program<Variant>::resolve_prefill_lane(std::uint32_t lane, bool terminal) {
    impl->resolve_prefill_lane(lane, terminal);
}

template <>
void Program<Variant>::abort_lane(std::uint32_t lane) noexcept {
    impl->abort_lane(lane);
}

template <>
bool Program<Variant>::has_retained_lane(std::uint32_t lane) const noexcept {
    return impl->has_retained_lane(lane);
}

template <>
void Program<Variant>::evict_retained_lane(std::uint32_t lane) noexcept {
    impl->evict_retained_lane(lane);
}

template <>
GenerationTimings Program<Variant>::generation_timings_lane(std::uint32_t lane) const noexcept {
    return impl->generation_timings_lane(lane);
}

template <>
SpeculativeStats Program<Variant>::speculative_stats_lane(std::uint32_t lane) const noexcept {
    return impl->speculative_stats_lane(lane);
}

template <>
MemorySummary Program<Variant>::memory_summary() const noexcept {
    return impl->memory_summary();
}

template <>
void Program<Variant>::reset_memory_peaks() noexcept {
    impl->reset_memory_peaks();
}

template <>
SequencePlanner<Variant> make_sequence_planner<Variant>(DeviceContext& device,
                                                        const EngineOptions& options,
                                                        Variant::WeightsProfile weights_profile) {
    return detail::NINFER_QWEN36_RUNTIME_NS::build_sequence_planner(device, options,
                                                                    weights_profile);
}

template <>
std::unique_ptr<Program<Variant>>
create_program<Variant>(const Variant::ModelView& model, Variant::WeightsProfile weights_profile,
                        SequencePlan<Variant>&& plan, DeviceContext& device) {
    if (plan.capacity == 0) { throw std::invalid_argument("sequence plan is empty"); }
    if (plan.weights_profile != static_cast<std::uint32_t>(weights_profile)) {
        throw std::invalid_argument(
            "loaded model weights profile does not match the sequence plan");
    }
    auto impl = std::make_unique<detail::ProgramImpl<Variant>>(model, plan, device);
    plan      = SequencePlan<Variant>{};
    return std::unique_ptr<Program<Variant>>(new Program<Variant>(std::move(impl)));
}

} // namespace ninfer::targets::qwen3_6
