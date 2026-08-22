#include "targets/qwen3_6/impl/runtime/instance.h"

#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include "targets/qwen3_6/impl/runtime/layouts.h"
#include "targets/qwen3_6/impl/runtime/program.h"

#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3_6 {

using detail::NINFER_QWEN36_RUNTIME_NS::Variant;

template <>
SequencePlan<Variant>::SequencePlan(
    std::unique_ptr<detail::SequencePlanImpl<Variant>> impl) noexcept
    : impl_(std::move(impl)) {}

template <>
SequencePlan<Variant>::SequencePlan(SequencePlan&&) noexcept = default;
template <>
SequencePlan<Variant>& SequencePlan<Variant>::operator=(SequencePlan&&) noexcept = default;
template <>
SequencePlan<Variant>::~SequencePlan() = default;

template <>
std::uint32_t SequencePlan<Variant>::capacity() const noexcept {
    return impl_ != nullptr ? impl_->capacity : 0;
}

template <>
std::uint32_t SequencePlan<Variant>::kv_capacity() const noexcept {
    return impl_ != nullptr ? impl_->kv_capacity : 0;
}

template <>
std::uint32_t SequencePlan<Variant>::max_concurrency() const noexcept {
    return impl_ != nullptr ? impl_->max_concurrency : 0;
}

template <>
std::size_t SequencePlan<Variant>::device_reservation_bytes() const noexcept {
    return impl_ != nullptr ? impl_->device_reservation_bytes : 0;
}

template <>
std::size_t SequencePlan<Variant>::workspace_capacity_bytes() const noexcept {
    return impl_ != nullptr ? impl_->workspace.capacity : 0;
}

template <>
std::size_t SequencePlan<Variant>::request_transient_capacity_bytes() const noexcept {
    return impl_ != nullptr ? impl_->request_transient_capacity_bytes : 0;
}

template <>
SequencePlanner<Variant>::SequencePlanner(
    std::unique_ptr<detail::SequencePlannerImpl<Variant>> impl) noexcept
    : impl_(std::move(impl)) {}

template <>
SequencePlanner<Variant>::SequencePlanner(SequencePlanner&&) noexcept = default;
template <>
SequencePlanner<Variant>& SequencePlanner<Variant>::operator=(SequencePlanner&&) noexcept = default;
template <>
SequencePlanner<Variant>::~SequencePlanner() = default;

template <>
const runtime::SequenceCapacityCurve& SequencePlanner<Variant>::capacity_curve() const noexcept {
    static const runtime::SequenceCapacityCurve empty;
    return impl_ != nullptr ? impl_->curve : empty;
}

template <>
SequencePlan<Variant> SequencePlanner<Variant>::finalize(std::uint32_t main_page_groups) && {
    if (impl_ == nullptr) { throw std::logic_error("sequence planner is empty"); }
    return SequencePlan<Variant>(detail::NINFER_QWEN36_RUNTIME_NS::finalize_sequence_plan_impl(
        std::move(impl_), main_page_groups));
}

template <>
RequestBasePlan<Variant>::RequestBasePlan(
    std::unique_ptr<detail::RequestBasePlanImpl<Variant>> impl) noexcept
    : impl_(std::move(impl)) {}

template <>
RequestBasePlan<Variant>::RequestBasePlan(RequestBasePlan&&) noexcept = default;
template <>
RequestBasePlan<Variant>& RequestBasePlan<Variant>::operator=(RequestBasePlan&&) noexcept = default;
template <>
RequestBasePlan<Variant>::~RequestBasePlan() = default;

template <>
const runtime::RequestPlanSummary& RequestBasePlan<Variant>::summary() const noexcept {
    static const runtime::RequestPlanSummary empty;
    return impl_ != nullptr ? impl_->summary : empty;
}

template <>
AdmissionPlan<Variant>::AdmissionPlan(
    std::unique_ptr<detail::AdmissionPlanImpl<Variant>> impl) noexcept
    : impl_(std::move(impl)) {}

template <>
AdmissionPlan<Variant>::AdmissionPlan(AdmissionPlan&&) noexcept = default;
template <>
AdmissionPlan<Variant>& AdmissionPlan<Variant>::operator=(AdmissionPlan&&) noexcept = default;
template <>
AdmissionPlan<Variant>::~AdmissionPlan() = default;

template <>
const runtime::RequestPlanSummary& AdmissionPlan<Variant>::summary() const noexcept {
    static const runtime::RequestPlanSummary empty;
    return impl_ != nullptr ? impl_->summary : empty;
}

template <>
const runtime::ResourceDemand& AdmissionPlan<Variant>::demand() const noexcept {
    static const runtime::ResourceDemand empty;
    return impl_ != nullptr ? impl_->demand : empty;
}

template <>
Program<Variant>::Program(std::unique_ptr<detail::ProgramImpl<Variant>> impl) noexcept
    : impl_(std::move(impl)) {}

template <>
Program<Variant>::~Program() noexcept = default;

template <>
RequestBasePlan<Variant>
Program<Variant>::plan_request(const PreparedPrompt& prompt,
                               const runtime::ResolvedExecutionOptions& options) {
    return impl_->plan_request(PreparedPromptAccess::view(prompt), options);
}

template <>
std::optional<AdmissionPlan<Variant>> Program<Variant>::inspect_admission(
    const PreparedPrompt& prompt, const RequestBasePlan<Variant>& base, runtime::LaneId destination,
    const ContinuationHandle<Variant>* source, std::optional<runtime::CheckpointRef> checkpoint) {
    return impl_->inspect_admission(PreparedPromptAccess::view(prompt), base, destination, source,
                                    checkpoint);
}

template <>
runtime::DeviceResources Program<Variant>::admission_capacity() const noexcept {
    return impl_->admission_capacity();
}

template <>
MaterializationTicket<Variant> Program<Variant>::reserve_materialization(
    AdmissionPlan<Variant>&& plan, PreparedPrompt&& prompt,
    const ContinuationHandle<Variant>* source,
    std::span<const ContinuationHandle<Variant>* const> victims) {
    return impl_->reserve_materialization(
        std::move(plan), PreparedPromptAccess::take(std::move(prompt)), source, victims);
}

template <>
ReleaseResult<Variant>
Program<Variant>::release_materialization_victim(MaterializationTicket<Variant>& ticket,
                                                 ContinuationHandle<Variant>&& victim) noexcept {
    return impl_->release_materialization_victim(ticket, std::move(victim));
}

template <>
runtime::ConsumeStatus
Program<Variant>::abort_materialization(MaterializationTicket<Variant>&& ticket) noexcept {
    return impl_->abort_materialization(std::move(ticket));
}

template <>
void Program<Variant>::prepare_materialization(MaterializationTicket<Variant>& ticket) {
    impl_->prepare_materialization(ticket);
}

template <>
MaterializationResult<Variant>
Program<Variant>::publish_materialization(MaterializationTicket<Variant>&& ticket,
                                          std::optional<ContinuationHandle<Variant>>&& source,
                                          runtime::CancellationFlagView cancellation) {
    return impl_->publish_materialization(std::move(ticket), std::move(source), cancellation);
}

template <>
PrefillProgress<Variant> Program<Variant>::advance_prefill(SequenceHandle<Variant> sequence) {
    return impl_->advance_prefill(sequence);
}

template <>
PendingBatch<Variant> Program<Variant>::decode(std::span<const SequenceHandle<Variant>> sequences,
                                               std::span<const runtime::RoundBudget> budgets) {
    return impl_->decode(sequences, budgets);
}

template <>
CommitResult<Variant> Program<Variant>::commit(PendingBatch<Variant>&& pending,
                                               std::span<const runtime::CommitDecision> decisions,
                                               runtime::CommitObservation observation) {
    return impl_->commit(std::move(pending), decisions, observation);
}

template <>
DiscardResult<Variant> Program<Variant>::abort_pending(PendingBatch<Variant>&& pending) noexcept {
    return impl_->abort_pending(std::move(pending));
}

template <>
FinishResult<Variant> Program<Variant>::finish(SequenceHandle<Variant> sequence) noexcept {
    return impl_->finish(sequence);
}

template <>
AbortResult<Variant> Program<Variant>::abort(SequenceHandle<Variant> sequence) noexcept {
    return impl_->abort(sequence);
}

template <>
ReleaseResult<Variant>
Program<Variant>::release_continuation(ContinuationHandle<Variant>&& continuation) noexcept {
    return impl_->release_continuation(std::move(continuation));
}

template <>
void Program<Variant>::fail_all_cleanup() noexcept {
    impl_->fail_all_cleanup();
}

template <>
MemorySummary Program<Variant>::memory_summary() const noexcept {
    return impl_->memory_summary();
}

template <>
void Program<Variant>::reset_memory_peaks() noexcept {
    impl_->reset_memory_peaks();
}

template <>
SequencePlanner<Variant> make_sequence_planner<Variant>(DeviceContext& device,
                                                        const EngineOptions& options,
                                                        Variant::WeightsProfile weights_profile) {
    return SequencePlanner<Variant>(detail::NINFER_QWEN36_RUNTIME_NS::make_sequence_planner_impl(
        device, options, weights_profile));
}

template <>
std::unique_ptr<Program<Variant>>
create_program<Variant>(const Variant::ModelView& model, Variant::WeightsProfile weights_profile,
                        SequencePlan<Variant>&& plan, DeviceContext& device) {
    if (plan.impl_ == nullptr) { throw std::invalid_argument("sequence plan is empty"); }
    if (plan.impl_->weights_profile != weights_profile) {
        throw std::invalid_argument(
            "loaded model weights profile does not match the sequence plan");
    }
    auto impl = std::make_unique<detail::ProgramImpl<Variant>>(model, *plan.impl_, device);
    plan.impl_.reset();
    return std::unique_ptr<Program<Variant>>(new Program<Variant>(std::move(impl)));
}

} // namespace ninfer::targets::qwen3_6
