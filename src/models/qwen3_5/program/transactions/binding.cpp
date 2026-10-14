#include "models/common/qwen/vision/prefill.h"
#include "models/qwen3_5/program/program_impl.h"
#include "models/common/context/work.h"
#include "core/device.h"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {

void ProgramImpl::initialize_captures(std::uint32_t lane, std::uint32_t from,
                                      std::uint32_t through) {
    auto& request = requests[lane];
    request.capture_groups.clear();
    request.next_capture    = 0;
    request.capture_pending = false;
    for (const auto& group : request.base->capture_groups) {
        if (group.frontier > from && group.frontier <= through) {
            request.capture_groups.push_back(group);
        }
    }
}

void ProgramImpl::initialize_prefill(std::uint32_t lane, std::uint32_t base) {
    auto& request    = requests[lane];
    const auto& plan = *request.base;
    auto& staged     = request.prefill.emplace();
    staged.prompt    = *plan.prompt;
    staged.base = staged.cursor = base;
    staged.prompt_tokens        = plan.summary.prompt_tokens;
    staged.prepare_mtp          = speculative_backend == SpeculativeBackend::Mtp;
    staged.initial_mtp_extent   = initial_mtp_extent(plan);
    staged.reuse                = base ? PrefixReusePath::Checkpoint : PrefixReusePath::Root;
    staged.mtp_bridge           = !staged.prepare_mtp || !base   ? MtpBridgeMode::None
                                  : base == staged.prompt_tokens ? MtpBridgeMode::AfterExactHit
                                                                 : MtpBridgeMode::BeforeSuffix;
    initialize_captures(lane, base, plan.summary.prompt_tokens);
    if (plan.vision_control_plan) {
        auto& vision        = staged.vision_plan.emplace();
        vision.control_plan = plan.vision_control_plan;
        vision.control      = std::make_shared<qwen::VisionControl>(
            qwen::build_vision_control(staged.prompt, *vision.control_plan, 0));
        for (std::uint32_t i = 0; i < vision.control_plan->items.size(); ++i) {
            const auto& item = vision.control_plan->items[i];
            if (item.token_end <= base) { continue; }
            const auto begin =
                staged.prepare_mtp && item.token_begin ? item.token_begin - 1U : item.token_begin;
            vision.uses.push_back({begin, item.token_end, i, i});
            vision.max_merged_count = std::max(vision.max_merged_count, item.merged_count);
        }
        if (!vision.uses.empty()) {
            staged.vision = std::make_unique<qwen::VisionPrefillSession>(
                device, *parameters.model.config().vision, *parameters.vision,
                workspace_storage.span(), *workspace_plan.vision, staged.prompt, vision,
                vision_handoff, vision_handoff_peak_bytes);
        }
    }
}

BindingReservation ProgramImpl::start_binding(const RequestBasePlan& base, runtime::LaneId lane_id,
                                              const SourceCandidate& candidate, ResumeState* resume,
                                              ExecutionUnitKind resume_kind,
                                              std::uint32_t resume_tokens) {
    const auto lane = lane_id.value;
    if (context_transaction_ || lane >= max_concurrency ||
        requests[lane].lifecycle != Lifecycle::Empty || !base.impl_)
        throw std::logic_error("binding requires a free lane and context transaction slot");
    const bool own_snapshot = resume && resume->has_snapshot();
    const auto source       = own_snapshot ? resume->impl_->snapshot : candidate.checkpoint;
    if (source &&
        (!valid_checkpoint(*source) ||
         (!own_snapshot && (!checkpoint_matches(*source, base) ||
                            (resume && checkpoint(*source).frontier() > resume->frontier())))))
        return {.source_valid = false};
    ContextTransaction transaction;
    transaction.kind             = ContextOperationKind::Bind;
    transaction.lane             = lane;
    transaction.epoch            = lane_epochs[lane];
    transaction.base             = base.impl_;
    transaction.resume           = resume;
    transaction.source           = source;
    transaction.reuse_frontier   = source ? checkpoint(*source).frontier() : 0;
    transaction.backend_frontier = source ? checkpoint(*source).backend_frontier() : 0;
    transaction.source_tail_hidden =
        source && checkpoint(*source).continuation().metadata.tail_hidden_valid;
    transaction.resume_snapshot = own_snapshot;
    const auto select_source    = [&](const SourceCandidate& actual) {
        transaction.take_private   = candidate.take_private;
        transaction.carried_points = actual.private_points;
        if (actual.move_state) std::erase(transaction.carried_points, *source);
        transaction.consume_source =
            actual.consume_source &&
            std::find(transaction.carried_points.begin(), transaction.carried_points.end(),
                         *source) == transaction.carried_points.end();
        transaction.borrow_state = actual.move_state;
        transaction.split_state  = actual.split_state;
        transaction.backup_state = actual.backup_state;
    };
    if (source && !own_snapshot) {
        const auto actual = inspect_source(base, source, candidate.consume_source,
                                           candidate.private_points, candidate.retired_points);
        if (!actual) return {.source_valid = false};
        select_source(*actual);
    } else if (own_snapshot) {
        transaction.consume_source = true;
        transaction.take_private   = candidate.take_private;
        for (auto point : candidate.private_points)
            if (valid_checkpoint(point) &&
                checkpoint(point).frontier() <= checkpoint(*source).frontier())
                transaction.carried_points.push_back(point);
        const auto image         = checkpoint(*source).continuation().image;
        transaction.borrow_state = checkpoints[source->index].pins == 0 &&
                                   state_store->checkpoint_references(image) == 1 &&
                                   state_store->source_pins(image) == 0;
        transaction.split_state =
            !transaction.borrow_state && state_store->device_resident(image) &&
            state_store->host_resident(image) && state_store->source_pins(image) == 0 &&
            state_store->device_occupied() == state_store->device_capacity();
    }
    if (!own_snapshot && speculative_backend == SpeculativeBackend::Mtp &&
        transaction.reuse_frontier)
        transaction.backend_frontier = transaction.reuse_frontier - 1;
    auto& request = requests[lane];
    UnitDemand first;
    if (!resume)
        first = prefill_unit(base.summary().prompt_tokens, transaction.reuse_frontier,
                             initial_mtp_extent(*base.impl_));
    else if (!own_snapshot) {
        first                  = {.kind = ExecutionUnitKind::Replay,
                                  .main_frontier =
                                      std::min(transaction.reuse_frontier + prefill_chunk, resume->frontier())};
        first.backend_frontier = backend_kv_cache() ? first.main_frontier : 0;
    } else
        first =
            next_unit(resume->impl_->sequence, resume->impl_->control, resume_kind, resume_tokens);
    transaction.first_unit = transaction.reservation_demand = first;
    if (resume) {
        auto coverage =
            next_unit(resume->impl_->sequence, resume->impl_->control, resume_kind, resume_tokens);
        coverage.main_frontier = std::max(coverage.main_frontier, resume->frontier());
        if (backend_kv_cache())
            coverage.backend_frontier = std::max(coverage.backend_frontier, resume->frontier());
        transaction.recovery =
            RecoveryPermit{.coverage      = coverage,
                           .frontier      = resume->frontier(),
                           .ledger_tokens = resume->impl_->sequence.ledger.size()};
        transaction.reservation_demand = coverage;
    }
    const auto& coverage       = transaction.reservation_demand;
    const auto* main_source    = source ? &checkpoint(*source).text_prefix() : nullptr;
    const auto* backend_source = source ? checkpoint(*source).backend_prefix() : nullptr;
    std::vector<const context::KVPrefixRef*> retired_main, retired_backend;
    const auto retire = [&](CheckpointHandle point) {
        retired_main.push_back(&checkpoint(point).text_prefix());
        if (const auto* prefix = checkpoint(point).backend_prefix())
            retired_backend.push_back(prefix);
    };
    for (auto point : candidate.retired_points) retire(point);
    if (transaction.consume_source) retire(*source);
    const auto main = text_kv_histories->binding_demand(main_source, transaction.reuse_frontier,
                                                        coverage.main_frontier, retired_main);
    const auto backend =
        backend_kv_histories
            ? backend_kv_histories->binding_demand(backend_source, transaction.backend_frontier,
                                                   coverage.backend_frontier, retired_backend)
            : context::KVBindingDemand{};
    const auto usage      = physical_usage();
    const auto difference = [](auto needed, auto available) {
        return needed > available ? needed - available : 0U;
    };
    const bool needs_state =
        (!transaction.borrow_state && !transaction.split_state) ||
        (transaction.borrow_state &&
         !state_store->device_resident(checkpoint(*source).continuation().image));
    runtime::ContextResourceUsage shortage{
        .state_slots =
            needs_state && usage.occupied.state_slots == usage.capacity.state_slots ? 1U : 0U,
        .main_kv_pages =
            difference(main.device_pages(), text_kv_pages->physical_pool().available_pages()),
        .backend_kv_pages = backend_kv_pages
                                ? difference(backend.device_pages(),
                                             backend_kv_pages->physical_pool().available_pages())
                                : 0U};
    if (!candidate.retired_points.empty() &&
        (shortage.state_slots || shortage.main_kv_pages || shortage.backend_kv_pages)) {
        const auto credit = checkpoint_release_resources(candidate.retired_points, shortage);
        if (!credit) return {.source_valid = false};
        shortage.state_slots      = difference(shortage.state_slots, credit->state_slots);
        shortage.main_kv_pages    = difference(shortage.main_kv_pages, credit->main_kv_pages);
        shortage.backend_kv_pages = difference(shortage.backend_kv_pages, credit->backend_kv_pages);
    }
    if (shortage.state_slots || shortage.main_kv_pages || shortage.backend_kv_pages)
        return {
            .capacity_possible =
                context::kv_pages_for_frontier(coverage.main_frontier) + main.tail_copies <=
                    usage.capacity.main_kv_pages &&
                (!backend_kv_pages ||
                 context::kv_pages_for_frontier(coverage.backend_frontier) + backend.tail_copies <=
                     usage.capacity.backend_kv_pages),
            .shortage = shortage};
    BindingReservation result{.reserved       = true,
                              .retired_points = candidate.retired_points,
                              .consumed_source =
                                  transaction.consume_source ? source : std::nullopt};
    request.base      = base.impl_;
    request.lifecycle = Lifecycle::Prefilling;
    initialize_prefill(lane, transaction.reuse_frontier);
    sequences[lane].lane = lane;
    for (auto point : candidate.retired_points) {
        if (!release_checkpoint(point))
            throw std::logic_error("binding retirement changed after its quote");
        transaction.retired_points.push_back(point);
    }
    if (source && !own_snapshot && transaction.borrow_state && !candidate.retired_points.empty() &&
        std::find(candidate.private_points.begin(), candidate.private_points.end(), *source) !=
            candidate.private_points.end()) {
        const auto actual =
            inspect_source(base, source, candidate.consume_source, candidate.private_points);
        if (!actual) throw std::logic_error("accepted binding lost its source");
        select_source(*actual);
        result.consumed_source = transaction.consume_source ? source : std::nullopt;
    }
    if (source) transaction.source_lease = checkpoints.lease(*source);
    context_transaction_.emplace(std::move(transaction));
    auto& tx = *context_transaction_;
    try {
        for (auto point : tx.carried_points) {
            if (point != source) tx.carried_pins.push_back(checkpoints.lease(point));
            if (!tx.take_private)
                if (const auto clone = reserve_checkpoint())
                    tx.carried_clones.emplace_back(point, *clone);
        }
        tx.text_binding.emplace(text_kv_histories->prepare_binding(
            main_source, tx.reuse_frontier, tx.reservation_demand.main_frontier, lane,
            tx.consume_source));
        if (backend_kv_histories)
            tx.backend_binding.emplace(backend_kv_histories->prepare_binding(
                backend_source, tx.backend_frontier, tx.reservation_demand.backend_frontier, lane,
                tx.consume_source));
        const auto append_restore = [&](context::KVBinding& binding,
                                        context::LogicalKVPageStore& pages,
                                        runtime::ContextResourceClass resource) {
            if (binding.restores().empty()) return;
            KVTransfer transfer;
            transfer.pages    = &pages;
            transfer.resource = resource;
            transfer.restore  = true;
            transfer.logical.assign(binding.restores().begin(), binding.restores().end());
            transfer.physical.assign(binding.restore_destinations().begin(),
                                     binding.restore_destinations().end());
            tx.kv_transfers.push_back(std::move(transfer));
        };
        append_restore(*tx.text_binding, *text_kv_pages, runtime::ContextResourceClass::MainKV);
        if (tx.backend_binding)
            append_restore(*tx.backend_binding, *backend_kv_pages,
                           runtime::ContextResourceClass::BackendKV);
        const bool host_state =
            source && !state_store->device_resident(checkpoint(*source).continuation().image);
        tx.reserved_state =
            tx.borrow_state                ? std::optional(checkpoint(*source).continuation().image)
            : host_state || tx.split_state ? state_store->reserve_logical_destination()
                                           : state_store->reserve_destination();
        if (!tx.reserved_state) throw std::logic_error("binding lost its reserved State slot");
        context_source_ready_.record(device.stream());
        context_source_ready_.wait(device.transfer_stream());
        if (tx.backup_state) {
            auto backup =
                state_store->reserve_device_to_host(checkpoint(*source).continuation().image);
            if (!backup) throw std::logic_error("source backup lost its Host reservation");
            tx.state_transfer.emplace(std::move(*backup));
            enqueue_state_backup(tx);
        }
        if (host_state) {
            tx.transfers.push_back(context::state_transfer_requirement(
                state_images->storage().layout().transfer_work(),
                runtime::ContextTransferDirection::HostToDevice));
            start_context_transfer_timer(runtime::ContextResourceClass::State);
            auto transfer =
                tx.borrow_state
                    ? state_store->begin_host_to_device(*tx.reserved_state, device.transfer_view())
                    : state_store->begin_host_to_device(checkpoint(*source).continuation().image,
                                                        *tx.reserved_state, device.transfer_view());
            if (!transfer) throw std::logic_error("Host State restore reservation failed");
            tx.state_transfer.emplace(std::move(*transfer));
            stop_context_transfer_timer(runtime::ContextResourceClass::State);
            ++tx.operations.state_restores;
        }
        enqueue_context_transfers(tx);
        request.lifecycle = Lifecycle::Binding;
        return result;
    } catch (...) {
        abort_context();
        throw;
    }
}

void ProgramImpl::install_binding(ContextTransaction& tx) {
    auto& state       = sequences[tx.lane];
    auto& request     = requests[tx.lane];
    const auto image  = state.continuation.image;
    const auto writer = state.pending_write;
    auto kv           = std::move(state.kv);
    if (tx.resume) {
        auto& saved              = *tx.resume->impl_;
        request                  = std::move(saved.control);
        state                    = std::move(saved.sequence);
        state.kv                 = std::move(kv);
        state.continuation.image = image;
        state.pending_write      = writer;
        state.lane               = tx.lane;
        state.mtp_draft_count    = 0;
        const auto rope_delta    = state.continuation.metadata.rope_delta;
        state.continuation.metadata =
            tx.source ? checkpoint(*tx.source).continuation().metadata : ContinuationMetadata{};
        state.continuation.metadata.rope_delta = rope_delta;
        if (!tx.resume_snapshot) {
            request.resume_lifecycle                            = request.lifecycle;
            request.lifecycle                                   = Lifecycle::Replaying;
            request.replay_target                               = saved.frontier;
            request.replay_cursor                               = tx.reuse_frontier;
            state.continuation.metadata.execution_frontier      = tx.reuse_frontier;
            state.continuation.metadata.text_kv_valid           = tx.reuse_frontier;
            state.continuation.metadata.backend_kv_valid        = tx.backend_frontier;
            state.continuation.metadata.dflash_context_frontier = tx.reuse_frontier;
            initialize_captures(tx.lane, tx.reuse_frontier, request.replay_target);
        }
        // Reconstruct references into the moved durable prefill object.
        if (request.prefill && request.prefill->vision_plan &&
            !request.prefill->vision_plan->uses.empty()) {
            request.prefill->vision = std::make_unique<qwen::VisionPrefillSession>(
                device, *parameters.model.config().vision, *parameters.vision,
                workspace_storage.span(), *workspace_plan.vision, request.prefill->prompt,
                *request.prefill->vision_plan, vision_handoff, vision_handoff_peak_bytes);
        }
        install_resume_sampling(state, request);
    } else {
        state.continuation.metadata =
            tx.source ? checkpoint(*tx.source).continuation().metadata : ContinuationMetadata{};
        state.ledger = tx.base->prompt->token_ids;
        state.prefix_identity.assign(*tx.base->prompt);
        state.prefix_digests                                = tx.base->prefix_digests;
        state.continuation.metadata.rope_delta              = tx.base->prompt->rope_delta;
        state.continuation.metadata.text_kv_valid           = tx.reuse_frontier;
        state.continuation.metadata.backend_kv_valid        = tx.backend_frontier;
        state.continuation.metadata.dflash_context_frontier = tx.reuse_frontier;
        state.continuation.metadata.tail_hidden_valid       = tx.source_tail_hidden;
        request.lifecycle                                   = Lifecycle::Prefilling;
        request.publish_continuation                        = tx.base->summary.publish_continuation;
        install_sampling(state, request, tx.base->sampling);
    }
    request.permit   = tx.first_unit;
    request.recovery = tx.recovery;
}

void ProgramImpl::prepare_binding(ContextTransaction& tx) {
    const auto prepare = [&](context::KVBinding& binding, context::LogicalKVPageStore& pages,
                             runtime::ContextResourceClass resource) {
        binding.prepare_tail_copy();
        if (binding.needs_tail_copy())
            copy_context_tail(tx, pages, binding.tail_source(), binding.tail_destination(),
                              resource);
    };
    prepare(*tx.text_binding, *text_kv_pages, runtime::ContextResourceClass::MainKV);
    if (tx.backend_binding)
        prepare(*tx.backend_binding, *backend_kv_pages, runtime::ContextResourceClass::BackendKV);
    auto& state = sequences[tx.lane];
    state.continuation.image =
        tx.split_state ? checkpoint(*tx.source).continuation().image : *tx.reserved_state;
    state.pending_write = {};
    if (tx.source && !tx.borrow_state && !tx.split_state &&
        state_store->device_resident(checkpoint(*tx.source).continuation().image)) {
        const auto from      = checkpoint(*tx.source).continuation().image;
        const auto selectors = state_store->begin_write(from, *tx.reserved_state);
        ++tx.operations.state_forks;
        state.continuation.image = from;
        state.pending_write      = *tx.reserved_state;
        if (is_masked_draft_backend(speculative_backend))
            copy_local_for_context(tx, selectors.source, selectors.destination);
    } else if (tx.source && !tx.borrow_state && !tx.split_state) {
        state_store->complete_initialization(*tx.reserved_state);
    } else if (!tx.source) {
        state_images->storage().clear(state_store->physical_slot(*tx.reserved_state),
                                      device.transfer_view());
        state_store->complete_initialization(*tx.reserved_state);
    }
    tx.submitted = !tx.source || !tx.transfers.empty();
    if (tx.submitted) context_completion_.record(device.transfer_stream());
}

void ProgramImpl::complete_binding(ContextTransaction& tx, ContextProgress& out) {
    install_binding(tx);
    if (tx.split_state) {
        state_store->split_device_replica_identity(checkpoint(*tx.source).continuation().image,
                                                   *tx.reserved_state);
        state_store->complete_initialization(*tx.reserved_state);
        sequences[tx.lane].continuation.image = *tx.reserved_state;
        sequences[tx.lane].pending_write      = {};
        ++tx.operations.state_moves;
    }
    tx.adopted = true;
    if (tx.consume_source) {
        const auto old = *tx.source;
        if (tx.borrow_state) {
            (void)checkpoint(old).snapshot.take_mutable_image();
            ++tx.operations.state_moves;
        }
        tx.source_lease.reset();
        if (!checkpoints.erase(old))
            throw std::logic_error("consumed checkpoint still has readers");
        tx.retired_points.push_back(old);
        tx.source.reset();
        if (tx.resume) tx.resume->impl_->snapshot.reset();
        tx.borrow_state = false;
    }
    SequenceKVBundle kv{.text = tx.text_binding->commit(device.transfer_stream())};
    if (tx.backend_binding)
        kv.backend.emplace(tx.backend_binding->commit(device.transfer_stream()));
    // Table publication belongs to the same transfer stream as the binding copies.
    context_completion_.record(device.transfer_stream());
    context_completion_.wait(device.stream());
    sequences[tx.lane].kv.emplace(std::move(kv));
    for (const auto& [from, to] : tx.carried_clones) checkpoints.publish(to, checkpoint(from));
    if (!tx.take_private) {
        tx.carried_points.clear();
        for (const auto& [from, to] : tx.carried_clones) tx.carried_points.push_back(to);
    }
    tx.reserved_state.reset();
    out.private_points      = std::move(tx.carried_points);
    out.retired_checkpoints = std::move(tx.retired_points);
    out.sequence            = sequence_handle(tx.lane);
    out.replaying           = requests[tx.lane].lifecycle == Lifecycle::Replaying;
}

} // namespace ninfer::models::qwen3_5::detail
