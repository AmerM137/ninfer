#include "models/qwen3_5/program/program_impl.h"
#include "models/common/context/work.h"
#include "core/device.h"

#include <array>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {

std::optional<CheckpointHandle> ProgramImpl::detach_checkpoint(SequenceState& state) {
    const auto frontier = state.continuation.metadata.text_kv_valid;
    if (!frontier || !state.kv) return std::nullopt;
    auto backing     = std::make_shared<PreparedCaptureBacking>();
    backing->digests = state.prefix_digests;
    backing->digests.truncate(frontier);
    backing->ledger.assign(state.ledger.begin(), state.ledger.begin() + frontier);
    backing->prefix_identity = state.prefix_identity;
    backing->prefix_identity.truncate(frontier);
    auto handle = reserve_checkpoint();
    if (!handle) return std::nullopt;
    if (state_store->role(state.continuation.image) == context::StateImageRole::ActiveMutable)
        state_store->freeze(state.continuation.image);
    auto captured                        = state.continuation;
    captured.metadata.execution_frontier = frontier;
    captured.metadata.backend_kv_valid   = backend_kv_valid(state);
    std::vector<context::KVPrefixRef> prefixes;
    prefixes.push_back(text_kv_histories->retain_prefix(state.kv->text.handle(), frontier));
    if (state.kv->backend)
        prefixes.push_back(backend_kv_histories->retain_prefix(state.kv->backend->handle(),
                                                               backend_kv_valid(state)));
    CheckpointState snapshot{
        .snapshot = Snapshot(*state_store, captured, std::move(prefixes), std::move(backing)),
        .key      = {state.prefix_digests.at(frontier), frontier,
                     requests[state.lane].base->prefix_identity_tag}};
    // The complete snapshot now owns the base before an outstanding writer is discarded.
    if (state.pending_write.valid()) {
        state_store->abort_write(state.continuation.image, state.write_state());
        if (!state_store->release(state.write_state()))
            throw std::logic_error("unused writer could not be released");
    }
    checkpoints.publish(*handle, std::move(snapshot));
    state.kv.reset();
    state.continuation.image = {};
    state.pending_write      = {};
    return handle;
}

bool ProgramImpl::prepare_backup(ContextTransaction& tx, CheckpointState& record,
                                 bool shared_device) {
    if (!host_context_arena) { return false; }
    // Both copies already present cost no transfer. A source still read by another resident stays
    // resident; the paused record is an evictable reference, never a permanent Device pin.
    tx.preserve_state_device =
        shared_device && state_store->source_pins(record.continuation().image) != 0;
    if (!state_store->host_resident(record.continuation().image)) {
        auto transfer = state_store->reserve_device_to_host(record.continuation().image);
        if (transfer) { tx.state_transfer.emplace(std::move(*transfer)); }
        if (!tx.state_transfer) { return false; }
    }
    const auto collect = [&](context::KVHistoryStore& histories, context::LogicalKVPageStore& pages,
                             context::KVHistoryHandle address,
                             runtime::ContextResourceClass resource) {
        KVTransfer transfer;
        transfer.pages       = &pages;
        transfer.resource    = resource;
        transfer.drop_device = true;
        for (std::uint32_t i = 0; i < histories.mapped_pages(address); ++i) {
            const auto page = histories.logical_page(address, i);
            if (!pages.device_resident(page) ||
                (shared_device && pages.active_history_references(page))) {
                continue;
            }
            if (pages.host_replica_current(page)) {
                if (!pages.drop_device_replica(page)) { return false; }
            } else {
                transfer.logical.push_back(page);
            }
        }
        if (!transfer.logical.empty()) {
            auto destination =
                host_kv_extents ? host_kv_extents->prepare(pages, transfer.logical) : std::nullopt;
            if (destination) { transfer.host_destination.emplace(std::move(*destination)); }
            if (!transfer.host_destination) { return false; }
            transfer.physical = host_kv_extents->device_sources(*transfer.host_destination);
        }
        tx.kv_transfers.push_back(std::move(transfer));
        return true;
    };
    if (!collect(*text_kv_histories, *text_kv_pages, record.history().text,
                 runtime::ContextResourceClass::MainKV)) {
        return false;
    }
    return !record.history().backend ||
           collect(*backend_kv_histories, *backend_kv_pages, *record.history().backend,
                   runtime::ContextResourceClass::BackendKV);
}

bool ProgramImpl::start_pause(SequenceHandle handle, bool save_snapshot,
                              runtime::ExecutionTiming* observation) {
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, observation);
    if (context_transaction_ || pending_transaction_) { return false; }
    if (!valid_sequence(handle)) { throw std::logic_error("pause has a stale sequence"); }
    const auto lane = ContractAccess::lane(handle).value;
    auto& state     = sequences[lane];
    auto& control   = requests[lane];
    if (control.lifecycle != Lifecycle::Active && control.lifecycle != Lifecycle::Prefilling &&
        control.lifecycle != Lifecycle::Replaying) {
        throw std::logic_error("pause requires a committed boundary");
    }
    timing.begin_wait();
    device.synchronize();
    timing.end_wait();
    control.capture_reservation.reset();
    if (control.capture_pending) { skip_capture(handle); }
    control.recovery.reset();
    if (control.permit) { settle_unit(lane); }
    if (control.lifecycle == Lifecycle::Replaying) { save_snapshot = false; }
    if (save_snapshot && is_masked_draft_backend(speculative_backend) &&
        state.continuation.metadata.dflash_context_frontier <
            state.continuation.metadata.execution_frontier) {
        const std::array<ExecutionUnit, 1> units{{handle, ExecutionUnitKind::Normalize, 0}};
        const auto reservation = reserve_units(units);
        if (!reservation) {
            save_snapshot = false;
        } else {
            const std::array<std::uint32_t, 1> lanes{lane};
            const std::array<std::uint32_t, 1> begin{
                state.continuation.metadata.dflash_context_frontier};
            const std::array<std::uint32_t, 1> count{
                state.continuation.metadata.execution_frontier -
                state.continuation.metadata.dflash_context_frontier};
            ensure_sequence_kv_mapped(
                state, state.continuation.metadata.execution_frontier,
                backend_kv_cache() ? state.continuation.metadata.execution_frontier : 0);
            enqueue_dflash_context_append(lanes, begin, count);
            timing.begin_wait();
            device.synchronize();
            timing.end_wait();
            state.continuation.metadata.dflash_context_frontier =
                state.continuation.metadata.execution_frontier;
            commit_sequence_kv(state, state.continuation.metadata.text_kv_valid,
                               backend_kv_valid(state));
            settle_unit(lane);
        }
    }
    control.capture_reservation.reset();
    if (control.prefill && control.prefill->vision) {
        control.prefill->retired_vision_seconds += control.prefill->vision->elapsed_seconds();
        control.prefill->vision.reset();
    }
    if (control.replay) { control.replay->vision.reset(); }
    auto saved      = std::make_unique<ResumeStateImpl>();
    saved->owner    = this;
    saved->frontier = control.lifecycle == Lifecycle::Prefilling ? control.prefill->cursor
                      : control.lifecycle == Lifecycle::Replaying
                          ? control.replay_target
                          : state.continuation.metadata.execution_frontier;
    if (control.lifecycle == Lifecycle::Replaying) {
        control.lifecycle = control.resume_lifecycle;
        control.replay.reset();
    }
    if (save_snapshot && control.base->vision_control_plan) {
        for (const auto& item : control.base->vision_control_plan->items) {
            if (item.token_begin < saved->frontier && saved->frontier < item.token_end) {
                save_snapshot = false;
                break;
            }
        }
    }
    if (save_snapshot && saved->frontier) { saved->snapshot = detach_checkpoint(state); }
    if (!saved->snapshot) {
        release_sequence_kv(state);
        release_sequence_state(state);
    }
    state.mtp_draft_count = 0;
    saved->sequence       = std::move(state);
    saved->control        = std::move(control);
    state                 = {};
    state.lane            = lane;
    control               = {};
    control.lifecycle     = Lifecycle::Pausing;
    ContextTransaction tx;
    tx.kind  = ContextOperationKind::Pause;
    tx.lane  = lane;
    tx.epoch = lane_epochs[lane];
    tx.paused.emplace(ResumeState(std::move(saved)));
    if (tx.paused->has_snapshot()) {
        auto& snapshot = checkpoint(*tx.paused->impl_->snapshot);
        if (!prepare_backup(tx, snapshot, true)) {
            tx.state_transfer.reset();
            tx.kv_transfers.clear();
            if (!revoke_snapshot(*tx.paused)) {
                throw std::logic_error("failed pause snapshot could not be released");
            }
        }
    }
    context_transaction_.emplace(std::move(tx));
    auto& active = *context_transaction_;
    if (active.state_transfer) { enqueue_state_backup(active); }
    enqueue_context_transfers(active);
    return true;
}

ResumeState ProgramImpl::complete_pause(ContextTransaction& tx) {
    if (!tx.paused) { throw std::logic_error("pause operation lost its durable request"); }
    if (tx.paused->has_snapshot()) {
        const auto& record = checkpoint(*tx.paused->impl_->snapshot);
        if (!tx.preserve_state_device &&
            state_store->device_resident(record.continuation().image) &&
            !state_store->drop_device_replica(record.continuation().image)) {
            throw std::logic_error("paused state is still Device-pinned");
        }
    }
    requests[tx.lane]       = {};
    sequences[tx.lane]      = {};
    sequences[tx.lane].lane = tx.lane;
    invalidate_lane(tx.lane);
    return std::move(*tx.paused);
}
} // namespace ninfer::models::qwen3_5::detail
