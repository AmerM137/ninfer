#include "models/common/context/state_store.h"

#include <limits>
#include <stdexcept>

namespace ninfer::models::context {

StateImageStore::StateImageStore(StateImageStorage& device, HostContextArena* host,
                                 std::uint32_t logical_capacity)
    : device_(&device), host_(host), objects_(logical_capacity), free_objects_(logical_capacity),
      free_device_slots_(static_cast<std::size_t>(device.slot_count())),
      free_object_count_(logical_capacity),
      free_device_count_(static_cast<std::uint32_t>(device.slot_count())) {
    if (logical_capacity == 0 || device.slot_count() <= 0 ||
        logical_capacity < static_cast<std::uint32_t>(device.slot_count()) ||
        (host != nullptr && logical_capacity < static_cast<std::uint32_t>(device.slot_count()) +
                                                   host->capacity_bytes() / device.image_bytes())) {
        throw std::invalid_argument("StateImageStore capacity is inconsistent");
    }
    for (std::uint32_t index = 0; index < logical_capacity; ++index) {
        free_objects_[index] = logical_capacity - 1U - index;
    }
    for (std::uint32_t index = 0; index < free_device_slots_.size(); ++index) {
        free_device_slots_[index] =
            static_cast<std::int32_t>(free_device_slots_.size() - 1U - index);
    }
}

std::uint32_t StateImageStore::capacity() const noexcept {
    return static_cast<std::uint32_t>(objects_.size());
}

std::uint32_t StateImageStore::occupied() const noexcept { return capacity() - free_object_count_; }

std::uint32_t StateImageStore::device_occupied() const noexcept {
    return static_cast<std::uint32_t>(free_device_slots_.size()) - free_device_count_;
}

std::uint32_t StateImageStore::device_capacity() const noexcept {
    return static_cast<std::uint32_t>(free_device_slots_.size());
}

std::uint32_t StateImageStore::host_occupied() const noexcept {
    std::uint32_t count = 0;
    for (const auto& object : objects_) {
        count += object.host_slot.has_value() + object.pending_host_slot.has_value();
    }
    return count;
}

std::optional<StateImageStore::Handle> StateImageStore::reserve_destination() noexcept {
    return allocate(StateImageRole::ReservedDestination, true);
}

std::optional<StateImageStore::Handle> StateImageStore::reserve_logical_destination() noexcept {
    return allocate(StateImageRole::ReservedDestination, false);
}

void StateImageStore::complete_initialization(Handle handle) {
    Object& object = require(handle);
    if (object.role != StateImageRole::ReservedDestination || !object.device_slot ||
        object.source_pins != 0 || object.destination_pinned || has_pending_replica(object)) {
        throw std::logic_error("StateImage initialization has no exclusive reservation");
    }
    if (object.content_epoch == 0) { object.content_epoch = next_epoch(); }
    object.role = StateImageRole::ActiveMutable;
}

bool StateImageStore::valid(Handle handle) const noexcept {
    return handle.owner_ == this && handle.index_ < objects_.size() &&
           objects_[handle.index_].role != StateImageRole::Free &&
           objects_[handle.index_].generation == handle.generation_;
}

std::uint32_t StateImageStore::descriptor_index(Handle handle) const {
    (void)require(handle);
    return handle.index_;
}

StateImageRole StateImageStore::role(Handle handle) const { return require(handle).role; }

StateReplicaResidency StateImageStore::residency(Handle handle) const {
    const Object& object = require(handle);
    if (object.device_slot && object.host_slot) { return StateReplicaResidency::Both; }
    if (object.device_slot) { return StateReplicaResidency::DeviceOnly; }
    if (object.host_slot) { return StateReplicaResidency::HostOnly; }
    return StateReplicaResidency::None;
}

bool StateImageStore::device_resident(Handle handle) const {
    return require(handle).device_slot.has_value();
}

bool StateImageStore::host_resident(Handle handle) const {
    return require(handle).host_slot.has_value();
}

std::uint64_t StateImageStore::content_epoch(Handle handle) const {
    return require(handle).content_epoch;
}

std::uint32_t StateImageStore::source_pins(Handle handle) const {
    return require(handle).source_pins;
}

std::uint32_t StateImageStore::checkpoint_references(Handle handle) const {
    return require(handle).checkpoint_references;
}

void StateImageStore::retain_checkpoint_reference(Handle handle) {
    Object& object = require(handle);
    if (object.role != StateImageRole::CheckpointImmutable ||
        object.checkpoint_references == std::numeric_limits<std::uint32_t>::max()) {
        throw std::logic_error("StateImage checkpoint reference is not retainable");
    }
    ++object.checkpoint_references;
    object.release_pending = false;
}

bool StateImageStore::can_release_checkpoint_owner(Handle handle) const noexcept {
    if (!valid(handle)) { return false; }
    const Object& object = objects_[handle.index_];
    return object.role == StateImageRole::CheckpointImmutable && object.checkpoint_references != 0;
}

bool StateImageStore::release_checkpoint_owner(Handle handle) noexcept {
    if (!can_release_checkpoint_owner(handle)) { return false; }
    Object& object = objects_[handle.index_];
    if (--object.checkpoint_references == 0) {
        object.release_pending = true;
        retire_released_owner(handle);
    }
    return true;
}

void StateImageStore::release_checkpoint_reference(Handle handle) {
    Object& object = require(handle);
    if (object.role != StateImageRole::CheckpointImmutable || object.checkpoint_references == 0) {
        throw std::logic_error("StateImage checkpoint reference is not releasable");
    }
    --object.checkpoint_references;
}

std::int32_t StateImageStore::physical_slot(Handle handle) const {
    const Object& object = require(handle);
    if (!object.device_slot) { throw std::logic_error("StateImage has no Device storage"); }
    return *object.device_slot;
}

void StateImageStore::make_mutable(Handle handle) {
    Object& object = require(handle);
    if (object.role != StateImageRole::CheckpointImmutable || !object.device_slot ||
        object.checkpoint_references != 0 || object.source_pins != 0 || object.destination_pinned ||
        has_pending_replica(object)) {
        throw std::logic_error("StateImage checkpoint is not movable");
    }
    object.host_slot.reset();
    object.role = StateImageRole::ActiveMutable;
}

void StateImageStore::consume_checkpoint(Handle handle) {
    Object& object = require(handle);
    if (object.role != StateImageRole::CheckpointImmutable || !object.device_slot ||
        object.checkpoint_references != 1 || object.source_pins != 0 || object.destination_pinned ||
        has_pending_replica(object)) {
        throw std::logic_error("StateImage checkpoint has other owners or readers");
    }
    object.checkpoint_references = 0;
    object.release_pending       = false;
    object.host_slot.reset();
    object.role = StateImageRole::ActiveMutable;
}

bool StateImageStore::drop_device_replica(Handle handle) noexcept {
    if (!valid(handle)) { return false; }
    Object& object = objects_[handle.index_];
    if (object.role != StateImageRole::CheckpointImmutable || !object.device_slot ||
        !object.host_slot || object.source_pins != 0 || object.destination_pinned ||
        has_pending_replica(object)) {
        return false;
    }
    return_device_slot(*object.device_slot);
    object.device_slot.reset();
    return true;
}

bool StateImageStore::drop_host_replica(Handle handle) noexcept {
    if (!valid(handle)) { return false; }
    Object& object = objects_[handle.index_];
    if (host_ == nullptr || object.role != StateImageRole::CheckpointImmutable ||
        !object.device_slot || !object.host_slot || object.source_pins != 0 ||
        object.destination_pinned || has_pending_replica(object)) {
        return false;
    }
    object.host_slot.reset();
    return true;
}

void StateImageStore::split_device_replica_identity(Handle source, Handle destination) {
    Object& source_object      = require(source);
    Object& destination_object = require(destination);
    if (source == destination || source_object.role != StateImageRole::CheckpointImmutable ||
        !source_object.device_slot || !source_object.host_slot || source_object.source_pins != 0 ||
        source_object.destination_pinned || has_pending_replica(source_object) ||
        destination_object.role != StateImageRole::ReservedDestination ||
        destination_object.device_slot || destination_object.host_slot ||
        has_pending_replica(destination_object) || destination_object.source_pins != 0 ||
        destination_object.destination_pinned) {
        throw std::logic_error("StateImage replica identity split is invalid");
    }
    destination_object.device_slot   = source_object.device_slot;
    destination_object.content_epoch = source_object.content_epoch;
    source_object.device_slot.reset();
}

void StateImageStore::freeze(Handle handle) {
    Object& object = require(handle);
    if (object.role != StateImageRole::ActiveMutable || !object.device_slot ||
        object.source_pins != 0 || object.destination_pinned || has_pending_replica(object)) {
        throw std::logic_error("StateImage active image is not freezable");
    }
    object.role = StateImageRole::CheckpointImmutable;
}

StateImageSelectors StateImageStore::begin_write(Handle source, Handle destination) {
    Object& source_object      = require(source);
    Object& destination_object = require(destination);
    if (source == destination || source_object.role != StateImageRole::CheckpointImmutable ||
        !source_object.device_slot ||
        source_object.source_pins == std::numeric_limits<std::uint32_t>::max() ||
        destination_object.role != StateImageRole::ReservedDestination ||
        !destination_object.device_slot || destination_object.source_pins != 0 ||
        destination_object.destination_pinned || has_pending_replica(destination_object)) {
        throw std::logic_error("StateImage fork binding is invalid");
    }
    ++source_object.source_pins;
    destination_object.destination_pinned = true;
    destination_object.role               = StateImageRole::Writing;
    destination_object.write_source       = source;
    destination_object.content_epoch      = source_object.content_epoch;
    return {.source = *source_object.device_slot, .destination = *destination_object.device_slot};
}

void StateImageStore::complete_write(Handle source, Handle destination) {
    if (source == destination) {
        (void)selectors(source, destination);
        require(destination).content_epoch = next_epoch();
        return;
    }
    Object& source_object      = require(source);
    Object& destination_object = require(destination);
    if (source_object.role != StateImageRole::CheckpointImmutable ||
        source_object.source_pins == 0 || destination_object.role != StateImageRole::Writing ||
        !destination_object.device_slot || !destination_object.destination_pinned ||
        destination_object.write_source != source) {
        throw std::logic_error("StateImage fork commit is invalid");
    }
    --source_object.source_pins;
    destination_object.destination_pinned = false;
    destination_object.write_source       = {};
    destination_object.role               = StateImageRole::ActiveMutable;
    destination_object.content_epoch      = next_epoch();
    retire_released_owner(source);
}

void StateImageStore::abort_write(Handle source, Handle destination) {
    if (!can_abort_write(source, destination)) {
        throw std::logic_error("StateImage fork abort is invalid");
    }
    Object& source_object      = require(source);
    Object& destination_object = require(destination);
    --source_object.source_pins;
    destination_object.destination_pinned = false;
    destination_object.write_source       = {};
    destination_object.role               = StateImageRole::ReservedDestination;
    destination_object.content_epoch      = 0;
    retire_released_owner(source);
}

bool StateImageStore::can_abort_write(Handle source, Handle destination) const noexcept {
    if (!valid(source) || !valid(destination)) { return false; }
    const Object& source_object      = objects_[source.index_];
    const Object& destination_object = objects_[destination.index_];
    return source != destination && source_object.role == StateImageRole::CheckpointImmutable &&
           source_object.source_pins != 0 && destination_object.role == StateImageRole::Writing &&
           destination_object.device_slot.has_value() && destination_object.destination_pinned &&
           destination_object.write_source == source;
}

StateImageSelectors StateImageStore::selectors(Handle source, Handle destination) const {
    const Object& source_object      = require(source);
    const Object& destination_object = require(destination);
    const bool inplace               = source == destination;
    if (!source_object.device_slot || !destination_object.device_slot ||
        (inplace && (source_object.role != StateImageRole::ActiveMutable ||
                     source_object.source_pins != 0 || source_object.destination_pinned)) ||
        (!inplace &&
         (source_object.role != StateImageRole::CheckpointImmutable ||
          source_object.source_pins == 0 || destination_object.role != StateImageRole::Writing ||
          !destination_object.destination_pinned || destination_object.write_source != source))) {
        throw std::logic_error("StateImage execution binding is invalid");
    }
    return {.source = *source_object.device_slot, .destination = *destination_object.device_slot};
}

std::optional<StateImageStore::Transfer>
StateImageStore::reserve_device_to_host(Handle source, HostContextAllocation reserved) {
    Object& object = require(source);
    if (host_ == nullptr || object.role != StateImageRole::CheckpointImmutable ||
        !object.device_slot || object.host_slot || has_pending_replica(object) ||
        object.source_pins == std::numeric_limits<std::uint32_t>::max()) {
        return std::nullopt;
    }
    if (!reserved.valid()) {
        auto target = host_->allocate(device_->image_bytes());
        if (!target) { return std::nullopt; }
        reserved = std::move(*target);
    }
    if (!reserved.reserved() || reserved.bytes() != device_->image_bytes()) {
        throw std::invalid_argument("StateImage Host reservation has incompatible geometry");
    }
    const std::uint64_t transfer = next_transfer();
    object.pending_host_slot.emplace(std::move(reserved));
    object.transfer_id        = transfer;
    object.destination_pinned = true;
    ++object.source_pins;
    return Transfer(*this, transfer, StateTransferDirection::DeviceToHost, source, source);
}

void StateImageStore::enqueue_device_to_host(const Transfer& transfer,
                                             CudaExecutionView execution) {
    validate_transfer(transfer);
    Object& object = require(transfer.source_);
    if (transfer.direction_ != StateTransferDirection::DeviceToHost || host_ == nullptr ||
        !object.device_slot || !object.pending_host_slot) {
        throw std::logic_error("StateImage Device-to-Host transfer is not enqueueable");
    }
    device_->copy_to_host(*object.device_slot, object.pending_host_slot->span(), execution);
}

std::optional<StateImageStore::Transfer>
StateImageStore::begin_device_to_host(Handle source, CudaExecutionView execution) {
    std::optional<Transfer> transfer = reserve_device_to_host(source);
    if (!transfer) { return std::nullopt; }
    try {
        enqueue_device_to_host(*transfer, execution);
    } catch (...) {
        abort_transfer(std::move(*transfer));
        throw;
    }
    return transfer;
}

std::optional<StateImageStore::Transfer>
StateImageStore::begin_host_to_device(Handle source, CudaExecutionView execution) {
    Object& object = require(source);
    if (host_ == nullptr || object.role != StateImageRole::CheckpointImmutable ||
        !object.host_slot || object.device_slot || has_pending_replica(object) ||
        object.source_pins == std::numeric_limits<std::uint32_t>::max()) {
        return std::nullopt;
    }
    const std::optional<std::int32_t> target = take_device_slot();
    if (!target) { return std::nullopt; }
    const std::uint64_t transfer = next_transfer();
    object.pending_device_slot   = *target;
    object.transfer_id           = transfer;
    object.destination_pinned    = true;
    ++object.source_pins;
    try {
        device_->copy_from_host(object.host_slot->span(), *target, execution);
    } catch (...) {
        --object.source_pins;
        object.destination_pinned = false;
        object.transfer_id        = 0;
        object.pending_device_slot.reset();
        return_device_slot(*target);
        throw;
    }
    return Transfer(*this, transfer, StateTransferDirection::HostToDevice, source, source);
}

std::optional<StateImageStore::Transfer>
StateImageStore::begin_host_to_device(Handle source, Handle destination,
                                      CudaExecutionView execution) {
    Object& source_object      = require(source);
    Object& destination_object = require(destination);
    if (host_ == nullptr || source == destination ||
        source_object.role != StateImageRole::CheckpointImmutable || !source_object.host_slot ||
        source_object.source_pins == std::numeric_limits<std::uint32_t>::max() ||
        destination_object.role != StateImageRole::ReservedDestination ||
        destination_object.device_slot || destination_object.host_slot ||
        has_pending_replica(destination_object) || destination_object.source_pins != 0 ||
        destination_object.destination_pinned) {
        return std::nullopt;
    }
    const std::optional<std::int32_t> target = take_device_slot();
    if (!target) { return std::nullopt; }
    const std::uint64_t transfer           = next_transfer();
    destination_object.pending_device_slot = *target;
    destination_object.transfer_id         = transfer;
    destination_object.destination_pinned  = true;
    destination_object.content_epoch       = source_object.content_epoch;
    ++source_object.source_pins;
    try {
        device_->copy_from_host(source_object.host_slot->span(), *target, execution);
    } catch (...) {
        --source_object.source_pins;
        destination_object.pending_device_slot.reset();
        destination_object.transfer_id        = 0;
        destination_object.destination_pinned = false;
        destination_object.role               = StateImageRole::ReservedDestination;
        destination_object.content_epoch      = 0;
        return_device_slot(*target);
        throw;
    }
    return Transfer(*this, transfer, StateTransferDirection::HostToDevice, source, destination);
}

void StateImageStore::publish_transfer(Transfer&& transfer, bool keep_source_replica) {
    validate_transfer(transfer);
    const Handle source_handle = transfer.source_;
    Object& source             = require(transfer.source_);
    Object& destination        = require(transfer.destination_);
    if (transfer.direction_ == StateTransferDirection::DeviceToHost) {
        if (!source.pending_host_slot) {
            throw std::logic_error("StateImage Host destination publication failed");
        }
        if (!keep_source_replica && source.source_pins != 1) {
            throw std::logic_error("StateImage Device replica still has readers");
        }
        source.pending_host_slot->publish();
        source.host_slot = std::move(source.pending_host_slot);
        source.pending_host_slot.reset();
        if (!keep_source_replica) {
            return_device_slot(*source.device_slot);
            source.device_slot.reset();
        }
    } else {
        if (!keep_source_replica && transfer.source_ == transfer.destination_ &&
            source.source_pins != 1) {
            throw std::logic_error("StateImage Host replica still has readers");
        }
        destination.device_slot = destination.pending_device_slot;
        destination.pending_device_slot.reset();
        if (!keep_source_replica && transfer.source_ == transfer.destination_) {
            source.host_slot.reset();
        }
    }
    --source.source_pins;
    destination.destination_pinned = false;
    destination.transfer_id        = 0;
    consume(transfer);
    retire_released_owner(source_handle);
}

void StateImageStore::abort_transfer(Transfer&& transfer) noexcept {
    if (!valid_transfer(transfer)) {
        consume(transfer);
        return;
    }
    const Handle source_handle = transfer.source_;
    Object& source             = objects_[transfer.source_.index_];
    Object& destination        = objects_[transfer.destination_.index_];
    if (source.source_pins != 0) { --source.source_pins; }
    if (transfer.direction_ == StateTransferDirection::DeviceToHost) {
        source.pending_host_slot.reset();
        source.destination_pinned = false;
        source.transfer_id        = 0;
    } else {
        if (destination.pending_device_slot) {
            return_device_slot(*destination.pending_device_slot);
        }
        destination.pending_device_slot.reset();
        destination.destination_pinned = false;
        destination.transfer_id        = 0;
        if (transfer.source_ != transfer.destination_) {
            destination.role          = StateImageRole::ReservedDestination;
            destination.content_epoch = 0;
        }
    }
    consume(transfer);
    retire_released_owner(source_handle);
}

bool StateImageStore::release(Handle handle) noexcept {
    if (!can_release(handle)) { return false; }
    Object& object = objects_[handle.index_];
    object.host_slot.reset();
    if (object.device_slot) {
        return_device_slot(*object.device_slot);
        object.device_slot.reset();
    }
    object.role          = StateImageRole::Free;
    object.content_epoch = 0;
    if (++object.generation == 0) { ++object.generation; }
    free_objects_[free_object_count_++] = handle.index_;
    return true;
}

bool StateImageStore::can_release(Handle handle) const noexcept {
    if (!valid(handle)) { return false; }
    const Object& object = objects_[handle.index_];
    return object.checkpoint_references == 0 && object.source_pins == 0 &&
           !object.destination_pinned && !has_pending_replica(object);
}

bool StateImageStore::can_release_after_checkpoint_references(
    Handle handle, std::uint32_t released_references) const {
    const Object& object = require(handle);
    if (released_references > object.checkpoint_references) { return false; }
    if (object.checkpoint_references != released_references) { return true; }
    if (object.source_pins != 0 || object.destination_pinned || has_pending_replica(object)) {
        return false;
    }
    return true;
}

void StateImageStore::retire_released_owner(Handle handle) noexcept {
    if (valid(handle) && objects_[handle.index_].release_pending && can_release(handle) &&
        !release(handle)) {
        std::terminate();
    }
}

bool StateImageStore::has_pending_replica(const Object& object) noexcept {
    return object.pending_device_slot.has_value() || object.pending_host_slot.has_value() ||
           object.transfer_id != 0;
}

std::optional<StateImageStore::Handle> StateImageStore::allocate(StateImageRole role,
                                                                 bool with_device) noexcept {
    if (free_object_count_ == 0 || role == StateImageRole::Free ||
        (with_device && free_device_count_ == 0)) {
        return std::nullopt;
    }
    const std::uint32_t index = free_objects_[--free_object_count_];
    Object& object            = objects_[index];
    object                    = Object{.generation = object.generation, .role = role};
    if (with_device) { object.device_slot = free_device_slots_[--free_device_count_]; }
    return Handle(this, index, object.generation);
}

std::optional<std::int32_t> StateImageStore::take_device_slot() noexcept {
    if (free_device_count_ == 0) { return std::nullopt; }
    return free_device_slots_[--free_device_count_];
}

void StateImageStore::return_device_slot(std::int32_t slot) noexcept {
    free_device_slots_[free_device_count_++] = slot;
}

std::uint64_t StateImageStore::next_epoch() noexcept {
    if (++next_content_epoch_ == 0) { ++next_content_epoch_; }
    return next_content_epoch_;
}

std::uint64_t StateImageStore::next_transfer() noexcept {
    if (++next_transfer_id_ == 0) { ++next_transfer_id_; }
    return next_transfer_id_;
}

StateImageStore::Object& StateImageStore::require(Handle handle) {
    if (!valid(handle)) { throw std::invalid_argument("StateImage handle is stale"); }
    return objects_[handle.index_];
}

const StateImageStore::Object& StateImageStore::require(Handle handle) const {
    if (!valid(handle)) { throw std::invalid_argument("StateImage handle is stale"); }
    return objects_[handle.index_];
}

bool StateImageStore::valid_transfer(const Transfer& transfer) const noexcept {
    if (transfer.owner_ != this || transfer.id_ == 0 || !valid(transfer.source_) ||
        !valid(transfer.destination_)) {
        return false;
    }
    const Object& destination = objects_[transfer.destination_.index_];
    return destination.transfer_id == transfer.id_;
}

void StateImageStore::validate_transfer(const Transfer& transfer) const {
    if (!valid_transfer(transfer)) {
        throw std::logic_error("StateImage transfer capability is stale");
    }
}

void StateImageStore::consume(Transfer& transfer) noexcept {
    transfer.owner_ = nullptr;
    transfer.id_    = 0;
}

} // namespace ninfer::models::context
