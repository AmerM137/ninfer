#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::models::context {

enum class StateImageRole : std::uint8_t {
    Free,
    ActiveMutable,
    CheckpointImmutable,
    ReservedDestination,
};

enum class StateReplicaResidency : std::uint8_t {
    None,
    DeviceOnly,
    HostOnly,
    Both,
};

enum class StateTransferDirection : std::uint8_t {
    DeviceToHost,
    HostToDevice,
    HostToFork,
};

struct StateImageSelectors {
    std::int32_t source      = -1;
    std::int32_t destination = -1;
};

// Logical ownership is independent of the typed Device and Host state layouts.
template <class DevicePool, class HostPool>
class StateImageStore {
public:
    class Handle {
    public:
        Handle() noexcept = default;

        [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

        [[nodiscard]] friend bool operator==(Handle, Handle) noexcept = default;

    private:
        Handle(const StateImageStore* owner, std::uint32_t index, std::uint32_t generation) noexcept
            : owner_(owner), index_(index), generation_(generation) {}

        const StateImageStore* owner_ = nullptr;
        std::uint32_t index_          = 0;
        std::uint32_t generation_     = 0;

        friend class StateImageStore;
    };

    class Transfer {
    public:
        Transfer() noexcept = default;

        ~Transfer() {
            if (owner_ != nullptr) { owner_->abort_transfer(std::move(*this)); }
        }

        Transfer(Transfer&& other) noexcept
            : owner_(std::exchange(other.owner_, nullptr)), id_(std::exchange(other.id_, 0)),
              direction_(other.direction_), source_(other.source_),
              destination_(other.destination_) {}

        Transfer& operator=(Transfer&&)      = delete;
        Transfer(const Transfer&)            = delete;
        Transfer& operator=(const Transfer&) = delete;

        [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

        [[nodiscard]] StateTransferDirection direction() const noexcept { return direction_; }

    private:
        Transfer(StateImageStore& owner, std::uint64_t id, StateTransferDirection direction,
                 Handle source, Handle destination) noexcept
            : owner_(&owner), id_(id), direction_(direction), source_(source),
              destination_(destination) {}

        StateImageStore* owner_           = nullptr;
        std::uint64_t id_                 = 0;
        StateTransferDirection direction_ = StateTransferDirection::DeviceToHost;
        Handle source_;
        Handle destination_;

        friend class StateImageStore;
    };

    struct ActiveBinding {
        Handle read;
        Handle write;
        bool fork_pending = false;
    };

    StateImageStore(DevicePool& device, HostPool* host, std::uint32_t logical_capacity)
        : device_(&device), host_(host), objects_(logical_capacity),
          free_objects_(logical_capacity),
          free_device_slots_(static_cast<std::size_t>(device.slot_count())),
          free_object_count_(logical_capacity),
          free_device_count_(static_cast<std::uint32_t>(device.slot_count())) {
        if (logical_capacity == 0 || device.slot_count() <= 0 ||
            logical_capacity < static_cast<std::uint32_t>(device.slot_count()) ||
            (host != nullptr && logical_capacity < static_cast<std::uint32_t>(device.slot_count()) +
                                                       host->capacity())) {
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

    StateImageStore(const StateImageStore&)            = delete;
    StateImageStore& operator=(const StateImageStore&) = delete;
    StateImageStore(StateImageStore&&)                 = delete;
    StateImageStore& operator=(StateImageStore&&)      = delete;

    [[nodiscard]] std::uint32_t capacity() const noexcept {
        return static_cast<std::uint32_t>(objects_.size());
    }

    [[nodiscard]] std::uint32_t occupied() const noexcept {
        return capacity() - free_object_count_;
    }

    [[nodiscard]] std::uint32_t device_occupied() const noexcept {
        return static_cast<std::uint32_t>(free_device_slots_.size()) - free_device_count_;
    }

    [[nodiscard]] std::uint32_t device_capacity() const noexcept {
        return static_cast<std::uint32_t>(free_device_slots_.size());
    }

    [[nodiscard]] std::uint32_t host_occupied() const noexcept {
        return host_ == nullptr ? 0U : host_->occupied();
    }

    [[nodiscard]] std::optional<Handle> reserve_destination() noexcept {
        return allocate(StateImageRole::ReservedDestination, true);
    }

    [[nodiscard]] std::optional<Handle> reserve_logical_destination() noexcept {
        return allocate(StateImageRole::ReservedDestination, false);
    }

    [[nodiscard]] std::optional<Handle> reserve_reset(cudaStream_t stream = nullptr) {
        std::optional<Handle> handle = allocate(StateImageRole::ActiveMutable, true);
        if (!handle) { return std::nullopt; }
        try {
            Object& object = require(*handle);
            device_->zero_slot(*object.device_slot, stream);
            object.content_epoch = next_epoch();
        } catch (...) {
            (void)release(*handle);
            throw;
        }
        return handle;
    }

    void activate_reset(Handle handle, cudaStream_t stream = nullptr) {
        Object& object = require(handle);
        if (object.role != StateImageRole::ReservedDestination || !object.device_slot ||
            object.source_pins != 0 || object.destination_pinned || has_pending_replica(object)) {
            throw std::logic_error("StateImage reset reservation is not activatable");
        }
        device_->zero_slot(*object.device_slot, stream);
        object.content_epoch = next_epoch();
        object.role          = StateImageRole::ActiveMutable;
    }

    [[nodiscard]] bool valid(Handle handle) const noexcept {
        return handle.owner_ == this && handle.index_ < objects_.size() &&
               objects_[handle.index_].role != StateImageRole::Free &&
               objects_[handle.index_].generation == handle.generation_;
    }

    [[nodiscard]] std::uint32_t descriptor_index(Handle handle) const {
        (void)require(handle);
        return handle.index_;
    }

    [[nodiscard]] StateImageRole role(Handle handle) const { return require(handle).role; }

    [[nodiscard]] StateReplicaResidency residency(Handle handle) const {
        const Object& object = require(handle);
        if (object.device_slot && object.host_slot) { return StateReplicaResidency::Both; }
        if (object.device_slot) { return StateReplicaResidency::DeviceOnly; }
        if (object.host_slot) { return StateReplicaResidency::HostOnly; }
        return StateReplicaResidency::None;
    }

    [[nodiscard]] bool device_resident(Handle handle) const {
        return require(handle).device_slot.has_value();
    }

    [[nodiscard]] bool host_resident(Handle handle) const {
        return require(handle).host_slot.has_value();
    }

    [[nodiscard]] std::uint64_t content_epoch(Handle handle) const {
        return require(handle).content_epoch;
    }

    [[nodiscard]] std::uint32_t source_pins(Handle handle) const {
        return require(handle).source_pins;
    }

    [[nodiscard]] std::uint32_t checkpoint_references(Handle handle) const {
        return require(handle).checkpoint_references;
    }

    void retain_checkpoint_reference(Handle handle) {
        Object& object = require(handle);
        if (object.role != StateImageRole::CheckpointImmutable ||
            object.checkpoint_references == std::numeric_limits<std::uint32_t>::max()) {
            throw std::logic_error("StateImage checkpoint reference is not retainable");
        }
        ++object.checkpoint_references;
        object.release_pending = false;
    }

    [[nodiscard]] bool can_release_checkpoint_owner(Handle handle) const noexcept {
        if (!valid(handle)) { return false; }
        const Object& object = objects_[handle.index_];
        return object.role == StateImageRole::CheckpointImmutable &&
               object.checkpoint_references != 0;
    }

    // Content owners and readers retire independently. A last checkpoint can disappear while
    // another request still reads this immutable image; that read lease keeps its replicas alive.
    [[nodiscard]] bool release_checkpoint_owner(Handle handle) noexcept {
        if (!can_release_checkpoint_owner(handle)) { return false; }
        Object& object = objects_[handle.index_];
        if (--object.checkpoint_references == 0) {
            object.release_pending = true;
            retire_released_owner(handle);
        }
        return true;
    }

    void release_checkpoint_reference(Handle handle) {
        Object& object = require(handle);
        if (object.role != StateImageRole::CheckpointImmutable ||
            object.checkpoint_references == 0) {
            throw std::logic_error("StateImage checkpoint reference is not releasable");
        }
        --object.checkpoint_references;
    }

    [[nodiscard]] std::int32_t physical_slot(Handle handle) const {
        const Object& object = require(handle);
        if (!object.device_slot) {
            throw std::logic_error("StateImage has no published Device replica");
        }
        return *object.device_slot;
    }

    void move_checkpoint_to_active(Handle handle) {
        Object& object = require(handle);
        if (object.role != StateImageRole::CheckpointImmutable || !object.device_slot ||
            object.checkpoint_references != 0 || object.source_pins != 0 ||
            object.destination_pinned || has_pending_replica(object)) {
            throw std::logic_error("StateImage checkpoint is not movable");
        }
        if (object.host_slot) {
            if (host_ == nullptr || !host_->release(*object.host_slot)) {
                throw std::logic_error("StateImage Host replica could not be consumed by Move");
            }
            object.host_slot.reset();
        }
        object.role = StateImageRole::ActiveMutable;
    }

    [[nodiscard]] bool drop_device_replica(Handle handle) noexcept {
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

    [[nodiscard]] bool drop_host_replica(Handle handle) noexcept {
        if (!valid(handle)) { return false; }
        Object& object = objects_[handle.index_];
        if (host_ == nullptr || object.role != StateImageRole::CheckpointImmutable ||
            !object.device_slot || !object.host_slot || object.source_pins != 0 ||
            object.destination_pinned || has_pending_replica(object) ||
            !host_->release(*object.host_slot)) {
            return false;
        }
        object.host_slot.reset();
        return true;
    }

    // Transfers the Device replica to a new logical identity while the old immutable identity
    // keeps its Host replica. This is the zero-copy preserving branch for a Both source.
    void split_device_replica_identity(Handle source, Handle destination) {
        Object& source_object      = require(source);
        Object& destination_object = require(destination);
        if (source == destination || source_object.role != StateImageRole::CheckpointImmutable ||
            !source_object.device_slot || !source_object.host_slot ||
            source_object.source_pins != 0 || source_object.destination_pinned ||
            has_pending_replica(source_object) ||
            destination_object.role != StateImageRole::ReservedDestination ||
            destination_object.device_slot || destination_object.host_slot ||
            has_pending_replica(destination_object) || destination_object.source_pins != 0 ||
            destination_object.destination_pinned) {
            throw std::logic_error("StateImage replica identity split is invalid");
        }
        destination_object.device_slot   = source_object.device_slot;
        destination_object.content_epoch = source_object.content_epoch;
        destination_object.role          = StateImageRole::ActiveMutable;
        source_object.device_slot.reset();
    }

    void freeze(Handle handle) {
        Object& object = require(handle);
        if (object.role != StateImageRole::ActiveMutable || !object.device_slot ||
            object.source_pins != 0 || object.destination_pinned || has_pending_replica(object)) {
            throw std::logic_error("StateImage active image is not freezable");
        }
        object.content_epoch = next_epoch();
        object.role          = StateImageRole::CheckpointImmutable;
    }

    void thaw(Handle handle) {
        Object& object = require(handle);
        if (object.role != StateImageRole::CheckpointImmutable || !object.device_slot ||
            object.checkpoint_references != 0 || object.source_pins != 0 ||
            object.destination_pinned || has_pending_replica(object)) {
            throw std::logic_error("StateImage checkpoint is not thawable");
        }
        object.role = StateImageRole::ActiveMutable;
    }

    [[nodiscard]] StateImageSelectors begin_fork(Handle source, Handle destination) {
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
        destination_object.role               = StateImageRole::ActiveMutable;
        destination_object.content_epoch      = source_object.content_epoch;
        return {.source      = *source_object.device_slot,
                .destination = *destination_object.device_slot};
    }

    void commit_fork(Handle source, Handle destination) {
        Object& source_object      = require(source);
        Object& destination_object = require(destination);
        if (source_object.role != StateImageRole::CheckpointImmutable ||
            source_object.source_pins == 0 ||
            destination_object.role != StateImageRole::ActiveMutable ||
            !destination_object.device_slot || !destination_object.destination_pinned) {
            throw std::logic_error("StateImage fork commit is invalid");
        }
        --source_object.source_pins;
        destination_object.destination_pinned = false;
        retire_released_owner(source);
    }

    void abort_fork(Handle source, Handle destination) {
        if (!can_abort_fork(source, destination)) {
            throw std::logic_error("StateImage fork abort is invalid");
        }
        Object& source_object      = require(source);
        Object& destination_object = require(destination);
        --source_object.source_pins;
        destination_object.destination_pinned = false;
        destination_object.role               = StateImageRole::ReservedDestination;
        destination_object.content_epoch      = 0;
        retire_released_owner(source);
    }

    [[nodiscard]] bool can_abort_fork(Handle source, Handle destination) const noexcept {
        if (!valid(source) || !valid(destination)) { return false; }
        const Object& source_object      = objects_[source.index_];
        const Object& destination_object = objects_[destination.index_];
        return source != destination && source_object.role == StateImageRole::CheckpointImmutable &&
               source_object.source_pins != 0 &&
               destination_object.role == StateImageRole::ActiveMutable &&
               destination_object.device_slot.has_value() && destination_object.destination_pinned;
    }

    [[nodiscard]] StateImageSelectors selectors(Handle source, Handle destination) const {
        const Object& source_object      = require(source);
        const Object& destination_object = require(destination);
        const bool inplace               = source == destination;
        if (!source_object.device_slot || !destination_object.device_slot ||
            (inplace && (source_object.role != StateImageRole::ActiveMutable ||
                         source_object.source_pins != 0 || source_object.destination_pinned)) ||
            (!inplace && (source_object.role != StateImageRole::CheckpointImmutable ||
                          source_object.source_pins == 0 ||
                          destination_object.role != StateImageRole::ActiveMutable ||
                          !destination_object.destination_pinned))) {
            throw std::logic_error("StateImage execution binding is invalid");
        }
        return {.source      = *source_object.device_slot,
                .destination = *destination_object.device_slot};
    }

    [[nodiscard]] std::optional<Transfer> reserve_device_to_host(Handle source) {
        Object& object = require(source);
        if (host_ == nullptr || object.role != StateImageRole::CheckpointImmutable ||
            !object.device_slot || object.host_slot || has_pending_replica(object) ||
            object.source_pins == std::numeric_limits<std::uint32_t>::max()) {
            return std::nullopt;
        }
        std::optional<typename HostPool::Handle> target = host_->allocate();
        if (!target) { return std::nullopt; }
        const std::uint64_t transfer = next_transfer();
        object.pending_host_slot     = *target;
        object.transfer_id           = transfer;
        object.destination_pinned    = true;
        ++object.source_pins;
        return Transfer(*this, transfer, StateTransferDirection::DeviceToHost, source, source);
    }

    void enqueue_device_to_host(const Transfer& transfer, cudaStream_t stream = nullptr) {
        validate_transfer(transfer);
        Object& object = require(transfer.source_);
        if (transfer.direction_ != StateTransferDirection::DeviceToHost || host_ == nullptr ||
            !object.device_slot || !object.pending_host_slot) {
            throw std::logic_error("StateImage Device-to-Host transfer is not enqueueable");
        }
        device_->copy_to_host(*object.device_slot, host_->writable_view(*object.pending_host_slot),
                              stream);
    }

    [[nodiscard]] std::optional<Transfer> begin_device_to_host(Handle source,
                                                               cudaStream_t stream = nullptr) {
        std::optional<Transfer> transfer = reserve_device_to_host(source);
        if (!transfer) { return std::nullopt; }
        try {
            enqueue_device_to_host(*transfer, stream);
        } catch (...) {
            abort_transfer(std::move(*transfer));
            throw;
        }
        return transfer;
    }

    [[nodiscard]] std::optional<Transfer> begin_host_to_device(Handle source,
                                                               cudaStream_t stream = nullptr) {
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
            device_->copy_from_host(host_->view(*object.host_slot), *target, stream);
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

    [[nodiscard]] std::optional<Transfer> begin_host_fork(Handle source, Handle destination,
                                                          cudaStream_t stream = nullptr) {
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
        destination_object.role                = StateImageRole::ActiveMutable;
        destination_object.content_epoch       = source_object.content_epoch;
        ++source_object.source_pins;
        try {
            device_->copy_from_host(host_->view(*source_object.host_slot), *target, stream);
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
        return Transfer(*this, transfer, StateTransferDirection::HostToFork, source, destination);
    }

    void publish_transfer(Transfer&& transfer, bool keep_source_replica) {
        validate_transfer(transfer);
        const Handle source_handle = transfer.source_;
        Object& source             = require(transfer.source_);
        Object& destination        = require(transfer.destination_);
        switch (transfer.direction_) {
        case StateTransferDirection::DeviceToHost:
            if (host_ == nullptr || !source.pending_host_slot ||
                !host_->publish(*source.pending_host_slot)) {
                throw std::logic_error("StateImage Host destination publication failed");
            }
            source.host_slot = source.pending_host_slot;
            source.pending_host_slot.reset();
            if (!keep_source_replica) {
                return_device_slot(*source.device_slot);
                source.device_slot.reset();
            }
            --source.source_pins;
            source.destination_pinned = false;
            source.transfer_id        = 0;
            break;
        case StateTransferDirection::HostToDevice:
            source.device_slot = source.pending_device_slot;
            source.pending_device_slot.reset();
            if (!keep_source_replica) {
                if (host_ == nullptr || !host_->release(*source.host_slot)) {
                    throw std::logic_error("StateImage Host replica release failed at publication");
                }
                source.host_slot.reset();
            }
            --source.source_pins;
            source.destination_pinned = false;
            source.transfer_id        = 0;
            break;
        case StateTransferDirection::HostToFork:
            destination.device_slot = destination.pending_device_slot;
            destination.pending_device_slot.reset();
            --source.source_pins;
            destination.destination_pinned = false;
            destination.transfer_id        = 0;
            break;
        }
        consume(transfer);
        retire_released_owner(source_handle);
    }

    void abort_transfer(Transfer&& transfer) noexcept {
        if (!valid_transfer(transfer)) {
            consume(transfer);
            return;
        }
        const Handle source_handle = transfer.source_;
        Object& source             = objects_[transfer.source_.index_];
        Object& destination        = objects_[transfer.destination_.index_];
        if (source.source_pins != 0) { --source.source_pins; }
        if (transfer.direction_ == StateTransferDirection::DeviceToHost) {
            if (host_ != nullptr && source.pending_host_slot) {
                (void)host_->release(*source.pending_host_slot);
            }
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
            if (transfer.direction_ == StateTransferDirection::HostToFork) {
                destination.role          = StateImageRole::ReservedDestination;
                destination.content_epoch = 0;
            }
        }
        consume(transfer);
        retire_released_owner(source_handle);
    }

    [[nodiscard]] bool release(Handle handle) noexcept {
        if (!can_release(handle)) { return false; }
        Object& object = objects_[handle.index_];
        if (object.host_slot) {
            if (host_ == nullptr || !host_->release(*object.host_slot)) { return false; }
            object.host_slot.reset();
        }
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

    [[nodiscard]] bool can_release(Handle handle) const noexcept {
        if (!valid(handle)) { return false; }
        const Object& object = objects_[handle.index_];
        return object.checkpoint_references == 0 && object.source_pins == 0 &&
               !object.destination_pinned && !has_pending_replica(object);
    }

    [[nodiscard]] bool
    can_release_after_checkpoint_references(Handle handle,
                                            std::uint32_t released_references) const {
        const Object& object = require(handle);
        if (released_references > object.checkpoint_references) { return false; }
        if (object.checkpoint_references != released_references) { return true; }
        if (object.source_pins != 0 || object.destination_pinned || has_pending_replica(object)) {
            return false;
        }
        if (object.host_slot) {
            if (host_ == nullptr) { return false; }
            (void)host_->view(*object.host_slot);
        }
        return true;
    }

private:
    struct Object {
        std::uint32_t generation    = 1;
        std::uint64_t content_epoch = 0;
        std::optional<std::int32_t> device_slot;
        std::optional<typename HostPool::Handle> host_slot;
        std::optional<std::int32_t> pending_device_slot;
        std::optional<typename HostPool::Handle> pending_host_slot;
        std::uint64_t transfer_id           = 0;
        std::uint32_t checkpoint_references = 0;
        std::uint32_t source_pins           = 0;
        bool destination_pinned             = false;
        bool release_pending                = false;
        StateImageRole role                 = StateImageRole::Free;
    };

    void retire_released_owner(Handle handle) noexcept {
        if (valid(handle) && objects_[handle.index_].release_pending && can_release(handle) &&
            !release(handle)) {
            std::terminate();
        }
    }

    [[nodiscard]] static bool has_pending_replica(const Object& object) noexcept {
        return object.pending_device_slot.has_value() || object.pending_host_slot.has_value() ||
               object.transfer_id != 0;
    }

    [[nodiscard]] std::optional<Handle> allocate(StateImageRole role, bool with_device) noexcept {
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

    [[nodiscard]] std::optional<std::int32_t> take_device_slot() noexcept {
        if (free_device_count_ == 0) { return std::nullopt; }
        return free_device_slots_[--free_device_count_];
    }

    void return_device_slot(std::int32_t slot) noexcept {
        free_device_slots_[free_device_count_++] = slot;
    }

    [[nodiscard]] std::uint64_t next_epoch() noexcept {
        if (++next_content_epoch_ == 0) { ++next_content_epoch_; }
        return next_content_epoch_;
    }

    [[nodiscard]] std::uint64_t next_transfer() noexcept {
        if (++next_transfer_id_ == 0) { ++next_transfer_id_; }
        return next_transfer_id_;
    }

    [[nodiscard]] Object& require(Handle handle) {
        if (!valid(handle)) { throw std::invalid_argument("StateImage handle is stale"); }
        return objects_[handle.index_];
    }

    [[nodiscard]] const Object& require(Handle handle) const {
        if (!valid(handle)) { throw std::invalid_argument("StateImage handle is stale"); }
        return objects_[handle.index_];
    }

    [[nodiscard]] bool valid_transfer(const Transfer& transfer) const noexcept {
        if (transfer.owner_ != this || transfer.id_ == 0 || !valid(transfer.source_) ||
            !valid(transfer.destination_)) {
            return false;
        }
        const Object& destination = objects_[transfer.destination_.index_];
        return destination.transfer_id == transfer.id_;
    }

    void validate_transfer(const Transfer& transfer) const {
        if (!valid_transfer(transfer)) {
            throw std::logic_error("StateImage transfer capability is stale");
        }
    }

    static void consume(Transfer& transfer) noexcept {
        transfer.owner_ = nullptr;
        transfer.id_    = 0;
    }

    DevicePool* device_ = nullptr;
    HostPool* host_     = nullptr;
    std::vector<Object> objects_;
    std::vector<std::uint32_t> free_objects_;
    std::vector<std::int32_t> free_device_slots_;
    std::uint32_t free_object_count_  = 0;
    std::uint32_t free_device_count_  = 0;
    std::uint64_t next_content_epoch_ = 0;
    std::uint64_t next_transfer_id_   = 0;
};

} // namespace ninfer::models::context
