#pragma once

#include <ninfer/targets/qwen3_6/state_image.h>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

class StateImageStore;

class StateImageHandle {
public:
    StateImageHandle() noexcept = default;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

    [[nodiscard]] friend bool operator==(StateImageHandle, StateImageHandle) noexcept = default;

private:
    StateImageHandle(const StateImageStore* owner, std::uint32_t index,
                     std::uint32_t generation) noexcept
        : owner_(owner), index_(index), generation_(generation) {}

    const StateImageStore* owner_ = nullptr;
    std::uint32_t index_          = 0;
    std::uint32_t generation_     = 0;

    friend class StateImageStore;
};

enum class StateImageRole : std::uint8_t {
    Free,
    ActiveMutable,
    CheckpointImmutable,
    ReservedDestination,
};

struct StateImageSelectors {
    std::int32_t source      = -1;
    std::int32_t destination = -1;
};

// Program-private logical ownership over a caller-backed StateImageDevicePool. The store assigns
// logical generations and roles; the physical pool remains a storage/transfer primitive and has
// no continuation or scheduling semantics.
class StateImageStore {
public:
    explicit StateImageStore(qwen3_6::StateImageDevicePool& pool)
        : pool_(&pool), objects_(static_cast<std::size_t>(pool.slot_count())),
          free_objects_(objects_.size()), free_physical_slots_(objects_.size()),
          free_object_count_(static_cast<std::uint32_t>(objects_.size())),
          free_physical_count_(static_cast<std::uint32_t>(objects_.size())) {
        if (objects_.empty()) { throw std::invalid_argument("StateImageStore capacity is zero"); }
        for (std::uint32_t index = 0; index < objects_.size(); ++index) {
            free_objects_[index]        = static_cast<std::uint32_t>(objects_.size()) - 1U - index;
            free_physical_slots_[index] = static_cast<std::int32_t>(objects_.size() - 1U - index);
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

    [[nodiscard]] std::optional<StateImageHandle> reserve_destination() noexcept {
        return allocate(StateImageRole::ReservedDestination);
    }

    [[nodiscard]] std::optional<StateImageHandle> reserve_reset(cudaStream_t stream = nullptr) {
        std::optional<StateImageHandle> handle = allocate(StateImageRole::ActiveMutable);
        if (!handle) { return std::nullopt; }
        try {
            pool_->zero_slot(physical_slot(*handle), stream);
        } catch (...) {
            (void)release(*handle);
            throw;
        }
        return handle;
    }

    void activate_reset(StateImageHandle handle, cudaStream_t stream = nullptr) {
        Object& object = require(handle);
        if (object.role != StateImageRole::ReservedDestination || object.source_pins != 0 ||
            object.destination_pinned) {
            throw std::logic_error("StateImage reset reservation is not activatable");
        }
        pool_->zero_slot(object.physical_slot, stream);
        object.role = StateImageRole::ActiveMutable;
    }

    [[nodiscard]] bool valid(StateImageHandle handle) const noexcept {
        return handle.owner_ == this && handle.index_ < objects_.size() &&
               objects_[handle.index_].role != StateImageRole::Free &&
               objects_[handle.index_].generation == handle.generation_;
    }

    [[nodiscard]] StateImageRole role(StateImageHandle handle) const {
        return require(handle).role;
    }

    [[nodiscard]] std::int32_t physical_slot(StateImageHandle handle) const {
        return require(handle).physical_slot;
    }

    void move_checkpoint_to_active(StateImageHandle handle) {
        Object& object = require(handle);
        if (object.role != StateImageRole::CheckpointImmutable || object.source_pins != 0 ||
            object.destination_pinned) {
            throw std::logic_error("StateImage checkpoint is not movable");
        }
        object.role = StateImageRole::ActiveMutable;
    }

    void freeze(StateImageHandle handle) {
        Object& object = require(handle);
        if (object.role != StateImageRole::ActiveMutable || object.source_pins != 0 ||
            object.destination_pinned) {
            throw std::logic_error("StateImage active image is not freezable");
        }
        object.role = StateImageRole::CheckpointImmutable;
    }

    [[nodiscard]] StateImageSelectors begin_fork(StateImageHandle source,
                                                 StateImageHandle destination) {
        Object& source_object      = require(source);
        Object& destination_object = require(destination);
        if (source == destination || source_object.role != StateImageRole::CheckpointImmutable ||
            source_object.source_pins != 0 ||
            destination_object.role != StateImageRole::ReservedDestination ||
            destination_object.source_pins != 0 || destination_object.destination_pinned) {
            throw std::logic_error("StateImage fork binding is invalid");
        }
        ++source_object.source_pins;
        destination_object.destination_pinned = true;
        destination_object.role               = StateImageRole::ActiveMutable;
        return {.source      = source_object.physical_slot,
                .destination = destination_object.physical_slot};
    }

    void commit_fork(StateImageHandle source, StateImageHandle destination) {
        Object& source_object      = require(source);
        Object& destination_object = require(destination);
        if (source_object.role != StateImageRole::CheckpointImmutable ||
            source_object.source_pins == 0 ||
            destination_object.role != StateImageRole::ActiveMutable ||
            !destination_object.destination_pinned) {
            throw std::logic_error("StateImage fork commit is invalid");
        }
        --source_object.source_pins;
        destination_object.destination_pinned = false;
    }

    void abort_fork(StateImageHandle source, StateImageHandle destination) {
        Object& source_object      = require(source);
        Object& destination_object = require(destination);
        if (source_object.role != StateImageRole::CheckpointImmutable ||
            source_object.source_pins == 0 ||
            destination_object.role != StateImageRole::ActiveMutable ||
            !destination_object.destination_pinned) {
            throw std::logic_error("StateImage fork abort is invalid");
        }
        --source_object.source_pins;
        destination_object.destination_pinned = false;
        destination_object.role               = StateImageRole::ReservedDestination;
    }

    [[nodiscard]] StateImageSelectors selectors(StateImageHandle source,
                                                StateImageHandle destination) const {
        const Object& source_object      = require(source);
        const Object& destination_object = require(destination);
        const bool inplace               = source == destination;
        if ((inplace && (source_object.role != StateImageRole::ActiveMutable ||
                         source_object.source_pins != 0 || source_object.destination_pinned)) ||
            (!inplace && (source_object.role != StateImageRole::CheckpointImmutable ||
                          source_object.source_pins == 0 ||
                          destination_object.role != StateImageRole::ActiveMutable ||
                          !destination_object.destination_pinned))) {
            throw std::logic_error("StateImage execution binding is invalid");
        }
        return {.source      = source_object.physical_slot,
                .destination = destination_object.physical_slot};
    }

    [[nodiscard]] bool release(StateImageHandle handle) noexcept {
        if (!valid(handle)) { return false; }
        Object& object = objects_[handle.index_];
        if (object.source_pins != 0 || object.destination_pinned) { return false; }
        free_physical_slots_[free_physical_count_++] = object.physical_slot;
        object.physical_slot                         = -1;
        object.role                                  = StateImageRole::Free;
        if (++object.generation == 0) { ++object.generation; }
        free_objects_[free_object_count_++] = handle.index_;
        return true;
    }

private:
    struct Object {
        std::uint32_t generation   = 1;
        std::int32_t physical_slot = -1;
        std::uint16_t source_pins  = 0;
        bool destination_pinned    = false;
        StateImageRole role        = StateImageRole::Free;
    };

    [[nodiscard]] std::optional<StateImageHandle> allocate(StateImageRole role) noexcept {
        if (free_object_count_ == 0 || free_physical_count_ == 0 || role == StateImageRole::Free) {
            return std::nullopt;
        }
        const std::uint32_t index = free_objects_[--free_object_count_];
        Object& object            = objects_[index];
        object.physical_slot      = free_physical_slots_[--free_physical_count_];
        object.role               = role;
        object.source_pins        = 0;
        object.destination_pinned = false;
        return StateImageHandle(this, index, object.generation);
    }

    [[nodiscard]] Object& require(StateImageHandle handle) {
        if (!valid(handle)) { throw std::invalid_argument("StateImage handle is stale"); }
        return objects_[handle.index_];
    }

    [[nodiscard]] const Object& require(StateImageHandle handle) const {
        if (!valid(handle)) { throw std::invalid_argument("StateImage handle is stale"); }
        return objects_[handle.index_];
    }

    qwen3_6::StateImageDevicePool* pool_ = nullptr;
    std::vector<Object> objects_;
    std::vector<std::uint32_t> free_objects_;
    std::vector<std::int32_t> free_physical_slots_;
    std::uint32_t free_object_count_   = 0;
    std::uint32_t free_physical_count_ = 0;
};

struct ActiveStateBinding {
    StateImageHandle read;
    StateImageHandle write;
    bool fork_pending = false;
};

} // namespace ninfer::targets::qwen3_6::detail
