#pragma once

#include "core/host_kv_arena.h"
#include "targets/qwen3_6/impl/runtime/logical_kv_store.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

class HostKVExtentStore;

struct HostKVPageReplicaRelease {
    LogicalKVPageStore* pages = nullptr;
    LogicalKVPageHandle page;
};

class HostKVExtentReservation {
public:
    HostKVExtentReservation() noexcept = default;
    ~HostKVExtentReservation();

    HostKVExtentReservation(HostKVExtentReservation&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), descriptor_(other.descriptor_),
          generation_(other.generation_), page_store_(other.page_store_) {}

    HostKVExtentReservation& operator=(HostKVExtentReservation&&)      = delete;
    HostKVExtentReservation(const HostKVExtentReservation&)            = delete;
    HostKVExtentReservation& operator=(const HostKVExtentReservation&) = delete;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

private:
    HostKVExtentStore* owner_       = nullptr;
    std::uint32_t descriptor_       = 0;
    std::uint32_t generation_       = 0;
    LogicalKVPageStore* page_store_ = nullptr;

    friend class HostKVExtentStore;
};

// Owns typed Host extents and the ordered logical pages represented by each allocation. It has no
// checkpoint, retention, or scheduling policy.
class HostKVExtentStore {
public:
    HostKVExtentStore(HostKVArena& arena, std::uint32_t descriptor_capacity)
        : arena_(&arena), extents_(descriptor_capacity), free_(descriptor_capacity),
          free_count_(descriptor_capacity), memberships_(descriptor_capacity),
          free_memberships_(descriptor_capacity), free_membership_count_(descriptor_capacity) {
        if (descriptor_capacity == 0) {
            throw std::invalid_argument("Host KV extent descriptor capacity is zero");
        }
        for (std::uint32_t index = 0; index < descriptor_capacity; ++index) {
            free_[index]             = descriptor_capacity - 1U - index;
            free_memberships_[index] = descriptor_capacity - 1U - index;
        }
    }

    HostKVExtentStore(const HostKVExtentStore&)            = delete;
    HostKVExtentStore& operator=(const HostKVExtentStore&) = delete;
    HostKVExtentStore(HostKVExtentStore&&)                 = delete;
    HostKVExtentStore& operator=(HostKVExtentStore&&)      = delete;

    [[nodiscard]] std::uint32_t capacity() const noexcept {
        return static_cast<std::uint32_t>(extents_.size());
    }

    [[nodiscard]] std::uint32_t occupied() const noexcept { return capacity() - free_count_; }

    [[nodiscard]] const HostKVPageLayout& page_layout(const LogicalKVPageStore& pages) const {
        const HostKVPageLayout* layout = arena_->layout_for(pages.physical_pool().geometry());
        if (layout == nullptr) {
            throw std::invalid_argument("Host KV page store has no arena layout");
        }
        return *layout;
    }

    [[nodiscard]] std::optional<HostKVExtentReservation>
    prepare(LogicalKVPageStore& pages, std::span<const LogicalKVPageHandle> membership) {
        if (membership.empty() || free_count_ == 0 || membership.size() > free_membership_count_) {
            return std::nullopt;
        }
        for (const LogicalKVPageHandle page : membership) {
            if (!pages.can_pin_source(page) || pages.host_resident(page)) { return std::nullopt; }
        }

        const HostKVPageLayout& layout = page_layout(pages);
        std::optional<HostKVAllocation> allocation =
            arena_->allocate(layout, static_cast<std::uint32_t>(membership.size()));
        if (!allocation) { return std::nullopt; }

        const std::uint32_t descriptor = free_[--free_count_];
        Extent& extent                 = extents_[descriptor];
        if (extent.state != ExtentState::Free) { std::terminate(); }
        extent.state      = ExtentState::Reserved;
        extent.page_store = &pages;
        extent.allocation = std::move(allocation);
        for (const LogicalKVPageHandle page : membership) {
            const std::uint32_t node = take_membership();
            if (node == kInvalidIndex) { std::terminate(); }
            Membership& entry = memberships_[node];
            entry.page        = page;
            entry.epoch       = pages.content_epoch(page);
            entry.coverage    = pages.committed_columns(page);
            entry.next        = kInvalidIndex;
            if (extent.tail == kInvalidIndex) {
                extent.head = node;
            } else {
                memberships_[extent.tail].next = node;
            }
            extent.tail = node;
            ++extent.page_count;
        }
        for (const LogicalKVPageHandle page : membership) { pages.pin_source(page); }

        HostKVExtentReservation reservation;
        reservation.owner_      = this;
        reservation.descriptor_ = descriptor;
        reservation.generation_ = extent.generation;
        reservation.page_store_ = &pages;
        return reservation;
    }

    [[nodiscard]] HostKVAllocationView writable_view(HostKVExtentReservation& reservation) {
        validate(reservation);
        return arena_->writable_view(*extents_[reservation.descriptor_].allocation);
    }

    [[nodiscard]] std::vector<DeviceKVPageHandle>
    device_sources(const HostKVExtentReservation& reservation) const {
        validate(reservation);
        std::vector<DeviceKVPageHandle> out;
        out.resize(extents_[reservation.descriptor_].page_count);
        device_sources(reservation, out);
        return out;
    }

    void device_sources(const HostKVExtentReservation& reservation,
                        std::span<DeviceKVPageHandle> out) const {
        validate(reservation);
        const Extent& extent = extents_[reservation.descriptor_];
        if (out.size() != extent.page_count) {
            throw std::invalid_argument("Host KV device-source output has the wrong size");
        }
        std::uint32_t node = extent.head;
        for (std::size_t index = 0; index < out.size(); ++index) {
            if (node == kInvalidIndex) { std::terminate(); }
            out[index] = extent.page_store->physical(memberships_[node].page);
            node       = memberships_[node].next;
        }
        if (node != kInvalidIndex) { std::terminate(); }
    }

    [[nodiscard]] std::uint32_t page_count(const HostKVExtentReservation& reservation) const {
        validate(reservation);
        return extents_[reservation.descriptor_].page_count;
    }

    [[nodiscard]] HostKVExtentCapability publish(HostKVExtentReservation&& reservation) noexcept {
        if (!valid(reservation)) { std::terminate(); }
        Extent& extent     = extents_[reservation.descriptor_];
        std::uint32_t node = extent.head;
        for (std::uint32_t index = 0; index < extent.page_count; ++index) {
            if (node == kInvalidIndex) { std::terminate(); }
            const Membership& entry = memberships_[node];
            if (!extent.page_store->can_attach_host_replica(entry.page, entry.epoch,
                                                            entry.coverage)) {
                std::terminate();
            }
            node = entry.next;
        }
        if (node != kInvalidIndex) { std::terminate(); }
        extent.state = ExtentState::Published;
        const HostKVExtentCapability capability(this, reservation.descriptor_, extent.generation);
        node = extent.head;
        for (std::uint32_t index = 0; index < extent.page_count; ++index) {
            Membership& entry = memberships_[node];
            try {
                extent.page_store->attach_host_replica(
                    entry.page, HostKVPageReplica{.extent            = capability,
                                                  .page_offset       = index,
                                                  .content_epoch     = entry.epoch,
                                                  .committed_columns = entry.coverage});
            } catch (...) { std::terminate(); }
            extent.page_store->unpin_source(entry.page);
            node = entry.next;
        }
        consume(reservation);
        return capability;
    }

    void abort(HostKVExtentReservation& reservation) noexcept {
        if (!valid(reservation)) {
            consume(reservation);
            return;
        }
        Extent& extent     = extents_[reservation.descriptor_];
        std::uint32_t node = extent.head;
        for (std::uint32_t index = 0; index < extent.page_count; ++index) {
            if (node == kInvalidIndex) { std::terminate(); }
            const LogicalKVPageHandle page = memberships_[node].page;
            if (!extent.page_store->valid(page) || extent.page_store->source_pins(page) == 0) {
                std::terminate();
            }
            try {
                extent.page_store->unpin_source(page);
            } catch (...) { std::terminate(); }
            node = memberships_[node].next;
        }
        release_descriptor(reservation.descriptor_, extent);
        consume(reservation);
    }

    [[nodiscard]] bool valid(HostKVExtentCapability capability) const noexcept {
        return capability.owner_ == this && capability.index_ < extents_.size() &&
               extents_[capability.index_].state == ExtentState::Published &&
               extents_[capability.index_].generation == capability.generation_;
    }

    [[nodiscard]] HostKVAllocationConstView view(HostKVExtentCapability capability) const {
        const Extent& extent = require(capability);
        return arena_->view(*extent.allocation);
    }

    [[nodiscard]] std::pair<HostKVExtentCapability, HostKVExtentCapability>
    split(HostKVExtentCapability capability, std::uint32_t page_offset) {
        Extent& original = require(capability);
        if (free_count_ == 0 || page_offset == 0 || page_offset >= original.page_count) {
            throw std::out_of_range("Host KV extent split is not representable");
        }

        const HostKVExtentCapability old = capability;
        if (!original.allocation || original.allocation->page_count() != original.page_count) {
            throw std::logic_error("Host KV extent allocation and membership disagree");
        }
        std::uint32_t validated = original.head;
        for (std::uint32_t index = 0; index < original.page_count; ++index) {
            if (validated == kInvalidIndex) {
                throw std::logic_error("Host KV extent membership is truncated");
            }
            const Membership& entry = memberships_[validated];
            if (!original.page_store->valid(entry.page) ||
                !original.page_store->host_resident(entry.page)) {
                throw std::logic_error("Host KV extent membership changed before split");
            }
            const HostKVPageReplica replica = original.page_store->host_replica(entry.page);
            if (replica.extent != old || replica.page_offset != index ||
                replica.content_epoch != entry.epoch ||
                replica.committed_columns != entry.coverage) {
                throw std::logic_error("Host KV extent membership changed before split");
            }
            validated = entry.next;
        }
        if (validated != kInvalidIndex) {
            throw std::logic_error("Host KV extent membership exceeds its page count");
        }
        auto [left_allocation, right_allocation] =
            arena_->split(std::move(*original.allocation), page_offset);
        const std::uint32_t right_index = free_[--free_count_];
        Extent& right                   = extents_[right_index];
        if (right.state != ExtentState::Free) { std::terminate(); }
        const std::uint32_t old_count  = original.page_count;
        const std::uint32_t left_tail  = node_at(original, page_offset - 1U);
        const std::uint32_t right_head = memberships_[left_tail].next;
        if (right_head == kInvalidIndex) { std::terminate(); }
        const std::uint32_t old_tail = original.tail;
        memberships_[left_tail].next = kInvalidIndex;
        increment_generation(original.generation);
        original.allocation = std::move(left_allocation);
        original.tail       = left_tail;
        original.page_count = page_offset;
        right.state         = ExtentState::Published;
        right.page_store    = original.page_store;
        right.allocation    = std::move(right_allocation);
        right.head          = right_head;
        right.tail          = old_tail;
        right.page_count    = old_count - page_offset;

        const HostKVExtentCapability left_capability(this, capability.index_, original.generation);
        const HostKVExtentCapability right_capability(this, right_index, right.generation);
        std::uint32_t node = original.head;
        for (std::uint32_t index = 0; index < original.page_count; ++index) {
            Membership& entry = memberships_[node];
            original.page_store->rebind_host_replica(
                entry.page,
                HostKVPageReplica{.extent            = old,
                                  .page_offset       = index,
                                  .content_epoch     = entry.epoch,
                                  .committed_columns = entry.coverage},
                HostKVPageReplica{.extent            = left_capability,
                                  .page_offset       = index,
                                  .content_epoch     = entry.epoch,
                                  .committed_columns = entry.coverage});
            node = entry.next;
        }
        node = right.head;
        for (std::uint32_t index = 0; index < right.page_count; ++index) {
            Membership& entry = memberships_[node];
            right.page_store->rebind_host_replica(
                entry.page,
                HostKVPageReplica{.extent            = old,
                                  .page_offset       = page_offset + index,
                                  .content_epoch     = entry.epoch,
                                  .committed_columns = entry.coverage},
                HostKVPageReplica{.extent            = right_capability,
                                  .page_offset       = index,
                                  .content_epoch     = entry.epoch,
                                  .committed_columns = entry.coverage});
            node = entry.next;
        }
        return {left_capability, right_capability};
    }

    [[nodiscard]] bool release(HostKVExtentCapability capability) noexcept {
        if (!valid(capability)) { return false; }
        Extent& extent     = extents_[capability.index_];
        std::uint32_t node = extent.head;
        for (std::uint32_t index = 0; index < extent.page_count; ++index) {
            if (node == kInvalidIndex) { return false; }
            const LogicalKVPageHandle page = memberships_[node].page;
            if (!extent.page_store->valid(page) || !extent.page_store->host_resident(page) ||
                extent.page_store->host_replica(page).extent != capability ||
                (!extent.page_store->device_resident(page) &&
                 extent.page_store->address_references(page) != 0)) {
                return false;
            }
            node = memberships_[node].next;
        }
        if (node != kInvalidIndex) { return false; }
        node = extent.head;
        for (std::uint32_t index = 0; index < extent.page_count; ++index) {
            const LogicalKVPageHandle page = memberships_[node].page;
            if (!extent.page_store->detach_host_replica(page, capability)) { std::terminate(); }
            node = memberships_[node].next;
        }
        release_descriptor(capability.index_, extent);
        return true;
    }

    [[nodiscard]] bool can_release_page_replica(LogicalKVPageStore& pages,
                                                LogicalKVPageHandle page) const noexcept {
        if (!pages.valid(page) || !pages.host_resident(page) || pages.source_pins(page) != 0 ||
            (!pages.device_resident(page) && pages.address_references(page) != 0)) {
            return false;
        }
        const HostKVPageReplica replica = pages.host_replica(page);
        if (!valid(replica.extent)) { return false; }
        const Extent& extent     = extents_[replica.extent.index_];
        const std::uint32_t node = node_at(extent, replica.page_offset);
        if (extent.page_store != &pages || replica.page_offset >= extent.page_count ||
            node == kInvalidIndex || memberships_[node].page != page) {
            return false;
        }
        return true;
    }

    [[nodiscard]] bool
    can_release_page_replicas(std::span<const HostKVPageReplicaRelease> releases) const noexcept {
        for (std::size_t index = 0; index < releases.size(); ++index) {
            const HostKVPageReplicaRelease& release = releases[index];
            if (release.pages == nullptr ||
                !can_release_page_replica(*release.pages, release.page)) {
                return false;
            }
            for (std::size_t prior = 0; prior < index; ++prior) {
                if (releases[prior].pages == release.pages &&
                    releases[prior].page == release.page) {
                    return false;
                }
            }
        }
        return true;
    }

    [[nodiscard]] bool
    can_release_page_replicas(LogicalKVPageStore& pages,
                              std::span<const LogicalKVPageHandle> releases) const noexcept {
        for (std::size_t index = 0; index < releases.size(); ++index) {
            if (!can_release_page_replica(pages, releases[index]) ||
                std::find(releases.begin(), releases.begin() + static_cast<std::ptrdiff_t>(index),
                          releases[index]) !=
                    releases.begin() + static_cast<std::ptrdiff_t>(index)) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] bool
    can_allocate_after_page_releases(std::span<const HostKVPageReplicaRelease> releases,
                                     std::span<const HostKVAllocationRequest> allocations) const {
        return can_allocate_after_page_releases(releases, {}, allocations);
    }

    // Simulates both immediately droppable Host duplicates and Host replicas that become
    // unreferenced when a transaction removes their last address-space membership.
    [[nodiscard]] bool can_allocate_after_page_releases(
        std::span<const HostKVPageReplicaRelease> releases,
        std::span<const HostKVPageReplicaRelease> last_reference_releases,
        std::span<const HostKVAllocationRequest> allocations) const {
        if (!can_release_page_replicas(releases)) { return false; }
        for (std::size_t index = 0; index < last_reference_releases.size(); ++index) {
            const HostKVPageReplicaRelease& release = last_reference_releases[index];
            if (release.pages == nullptr || !release.pages->valid(release.page) ||
                !release.pages->host_resident(release.page) ||
                release.pages->source_pins(release.page) != 0 ||
                release.pages->address_references(release.page) != 1) {
                return false;
            }
            const HostKVPageReplica replica = release.pages->host_replica(release.page);
            if (!valid(replica.extent)) { return false; }
            const Extent& extent     = extents_[replica.extent.index_];
            const std::uint32_t node = node_at(extent, replica.page_offset);
            if (extent.page_store != release.pages || replica.page_offset >= extent.page_count ||
                node == kInvalidIndex || memberships_[node].page != release.page) {
                return false;
            }
            for (std::size_t prior = 0; prior < index; ++prior) {
                if (last_reference_releases[prior].pages == release.pages &&
                    last_reference_releases[prior].page == release.page) {
                    return false;
                }
            }
            for (const HostKVPageReplicaRelease& immediate : releases) {
                if (immediate.pages == release.pages && immediate.page == release.page) {
                    return false;
                }
            }
        }
        std::vector<HostKVSuballocationRelease> physical;
        physical.reserve(releases.size() + last_reference_releases.size());
        const auto append = [&](const HostKVPageReplicaRelease& release) {
            const HostKVPageReplica replica = release.pages->host_replica(release.page);
            const Extent& extent            = require(replica.extent);
            if (!extent.allocation) { return false; }
            physical.push_back(HostKVSuballocationRelease{
                .allocation = extent.allocation->handle(),
                .begin_page = replica.page_offset,
                .page_count = 1,
            });
            return true;
        };
        for (const HostKVPageReplicaRelease& release : releases) {
            if (!append(release)) { return false; }
        }
        for (const HostKVPageReplicaRelease& release : last_reference_releases) {
            if (!append(release)) { return false; }
        }
        return arena_->can_allocate_after_suballocation_releases(physical, allocations);
    }

    // Isolates one page from its extent, then releases only that Host replica. This is used by a
    // destructive private rewrite after the selected page has been restored to Device; it does
    // not infer checkpoint or pressure policy.
    [[nodiscard]] bool release_page_replica(LogicalKVPageStore& pages, LogicalKVPageHandle page) {
        if (!can_release_page_replica(pages, page)) { return false; }
        HostKVPageReplica replica     = pages.host_replica(page);
        HostKVExtentCapability target = replica.extent;
        std::uint32_t offset          = replica.page_offset;
        if (offset != 0) {
            auto split_extents = split(target, offset);
            target             = split_extents.second;
            offset             = 0;
        }
        const std::uint32_t pages_after_left = require(target).page_count;
        if (pages_after_left > 1U) {
            auto split_extents = split(target, 1U);
            target             = split_extents.first;
        }
        return release(target);
    }

    void release_page_replicas(std::span<const HostKVPageReplicaRelease> releases) {
        if (!can_release_page_replicas(releases)) {
            throw std::logic_error("Host KV page replicas are not atomically releasable");
        }
        for (const HostKVPageReplicaRelease& release : releases) {
            if (!release_page_replica(*release.pages, release.page)) { std::terminate(); }
        }
    }

    void release_page_replicas(LogicalKVPageStore& pages,
                               std::span<const LogicalKVPageHandle> releases) {
        if (!can_release_page_replicas(pages, releases)) {
            throw std::logic_error("Host KV page replicas are not atomically releasable");
        }
        for (const LogicalKVPageHandle page : releases) {
            if (!release_page_replica(pages, page)) { std::terminate(); }
        }
    }

    // Address-space teardown can leave part of an extent with no logical owner. Isolate each
    // zero-reference run before releasing it so a retained prefix does not pin an unrelated Host
    // suffix allocation. Descriptor capacity is provisioned per Host page by Program construction,
    // therefore every necessary split is representable.
    [[nodiscard]] std::size_t release_unreferenced() noexcept {
        std::size_t released_bytes = 0;
        try {
            for (;;) {
                bool released_run = false;
                for (std::uint32_t index = 0; index < extents_.size(); ++index) {
                    Extent& extent = extents_[index];
                    if (extent.state != ExtentState::Published || !extent.allocation ||
                        extent.page_store == nullptr) {
                        continue;
                    }
                    const auto unreferenced = [&](LogicalKVPageHandle page) {
                        return extent.page_store->valid(page) &&
                               extent.page_store->address_references(page) == 0 &&
                               extent.page_store->source_pins(page) == 0;
                    };
                    std::uint32_t begin = 0;
                    while (begin < extent.page_count &&
                           !unreferenced(memberships_[node_at(extent, begin)].page)) {
                        ++begin;
                    }
                    if (begin == extent.page_count) { continue; }
                    std::uint32_t end = begin + 1U;
                    while (end < extent.page_count &&
                           unreferenced(memberships_[node_at(extent, end)].page)) {
                        ++end;
                    }

                    HostKVExtentCapability target(this, index, extent.generation);
                    if (begin != 0) { target = split(target, begin).second; }
                    const std::uint32_t run_pages = end - begin;
                    if (run_pages < require(target).page_count) {
                        target = split(target, run_pages).first;
                    }
                    const HostKVAllocationConstView allocation = view(target);
                    const std::size_t bytes =
                        allocation.layout().page_stride * static_cast<std::size_t>(run_pages);
                    if (!release(target)) { std::terminate(); }
                    if (bytes > std::numeric_limits<std::size_t>::max() - released_bytes) {
                        std::terminate();
                    }
                    released_bytes += bytes;
                    released_run = true;
                    break;
                }
                if (!released_run) { break; }
            }
        } catch (...) { std::terminate(); }
        return released_bytes;
    }

private:
    static constexpr std::uint32_t kInvalidIndex = std::numeric_limits<std::uint32_t>::max();

    enum class ExtentState : std::uint8_t {
        Free,
        Reserved,
        Published,
    };

    struct Extent {
        ExtentState state              = ExtentState::Free;
        std::uint32_t generation       = 1;
        LogicalKVPageStore* page_store = nullptr;
        std::optional<HostKVAllocation> allocation;
        std::uint32_t head       = kInvalidIndex;
        std::uint32_t tail       = kInvalidIndex;
        std::uint32_t page_count = 0;
    };

    struct Membership {
        LogicalKVPageHandle page;
        std::uint64_t epoch    = 0;
        std::uint32_t coverage = 0;
        std::uint32_t next     = kInvalidIndex;
    };

    [[nodiscard]] bool valid(const HostKVExtentReservation& reservation) const noexcept {
        if (reservation.owner_ != this || reservation.descriptor_ >= extents_.size() ||
            reservation.page_store_ == nullptr) {
            return false;
        }
        const Extent& extent = extents_[reservation.descriptor_];
        return extent.state == ExtentState::Reserved &&
               extent.generation == reservation.generation_ &&
               extent.page_store == reservation.page_store_ && extent.allocation &&
               extent.page_count != 0 && extent.head != kInvalidIndex &&
               extent.tail != kInvalidIndex;
    }

    void validate(const HostKVExtentReservation& reservation) const {
        if (!valid(reservation)) { throw std::logic_error("Host KV extent reservation is stale"); }
    }

    [[nodiscard]] Extent& require(HostKVExtentCapability capability) {
        if (!valid(capability)) { throw std::invalid_argument("Host KV extent is stale"); }
        return extents_[capability.index_];
    }

    [[nodiscard]] const Extent& require(HostKVExtentCapability capability) const {
        if (!valid(capability)) { throw std::invalid_argument("Host KV extent is stale"); }
        return extents_[capability.index_];
    }

    static void increment_generation(std::uint32_t& generation) noexcept {
        ++generation;
        if (generation == 0) { ++generation; }
    }

    [[nodiscard]] std::uint32_t take_membership() noexcept {
        if (free_membership_count_ == 0) { return kInvalidIndex; }
        return free_memberships_[--free_membership_count_];
    }

    [[nodiscard]] std::uint32_t node_at(const Extent& extent, std::uint32_t offset) const noexcept {
        if (offset >= extent.page_count) { return kInvalidIndex; }
        std::uint32_t node = extent.head;
        while (offset-- != 0 && node != kInvalidIndex) { node = memberships_[node].next; }
        return node;
    }

    void release_descriptor(std::uint32_t index, Extent& extent) noexcept {
        std::uint32_t node = extent.head;
        for (std::uint32_t offset = 0; offset < extent.page_count; ++offset) {
            if (node == kInvalidIndex) { std::terminate(); }
            const std::uint32_t next                    = memberships_[node].next;
            memberships_[node]                          = {};
            free_memberships_[free_membership_count_++] = node;
            node                                        = next;
        }
        if (node != kInvalidIndex) { std::terminate(); }
        extent.state      = ExtentState::Free;
        extent.page_store = nullptr;
        extent.allocation.reset();
        extent.head       = kInvalidIndex;
        extent.tail       = kInvalidIndex;
        extent.page_count = 0;
        increment_generation(extent.generation);
        free_[free_count_++] = index;
    }

    static void consume(HostKVExtentReservation& reservation) noexcept {
        reservation.owner_      = nullptr;
        reservation.page_store_ = nullptr;
    }

    HostKVArena* arena_ = nullptr;
    std::vector<Extent> extents_;
    std::vector<std::uint32_t> free_;
    std::uint32_t free_count_ = 0;
    std::vector<Membership> memberships_;
    std::vector<std::uint32_t> free_memberships_;
    std::uint32_t free_membership_count_ = 0;
};

inline HostKVExtentReservation::~HostKVExtentReservation() {
    if (owner_ != nullptr) { owner_->abort(*this); }
}

} // namespace ninfer::targets::qwen3_6::detail
