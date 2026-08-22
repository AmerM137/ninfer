#pragma once

#include "core/paged_kv_cache.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

class LogicalKVPageStore;
class KVAddressSpaceStore;

class LogicalKVPageHandle {
public:
    LogicalKVPageHandle() noexcept = default;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

    [[nodiscard]] friend bool operator==(LogicalKVPageHandle,
                                         LogicalKVPageHandle) noexcept = default;

private:
    LogicalKVPageHandle(const LogicalKVPageStore* owner, std::uint32_t index,
                        std::uint32_t generation) noexcept
        : owner_(owner), index_(index), generation_(generation) {}

    const LogicalKVPageStore* owner_ = nullptr;
    std::uint32_t index_             = 0;
    std::uint32_t generation_        = 0;

    friend class LogicalKVPageStore;
};

class KVAddressSpaceHandle {
public:
    KVAddressSpaceHandle() noexcept = default;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

    [[nodiscard]] friend bool operator==(KVAddressSpaceHandle,
                                         KVAddressSpaceHandle) noexcept = default;

private:
    KVAddressSpaceHandle(const KVAddressSpaceStore* owner, std::uint32_t index,
                         std::uint32_t generation) noexcept
        : owner_(owner), index_(index), generation_(generation) {}

    const KVAddressSpaceStore* owner_ = nullptr;
    std::uint32_t index_              = 0;
    std::uint32_t generation_         = 0;

    friend class KVAddressSpaceStore;
};

class KVActivationReservation {
public:
    KVActivationReservation() noexcept = default;

    KVActivationReservation(KVActivationReservation&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), address_(other.address_),
          requested_entitlement_(other.requested_entitlement_),
          page_reservation_(std::move(other.page_reservation_)), row_(std::move(other.row_)) {}

    KVActivationReservation& operator=(KVActivationReservation&&)      = delete;
    KVActivationReservation(const KVActivationReservation&)            = delete;
    KVActivationReservation& operator=(const KVActivationReservation&) = delete;

private:
    KVActivationReservation(KVAddressSpaceStore& owner, KVAddressSpaceHandle address,
                            std::uint32_t requested_entitlement,
                            DeviceKVPageReservation&& page_reservation,
                            KVExecutionRowLease&& row) noexcept
        : owner_(&owner), address_(address), requested_entitlement_(requested_entitlement),
          page_reservation_(std::move(page_reservation)), row_(std::move(row)) {}

    KVAddressSpaceStore* owner_ = nullptr;
    KVAddressSpaceHandle address_;
    std::uint32_t requested_entitlement_ = 0;
    DeviceKVPageReservation page_reservation_;
    std::optional<KVExecutionRowLease> row_;

    friend class KVAddressSpaceStore;
};

class LogicalKVPageStore {
public:
    explicit LogicalKVPageStore(DeviceKVPagePool& physical)
        : physical_(&physical), pages_(physical.capacity_pages()), free_(pages_.size()),
          free_count_(static_cast<std::uint32_t>(pages_.size())) {
        if (pages_.empty()) { throw std::invalid_argument("Logical KV page capacity is zero"); }
        for (std::uint32_t index = 0; index < pages_.size(); ++index) {
            free_[index] = static_cast<std::uint32_t>(pages_.size()) - 1U - index;
        }
    }

    LogicalKVPageStore(const LogicalKVPageStore&)            = delete;
    LogicalKVPageStore& operator=(const LogicalKVPageStore&) = delete;
    LogicalKVPageStore(LogicalKVPageStore&&)                 = delete;
    LogicalKVPageStore& operator=(LogicalKVPageStore&&)      = delete;

    [[nodiscard]] DeviceKVPagePool& physical_pool() noexcept { return *physical_; }

    [[nodiscard]] std::uint32_t capacity() const noexcept {
        return static_cast<std::uint32_t>(pages_.size());
    }

    [[nodiscard]] std::uint32_t occupied() const noexcept { return capacity() - free_count_; }

    [[nodiscard]] LogicalKVPageHandle materialize(DeviceKVPageReservation& reservation) {
        if (free_count_ == 0) {
            throw std::logic_error("logical KV descriptors exhausted before physical capacity");
        }
        DeviceKVPageLease lease   = physical_->materialize_one(reservation);
        const std::uint32_t index = free_[--free_count_];
        Page& page                = pages_[index];
        page.replica.emplace(std::move(lease));
        page.content_epoch     = next_epoch(page.content_epoch);
        page.committed_columns = 0;
        page.references        = 1;
        page.write_protected   = false;
        return LogicalKVPageHandle(this, index, page.generation);
    }

    [[nodiscard]] bool valid(LogicalKVPageHandle handle) const noexcept {
        return handle.owner_ == this && handle.index_ < pages_.size() &&
               pages_[handle.index_].replica.has_value() &&
               pages_[handle.index_].generation == handle.generation_;
    }

    [[nodiscard]] DeviceKVPageHandle physical(LogicalKVPageHandle handle) const {
        return require(handle).replica->handle();
    }

    [[nodiscard]] std::uint64_t content_epoch(LogicalKVPageHandle handle) const {
        return require(handle).content_epoch;
    }

    [[nodiscard]] std::uint32_t committed_columns(LogicalKVPageHandle handle) const {
        return require(handle).committed_columns;
    }

    void commit_coverage(LogicalKVPageHandle handle, std::uint32_t columns) {
        Page& page = require(handle);
        if (columns < page.committed_columns ||
            columns > static_cast<std::uint32_t>(kPagedKVPageSize)) {
            throw std::invalid_argument("logical KV committed coverage is not monotonic");
        }
        page.committed_columns = columns;
        page.write_protected   = columns != 0;
    }

    void destructive_truncate(LogicalKVPageHandle handle, std::uint32_t columns) {
        Page& page = require(handle);
        if (columns > page.committed_columns) {
            throw std::invalid_argument("logical KV truncate extends committed coverage");
        }
        if (columns != page.committed_columns) {
            page.committed_columns = columns;
            page.content_epoch     = next_epoch(page.content_epoch);
            page.write_protected   = columns != 0;
        }
    }

    void dematerialize(LogicalKVPageHandle handle, DeviceKVPageReservation& reservation) {
        Page& page = require(handle);
        if (page.references != 1 || !page.replica) {
            throw std::logic_error("logical KV page is shared or has no Device replica");
        }
        physical_->dematerialize_one(reservation, std::move(*page.replica));
        page.replica.reset();
        release_descriptor(handle, page);
    }

    [[nodiscard]] bool release(LogicalKVPageHandle handle) noexcept {
        if (!valid(handle)) { return false; }
        Page& page = pages_[handle.index_];
        if (page.references != 1) { return false; }
        page.replica.reset();
        release_descriptor(handle, page);
        return true;
    }

private:
    struct Page {
        std::uint32_t generation        = 1;
        std::uint64_t content_epoch     = 0;
        std::uint32_t committed_columns = 0;
        std::uint32_t references        = 0;
        bool write_protected            = false;
        std::optional<DeviceKVPageLease> replica;
    };

    [[nodiscard]] static std::uint64_t next_epoch(std::uint64_t epoch) noexcept {
        ++epoch;
        return epoch == 0 ? 1 : epoch;
    }

    [[nodiscard]] Page& require(LogicalKVPageHandle handle) {
        if (!valid(handle)) { throw std::invalid_argument("logical KV page handle is stale"); }
        return pages_[handle.index_];
    }

    [[nodiscard]] const Page& require(LogicalKVPageHandle handle) const {
        if (!valid(handle)) { throw std::invalid_argument("logical KV page handle is stale"); }
        return pages_[handle.index_];
    }

    void release_descriptor(LogicalKVPageHandle handle, Page& page) noexcept {
        page.committed_columns = 0;
        page.references        = 0;
        page.write_protected   = false;
        if (++page.generation == 0) { ++page.generation; }
        free_[free_count_++] = handle.index_;
    }

    DeviceKVPagePool* physical_ = nullptr;
    std::vector<Page> pages_;
    std::vector<std::uint32_t> free_;
    std::uint32_t free_count_ = 0;
};

class KVAddressSpaceStore {
public:
    KVAddressSpaceStore(LogicalKVPageStore& pages, KVExecutionTablePool& tables,
                        std::uint32_t address_capacity, std::uint32_t page_capacity)
        : pages_(&pages), tables_(&tables), page_capacity_(page_capacity),
          addresses_(address_capacity), free_(address_capacity),
          memberships_(checked_membership_cells(address_capacity, page_capacity)),
          free_count_(address_capacity) {
        if (address_capacity == 0 || page_capacity == 0 ||
            page_capacity != tables.logical_page_capacity()) {
            throw std::invalid_argument("KV address-space geometry is invalid");
        }
        for (std::uint32_t index = 0; index < address_capacity; ++index) {
            free_[index] = address_capacity - 1U - index;
        }
    }

    KVAddressSpaceStore(const KVAddressSpaceStore&)            = delete;
    KVAddressSpaceStore& operator=(const KVAddressSpaceStore&) = delete;
    KVAddressSpaceStore(KVAddressSpaceStore&&)                 = delete;
    KVAddressSpaceStore& operator=(KVAddressSpaceStore&&)      = delete;

    [[nodiscard]] std::uint32_t capacity() const noexcept {
        return static_cast<std::uint32_t>(addresses_.size());
    }

    [[nodiscard]] std::uint32_t occupied() const noexcept { return capacity() - free_count_; }

    [[nodiscard]] std::optional<KVAddressSpaceHandle> create_active(std::uint32_t entitlement,
                                                                    std::int32_t execution_row) {
        if (entitlement == 0 || entitlement > page_capacity_) { return std::nullopt; }
        std::optional<KVAddressSpaceHandle> handle = create_inactive();
        if (!handle) { return std::nullopt; }
        try {
            activate(*handle, entitlement, execution_row);
            return handle;
        } catch (...) {
            (void)release(*handle);
            throw;
        }
    }

    [[nodiscard]] std::optional<KVAddressSpaceHandle> create_inactive() noexcept {
        if (free_count_ == 0) { return std::nullopt; }
        const std::uint32_t index = free_[--free_count_];
        Address& address          = addresses_[index];
        if (address.occupied) {
            free_[free_count_++] = index;
            return std::nullopt;
        }
        address.occupied = true;
        return KVAddressSpaceHandle(this, index, address.generation);
    }

    [[nodiscard]] bool valid(KVAddressSpaceHandle handle) const noexcept {
        return handle.owner_ == this && handle.index_ < addresses_.size() &&
               addresses_[handle.index_].occupied &&
               addresses_[handle.index_].generation == handle.generation_;
    }

    void activate(KVAddressSpaceHandle handle, std::uint32_t entitlement,
                  std::int32_t execution_row) {
        Address& address = require(handle);
        if (entitlement < address.page_count) {
            throw std::logic_error("KV address space is not activatable");
        }
        auto reservation = prepare_activation(handle, entitlement, execution_row);
        commit_activation(std::move(reservation));
    }

    [[nodiscard]] KVActivationReservation prepare_activation(KVAddressSpaceHandle handle,
                                                             std::uint32_t entitlement,
                                                             std::int32_t execution_row) {
        Address& address = require(handle);
        if (address.active || address.row || address.reservation.valid() || entitlement == 0 ||
            entitlement > page_capacity_) {
            throw std::logic_error("KV address space is not reservable for activation");
        }
        DeviceKVPageReservation reservation = pages_->physical_pool().make_empty_reservation();
        const std::uint32_t additional =
            entitlement > address.page_count ? entitlement - address.page_count : 0U;
        pages_->physical_pool().resize_reservation(reservation, additional);
        KVExecutionRowLease row = tables_->acquire(execution_row);
        return KVActivationReservation(*this, handle, entitlement, std::move(reservation),
                                       std::move(row));
    }

    void commit_activation(KVActivationReservation&& activation, cudaStream_t stream = nullptr) {
        if (activation.owner_ != this || !activation.row_ || !valid(activation.address_)) {
            throw std::logic_error("KV activation reservation is stale");
        }
        Address& address = require(activation.address_);
        if (address.active || address.row || address.reservation.valid() ||
            activation.requested_entitlement_ == 0 ||
            activation.requested_entitlement_ > page_capacity_) {
            throw std::logic_error("KV activation destination changed after reservation");
        }
        const std::uint32_t expected = activation.requested_entitlement_ > address.page_count
                                           ? activation.requested_entitlement_ - address.page_count
                                           : 0U;
        if (!activation.page_reservation_.belongs_to(pages_->physical_pool()) ||
            activation.page_reservation_.pages() != expected) {
            throw std::logic_error("KV activation capacity reservation changed");
        }
        address.reservation = std::move(activation.page_reservation_);
        address.row.emplace(std::move(*activation.row_));
        activation.row_.reset();
        address.active    = true;
        activation.owner_ = nullptr;
        publish_membership(address, stream);
    }

    void deactivate(KVAddressSpaceHandle handle) {
        Address& address = require_active(handle);
        address.row.reset();
        address.reservation.release();
        address.active = false;
    }

    void resize_entitlement(KVAddressSpaceHandle handle, std::uint32_t entitlement) {
        Address& address = require_active(handle);
        if (entitlement < address.page_count || entitlement > page_capacity_) {
            throw std::invalid_argument("KV entitlement is smaller than mapped pages");
        }
        pages_->physical_pool().resize_reservation(address.reservation,
                                                   entitlement - address.page_count);
    }

    void release_growth_entitlement(KVAddressSpaceHandle handle) {
        Address& address = require_active(handle);
        pages_->physical_pool().resize_reservation(address.reservation, 0);
    }

    void materialize_to_tokens(KVAddressSpaceHandle handle, std::uint32_t tokens,
                               cudaStream_t stream = nullptr) {
        Address& address           = require_active(handle);
        const std::uint32_t target = pages_for_tokens(tokens);
        if (target < address.page_count || target > entitlement(address)) {
            throw std::invalid_argument("KV materialization exceeds active entitlement");
        }
        while (address.page_count < target) {
            LogicalKVPageHandle page                = pages_->materialize(address.reservation);
            membership(address, address.page_count) = page;
            const std::array physical{pages_->physical(page)};
            tables_->publish(address.row->handle(), address.page_count, physical, stream);
            ++address.page_count;
        }
    }

    void commit_frontier(KVAddressSpaceHandle handle, std::uint32_t frontier) {
        Address& address = require_active(handle);
        if (frontier < address.committed_frontier ||
            pages_for_tokens(frontier) > address.page_count) {
            throw std::invalid_argument("KV committed frontier is invalid");
        }
        address.committed_frontier = frontier;
        for (std::uint32_t page = 0; page < address.page_count; ++page) {
            const std::uint32_t begin = page * static_cast<std::uint32_t>(kPagedKVPageSize);
            const std::uint32_t columns =
                frontier <= begin
                    ? 0U
                    : std::min(static_cast<std::uint32_t>(kPagedKVPageSize), frontier - begin);
            if (columns > pages_->committed_columns(membership(address, page))) {
                pages_->commit_coverage(membership(address, page), columns);
            }
        }
    }

    void destructive_truncate(KVAddressSpaceHandle handle, std::uint32_t frontier) {
        Address& address = require_active(handle);
        if (frontier > address.committed_frontier) {
            throw std::invalid_argument("KV destructive truncate extends the frontier");
        }
        const std::uint32_t target = pages_for_tokens(frontier);
        while (address.page_count > target) {
            const std::uint32_t index  = --address.page_count;
            LogicalKVPageHandle page   = membership(address, index);
            membership(address, index) = {};
            pages_->dematerialize(page, address.reservation);
        }
        if (target != 0) {
            const std::uint32_t columns =
                frontier - (target - 1U) * static_cast<std::uint32_t>(kPagedKVPageSize);
            pages_->destructive_truncate(membership(address, target - 1U), columns);
        }
        address.committed_frontier = frontier;
    }

    void set_checkpoint_requirements(KVAddressSpaceHandle handle, std::uint32_t endpoint,
                                     std::optional<std::uint32_t> rewrite) {
        Address& address = require(handle);
        if (endpoint > address.committed_frontier ||
            (rewrite && *rewrite > address.committed_frontier)) {
            throw std::invalid_argument("KV checkpoint requirement exceeds committed frontier");
        }
        address.endpoint_requirement = endpoint;
        address.rewrite_requirement  = rewrite;
    }

    [[nodiscard]] std::uint32_t mapped_pages(KVAddressSpaceHandle handle) const {
        return require(handle).page_count;
    }

    [[nodiscard]] std::uint32_t entitlement(KVAddressSpaceHandle handle) const {
        return entitlement(require(handle));
    }

    [[nodiscard]] std::uint32_t committed_frontier(KVAddressSpaceHandle handle) const {
        return require(handle).committed_frontier;
    }

    [[nodiscard]] std::int32_t bound_row(KVAddressSpaceHandle handle) const noexcept {
        if (!valid(handle) || !addresses_[handle.index_].row) { return -1; }
        return addresses_[handle.index_].row->row_index();
    }

    [[nodiscard]] bool active(KVAddressSpaceHandle handle) const noexcept {
        return valid(handle) && addresses_[handle.index_].active;
    }

    [[nodiscard]] const KVExecutionRowLease& execution_row(KVAddressSpaceHandle handle) const {
        const Address& address = require(handle);
        if (!address.active || !address.row) {
            throw std::logic_error("KV address space has no execution row");
        }
        return *address.row;
    }

    [[nodiscard]] DeviceKVPageHandle physical_page(KVAddressSpaceHandle handle,
                                                   std::uint32_t logical_page) const {
        const Address& address = require(handle);
        if (logical_page >= address.page_count) {
            throw std::out_of_range("KV logical page is outside the address space");
        }
        return pages_->physical(membership(address, logical_page));
    }

    [[nodiscard]] std::uint64_t content_epoch(KVAddressSpaceHandle handle,
                                              std::uint32_t logical_page) const {
        const Address& address = require(handle);
        if (logical_page >= address.page_count) {
            throw std::out_of_range("KV logical page is outside the address space");
        }
        return pages_->content_epoch(membership(address, logical_page));
    }

    [[nodiscard]] bool release(KVAddressSpaceHandle handle) noexcept {
        if (!valid(handle)) { return false; }
        Address& address = addresses_[handle.index_];
        if (address.active || address.row || address.reservation.valid()) { return false; }
        for (std::uint32_t page = 0; page < address.page_count; ++page) {
            if (!pages_->release(membership(address, page))) { return false; }
            membership(address, page) = {};
        }
        const std::uint32_t index      = handle.index_;
        const std::uint32_t generation = next_generation(address.generation);
        address                        = Address{};
        address.generation             = generation;
        free_[free_count_++]           = index;
        return true;
    }

private:
    struct Address {
        std::uint32_t generation           = 1;
        std::uint32_t page_count           = 0;
        std::uint32_t committed_frontier   = 0;
        std::uint32_t endpoint_requirement = 0;
        std::optional<std::uint32_t> rewrite_requirement;
        DeviceKVPageReservation reservation;
        std::optional<KVExecutionRowLease> row;
        bool occupied = false;
        bool active   = false;
    };

    [[nodiscard]] static std::size_t checked_membership_cells(std::uint32_t addresses,
                                                              std::uint32_t pages) {
        if (pages != 0 && addresses > std::numeric_limits<std::size_t>::max() / pages) {
            throw std::overflow_error("KV ordered-membership capacity overflow");
        }
        return static_cast<std::size_t>(addresses) * pages;
    }

    [[nodiscard]] static std::uint32_t pages_for_tokens(std::uint32_t tokens) noexcept {
        return tokens == 0 ? 0U : 1U + (tokens - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
    }

    [[nodiscard]] static std::uint32_t next_generation(std::uint32_t generation) noexcept {
        ++generation;
        return generation == 0 ? 1 : generation;
    }

    [[nodiscard]] Address& require(KVAddressSpaceHandle handle) {
        if (!valid(handle)) { throw std::invalid_argument("KV address-space handle is stale"); }
        return addresses_[handle.index_];
    }

    [[nodiscard]] const Address& require(KVAddressSpaceHandle handle) const {
        if (!valid(handle)) { throw std::invalid_argument("KV address-space handle is stale"); }
        return addresses_[handle.index_];
    }

    [[nodiscard]] Address& require_active(KVAddressSpaceHandle handle) {
        Address& address = require(handle);
        if (!address.active || !address.row || !address.reservation.valid()) {
            throw std::logic_error("KV address space is not active");
        }
        return address;
    }

    [[nodiscard]] std::uint32_t entitlement(const Address& address) const noexcept {
        return address.page_count +
               (address.reservation.valid() ? address.reservation.pages() : 0U);
    }

    [[nodiscard]] LogicalKVPageHandle& membership(Address& address, std::uint32_t page) noexcept {
        const std::size_t index = static_cast<std::size_t>(&address - addresses_.data());
        return memberships_[index * page_capacity_ + page];
    }

    [[nodiscard]] const LogicalKVPageHandle& membership(const Address& address,
                                                        std::uint32_t page) const noexcept {
        const std::size_t index = static_cast<std::size_t>(&address - addresses_.data());
        return memberships_[index * page_capacity_ + page];
    }

    void publish_membership(const Address& address, cudaStream_t stream = nullptr) {
        if (!address.row) { throw std::logic_error("KV address space has no execution row"); }
        for (std::uint32_t page = 0; page < address.page_count; ++page) {
            const std::array physical{pages_->physical(membership(address, page))};
            tables_->publish(address.row->handle(), page, physical, stream);
        }
    }

    LogicalKVPageStore* pages_    = nullptr;
    KVExecutionTablePool* tables_ = nullptr;
    std::uint32_t page_capacity_  = 0;
    std::vector<Address> addresses_;
    std::vector<std::uint32_t> free_;
    std::vector<LogicalKVPageHandle> memberships_;
    std::uint32_t free_count_ = 0;
};

} // namespace ninfer::targets::qwen3_6::detail
