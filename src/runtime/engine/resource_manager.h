#pragma once

#include "ninfer/types.h"
#include "runtime/contract/types.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>

namespace ninfer::runtime {

inline constexpr std::uint32_t kInvalidCatalogSlot = 2U * kMaximumConcurrency;

enum class LogicalLaneState : std::uint8_t {
    Free,
    Active,
};

struct LogicalLaneSnapshot {
    LogicalLaneState state = LogicalLaneState::Free;
    DeviceResources resources;
};

[[nodiscard]] constexpr bool resources_fit(DeviceResources value,
                                           DeviceResources capacity) noexcept {
    return value.active_lanes <= capacity.active_lanes &&
           value.state_slots <= capacity.state_slots &&
           value.main_kv_pages <= capacity.main_kv_pages &&
           value.backend_kv_pages <= capacity.backend_kv_pages;
}

[[nodiscard]] constexpr bool continuation_within_active(DeviceResources continuation,
                                                        DeviceResources active) noexcept {
    return continuation.active_lanes == 0 && continuation.state_slots <= active.state_slots &&
           continuation.main_kv_pages <= active.main_kv_pages &&
           continuation.backend_kv_pages <= active.backend_kv_pages;
}

namespace detail {

[[nodiscard]] constexpr bool add_resources(DeviceResources left, DeviceResources right,
                                           DeviceResources& out) noexcept {
    const std::uint64_t active = static_cast<std::uint64_t>(left.active_lanes) + right.active_lanes;
    const std::uint64_t state  = static_cast<std::uint64_t>(left.state_slots) + right.state_slots;
    const std::uint64_t main = static_cast<std::uint64_t>(left.main_kv_pages) + right.main_kv_pages;
    const std::uint64_t backend =
        static_cast<std::uint64_t>(left.backend_kv_pages) + right.backend_kv_pages;
    if (active > std::numeric_limits<std::uint32_t>::max() ||
        state > std::numeric_limits<std::uint32_t>::max() ||
        main > std::numeric_limits<std::uint32_t>::max() ||
        backend > std::numeric_limits<std::uint32_t>::max()) {
        return false;
    }
    out = DeviceResources{
        .active_lanes     = static_cast<std::uint32_t>(active),
        .state_slots      = static_cast<std::uint32_t>(state),
        .main_kv_pages    = static_cast<std::uint32_t>(main),
        .backend_kv_pages = static_cast<std::uint32_t>(backend),
    };
    return true;
}

[[nodiscard]] constexpr bool subtract_resources(DeviceResources value, DeviceResources removed,
                                                DeviceResources& out) noexcept {
    if (removed.active_lanes > value.active_lanes || removed.state_slots > value.state_slots ||
        removed.main_kv_pages > value.main_kv_pages ||
        removed.backend_kv_pages > value.backend_kv_pages) {
        return false;
    }
    out = DeviceResources{
        .active_lanes     = value.active_lanes - removed.active_lanes,
        .state_slots      = value.state_slots - removed.state_slots,
        .main_kv_pages    = value.main_kv_pages - removed.main_kv_pages,
        .backend_kv_pages = value.backend_kv_pages - removed.backend_kv_pages,
    };
    return true;
}

[[nodiscard]] constexpr bool transition_fits(DeviceResources used, DeviceResources removed,
                                             DeviceResources added,
                                             DeviceResources capacity) noexcept {
    DeviceResources remainder;
    if (!subtract_resources(used, removed, remainder)) { return false; }
    return static_cast<std::uint64_t>(remainder.active_lanes) + added.active_lanes <=
               capacity.active_lanes &&
           static_cast<std::uint64_t>(remainder.state_slots) + added.state_slots <=
               capacity.state_slots &&
           static_cast<std::uint64_t>(remainder.main_kv_pages) + added.main_kv_pages <=
               capacity.main_kv_pages &&
           static_cast<std::uint64_t>(remainder.backend_kv_pages) + added.backend_kv_pages <=
               capacity.backend_kv_pages;
}

[[nodiscard]] constexpr bool valid_demand(const ResourceDemand& demand,
                                          DeviceResources source) noexcept {
    if (demand.active_entitlement.active_lanes != 1 || source.active_lanes != 0 ||
        demand.source_conversions.active_lanes != 0) {
        return false;
    }
    DeviceResources unused;
    if (!subtract_resources(source, demand.source_conversions, unused)) { return false; }
    DeviceResources composed;
    return add_resources(demand.prepublish_additional, demand.source_conversions, composed) &&
           composed == demand.active_entitlement;
}

} // namespace detail

class ResourceLedger {
public:
    ResourceLedger(DeviceResources capacity, std::uint32_t lane_count)
        : capacity_(capacity), lane_count_(lane_count), catalog_count_(2U * lane_count) {
        if (lane_count == 0 || lane_count > kMaximumConcurrency ||
            capacity.active_lanes != lane_count) {
            throw std::invalid_argument("resource ledger capacity is invalid");
        }
    }

    [[nodiscard]] DeviceResources capacity() const noexcept { return capacity_; }

    [[nodiscard]] DeviceResources used() const {
        DeviceResources out;
        for (std::uint32_t lane = 0; lane < lane_count_; ++lane) {
            DeviceResources sum;
            if (!detail::add_resources(out, lanes_[lane].resources, sum)) {
                throw std::overflow_error("active Device resource ledger overflow");
            }
            out = sum;
        }
        for (std::uint32_t slot = 0; slot < catalog_count_; ++slot) {
            if (!catalog_occupied_[slot]) { continue; }
            DeviceResources sum;
            if (!detail::add_resources(out, catalog_resources_[slot], sum)) {
                throw std::overflow_error("catalog Device resource ledger overflow");
            }
            out = sum;
        }
        return out;
    }

    [[nodiscard]] const LogicalLaneSnapshot& lane(LaneId id) const noexcept {
        static const LogicalLaneSnapshot invalid;
        return id.value < lane_count_ ? lanes_[id.value] : invalid;
    }

    [[nodiscard]] bool catalogued(std::uint32_t slot) const noexcept {
        return slot < catalog_count_ && catalog_occupied_[slot];
    }

    [[nodiscard]] DeviceResources catalog_resources(std::uint32_t slot) const noexcept {
        return catalogued(slot) ? catalog_resources_[slot] : DeviceResources{};
    }

    [[nodiscard]] bool adopt_active(LaneId id, DeviceResources resources) {
        if (id.value >= lane_count_ || lanes_[id.value].state != LogicalLaneState::Free ||
            resources.active_lanes != 1 ||
            !detail::transition_fits(used(), {}, resources, capacity_)) {
            return false;
        }
        lanes_[id.value] = {LogicalLaneState::Active, resources};
        return true;
    }

    [[nodiscard]] bool adopt_materialization(LaneId lane, std::uint32_t source_slot,
                                             DeviceResources source, DeviceResources active) {
        if (lane.value >= lane_count_ || lanes_[lane.value].state != LogicalLaneState::Free ||
            active.active_lanes != 1) {
            return false;
        }
        if (source_slot != kInvalidCatalogSlot &&
            (source_slot >= catalog_count_ || !catalog_occupied_[source_slot] ||
             catalog_resources_[source_slot] != source)) {
            return false;
        }
        if (source_slot == kInvalidCatalogSlot && source != DeviceResources{}) { return false; }
        if (!detail::transition_fits(used(), source, active, capacity_)) { return false; }

        if (source_slot != kInvalidCatalogSlot) {
            catalog_resources_[source_slot] = {};
            catalog_occupied_[source_slot]  = false;
        }
        lanes_[lane.value] = {LogicalLaneState::Active, active};
        return true;
    }

    [[nodiscard]] bool release_active(LaneId id, DeviceResources released) noexcept {
        if (id.value >= lane_count_ || lanes_[id.value].state != LogicalLaneState::Active ||
            lanes_[id.value].resources != released) {
            return false;
        }
        lanes_[id.value] = {};
        return true;
    }

    [[nodiscard]] bool adopt_catalog(std::uint32_t slot, DeviceResources resources) {
        if (slot >= catalog_count_ || catalog_occupied_[slot] || resources.active_lanes != 0 ||
            !detail::transition_fits(used(), {}, resources, capacity_)) {
            return false;
        }
        catalog_resources_[slot] = resources;
        catalog_occupied_[slot]  = true;
        return true;
    }

    [[nodiscard]] bool release_catalog(std::uint32_t slot, DeviceResources released) noexcept {
        if (slot >= catalog_count_ || !catalog_occupied_[slot] ||
            catalog_resources_[slot] != released) {
            return false;
        }
        catalog_resources_[slot] = {};
        catalog_occupied_[slot]  = false;
        return true;
    }

    [[nodiscard]] bool publish_active_to_catalog(LaneId lane, std::uint32_t slot,
                                                 DeviceResources active,
                                                 DeviceResources continuation) {
        if (lane.value >= lane_count_ || slot >= catalog_count_ || catalog_occupied_[slot] ||
            lanes_[lane.value].state != LogicalLaneState::Active ||
            lanes_[lane.value].resources != active ||
            !continuation_within_active(continuation, active) ||
            !detail::transition_fits(used(), active, continuation, capacity_)) {
            return false;
        }
        lanes_[lane.value]       = {};
        catalog_resources_[slot] = continuation;
        catalog_occupied_[slot]  = true;
        return true;
    }

    void clear() noexcept {
        for (std::uint32_t lane = 0; lane < lane_count_; ++lane) { lanes_[lane] = {}; }
        for (std::uint32_t slot = 0; slot < catalog_count_; ++slot) {
            catalog_resources_[slot] = {};
            catalog_occupied_[slot]  = false;
        }
    }

private:
    DeviceResources capacity_;
    std::uint32_t lane_count_    = 0;
    std::uint32_t catalog_count_ = 0;
    std::array<LogicalLaneSnapshot, kMaximumConcurrency> lanes_{};
    std::array<DeviceResources, 2 * kMaximumConcurrency> catalog_resources_{};
    std::array<bool, 2 * kMaximumConcurrency> catalog_occupied_{};
};

struct CatalogResourceDescriptor {
    std::uint32_t slot = kInvalidCatalogSlot;
    ContinuationId id;
    DeviceResources resources;
    std::uint64_t rebuild_work_quanta = 0;
};

struct ResourceCandidateDescriptor {
    ResourceDemand demand;
    DeviceResources source_resources;
    std::uint32_t source_slot          = kInvalidCatalogSlot;
    std::uint32_t reused_prompt_tokens = 0;
    std::uint64_t service_work_quanta  = 0;
    CheckpointKind checkpoint_kind     = CheckpointKind::SessionEndpoint;
    bool publication_slot_available    = false;
};

struct ResourceCandidateSelection {
    bool found                  = false;
    std::size_t candidate_index = 0;
    std::array<std::uint32_t, 2 * kMaximumConcurrency> evictions{};
    std::array<ContinuationId, 2 * kMaximumConcurrency> eviction_ids{};
    std::size_t eviction_count      = 0;
    std::uint64_t total_work_quanta = 0;
};

namespace detail {

[[nodiscard]] inline bool feasible_candidate(DeviceResources used,
                                             const ResourceCandidateDescriptor& candidate,
                                             std::span<const CatalogResourceDescriptor> victims,
                                             DeviceResources capacity) noexcept {
    if (!valid_demand(candidate.demand, candidate.source_resources)) { return false; }
    DeviceResources victim_resources;
    for (const CatalogResourceDescriptor& victim : victims) {
        DeviceResources sum;
        if (!add_resources(victim_resources, victim.resources, sum)) { return false; }
        victim_resources = sum;
    }
    DeviceResources peak_base;
    if (!subtract_resources(used, victim_resources, peak_base) ||
        !transition_fits(peak_base, {}, candidate.demand.prepublish_additional, capacity)) {
        return false;
    }
    DeviceResources steady_removals;
    return add_resources(victim_resources, candidate.source_resources, steady_removals) &&
           transition_fits(used, steady_removals, candidate.demand.active_entitlement, capacity);
}

[[nodiscard]] inline bool
lexicographically_smaller_ids(const ResourceCandidateSelection& candidate,
                              const ResourceCandidateSelection& selected) noexcept {
    const std::size_t common = std::min(candidate.eviction_count, selected.eviction_count);
    for (std::size_t index = 0; index < common; ++index) {
        if (candidate.eviction_ids[index] == selected.eviction_ids[index]) { continue; }
        return candidate.eviction_ids[index] < selected.eviction_ids[index];
    }
    return candidate.eviction_count < selected.eviction_count;
}

[[nodiscard]] inline bool
better_selection(const ResourceCandidateSelection& candidate,
                 const ResourceCandidateDescriptor& candidate_desc,
                 const ResourceCandidateSelection& selected,
                 const ResourceCandidateDescriptor& selected_desc) noexcept {
    if (candidate.total_work_quanta != selected.total_work_quanta) {
        return candidate.total_work_quanta < selected.total_work_quanta;
    }
    if (candidate_desc.reused_prompt_tokens != selected_desc.reused_prompt_tokens) {
        return candidate_desc.reused_prompt_tokens > selected_desc.reused_prompt_tokens;
    }
    if (candidate.eviction_count != selected.eviction_count) {
        return candidate.eviction_count < selected.eviction_count;
    }
    if (!std::equal(candidate.eviction_ids.begin(),
                    candidate.eviction_ids.begin() +
                        static_cast<std::ptrdiff_t>(candidate.eviction_count),
                    selected.eviction_ids.begin())) {
        return lexicographically_smaller_ids(candidate, selected);
    }
    const bool candidate_endpoint =
        candidate_desc.checkpoint_kind == CheckpointKind::SessionEndpoint;
    const bool selected_endpoint = selected_desc.checkpoint_kind == CheckpointKind::SessionEndpoint;
    if (candidate_endpoint != selected_endpoint) { return candidate_endpoint; }
    return candidate.candidate_index < selected.candidate_index;
}

} // namespace detail

[[nodiscard]] inline ResourceCandidateSelection
select_resource_candidate(std::span<const ResourceCandidateDescriptor> candidates,
                          std::span<const CatalogResourceDescriptor> catalog, DeviceResources used,
                          DeviceResources capacity) noexcept {
    if (catalog.size() > 2U * kMaximumConcurrency) { return {}; }

    for (int victim_pass = 0; victim_pass < 2; ++victim_pass) {
        ResourceCandidateSelection selected;
        const std::uint64_t subset_count = 1ULL << catalog.size();
        for (std::size_t candidate_index = 0; candidate_index < candidates.size();
             ++candidate_index) {
            const ResourceCandidateDescriptor& descriptor = candidates[candidate_index];
            for (std::uint64_t mask = 0; mask < subset_count; ++mask) {
                if ((mask != 0) != (victim_pass != 0)) { continue; }

                std::array<CatalogResourceDescriptor, 2 * kMaximumConcurrency> victims{};
                std::size_t victim_count  = 0;
                bool consumes_source      = false;
                std::uint64_t victim_work = 0;
                for (std::size_t bit = 0; bit < catalog.size(); ++bit) {
                    if ((mask & (1ULL << bit)) == 0) { continue; }
                    if (catalog[bit].slot == descriptor.source_slot) {
                        consumes_source = true;
                        break;
                    }
                    victims[victim_count++] = catalog[bit];
                    victim_work = victim_work > std::numeric_limits<std::uint64_t>::max() -
                                                    catalog[bit].rebuild_work_quanta
                                      ? std::numeric_limits<std::uint64_t>::max()
                                      : victim_work + catalog[bit].rebuild_work_quanta;
                }
                if (consumes_source ||
                    (!descriptor.publication_slot_available && victim_count == 0) ||
                    !detail::feasible_candidate(
                        used, descriptor,
                        std::span<const CatalogResourceDescriptor>(victims.data(), victim_count),
                        capacity)) {
                    continue;
                }

                ResourceCandidateSelection current;
                current.found           = true;
                current.candidate_index = candidate_index;
                current.eviction_count  = victim_count;
                current.total_work_quanta =
                    descriptor.service_work_quanta >
                            std::numeric_limits<std::uint64_t>::max() - victim_work
                        ? std::numeric_limits<std::uint64_t>::max()
                        : descriptor.service_work_quanta + victim_work;
                std::sort(
                    victims.begin(), victims.begin() + static_cast<std::ptrdiff_t>(victim_count),
                    [](const CatalogResourceDescriptor& left,
                       const CatalogResourceDescriptor& right) { return left.id < right.id; });
                for (std::size_t index = 0; index < victim_count; ++index) {
                    current.evictions[index]    = victims[index].slot;
                    current.eviction_ids[index] = victims[index].id;
                }
                if (!selected.found ||
                    detail::better_selection(current, descriptor, selected,
                                             candidates[selected.candidate_index])) {
                    selected = current;
                }
            }
        }
        if (selected.found) { return selected; }
    }
    return {};
}

template <class Package>
class ResourceManager {
public:
    using Program                      = typename Package::Program;
    using PreparedPrompt               = typename Package::PreparedPrompt;
    using RequestBasePlan              = typename Package::RequestBasePlan;
    using AdmissionPlan                = typename Package::AdmissionPlan;
    using SequenceHandle               = typename Package::SequenceHandle;
    using ContinuationHandle           = typename Package::ContinuationHandle;
    using MaterializationTicket        = typename Package::MaterializationTicket;
    using ProgramMaterializationResult = typename Package::MaterializationResult;
    using StartResult                  = typename Package::StartResult;
    using FinishResult                 = typename Package::FinishResult;
    using AbortResult                  = typename Package::AbortResult;

    enum class CatalogState : std::uint8_t {
        Vacant,
        Catalogued,
        Claimed,
        ReservedForActive,
    };

    class Choice {
    public:
        Choice(Choice&&) noexcept        = default;
        Choice& operator=(Choice&&)      = delete;
        Choice(const Choice&)            = delete;
        Choice& operator=(const Choice&) = delete;

        [[nodiscard]] const RequestPlanSummary& summary() const noexcept {
            return plan_->summary();
        }

        [[nodiscard]] LaneId destination() const noexcept { return destination_; }

    private:
        Choice(LaneId destination, AdmissionPlan&& plan) : destination_(destination) {
            plan_.emplace(std::move(plan));
        }

        LaneId destination_{};
        std::optional<AdmissionPlan> plan_;
        ResourceDemand demand_;
        DeviceResources source_resources_;
        std::uint32_t source_slot_      = kInvalidCatalogSlot;
        std::uint64_t source_id_        = 0;
        std::uint64_t source_revision_  = 0;
        std::uint32_t publication_slot_ = kInvalidCatalogSlot;
        std::array<std::uint32_t, 2 * kMaximumConcurrency> evictions_{};
        std::array<std::uint64_t, 2 * kMaximumConcurrency> eviction_ids_{};
        std::array<std::uint64_t, 2 * kMaximumConcurrency> eviction_revisions_{};
        std::size_t eviction_count_ = 0;

        friend class ResourceManager;
    };

    class PublishedActivation {
    public:
        PublishedActivation(PublishedActivation&& other) noexcept
            : owner_(std::exchange(other.owner_, nullptr)), result_(std::move(other.result_)),
              destination_(other.destination_), demand_(other.demand_),
              source_resources_(other.source_resources_), source_slot_(other.source_slot_),
              publication_slot_(other.publication_slot_), continuation_id_(other.continuation_id_) {
        }

        PublishedActivation& operator=(PublishedActivation&&)      = delete;
        PublishedActivation(const PublishedActivation&)            = delete;
        PublishedActivation& operator=(const PublishedActivation&) = delete;

        [[nodiscard]] const SequenceHandle& sequence() const {
            if (!result_) { throw std::logic_error("published activation is empty"); }
            return result_->sequence;
        }

        [[nodiscard]] DeviceResources active_resources() const {
            if (!result_) { throw std::logic_error("published activation is empty"); }
            return result_->active_resources;
        }

    private:
        PublishedActivation(ResourceManager& owner, StartResult&& result, LaneId destination,
                            ResourceDemand demand, DeviceResources source_resources,
                            std::uint32_t source_slot, std::uint32_t publication_slot,
                            std::uint64_t continuation_id)
            : owner_(&owner), result_(std::move(result)), destination_(destination),
              demand_(demand), source_resources_(source_resources), source_slot_(source_slot),
              publication_slot_(publication_slot), continuation_id_(continuation_id) {}

        ResourceManager* owner_ = nullptr;
        std::optional<StartResult> result_;
        LaneId destination_{};
        ResourceDemand demand_;
        DeviceResources source_resources_;
        std::uint32_t source_slot_      = kInvalidCatalogSlot;
        std::uint32_t publication_slot_ = kInvalidCatalogSlot;
        std::uint64_t continuation_id_  = 0;

        friend class ResourceManager;
    };

    struct MaterializationOutcome {
        MaterializationStatus status = MaterializationStatus::Aborted;
        std::optional<PublishedActivation> activation;
    };

    struct Inspection {
        Readiness readiness = Readiness::TemporarilyBlocked;
        std::optional<Choice> choice;
    };

    ResourceManager(DeviceResources capacity, std::uint32_t lane_count)
        : ledger_(capacity, lane_count), lane_count_(lane_count), catalog_count_(2U * lane_count) {}

    [[nodiscard]] Inspection inspect(Program& program, const PreparedPrompt& prompt,
                                     const RequestBasePlan& base) {
        std::optional<LaneId> destination;
        for (std::uint32_t lane = 0; lane < lane_count_; ++lane) {
            if (ledger_.lane(LaneId{lane}).state == LogicalLaneState::Free) {
                destination = LaneId{lane};
                break;
            }
        }
        if (!resources_fit(base.summary().admission, ledger_.capacity())) {
            return {.readiness = Readiness::PermanentlyInfeasible};
        }
        if (!destination) { return {.readiness = Readiness::TemporarilyBlocked}; }

        constexpr std::size_t kCandidateCapacity = 1U + 4U * kMaximumConcurrency;
        std::array<std::optional<AdmissionPlan>, kCandidateCapacity> plans{};
        std::array<ResourceCandidateDescriptor, kCandidateCapacity> candidates{};
        std::array<std::uint32_t, kCandidateCapacity> source_slots{};
        std::size_t candidate_count = 0;
        std::array<CatalogResourceDescriptor, 2 * kMaximumConcurrency> residents{};
        std::size_t resident_count = 0;

        bool vacant_catalog_slot = false;
        for (std::uint32_t slot = 0; slot < catalog_count_; ++slot) {
            const CatalogEntry& entry = catalog_[slot];
            if (entry.state == CatalogState::Vacant) {
                vacant_catalog_slot = true;
                continue;
            }
            if (entry.state != CatalogState::Catalogued || !entry.handle) { continue; }
            residents[resident_count++] = CatalogResourceDescriptor{
                .slot                = slot,
                .id                  = ContinuationId{entry.id},
                .resources           = entry.summary.footprint,
                .rebuild_work_quanta = entry.summary.rebuild_work_quanta,
            };
        }

        std::optional<AdmissionPlan> root =
            program.inspect_admission(prompt, base, *destination, nullptr, std::nullopt);
        if (!root) { throw std::logic_error("Program rejected its root admission assessment"); }
        plans[candidate_count].emplace(std::move(*root));
        candidates[candidate_count] = ResourceCandidateDescriptor{
            .demand                     = plans[candidate_count]->demand(),
            .service_work_quanta        = plans[candidate_count]->summary().service_work_quanta,
            .publication_slot_available = vacant_catalog_slot,
        };
        source_slots[candidate_count++] = kInvalidCatalogSlot;

        for (std::uint32_t slot = 0; slot < catalog_count_; ++slot) {
            const CatalogEntry& entry = catalog_[slot];
            if (entry.state != CatalogState::Catalogued || !entry.handle) { continue; }
            const auto add_checkpoint = [&](CheckpointRef checkpoint) {
                std::optional<AdmissionPlan> assessment = program.inspect_admission(
                    prompt, base, *destination, &*entry.handle, checkpoint);
                if (!assessment) { return; }
                if (candidate_count >= kCandidateCapacity ||
                    assessment->summary().reusable_prompt_tokens == 0) {
                    throw std::logic_error("Program returned an invalid private assessment");
                }
                plans[candidate_count].emplace(std::move(*assessment));
                const RequestPlanSummary& summary = plans[candidate_count]->summary();
                candidates[candidate_count]       = ResourceCandidateDescriptor{
                          .demand                     = plans[candidate_count]->demand(),
                          .source_resources           = entry.summary.footprint,
                          .source_slot                = slot,
                          .reused_prompt_tokens       = summary.reusable_prompt_tokens,
                          .service_work_quanta        = summary.service_work_quanta,
                          .checkpoint_kind            = checkpoint.kind,
                          .publication_slot_available = true,
                };
                source_slots[candidate_count] = slot;
                ++candidate_count;
            };
            add_checkpoint(entry.summary.endpoint);
            if (entry.summary.rewrite) { add_checkpoint(*entry.summary.rewrite); }
        }

        const ResourceCandidateSelection selected = select_resource_candidate(
            std::span<const ResourceCandidateDescriptor>(candidates.data(), candidate_count),
            std::span<const CatalogResourceDescriptor>(residents.data(), resident_count),
            ledger_.used(), ledger_.capacity());
        if (!selected.found) { return {.readiness = Readiness::TemporarilyBlocked}; }

        const std::size_t index = selected.candidate_index;
        Choice choice(*destination, std::move(*plans[index]));
        choice.demand_           = candidates[index].demand;
        choice.source_resources_ = candidates[index].source_resources;
        choice.source_slot_      = source_slots[index];
        if (choice.source_slot_ != kInvalidCatalogSlot) {
            const CatalogEntry& source = catalog_[choice.source_slot_];
            choice.source_id_          = source.id;
            choice.source_revision_    = source.revision;
            choice.publication_slot_   = choice.source_slot_;
        } else {
            for (std::uint32_t slot = 0; slot < catalog_count_; ++slot) {
                if (catalog_[slot].state == CatalogState::Vacant) {
                    choice.publication_slot_ = slot;
                    break;
                }
            }
            if (choice.publication_slot_ == kInvalidCatalogSlot) {
                choice.publication_slot_ =
                    *std::min_element(selected.evictions.begin(),
                                      selected.evictions.begin() +
                                          static_cast<std::ptrdiff_t>(selected.eviction_count));
            }
        }
        choice.eviction_count_ = selected.eviction_count;
        for (std::size_t victim = 0; victim < selected.eviction_count; ++victim) {
            const std::uint32_t slot           = selected.evictions[victim];
            choice.evictions_[victim]          = slot;
            choice.eviction_ids_[victim]       = catalog_[slot].id;
            choice.eviction_revisions_[victim] = catalog_[slot].revision;
        }
        return {.readiness = Readiness::Ready, .choice = std::optional<Choice>(std::move(choice))};
    }

    [[nodiscard]] MaterializationOutcome materialize(Program& program, Choice&& choice,
                                                     PreparedPrompt&& prompt,
                                                     CancellationFlagView cancellation) {
        validate_choice(choice);
        if (cancellation.requested()) { return {.status = MaterializationStatus::Aborted}; }

        std::array<const ContinuationHandle*, 2 * kMaximumConcurrency> victim_handles{};
        for (std::size_t index = 0; index < choice.eviction_count_; ++index) {
            victim_handles[index] = &*catalog_[choice.evictions_[index]].handle;
        }
        const ContinuationHandle* source_handle = choice.source_slot_ == kInvalidCatalogSlot
                                                      ? nullptr
                                                      : &*catalog_[choice.source_slot_].handle;
        MaterializationTicket ticket            = program.reserve_materialization(
            std::move(*choice.plan_), std::move(prompt), source_handle,
            std::span<const ContinuationHandle* const>(victim_handles.data(),
                                                                  choice.eviction_count_));
        choice.plan_.reset();

        const auto mark_claimed = [](CatalogEntry& entry) noexcept {
            entry.state = CatalogState::Claimed;
            if (++entry.revision == 0) { ++entry.revision; }
        };
        const auto restore_catalogued = [](CatalogEntry& entry) noexcept {
            entry.state = CatalogState::Catalogued;
            if (++entry.revision == 0) { ++entry.revision; }
        };
        if (choice.source_slot_ != kInvalidCatalogSlot) {
            mark_claimed(catalog_[choice.source_slot_]);
        }
        for (std::size_t index = 0; index < choice.eviction_count_; ++index) {
            mark_claimed(catalog_[choice.evictions_[index]]);
        }

        if (cancellation.requested()) {
            if (program.abort_materialization(std::move(ticket)) != ConsumeStatus::Consumed) {
                throw std::logic_error("Program did not abort its reserved materialization");
            }
            if (choice.source_slot_ != kInvalidCatalogSlot) {
                restore_catalogued(catalog_[choice.source_slot_]);
            }
            for (std::size_t index = 0; index < choice.eviction_count_; ++index) {
                restore_catalogued(catalog_[choice.evictions_[index]]);
            }
            return {.status = MaterializationStatus::Aborted};
        }

        std::array<DeviceResources, 2 * kMaximumConcurrency> released_victims{};
        for (std::size_t index = 0; index < choice.eviction_count_; ++index) {
            const std::uint32_t slot        = choice.evictions_[index];
            CatalogEntry& entry             = catalog_[slot];
            const DeviceResources resources = entry.summary.footprint;
            auto released =
                program.release_materialization_victim(ticket, std::move(*entry.handle));
            entry.handle.reset();
            if (released.status != ConsumeStatus::Consumed ||
                released.released_resources != resources ||
                !ledger_.release_catalog(slot, resources)) {
                throw std::logic_error("continuation eviction violated the materialization ledger");
            }
            released_victims[index] = resources;
            clear_catalog_entry(entry);
        }

        program.prepare_materialization(ticket);

        std::optional<ContinuationHandle> source;
        if (choice.source_slot_ != kInvalidCatalogSlot) {
            CatalogEntry& entry = catalog_[choice.source_slot_];
            source.emplace(std::move(*entry.handle));
            entry.handle.reset();
        }
        ProgramMaterializationResult result =
            program.publish_materialization(std::move(ticket), std::move(source), cancellation);
        if (result.victim_count != choice.eviction_count_) {
            throw std::logic_error("Program returned an incomplete victim acknowledgement");
        }
        for (std::size_t index = 0; index < choice.eviction_count_; ++index) {
            if (result.released_victims[index] != released_victims[index]) {
                throw std::logic_error("Program victim acknowledgement changed at publication");
            }
        }

        if (result.status == MaterializationStatus::Aborted) {
            if (result.published ||
                result.source.has_value() != (choice.source_slot_ != kInvalidCatalogSlot)) {
                throw std::logic_error("Program returned an invalid aborted materialization");
            }
            if (choice.source_slot_ != kInvalidCatalogSlot) {
                CatalogEntry& entry = catalog_[choice.source_slot_];
                if (entry.state != CatalogState::Claimed || entry.handle) {
                    throw std::logic_error("aborted materialization lost its source claim");
                }
                entry.handle.emplace(std::move(*result.source));
                result.source.reset();
                restore_catalogued(entry);
            }
            return {.status = MaterializationStatus::Aborted};
        }
        if (result.status != MaterializationStatus::Published || !result.published ||
            result.source) {
            throw std::logic_error("Program returned an invalid published materialization");
        }
        StartResult started = std::move(*result.published);
        result.published.reset();
        if (started.active_resources != choice.demand_.active_entitlement ||
            started.resource_delta.removed != choice.source_resources_ ||
            started.resource_delta.added != choice.demand_.active_entitlement) {
            throw std::logic_error("Program publication violated the selected resource delta");
        }
        const std::uint64_t lineage_id = choice.source_slot_ != kInvalidCatalogSlot
                                             ? choice.source_id_
                                             : next_continuation_id_++;
        MaterializationOutcome outcome{.status = MaterializationStatus::Published};
        outcome.activation.emplace(PublishedActivation(
            *this, std::move(started), choice.destination_, choice.demand_,
            choice.source_resources_, choice.source_slot_, choice.publication_slot_, lineage_id));
        return outcome;
    }

    void adopt(PublishedActivation&& activation) {
        if (activation.owner_ != this || !activation.result_ ||
            activation.destination_.value >= lane_count_ ||
            active_[activation.destination_.value].occupied ||
            activation.publication_slot_ >= catalog_count_) {
            throw std::logic_error("published activation token is stale");
        }
        const StartResult& result = *activation.result_;
        if (result.active_resources != activation.demand_.active_entitlement ||
            result.resource_delta.removed != activation.source_resources_ ||
            result.resource_delta.added != activation.demand_.active_entitlement) {
            throw std::logic_error("published activation token has an invalid resource delta");
        }

        CatalogEntry& publication   = catalog_[activation.publication_slot_];
        const CatalogState expected = activation.source_slot_ == kInvalidCatalogSlot
                                          ? CatalogState::Vacant
                                          : CatalogState::Claimed;
        if (publication.state != expected || publication.handle ||
            (activation.source_slot_ != kInvalidCatalogSlot &&
             (activation.source_slot_ != activation.publication_slot_ ||
              publication.id != activation.continuation_id_)) ||
            !ledger_.adopt_materialization(activation.destination_, activation.source_slot_,
                                           activation.source_resources_, result.active_resources)) {
            throw std::logic_error("published activation could not be adopted by the ledger");
        }

        publication.state   = CatalogState::ReservedForActive;
        publication.id      = activation.continuation_id_;
        publication.summary = {};
        if (++publication.revision == 0) { ++publication.revision; }
        active_[activation.destination_.value] = ActiveEntry{
            .occupied         = true,
            .publication_slot = activation.publication_slot_,
            .continuation_id  = activation.continuation_id_,
            .resources        = result.active_resources,
        };
        activation.result_.reset();
        activation.owner_ = nullptr;
    }

    [[nodiscard]] FinishResult finish(Program& program, LaneId lane, SequenceHandle sequence) {
        const DeviceResources active = require_active(lane);
        ActiveEntry& active_entry    = active_[lane.value];
        FinishResult result          = program.finish(sequence);
        if (result.status != ConsumeStatus::Consumed || result.released_resources != active) {
            throw std::logic_error("Runtime finish did not consume the active entitlement");
        }
        CatalogEntry& publication = catalog_[active_entry.publication_slot];
        const bool valid_summary =
            result.summary.footprint == result.resident_resources &&
            result.summary.endpoint.kind == CheckpointKind::SessionEndpoint &&
            result.summary.endpoint.frontier != 0 && result.summary.endpoint.ordinal == 0 &&
            (!result.summary.rewrite ||
             (result.summary.rewrite->kind != CheckpointKind::SessionEndpoint &&
              result.summary.rewrite->frontier != 0 &&
              result.summary.rewrite->frontier <= result.summary.endpoint.frontier &&
              result.summary.rewrite->ordinal == 0)) &&
            result.summary.rebuild_work_quanta != 0;
        if (!result.continuation || !valid_summary ||
            !continuation_within_active(result.resident_resources, active) ||
            publication.state != CatalogState::ReservedForActive ||
            publication.id != active_entry.continuation_id ||
            !ledger_.publish_active_to_catalog(lane, active_entry.publication_slot, active,
                                               result.resident_resources)) {
            if (result.continuation) {
                (void)program.release_continuation(std::move(*result.continuation));
                result.continuation.reset();
            }
            throw std::logic_error("Runtime continuation publication violated the ledger");
        }
        publication.state   = CatalogState::Catalogued;
        publication.summary = result.summary;
        publication.handle.emplace(std::move(*result.continuation));
        result.continuation.reset();
        ++publication.revision;
        active_entry = {};
        return result;
    }

    [[nodiscard]] AbortResult abort(Program& program, LaneId lane, SequenceHandle sequence) {
        const DeviceResources active = require_active(lane);
        AbortResult result           = program.abort(sequence);
        if (result.status != ConsumeStatus::Consumed || result.released_resources != active ||
            !ledger_.release_active(lane, active)) {
            throw std::logic_error("Runtime abort violated the resource ledger");
        }
        clear_catalog_entry(catalog_[active_[lane.value].publication_slot]);
        active_[lane.value] = {};
        return result;
    }

    void apply_commit(std::span<const LaneId> lanes, const typename Package::CommitResult& result) {
        if (lanes.size() != result.row_count) {
            throw std::logic_error("commit result membership is not row aligned");
        }
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const DeviceResources active = require_active(lanes[row]);
            if (result.rows[row].disposition == CommitDisposition::CancelledReleased &&
                result.rows[row].released_resources != active) {
                throw std::logic_error("cancelled commit did not release its active entitlement");
            }
        }
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            if (result.rows[row].disposition != CommitDisposition::CancelledReleased) { continue; }
            release_cancelled_lane(lanes[row]);
        }
    }

    void apply_discard(std::span<const LaneId> lanes,
                       const typename Package::DiscardResult& result) {
        if (lanes.size() != result.row_count || result.status != ConsumeStatus::Consumed) {
            throw std::logic_error("pending discard did not consume its membership");
        }
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            if (result.released_resources[row] != require_active(lanes[row])) {
                throw std::logic_error("pending discard release acknowledgement is invalid");
            }
        }
        for (const LaneId lane : lanes) { release_cancelled_lane(lane); }
    }

    void release_failed_commit(std::span<const LaneId> lanes) noexcept {
        for (const LaneId lane : lanes) {
            if (lane.value >= lane_count_ || ledger_.lane(lane).state != LogicalLaneState::Active) {
                continue;
            }
            release_cancelled_lane(lane);
        }
    }

    [[nodiscard]] const ResourceLedger& ledger() const noexcept { return ledger_; }

    [[nodiscard]] CatalogState catalog_state(std::uint32_t slot) const noexcept {
        return slot < catalog_count_ ? catalog_[slot].state : CatalogState::Vacant;
    }

    void clear_after_program_cleanup() noexcept {
        for (std::uint32_t slot = 0; slot < catalog_count_; ++slot) {
            catalog_[slot].handle.reset();
            clear_catalog_entry(catalog_[slot]);
        }
        for (std::uint32_t lane = 0; lane < lane_count_; ++lane) { active_[lane] = {}; }
        ledger_.clear();
    }

private:
    struct CatalogEntry {
        CatalogState state     = CatalogState::Vacant;
        std::uint64_t id       = 0;
        std::uint64_t revision = 1;
        ContinuationSummary summary;
        std::optional<ContinuationHandle> handle;
    };

    struct ActiveEntry {
        bool occupied                  = false;
        std::uint32_t publication_slot = kInvalidCatalogSlot;
        std::uint64_t continuation_id  = 0;
        DeviceResources resources;
    };

    static void clear_catalog_entry(CatalogEntry& entry) noexcept {
        entry.state   = CatalogState::Vacant;
        entry.id      = 0;
        entry.summary = {};
        entry.handle.reset();
        if (++entry.revision == 0) { ++entry.revision; }
    }

    [[nodiscard]] DeviceResources require_active(LaneId lane) const {
        if (lane.value >= lane_count_ || ledger_.lane(lane).state != LogicalLaneState::Active ||
            !active_[lane.value].occupied ||
            active_[lane.value].resources != ledger_.lane(lane).resources) {
            throw std::logic_error("resource lane is not active");
        }
        return active_[lane.value].resources;
    }

    void release_cancelled_lane(LaneId lane) noexcept {
        if (lane.value >= lane_count_ || !active_[lane.value].occupied) { return; }
        const DeviceResources active = active_[lane.value].resources;
        (void)ledger_.release_active(lane, active);
        const std::uint32_t publication_slot = active_[lane.value].publication_slot;
        if (publication_slot < catalog_count_) { clear_catalog_entry(catalog_[publication_slot]); }
        active_[lane.value] = {};
    }

    void validate_choice(const Choice& choice) const {
        if (!choice.plan_ || choice.destination_.value >= lane_count_ ||
            ledger_.lane(choice.destination_).state != LogicalLaneState::Free ||
            choice.publication_slot_ >= catalog_count_ ||
            choice.plan_->demand() != choice.demand_ ||
            choice.plan_->summary().admission != choice.demand_.active_entitlement ||
            !detail::valid_demand(choice.demand_, choice.source_resources_)) {
            throw std::logic_error("admission choice is empty or its destination changed");
        }
        if (choice.source_slot_ != kInvalidCatalogSlot) {
            const CatalogEntry& source = catalog_[choice.source_slot_];
            if (source.state != CatalogState::Catalogued || source.id != choice.source_id_ ||
                source.revision != choice.source_revision_ || !source.handle ||
                source.summary.footprint != choice.source_resources_) {
                throw std::logic_error("admission source changed after inspection");
            }
        } else if (catalog_[choice.publication_slot_].state != CatalogState::Vacant &&
                   std::find(choice.evictions_.begin(),
                             choice.evictions_.begin() +
                                 static_cast<std::ptrdiff_t>(choice.eviction_count_),
                             choice.publication_slot_) ==
                       choice.evictions_.begin() +
                           static_cast<std::ptrdiff_t>(choice.eviction_count_)) {
            throw std::logic_error("root publication slot is no longer available");
        }
        for (std::size_t index = 0; index < choice.eviction_count_; ++index) {
            const std::uint32_t slot  = choice.evictions_[index];
            const CatalogEntry& entry = catalog_[slot];
            if (slot == choice.source_slot_ || entry.state != CatalogState::Catalogued ||
                entry.id != choice.eviction_ids_[index] ||
                entry.revision != choice.eviction_revisions_[index] || !entry.handle) {
                throw std::logic_error("admission victim changed after inspection");
            }
        }

        std::array<CatalogResourceDescriptor, 2 * kMaximumConcurrency> victims{};
        for (std::size_t index = 0; index < choice.eviction_count_; ++index) {
            const std::uint32_t slot = choice.evictions_[index];
            victims[index]           = CatalogResourceDescriptor{
                          .slot      = slot,
                          .id        = ContinuationId{catalog_[slot].id},
                          .resources = catalog_[slot].summary.footprint,
            };
        }
        const ResourceCandidateDescriptor candidate{
            .demand           = choice.demand_,
            .source_resources = choice.source_resources_,
            .source_slot      = choice.source_slot_,
        };
        if (!detail::feasible_candidate(
                ledger_.used(), candidate,
                std::span<const CatalogResourceDescriptor>(victims.data(), choice.eviction_count_),
                ledger_.capacity())) {
            throw std::logic_error("admission resource transaction changed after inspection");
        }
    }

    ResourceLedger ledger_;
    std::uint32_t lane_count_    = 0;
    std::uint32_t catalog_count_ = 0;
    std::array<CatalogEntry, 2 * kMaximumConcurrency> catalog_{};
    std::array<ActiveEntry, kMaximumConcurrency> active_{};
    std::uint64_t next_continuation_id_ = 1;
};

} // namespace ninfer::runtime
