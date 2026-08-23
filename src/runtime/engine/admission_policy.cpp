#include "runtime/engine/admission_policy.h"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace ninfer::runtime {
namespace {

struct ResourceTotals {
    std::uint64_t active_lanes     = 0;
    std::uint64_t state_slots      = 0;
    std::uint64_t main_kv_pages    = 0;
    std::uint64_t backend_kv_pages = 0;
};

void add(ResourceTotals& total, const DeviceResources& resources) noexcept {
    total.active_lanes += resources.active_lanes;
    total.state_slots += resources.state_slots;
    total.main_kv_pages += resources.main_kv_pages;
    total.backend_kv_pages += resources.backend_kv_pages;
}

bool fits(const ResourceTotals& used, const DeviceResources& capacity) noexcept {
    return used.active_lanes <= capacity.active_lanes && used.state_slots <= capacity.state_slots &&
           used.main_kv_pages <= capacity.main_kv_pages &&
           used.backend_kv_pages <= capacity.backend_kv_pages;
}

bool contains(std::span<const std::uint64_t> ids, std::uint64_t id) noexcept {
    return std::find(ids.begin(), ids.end(), id) != ids.end();
}

bool is_donor(const AdmissionProtection& protection, std::uint64_t id) noexcept {
    return contains(
        std::span<const std::uint64_t>(protection.donor_ids.data(), protection.donor_count), id);
}

std::uint32_t projection_owner_mask(const ProtectedHeadResourceProjection& projection) noexcept {
    std::uint32_t mask = 0;
    for (std::uint32_t owners = 1; owners < projection.release_by_last_owner_mask.size();
         ++owners) {
        const DeviceResources& resources = projection.release_by_last_owner_mask[owners];
        if (resources != DeviceResources{}) { mask |= owners; }
    }
    return mask;
}

ResourceTotals projected_resources(const ProtectedHeadResourceProjection& projection,
                                   std::uint32_t surviving_owners) noexcept {
    ResourceTotals total;
    add(total, projection.fixed_non_reclaimable);
    for (std::uint32_t owners = 1; owners < projection.release_by_last_owner_mask.size();
         ++owners) {
        if ((owners & surviving_owners) != 0) {
            add(total, projection.release_by_last_owner_mask[owners]);
        }
    }
    return total;
}

} // namespace

bool admission_resources_fit(const DeviceResources& used,
                             const DeviceResources& capacity) noexcept {
    ResourceTotals total;
    add(total, used);
    return fits(total, capacity);
}

AdmissionProtection make_admission_protection(std::uint64_t epoch_id, std::uint64_t head_request_id,
                                              const DeviceResources& head_resources,
                                              std::span<const ActiveAdmissionSnapshot> active,
                                              const ProtectedHeadResourceProjection& projection,
                                              const DeviceResources& capacity) {
    if (epoch_id == 0 || head_request_id == 0 || active.empty() ||
        active.size() > kMaximumConcurrency || head_resources.active_lanes == 0 ||
        !admission_resources_fit(head_resources, capacity)) {
        throw std::invalid_argument("invalid protected-admission frontier");
    }

    AdmissionProtection out;
    out.epoch_id        = epoch_id;
    out.head_request_id = head_request_id;
    out.head_resources  = head_resources;

    std::uint32_t surviving_owners = projection_owner_mask(projection);
    std::uint32_t observed_bits    = 0;
    for (std::size_t i = 0; i < active.size(); ++i) {
        if (active[i].request_id == 0 || active[i].remaining_work_quanta == 0 ||
            active[i].owner_bit == 0 || (active[i].owner_bit & (active[i].owner_bit - 1U)) != 0 ||
            (observed_bits & active[i].owner_bit) != 0 ||
            (surviving_owners & active[i].owner_bit) == 0) {
            throw std::invalid_argument("protected incumbent has invalid progress state");
        }
        observed_bits |= active[i].owner_bit;
    }
    ResourceTotals survivors = projected_resources(projection, surviving_owners);
    add(survivors, head_resources);
    if (fits(survivors, capacity)) {
        throw std::invalid_argument("protected head is not blocked by frozen incumbents");
    }

    std::array<std::size_t, kMaximumConcurrency> order{};
    for (std::size_t i = 0; i < active.size(); ++i) { order[i] = i; }
    std::sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(active.size()),
              [&](std::size_t lhs, std::size_t rhs) {
                  if (active[lhs].remaining_work_quanta != active[rhs].remaining_work_quanta) {
                      return active[lhs].remaining_work_quanta < active[rhs].remaining_work_quanta;
                  }
                  return active[lhs].request_id < active[rhs].request_id;
              });

    for (std::size_t i = 0; i < active.size(); ++i) {
        const ActiveAdmissionSnapshot& donor = active[order[i]];
        surviving_owners &= ~donor.owner_bit;
        survivors = projected_resources(projection, surviving_owners);
        add(survivors, head_resources);
        out.donor_ids[out.donor_count++] = donor.request_id;
        out.temporal_credit              = donor.remaining_work_quanta;
        if (fits(survivors, capacity)) { return out; }
    }
    throw std::logic_error("exclusive-feasible head has no releasing incumbent frontier");
}

bool persistent_backfill_is_safe(const AdmissionProtection& protection,
                                 std::span<const ActiveAdmissionSnapshot> active,
                                 const ProtectedHeadResourceProjection& candidate,
                                 const DeviceResources& capacity) noexcept {
    std::uint32_t surviving_owners = projection_owner_mask(candidate);
    for (const ActiveAdmissionSnapshot& request : active) {
        if (is_donor(protection, request.request_id)) { surviving_owners &= ~request.owner_bit; }
    }
    ResourceTotals future = projected_resources(candidate, surviving_owners);
    add(future, protection.head_resources);
    return fits(future, capacity);
}

std::uint64_t
protection_frontier_distance(const AdmissionProtection& protection,
                             std::span<const ActiveAdmissionSnapshot> active) noexcept {
    std::uint64_t distance = 0;
    for (const ActiveAdmissionSnapshot& request : active) {
        if (is_donor(protection, request.request_id)) {
            distance = std::max(distance, request.remaining_work_quanta);
        }
    }
    return distance;
}

bool protected_head_safe_without_temporal(const AdmissionProtection& protection,
                                          std::span<const ActiveAdmissionSnapshot> active,
                                          const ProtectedHeadResourceProjection& projection,
                                          const DeviceResources& capacity) noexcept {
    std::uint32_t surviving_owners = projection_owner_mask(projection);
    for (const ActiveAdmissionSnapshot& request : active) {
        if (request.backfill_epoch == protection.epoch_id &&
            request.backfill_class == BackfillClass::Temporal) {
            surviving_owners &= ~request.owner_bit;
        }
    }
    ResourceTotals used = projected_resources(projection, surviving_owners);
    add(used, protection.head_resources);
    return fits(used, capacity);
}

} // namespace ninfer::runtime
