#pragma once

#include "ninfer/types.h"
#include "runtime/contract/types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::runtime {

enum class BackfillClass : std::uint8_t {
    None,
    Persistent,
    Temporal,
};

struct ActiveAdmissionSnapshot {
    std::uint64_t request_id            = 0;
    std::uint64_t remaining_work_quanta = 0;
    std::uint64_t backfill_epoch        = 0;
    BackfillClass backfill_class        = BackfillClass::None;
    std::uint32_t owner_bit             = 0;
};

struct ProtectedHeadResourceProjection {
    DeviceResources fixed_non_reclaimable;
    std::array<DeviceResources, 1U << kMaximumConcurrency> release_by_last_owner_mask{};
};

enum class ProtectionPhase : std::uint8_t {
    Open,
    Drain,
};

struct AdmissionProtection {
    std::uint64_t epoch_id        = 0;
    std::uint64_t head_request_id = 0;
    DeviceResources head_resources;
    std::array<std::uint64_t, kMaximumConcurrency> donor_ids{};
    std::size_t donor_count       = 0;
    std::uint64_t temporal_credit = 0;
    ProtectionPhase phase         = ProtectionPhase::Open;
};

[[nodiscard]] bool admission_resources_fit(const DeviceResources& used,
                                           const DeviceResources& capacity) noexcept;

// Freezes the currently active requests and selects the earliest projected completion prefix
// whose release makes the protected head componentwise feasible.
[[nodiscard]] AdmissionProtection make_admission_protection(
    std::uint64_t epoch_id, std::uint64_t head_request_id, const DeviceResources& head_resources,
    std::span<const ActiveAdmissionSnapshot> active,
    const ProtectedHeadResourceProjection& projection, const DeviceResources& capacity);

// Tests the cumulative future-frontier invariant, including every still-active persistent
// backfill from this epoch and the proposed candidate.
[[nodiscard]] bool persistent_backfill_is_safe(const AdmissionProtection& protection,
                                               std::span<const ActiveAdmissionSnapshot> active,
                                               const ProtectedHeadResourceProjection& candidate,
                                               const DeviceResources& capacity) noexcept;

// Projected distance to the last still-active frozen donor. Later admissions never contribute.
[[nodiscard]] std::uint64_t
protection_frontier_distance(const AdmissionProtection& protection,
                             std::span<const ActiveAdmissionSnapshot> active) noexcept;

// True once the head would fit if current-epoch temporal borrowers were absent. This recognizes
// both the frozen donor frontier and an earlier opportunity created by any incumbent release.
[[nodiscard]] bool protected_head_safe_without_temporal(
    const AdmissionProtection& protection, std::span<const ActiveAdmissionSnapshot> active,
    const ProtectedHeadResourceProjection& projection, const DeviceResources& capacity) noexcept;

} // namespace ninfer::runtime
