#include "runtime/engine/admission_policy.h"
#include "runtime/engine/scheduler.h"

#include <array>
#include <iostream>
#include <utility>

namespace {

struct SchedulerRequest {
    using SequenceHandle = std::uint64_t;
};

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

void add_candidate(ninfer::runtime::ProtectedHeadResourceProjection& projection,
                   std::uint32_t owner_bit, const ninfer::runtime::DeviceResources& resources) {
    auto& bucket = projection.release_by_last_owner_mask[owner_bit];
    bucket.active_lanes += resources.active_lanes;
    bucket.state_slots += resources.state_slots;
    bucket.main_kv_pages += resources.main_kv_pages;
    bucket.backend_kv_pages += resources.backend_kv_pages;
}

} // namespace

int main() {
    using ninfer::runtime::ActiveAdmissionSnapshot;
    using ninfer::runtime::DeviceResources;
    using ninfer::runtime::BackfillClass;

    int failures = 0;
    const DeviceResources capacity{
        .active_lanes     = 4,
        .state_slots      = 8,
        .main_kv_pages    = 160,
        .backend_kv_pages = 128,
    };
    const DeviceResources head{
        .active_lanes     = 1,
        .state_slots      = 2,
        .main_kv_pages    = 64,
        .backend_kv_pages = 48,
    };
    std::array<ActiveAdmissionSnapshot, 2> incumbents{
        ActiveAdmissionSnapshot{
            .request_id            = 1,
            .remaining_work_quanta = 100,
            .owner_bit             = 1U,
        },
        ActiveAdmissionSnapshot{
            .request_id            = 2,
            .remaining_work_quanta = 20,
            .owner_bit             = 2U,
        },
    };
    ninfer::runtime::ProtectedHeadResourceProjection incumbent_projection;
    add_candidate(incumbent_projection, 1U, DeviceResources{1, 2, 64, 32});
    add_candidate(incumbent_projection, 2U, DeviceResources{1, 2, 48, 64});

    const auto protection = ninfer::runtime::make_admission_protection(
        7, 10, head, std::span<const ActiveAdmissionSnapshot>(incumbents), incumbent_projection,
        capacity);
    failures += check(protection.donor_count == 1 && protection.donor_ids[0] == 2 &&
                          protection.temporal_credit == 20,
                      "release frontier did not select the earliest sufficient incumbent");
    failures += check(ninfer::runtime::protection_frontier_distance(protection, incumbents) == 20,
                      "frontier distance did not follow the frozen donor");

    const DeviceResources persistent_candidate{1, 1, 24, 40};
    auto persistent_projection = incumbent_projection;
    add_candidate(persistent_projection, 4U, persistent_candidate);
    failures += check(ninfer::runtime::persistent_backfill_is_safe(protection, incumbents,
                                                                   persistent_projection, capacity),
                      "future resource surplus rejected a persistent-safe backfill");
    auto unsafe_projection = incumbent_projection;
    add_candidate(unsafe_projection, 4U, DeviceResources{1, 1, 40, 60});
    failures += check(!ninfer::runtime::persistent_backfill_is_safe(protection, incumbents,
                                                                    unsafe_projection, capacity),
                      "persistent backfill borrowed protected future capacity");

    std::array<ActiveAdmissionSnapshot, 3> with_persistent{
        incumbents[0],
        incumbents[1],
        ActiveAdmissionSnapshot{
            .request_id            = 3,
            .remaining_work_quanta = 50,
            .backfill_epoch        = 7,
            .backfill_class        = BackfillClass::Persistent,
            .owner_bit             = 4U,
        },
    };
    auto second_candidate_projection = persistent_projection;
    add_candidate(second_candidate_projection, 8U, DeviceResources{1, 1, 9, 9});
    failures += check(!ninfer::runtime::persistent_backfill_is_safe(
                          protection, with_persistent, second_candidate_projection, capacity),
                      "persistent ledger failed to accumulate earlier backfills");

    std::array<ActiveAdmissionSnapshot, 2> after_donor{
        incumbents[0],
        ActiveAdmissionSnapshot{
            .request_id            = 4,
            .remaining_work_quanta = 8,
            .backfill_epoch        = 7,
            .backfill_class        = BackfillClass::Temporal,
            .owner_bit             = 4U,
        },
    };
    failures += check(ninfer::runtime::protection_frontier_distance(protection, after_donor) == 0,
                      "later temporal work changed the frozen frontier");
    ninfer::runtime::ProtectedHeadResourceProjection after_donor_projection;
    add_candidate(after_donor_projection, 1U, DeviceResources{1, 2, 64, 32});
    add_candidate(after_donor_projection, 4U, DeviceResources{1, 1, 32, 64});
    failures += check(ninfer::runtime::protected_head_safe_without_temporal(
                          protection, after_donor, after_donor_projection, capacity),
                      "released frontier did not mature behind a temporal borrower");

    failures += check(
        !ninfer::runtime::admission_resources_fit(DeviceResources{1, 9, 1, 1}, capacity) &&
            !ninfer::runtime::admission_resources_fit(DeviceResources{1, 1, 161, 1}, capacity) &&
            !ninfer::runtime::admission_resources_fit(DeviceResources{1, 1, 1, 129}, capacity),
        "independent Device resource dimensions were treated as interchangeable capacity");

    using Scheduler = ninfer::runtime::Scheduler<SchedulerRequest>;
    Scheduler scheduler;
    scheduler.observe_fifo_head(10);
    failures +=
        check(scheduler.protect_blocked_head(10, head, incumbents, incumbent_projection, capacity),
              "blocked FIFO head did not open a protection epoch");

    const DeviceResources persistent_safe_candidate{1, 1, 24, 24};
    auto persistent_safe_projection = incumbent_projection;
    add_candidate(persistent_safe_projection, 4U, persistent_safe_candidate);
    auto persistent =
        scheduler.qualify_backfill(11, persistent_safe_projection, 50, incumbents, capacity);
    failures += check(persistent && persistent->backfill_class() == BackfillClass::Persistent &&
                          scheduler.validate_grant(*persistent),
                      "persistent-safe candidate did not receive a valid Scheduler grant");
    if (!persistent) { return 1; }
    const std::uint64_t protection_epoch = persistent->protection_epoch();
    scheduler.commit_admission(std::move(*persistent));

    std::array<ActiveAdmissionSnapshot, 3> active_with_persistent{
        incumbents[0],
        incumbents[1],
        ActiveAdmissionSnapshot{
            .request_id            = 11,
            .remaining_work_quanta = 50,
            .backfill_epoch        = protection_epoch,
            .backfill_class        = BackfillClass::Persistent,
            .owner_bit             = 4U,
        },
    };
    const DeviceResources temporal_candidate{1, 1, 16, 8};
    auto temporal_projection = persistent_safe_projection;
    add_candidate(temporal_projection, 8U, temporal_candidate);
    auto temporal =
        scheduler.qualify_backfill(12, temporal_projection, 8, active_with_persistent, capacity);
    failures += check(temporal && temporal->backfill_class() == BackfillClass::Temporal &&
                          scheduler.validate_grant(*temporal),
                      "bounded temporal candidate did not receive a valid Scheduler grant");
    if (!temporal) { return 1; }
    scheduler.commit_admission(std::move(*temporal));
    failures += check(
        !scheduler.qualify_backfill(13, temporal_projection, 13, active_with_persistent, capacity),
        "committed temporal work did not consume protected credit");

    std::array<ActiveAdmissionSnapshot, 3> matured{
        incumbents[0],
        active_with_persistent[2],
        ActiveAdmissionSnapshot{
            .request_id            = 12,
            .remaining_work_quanta = 8,
            .backfill_epoch        = protection_epoch,
            .backfill_class        = BackfillClass::Temporal,
            .owner_bit             = 8U,
        },
    };
    ninfer::runtime::ProtectedHeadResourceProjection matured_projection;
    add_candidate(matured_projection, 1U, DeviceResources{1, 2, 64, 32});
    add_candidate(matured_projection, 4U, persistent_safe_candidate);
    add_candidate(matured_projection, 8U, temporal_candidate);
    failures +=
        check(!scheduler.protect_blocked_head(10, head, matured, matured_projection, capacity),
              "matured protected opportunity did not enter drain");
    scheduler.on_waiting_removed(10);
    scheduler.observe_fifo_head(14);
    auto head_grant = scheduler.grant_head(14, 1);
    failures += check(scheduler.validate_grant(head_grant),
                      "observed FIFO head did not receive a valid admission grant");
    scheduler.commit_admission(std::move(head_grant));

    using ExecutionAction = Scheduler::ExecutionAction;
    failures += check(scheduler.should_attempt_admission(true, true, false, false, false) &&
                          !scheduler.should_attempt_admission(false, true, true, true, false) &&
                          !scheduler.should_attempt_admission(true, false, true, true, false) &&
                          !scheduler.should_attempt_admission(true, true, true, false, false) &&
                          scheduler.should_attempt_admission(true, true, true, true, false) &&
                          !scheduler.should_attempt_admission(true, true, false, false, true) &&
                          scheduler.choose_execution(true, false, false) == ExecutionAction::Decode,
                      "admission invalidation and fairness gates were not independent");
    scheduler.set_prefill_lane(0);
    failures +=
        check(!scheduler.should_attempt_admission(true, true, true, true, false) &&
                  scheduler.choose_execution(true, true, false) == ExecutionAction::Decode &&
                  scheduler.choose_execution(true, true, true) == ExecutionAction::Prefill,
              "prefill/decode alternation did not follow the completed GPU unit");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
