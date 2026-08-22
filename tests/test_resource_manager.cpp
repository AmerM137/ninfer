#include "runtime/engine/resource_manager.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <utility>

namespace {

using ninfer::runtime::ConsumeStatus;
using ninfer::runtime::DeviceResources;
using ninfer::runtime::LaneId;
using ninfer::runtime::RequestPlanSummary;
using ninfer::runtime::ResourceDelta;
using ninfer::runtime::ResourceDemand;

int failures = 0;

void expect(bool condition, const char* message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

DeviceResources converted(DeviceResources active, DeviceResources source) {
    return DeviceResources{
        .state_slots      = std::min(active.state_slots, source.state_slots),
        .main_kv_pages    = std::min(active.main_kv_pages, source.main_kv_pages),
        .backend_kv_pages = std::min(active.backend_kv_pages, source.backend_kv_pages),
    };
}

DeviceResources additional(DeviceResources active, DeviceResources source_conversion) {
    return DeviceResources{
        .active_lanes     = active.active_lanes,
        .state_slots      = active.state_slots - source_conversion.state_slots,
        .main_kv_pages    = active.main_kv_pages - source_conversion.main_kv_pages,
        .backend_kv_pages = active.backend_kv_pages - source_conversion.backend_kv_pages,
    };
}

struct FakePreparedPrompt {
    std::uint32_t key = 0;
    bool allow_reuse  = true;
};

struct FakeRequestBasePlan {
    RequestPlanSummary value;

    [[nodiscard]] const RequestPlanSummary& summary() const noexcept { return value; }
};

struct FakeAdmissionPlan {
    RequestPlanSummary value;
    ResourceDemand resources;
    LaneId destination;
    std::uint32_t key       = 0;
    bool source             = false;
    std::uint64_t root_work = 0;

    FakeAdmissionPlan()                                        = default;
    FakeAdmissionPlan(FakeAdmissionPlan&&) noexcept            = default;
    FakeAdmissionPlan& operator=(FakeAdmissionPlan&&) noexcept = default;
    FakeAdmissionPlan(const FakeAdmissionPlan&)                = delete;
    FakeAdmissionPlan& operator=(const FakeAdmissionPlan&)     = delete;

    [[nodiscard]] const RequestPlanSummary& summary() const noexcept { return value; }

    [[nodiscard]] const ResourceDemand& demand() const noexcept { return resources; }
};

struct FakeSequenceHandle {
    LaneId lane;
    std::uint64_t generation = 0;
};

struct FakeContinuationHandle {
    std::uint32_t key = 0;
    DeviceResources resources;
    bool valid = false;

    FakeContinuationHandle() = default;

    FakeContinuationHandle(std::uint32_t key_, DeviceResources resources_)
        : key(key_), resources(resources_), valid(true) {}

    FakeContinuationHandle(FakeContinuationHandle&& other) noexcept
        : key(other.key), resources(other.resources), valid(std::exchange(other.valid, false)) {}

    FakeContinuationHandle& operator=(FakeContinuationHandle&&)      = delete;
    FakeContinuationHandle(const FakeContinuationHandle&)            = delete;
    FakeContinuationHandle& operator=(const FakeContinuationHandle&) = delete;
};

struct FakeMaterializationTicket {
    std::uint64_t id = 0;

    FakeMaterializationTicket() = default;

    explicit FakeMaterializationTicket(std::uint64_t value) : id(value) {}

    FakeMaterializationTicket(FakeMaterializationTicket&& other) noexcept
        : id(std::exchange(other.id, 0)) {}

    FakeMaterializationTicket& operator=(FakeMaterializationTicket&&)      = delete;
    FakeMaterializationTicket(const FakeMaterializationTicket&)            = delete;
    FakeMaterializationTicket& operator=(const FakeMaterializationTicket&) = delete;
};

struct FakeStartResult {
    FakeSequenceHandle sequence;
    DeviceResources active_resources;
    ResourceDelta resource_delta;
};

struct FakeMaterializationResult {
    ninfer::runtime::MaterializationStatus status = ninfer::runtime::MaterializationStatus::Aborted;
    std::optional<FakeStartResult> published;
    std::optional<FakeContinuationHandle> source;
    std::array<DeviceResources, 2 * ninfer::kMaximumConcurrency> released_victims{};
    std::size_t victim_count = 0;
};

struct FakeFinishResult {
    ConsumeStatus status = ConsumeStatus::InvariantMismatch;
    int timings          = 0;
    int speculative      = 0;
    DeviceResources released_resources;
    DeviceResources resident_resources;
    ninfer::runtime::ContinuationSummary summary;
    std::optional<FakeContinuationHandle> continuation;
};

struct FakeAbortResult {
    ConsumeStatus status = ConsumeStatus::InvariantMismatch;
    int timings          = 0;
    int speculative      = 0;
    DeviceResources released_resources;
};

struct FakeReleaseResult {
    ConsumeStatus status = ConsumeStatus::InvariantMismatch;
    DeviceResources released_resources;
};

struct FakeCommitRowResult {
    ninfer::runtime::CommitDisposition disposition = ninfer::runtime::CommitDisposition::Active;
    DeviceResources released_resources;
};

struct FakeCommitResult {
    std::array<FakeCommitRowResult, ninfer::kMaximumConcurrency> rows{};
    std::size_t row_count = 0;
};

struct FakeDiscardResult {
    ConsumeStatus status = ConsumeStatus::InvariantMismatch;
    std::array<DeviceResources, ninfer::kMaximumConcurrency> released_resources{};
    std::size_t row_count = 0;
};

class FakeProgram {
public:
    std::optional<FakeAdmissionPlan>
    inspect_admission(const FakePreparedPrompt& prompt, const FakeRequestBasePlan& base,
                      LaneId destination, const FakeContinuationHandle* source,
                      std::optional<ninfer::runtime::CheckpointRef> checkpoint) {
        if ((source == nullptr) != !checkpoint.has_value()) {
            throw std::logic_error("fake source/checkpoint mismatch");
        }
        FakeAdmissionPlan plan;
        plan.value       = base.summary();
        plan.destination = destination;
        plan.key         = prompt.key;
        plan.root_work   = base.summary().service_work_quanta;
        const bool hit   = source != nullptr && source->valid && prompt.allow_reuse &&
                         source->key == prompt.key &&
                         checkpoint->kind == ninfer::runtime::CheckpointKind::SessionEndpoint &&
                         checkpoint->frontier == 16;
        if (source != nullptr && !hit) { return std::nullopt; }
        plan.source = hit;
        if (hit) {
            plan.value.reusable_prompt_tokens = 16;
            plan.value.service_work_quanta    = 1;
            const DeviceResources conversion  = converted(plan.value.admission, source->resources);
            plan.resources                    = ResourceDemand{
                                   .active_entitlement    = plan.value.admission,
                                   .prepublish_additional = additional(plan.value.admission, conversion),
                                   .source_conversions    = conversion,
            };
        } else {
            plan.value.reusable_prompt_tokens = 0;
            plan.resources                    = ResourceDemand{
                                   .active_entitlement    = plan.value.admission,
                                   .prepublish_additional = plan.value.admission,
            };
        }
        return plan;
    }

    FakeMaterializationTicket
    reserve_materialization(FakeAdmissionPlan&& plan, FakePreparedPrompt&& prompt,
                            const FakeContinuationHandle* source,
                            std::span<const FakeContinuationHandle* const> victims) {
        if (transaction_ || plan.key != prompt.key || plan.source != (source != nullptr) ||
            (source != nullptr && (!source->valid || source->key != plan.key))) {
            throw std::logic_error("fake materialization reservation mismatch");
        }
        Transaction transaction;
        transaction.id           = next_transaction_++;
        transaction.source_key   = source != nullptr ? source->key : 0;
        transaction.has_source   = source != nullptr;
        transaction.victim_count = victims.size();
        transaction.plan.emplace(std::move(plan));
        transaction.prompt.emplace(std::move(prompt));
        for (std::size_t index = 0; index < victims.size(); ++index) {
            if (victims[index] == nullptr || !victims[index]->valid ||
                (source != nullptr && victims[index]->key == source->key)) {
                throw std::logic_error("fake materialization victim mismatch");
            }
            transaction.victim_keys[index] = victims[index]->key;
        }
        const std::uint64_t id = transaction.id;
        transaction_.emplace(std::move(transaction));
        return FakeMaterializationTicket{id};
    }

    FakeReleaseResult release_materialization_victim(FakeMaterializationTicket& ticket,
                                                     FakeContinuationHandle&& victim) noexcept {
        FakeReleaseResult result;
        if (!transaction_ || ticket.id != transaction_->id || !victim.valid) { return result; }
        std::size_t position = 0;
        for (; position < transaction_->victim_count; ++position) {
            if (!transaction_->released[position] &&
                transaction_->victim_keys[position] == victim.key) {
                break;
            }
        }
        if (position == transaction_->victim_count) { return result; }
        result.status                              = ConsumeStatus::Consumed;
        result.released_resources                  = victim.resources;
        transaction_->released[position]           = true;
        transaction_->released_resources[position] = victim.resources;
        victim.valid                               = false;
        ++release_count;
        last_released_key = victim.key;
        if (cancel_after_victim != nullptr) {
            cancel_after_victim->store(true, std::memory_order_release);
        }
        return result;
    }

    ConsumeStatus abort_materialization(FakeMaterializationTicket&& ticket) noexcept {
        if (!transaction_ || ticket.id != transaction_->id) {
            return ConsumeStatus::InvariantMismatch;
        }
        ticket.id = 0;
        transaction_.reset();
        return ConsumeStatus::Consumed;
    }

    void prepare_materialization(FakeMaterializationTicket& ticket) {
        if (!transaction_ || ticket.id != transaction_->id || transaction_->prepared) {
            throw std::logic_error("fake materialization preparation mismatch");
        }
        for (std::size_t index = 0; index < transaction_->victim_count; ++index) {
            if (!transaction_->released[index]) {
                throw std::logic_error("fake materialization victim was not prepared");
            }
        }
        transaction_->prepared = true;
    }

    FakeMaterializationResult
    publish_materialization(FakeMaterializationTicket&& ticket,
                            std::optional<FakeContinuationHandle>&& source,
                            ninfer::runtime::CancellationFlagView cancellation) {
        if (!transaction_ || ticket.id != transaction_->id || !transaction_->prepared ||
            transaction_->has_source != source.has_value() ||
            (source && source->key != transaction_->source_key)) {
            throw std::logic_error("fake materialization publication mismatch");
        }
        FakeMaterializationResult result;
        result.victim_count = transaction_->victim_count;
        for (std::size_t index = 0; index < transaction_->victim_count; ++index) {
            if (!transaction_->released[index]) {
                throw std::logic_error("fake materialization victim was not released");
            }
            result.released_victims[index] = transaction_->released_resources[index];
        }
        FakeAdmissionPlan plan(std::move(*transaction_->plan));
        FakePreparedPrompt prompt(std::move(*transaction_->prompt));
        transaction_.reset();
        ticket.id = 0;
        if (cancellation.requested()) {
            if (source) {
                result.source.emplace(std::move(*source));
                source.reset();
            }
            return result;
        }
        result.status = ninfer::runtime::MaterializationStatus::Published;
        result.published.emplace(
            start_request(std::move(plan), std::move(prompt), std::move(source)));
        return result;
    }

    FakeStartResult start_request(FakeAdmissionPlan&& plan, FakePreparedPrompt&& prompt,
                                  std::optional<FakeContinuationHandle>&& source) {
        const std::uint32_t lane = plan.destination.value;
        if (lane >= active_.size() || active_[lane].occupied || plan.key != prompt.key ||
            plan.source != source.has_value() ||
            (source && (!source->valid || source->key != prompt.key))) {
            throw std::logic_error("fake materialization contract mismatch");
        }
        const DeviceResources removed = source ? source->resources : DeviceResources{};
        source.reset();
        active_[lane]          = Active{.occupied     = true,
                                        .key          = prompt.key,
                                        .generation   = next_generation_++,
                                        .resources    = plan.value.admission,
                                        .rebuild_work = plan.root_work};
        last_start_lane        = lane;
        last_start_used_source = plan.source;
        return FakeStartResult{
            .sequence         = FakeSequenceHandle{LaneId{lane}, active_[lane].generation},
            .active_resources = active_[lane].resources,
            .resource_delta   = {.removed = removed, .added = active_[lane].resources},
        };
    }

    FakeFinishResult finish(FakeSequenceHandle sequence) noexcept {
        FakeFinishResult result;
        if (!valid(sequence)) { return result; }
        Active& active            = active_[sequence.lane.value];
        result.status             = ConsumeStatus::Consumed;
        result.released_resources = active.resources;
        result.resident_resources = DeviceResources{
            .state_slots      = 1,
            .main_kv_pages    = std::min(2U, active.resources.main_kv_pages),
            .backend_kv_pages = std::min(1U, active.resources.backend_kv_pages),
        };
        result.summary = ninfer::runtime::ContinuationSummary{
            .footprint = result.resident_resources,
            .endpoint  = {.kind = ninfer::runtime::CheckpointKind::SessionEndpoint, .frontier = 16},
            .rebuild_work_quanta = active.rebuild_work,
        };
        result.continuation.emplace(active.key, result.resident_resources);
        active = {};
        return result;
    }

    FakeAbortResult abort(FakeSequenceHandle sequence) noexcept {
        FakeAbortResult result;
        if (!valid(sequence)) { return result; }
        Active& active            = active_[sequence.lane.value];
        result.status             = ConsumeStatus::Consumed;
        result.released_resources = active.resources;
        active                    = {};
        return result;
    }

    FakeReleaseResult release_continuation(FakeContinuationHandle&& continuation) noexcept {
        FakeReleaseResult result;
        if (!continuation.valid) { return result; }
        result.status             = ConsumeStatus::Consumed;
        result.released_resources = continuation.resources;
        continuation.valid        = false;
        ++release_count;
        last_released_key = continuation.key;
        return result;
    }

    std::uint32_t last_start_lane          = ninfer::kMaximumConcurrency;
    bool last_start_used_source            = false;
    std::uint32_t release_count            = 0;
    std::uint32_t last_released_key        = 0;
    std::atomic<bool>* cancel_after_victim = nullptr;

private:
    struct Transaction {
        std::uint64_t id         = 0;
        bool has_source          = false;
        std::uint32_t source_key = 0;
        std::array<std::uint32_t, 2 * ninfer::kMaximumConcurrency> victim_keys{};
        std::array<DeviceResources, 2 * ninfer::kMaximumConcurrency> released_resources{};
        std::array<bool, 2 * ninfer::kMaximumConcurrency> released{};
        std::size_t victim_count = 0;
        std::optional<FakeAdmissionPlan> plan;
        std::optional<FakePreparedPrompt> prompt;
        bool prepared = false;
    };

    struct Active {
        bool occupied            = false;
        std::uint32_t key        = 0;
        std::uint64_t generation = 0;
        DeviceResources resources;
        std::uint64_t rebuild_work = 0;
    };

    [[nodiscard]] bool valid(FakeSequenceHandle sequence) const noexcept {
        return sequence.lane.value < active_.size() && active_[sequence.lane.value].occupied &&
               active_[sequence.lane.value].generation == sequence.generation;
    }

    std::array<Active, ninfer::kMaximumConcurrency> active_{};
    std::uint64_t next_generation_ = 1;
    std::optional<Transaction> transaction_;
    std::uint64_t next_transaction_ = 1;
};

struct FakePackage {
    using Program               = FakeProgram;
    using PreparedPrompt        = FakePreparedPrompt;
    using RequestBasePlan       = FakeRequestBasePlan;
    using AdmissionPlan         = FakeAdmissionPlan;
    using SequenceHandle        = FakeSequenceHandle;
    using ContinuationHandle    = FakeContinuationHandle;
    using MaterializationTicket = FakeMaterializationTicket;
    using MaterializationResult = FakeMaterializationResult;
    using StartResult           = FakeStartResult;
    using FinishResult          = FakeFinishResult;
    using AbortResult           = FakeAbortResult;
    using CommitResult          = FakeCommitResult;
    using DiscardResult         = FakeDiscardResult;
};

FakeRequestBasePlan make_base(std::uint64_t work = 10) {
    FakeRequestBasePlan base;
    base.value.admission = DeviceResources{
        .active_lanes = 1, .state_slots = 1, .main_kv_pages = 3, .backend_kv_pages = 2};
    base.value.service_work_quanta = work;
    return base;
}

using FakeManager = ninfer::runtime::ResourceManager<FakePackage>;

FakeStartResult materialize_and_adopt(FakeManager& manager, FakeProgram& program,
                                      FakeManager::Choice&& choice, FakePreparedPrompt prompt) {
    std::atomic<bool> cancelled{false};
    auto outcome = manager.materialize(program, std::move(choice), std::move(prompt),
                                       ninfer::runtime::CancellationFlagView{&cancelled});
    if (outcome.status != ninfer::runtime::MaterializationStatus::Published ||
        !outcome.activation) {
        throw std::logic_error("fake materialization did not publish");
    }
    FakeStartResult result{
        .sequence         = outcome.activation->sequence(),
        .active_resources = outcome.activation->active_resources(),
    };
    manager.adopt(std::move(*outcome.activation));
    outcome.activation.reset();
    return result;
}

void test_selection_and_ledger() {
    using ninfer::runtime::CatalogResourceDescriptor;
    using ninfer::runtime::CheckpointKind;
    using ninfer::runtime::ContinuationId;
    using ninfer::runtime::LogicalLaneState;
    using ninfer::runtime::ResourceCandidateDescriptor;
    using ninfer::runtime::ResourceLedger;

    const DeviceResources capacity{2, 4, 100, 100};
    const DeviceResources used{0, 3, 60, 60};
    const std::array<CatalogResourceDescriptor, 2> catalog{
        CatalogResourceDescriptor{0, ContinuationId{10}, {0, 1, 20, 20}, 5},
        CatalogResourceDescriptor{1, ContinuationId{20}, {0, 2, 40, 40}, 7},
    };
    const ResourceDemand demand{
        .active_entitlement    = {1, 2, 50, 50},
        .prepublish_additional = {1, 1, 30, 30},
        .source_conversions    = {0, 1, 20, 20},
    };
    const std::array<ResourceCandidateDescriptor, 2> tied{
        ResourceCandidateDescriptor{
            demand, {0, 1, 20, 20}, 0, 8, 10, CheckpointKind::TurnClosure, true},
        ResourceCandidateDescriptor{
            demand, {0, 1, 20, 20}, 0, 12, 10, CheckpointKind::SessionEndpoint, true},
    };
    const auto no_victim =
        ninfer::runtime::select_resource_candidate(tied, catalog, used, capacity);
    expect(no_victim.found && no_victim.candidate_index == 1 && no_victim.eviction_count == 0,
           "zero-victim candidate tie did not prefer more reuse");

    const DeviceResources pressured_capacity{1, 2, 8, 8};
    const DeviceResources pressured_used{0, 2, 4, 4};
    const std::array<CatalogResourceDescriptor, 2> victims{
        CatalogResourceDescriptor{3, ContinuationId{30}, {0, 1, 3, 1}, 9},
        CatalogResourceDescriptor{4, ContinuationId{40}, {0, 1, 1, 3}, 2},
    };
    const std::array<ResourceCandidateDescriptor, 1> root{
        ResourceCandidateDescriptor{ResourceDemand{.active_entitlement    = {1, 1, 5, 5},
                                                   .prepublish_additional = {1, 1, 5, 5}},
                                    {},
                                    ninfer::runtime::kInvalidCatalogSlot,
                                    0,
                                    4,
                                    CheckpointKind::SessionEndpoint,
                                    false},
    };
    const auto with_victim = ninfer::runtime::select_resource_candidate(
        root, victims, pressured_used, pressured_capacity);
    expect(with_victim.found && with_victim.eviction_count == 1 && with_victim.evictions[0] == 4 &&
               with_victim.eviction_ids[0] == ContinuationId{40},
           "whole-entry victim selection did not minimize rebuild work across dimensions");

    ResourceLedger ledger({2, 4, 10, 10}, 2);
    const DeviceResources resident{0, 1, 2, 1};
    const DeviceResources active{1, 2, 4, 3};
    expect(ledger.adopt_catalog(0, resident) && ledger.adopt_active(LaneId{1}, active) &&
               ledger.lane(LaneId{1}).state == LogicalLaneState::Active &&
               ledger.used() == DeviceResources{1, 3, 6, 4},
           "active and catalog ownership did not share one four-dimensional ledger");
    expect(!ledger.release_active(LaneId{1}, DeviceResources{1, 1, 4, 3}),
           "ledger accepted an inexact active release acknowledgement");
    const DeviceResources published{0, 1, 3, 2};
    expect(ledger.publish_active_to_catalog(LaneId{1}, 1, active, published) &&
               ledger.used() == DeviceResources{0, 2, 5, 3} &&
               ledger.release_catalog(0, resident) && ledger.release_catalog(1, published) &&
               ledger.used() == DeviceResources{},
           "active-to-catalog publication did not close the exact resource transition");
}

void test_global_catalog_lifecycle() {
    using Manager = ninfer::runtime::ResourceManager<FakePackage>;
    FakeProgram program;
    Manager manager({2, 4, 12, 8}, 2);
    const FakeRequestBasePlan base = make_base();

    auto first = manager.inspect(program, FakePreparedPrompt{1}, base);
    expect(first.readiness == ninfer::runtime::Readiness::Ready && first.choice &&
               first.choice->destination() == LaneId{0},
           "first root was not assigned to the lowest free lane");
    auto first_active =
        materialize_and_adopt(manager, program, std::move(*first.choice), FakePreparedPrompt{1});
    (void)manager.finish(program, LaneId{0}, first_active.sequence);

    auto blocker = manager.inspect(program, FakePreparedPrompt{2}, base);
    auto blocker_active =
        materialize_and_adopt(manager, program, std::move(*blocker.choice), FakePreparedPrompt{2});
    expect(blocker_active.sequence.lane == LaneId{0}, "root blocker did not occupy lane zero");

    auto resumed = manager.inspect(program, FakePreparedPrompt{1}, base);
    expect(resumed.readiness == ninfer::runtime::Readiness::Ready && resumed.choice &&
               resumed.choice->destination() == LaneId{1} &&
               resumed.choice->summary().reusable_prompt_tokens == 16,
           "global continuation was not reusable on a different destination lane");
    auto resumed_active =
        materialize_and_adopt(manager, program, std::move(*resumed.choice), FakePreparedPrompt{1});
    expect(program.last_start_used_source && program.last_start_lane == 1,
           "materialization did not consume the global source into lane one");
    (void)manager.abort(program, LaneId{1}, resumed_active.sequence);
    (void)manager.abort(program, LaneId{0}, blocker_active.sequence);
    expect(manager.ledger().used() == DeviceResources{},
           "abort did not release active ownership and publication reservations");
}

void test_catalog_pressure_reserves_publication() {
    using Manager = ninfer::runtime::ResourceManager<FakePackage>;
    FakeProgram program;
    Manager manager({1, 2, 6, 4}, 1);

    const auto publish_root = [&](std::uint32_t key, std::uint64_t work) {
        const FakeRequestBasePlan base = make_base(work);
        auto inspection                = manager.inspect(program, FakePreparedPrompt{key}, base);
        expect(inspection.readiness == ninfer::runtime::Readiness::Ready && inspection.choice,
               "root inspection failed under bounded catalog pressure");
        auto active = materialize_and_adopt(manager, program, std::move(*inspection.choice),
                                            FakePreparedPrompt{key});
        (void)manager.finish(program, LaneId{0}, active.sequence);
    };

    publish_root(1, 9);
    publish_root(2, 2);
    expect(manager.ledger().used() == DeviceResources{0, 2, 4, 2},
           "two catalogued continuations did not fill the bounded state catalog");
    publish_root(3, 4);
    expect(program.release_count == 1 && program.last_released_key == 2 &&
               manager.ledger().used() == DeviceResources{0, 2, 4, 2},
           "root materialization did not evict the cheapest whole entry and reuse its slot");
}

void test_materialization_abort_and_adoption() {
    FakeProgram program;
    FakeManager manager({1, 2, 4, 2}, 1);

    FakeRequestBasePlan seed = make_base();
    seed.value.admission     = DeviceResources{1, 1, 2, 1};
    const auto publish_seed  = [&](std::uint32_t key, std::uint64_t work) {
        seed.value.service_work_quanta = work;
        auto inspection                = manager.inspect(program, FakePreparedPrompt{key}, seed);
        expect(inspection.readiness == ninfer::runtime::Readiness::Ready && inspection.choice,
                "transaction seed was not ready");
        auto active = materialize_and_adopt(manager, program, std::move(*inspection.choice),
                                             FakePreparedPrompt{key});
        (void)manager.finish(program, LaneId{0}, active.sequence);
    };
    publish_seed(1, 8);
    publish_seed(2, 2);

    std::atomic<bool> cancelled{false};
    program.cancel_after_victim       = &cancelled;
    const FakeRequestBasePlan request = make_base(9);
    auto inspection                   = manager.inspect(program, FakePreparedPrompt{1}, request);
    expect(inspection.readiness == ninfer::runtime::Readiness::Ready && inspection.choice &&
               inspection.choice->summary().reusable_prompt_tokens == 16,
           "source materialization under pressure was not selected");
    auto aborted =
        manager.materialize(program, std::move(*inspection.choice), FakePreparedPrompt{1},
                            ninfer::runtime::CancellationFlagView{&cancelled});
    expect(aborted.status == ninfer::runtime::MaterializationStatus::Aborted &&
               !aborted.activation && program.release_count == 1 &&
               program.last_released_key == 2 &&
               manager.ledger().used() == DeviceResources{0, 1, 2, 1},
           "pre-publication abort did not retain committed victims and restore the source");

    program.cancel_after_victim = nullptr;
    cancelled.store(false, std::memory_order_release);
    auto retry = manager.inspect(program, FakePreparedPrompt{1}, request);
    expect(retry.readiness == ninfer::runtime::Readiness::Ready && retry.choice &&
               retry.choice->summary().reusable_prompt_tokens == 16,
           "aborted source was not reusable after its claim was restored");
    auto published = manager.materialize(program, std::move(*retry.choice), FakePreparedPrompt{1},
                                         ninfer::runtime::CancellationFlagView{&cancelled});
    expect(published.status == ninfer::runtime::MaterializationStatus::Published &&
               published.activation && manager.ledger().used() == DeviceResources{0, 1, 2, 1} &&
               manager.ledger().lane(LaneId{0}).state == ninfer::runtime::LogicalLaneState::Free,
           "Program publication changed the host ledger before activation adoption");
    const FakeSequenceHandle sequence = published.activation->sequence();
    manager.adopt(std::move(*published.activation));
    published.activation.reset();
    expect(manager.ledger().used() == DeviceResources{1, 1, 3, 2},
           "published activation did not adopt the exact source-to-active delta");
    (void)manager.abort(program, LaneId{0}, sequence);
}

} // namespace

int main() {
    test_selection_and_ledger();
    test_global_catalog_lifecycle();
    test_catalog_pressure_reserves_publication();
    test_materialization_abort_and_adoption();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
