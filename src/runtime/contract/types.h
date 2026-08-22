#pragma once

#include "ninfer/types.h"

#include <atomic>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace ninfer::runtime {

using ::ninfer::FinishReason;
using ::ninfer::KvCapacityMode;
using ::ninfer::KvCapacityPolicy;
using ::ninfer::OutputChannel;
using ::ninfer::ResolvedSamplingParameters;
using ::ninfer::StopPolicy;
using ::ninfer::StopString;
using ::ninfer::TokenId;

// Engine has already selected the registered model/mode preset, applied every explicit override,
// and validated these values before constructing the runtime request.
struct ResolvedExecutionOptions {
    ResolvedSamplingParameters sampling;
    std::uint32_t requested_output_tokens = 0;
    bool allow_prefix_reuse               = true;
};

struct ResolvedRequestOptions {
    ResolvedExecutionOptions execution;
    StopPolicy stop;
    OutputOptions output;
};

struct OutputDecision {
    std::uint32_t accepted_tokens = 0;
    FinishReason finish_reason    = FinishReason::None;

    [[nodiscard]] bool finished() const noexcept { return finish_reason != FinishReason::None; }
};

struct LaneId {
    std::uint32_t value = 0;

    [[nodiscard]] friend constexpr bool operator==(LaneId, LaneId) noexcept  = default;
    [[nodiscard]] friend constexpr auto operator<=>(LaneId, LaneId) noexcept = default;
};

enum class ConsumeStatus : std::uint8_t {
    Consumed,
    InvariantMismatch,
};

enum class CommitDisposition : std::uint8_t {
    Active,
    Finishable,
    CancelledReleased,
};

// The product Engine only needs statistics for rows whose sequence is released by commit.
// Direct diagnostic callers may temporarily request cumulative snapshots for every row.
enum class CommitObservation : std::uint8_t {
    ReleasedRowsOnly,
    AllRows,
};

struct CommitDecision {
    std::uint32_t accepted_tokens = 0;
    bool terminal                 = false;
    bool cancelled                = false;
};

// Device ownership in the independently exhausted runtime resource domains. Values are already
// rounded to the physical allocation granularity by the target.
struct DeviceResources {
    std::uint32_t active_lanes     = 0;
    std::uint32_t state_slots      = 0;
    std::uint32_t main_kv_pages    = 0;
    std::uint32_t backend_kv_pages = 0;

    [[nodiscard]] friend constexpr bool operator==(const DeviceResources&,
                                                   const DeviceResources&) noexcept = default;
};

// A candidate separates the steady-state active ownership from physical resources which must
// coexist with an intact source before publication, and from source resources reclassified at
// publication. None of these fields imply a physical free/reallocate cycle.
struct ResourceDemand {
    DeviceResources active_entitlement;
    DeviceResources prepublish_additional;
    DeviceResources source_conversions;

    [[nodiscard]] friend constexpr bool operator==(const ResourceDemand&,
                                                   const ResourceDemand&) noexcept = default;
};

struct ResourceDelta {
    DeviceResources removed;
    DeviceResources added;

    [[nodiscard]] friend constexpr bool operator==(const ResourceDelta&,
                                                   const ResourceDelta&) noexcept = default;
};

enum class Readiness : std::uint8_t {
    Ready,
    NeedsTransfer,
    TemporarilyBlocked,
    PermanentlyInfeasible,
};

// Non-owning cancellation observation used at the synchronous materialization publication point.
// The request record owns the flag for longer than the worker can retain this view.
struct CancellationFlagView {
    const std::atomic<bool>* flag = nullptr;

    [[nodiscard]] bool requested() const noexcept {
        return flag != nullptr && flag->load(std::memory_order_acquire);
    }
};

enum class MaterializationStatus : std::uint8_t {
    Published,
    Aborted,
};

enum class CheckpointKind : std::uint8_t {
    SessionEndpoint,
    TurnClosure,
    ResponseReplay,
};

struct CheckpointRef {
    CheckpointKind kind    = CheckpointKind::SessionEndpoint;
    std::uint32_t frontier = 0;
    std::uint32_t ordinal  = 0;

    [[nodiscard]] friend constexpr bool operator==(CheckpointRef, CheckpointRef) noexcept = default;
};

struct ContinuationSummary {
    DeviceResources footprint;
    CheckpointRef endpoint;
    std::optional<CheckpointRef> rewrite;
    std::uint64_t rebuild_work_quanta = 0;

    [[nodiscard]] friend bool operator==(const ContinuationSummary&,
                                         const ContinuationSummary&) noexcept = default;
};

struct ContinuationId {
    std::uint64_t value = 0;

    [[nodiscard]] friend constexpr bool operator==(ContinuationId,
                                                   ContinuationId) noexcept  = default;
    [[nodiscard]] friend constexpr auto operator<=>(ContinuationId,
                                                    ContinuationId) noexcept = default;
};

struct Revision {
    std::uint64_t value = 0;

    [[nodiscard]] friend constexpr bool operator==(Revision, Revision) noexcept = default;
};

struct RequestPlanSummary {
    std::uint32_t prompt_tokens           = 0;
    std::uint32_t reusable_prompt_tokens  = 0;
    std::uint32_t requested_output_tokens = 0;
    std::uint32_t effective_output_tokens = 0;
    FinishReason effective_limit_reason   = FinishReason::None;
    PrefixReusePath prefix_reuse_path     = PrefixReusePath::FullReset;
    DeviceResources admission;
    std::uint64_t service_work_quanta = 0;
};

struct BeginSummary {
    std::uint32_t prompt_tokens        = 0;
    std::uint32_t reused_prompt_tokens = 0;
    PrefixReusePath prefix_reuse_path  = PrefixReusePath::FullReset;
};

struct GeneratedRound {
    std::span<const TokenId> tokens;
};

struct BatchedGeneratedRound {
    std::span<const TokenId> tokens;
    std::span<const std::int32_t> row_counts;
    std::uint32_t row_stride = 1;
};

struct PrefillStepResult {
    BeginSummary summary;
    GeneratedRound round;
    std::uint32_t processed_prompt_tokens = 0;
    bool complete                         = false;
};

struct RoundBudget {
    std::uint32_t generated_tokens_remaining = 0;
};

// Target-produced affine reservation curve for one Main KV physical-capacity axis. The byte
// values come from complete target physical layout plans, not from a model geometry formula in
// the common runtime.
struct SequenceCapacityCurve {
    std::uint32_t main_page_tokens                   = 0;
    std::uint32_t minimum_main_page_groups           = 0;
    std::uint32_t maximum_main_page_groups           = 0;
    std::size_t minimum_device_reservation_bytes     = 0;
    std::size_t bytes_per_additional_main_page_group = 0;

    [[nodiscard]] std::size_t reservation_bytes(std::uint32_t main_page_groups) const;
    [[nodiscard]] std::uint32_t resolved_tokens(std::uint32_t main_page_groups) const;
};

struct KvCapacityResolution {
    KvCapacityMode mode                              = KvCapacityMode::Explicit;
    std::uint32_t main_page_groups                   = 0;
    std::uint32_t maximum_main_page_groups           = 0;
    std::uint32_t resolved_tokens                    = 0;
    std::size_t minimum_runtime_reservation_bytes    = 0;
    std::size_t bytes_per_additional_main_page_group = 0;
    std::size_t runtime_reservation_bytes            = 0;
    std::size_t available_after_weights_bytes        = 0;
    std::size_t available_after_startup_bytes        = 0;
    std::size_t automatic_headroom_bytes             = 0;
    std::size_t planned_slack_bytes                  = 0;
};

} // namespace ninfer::runtime
