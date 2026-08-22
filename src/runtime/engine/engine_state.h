#pragma once

#include "core/device.h"
#include "runtime/engine/concurrent_executor.h"
#include "targets/registry.h"

#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

namespace ninfer::runtime {

struct PreparedPromptState {
    PreparedPromptState(PromptSummary prompt_summary, PromptPreparationStats preparation,
                        SamplingMode mode, targets::qwen3_6::PreparedPrompt prepared)
        : summary(std::move(prompt_summary)), prepare(std::move(preparation)), sampling_mode(mode),
          value(std::move(prepared)) {}

    PromptSummary summary;
    PromptPreparationStats prepare;
    SamplingMode sampling_mode = SamplingMode::Thinking;
    targets::qwen3_6::PreparedPrompt value;
};

using Qwen3_6_27BExecutor    = ConcurrentExecutor<targets::Qwen3_6_27BInstance>;
using Qwen3_6_35BA3BExecutor = ConcurrentExecutor<targets::Qwen3_6_35BA3BInstance>;

struct Qwen3_6_27BRuntime {
    Qwen3_6_27BRuntime(std::unique_ptr<targets::Qwen3_6_27BInstance> target,
                       const EngineOptions& options)
        : instance(std::move(target)), executor(std::in_place, *instance, options) {}

    std::unique_ptr<targets::Qwen3_6_27BInstance> instance;
    // Reset before device synchronization while the target instance remains alive.
    std::optional<Qwen3_6_27BExecutor> executor;
};

struct Qwen3_6_35BA3BRuntime {
    Qwen3_6_35BA3BRuntime(std::unique_ptr<targets::Qwen3_6_35BA3BInstance> target,
                          const EngineOptions& options)
        : instance(std::move(target)), executor(std::in_place, *instance, options) {}

    std::unique_ptr<targets::Qwen3_6_35BA3BInstance> instance;
    // Reset before device synchronization while the target instance remains alive.
    std::optional<Qwen3_6_35BA3BExecutor> executor;
};

using ActiveRuntime = std::variant<std::monostate, Qwen3_6_27BRuntime, Qwen3_6_35BA3BRuntime>;

struct EngineState {
    explicit EngineState(EngineOptions engine_options);
    ~EngineState() noexcept;

    template <class Function>
    decltype(auto) with_active(Function&& function) {
        using Result = std::invoke_result_t<Function&, Qwen3_6_27BRuntime&>;
        return std::visit(
            [&](auto& active) -> Result {
                using Runtime = std::remove_cvref_t<decltype(active)>;
                if constexpr (std::is_same_v<Runtime, std::monostate>) {
                    throw std::logic_error("Engine target is not active");
                } else {
                    return function(active);
                }
            },
            runtime);
    }

    EngineOptions options;
    DeviceContext device;
    LoadSummary load;
    ModelSamplingDefaults sampling_defaults;
    ActiveRuntime runtime;
};

struct ImmediateSubmission {
    GenerationResult result;

    GenerationResult wait(OutputSink*, const CancellationView& cancellation) {
        if (cancellation.requested()) { result.finish_reason = FinishReason::Cancelled; }
        return std::move(result);
    }
};

using GenerationSubmissionValue = std::variant<ImmediateSubmission, Qwen3_6_27BExecutor::Submission,
                                               Qwen3_6_35BA3BExecutor::Submission>;

struct GenerationSubmission {
    template <class Submission>
    GenerationSubmission(std::shared_ptr<EngineState> engine_state, Submission value,
                         ResolvedSamplingParameters sampling)
        : engine(std::move(engine_state)), submission(std::move(value)), resolved(sampling) {}

    GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
        return std::visit([&](auto& value) { return value.wait(sink, cancellation); }, submission);
    }

    // ConcurrentExecutor::Submission refers back to its owning executor.
    std::shared_ptr<EngineState> engine;
    GenerationSubmissionValue submission;
    ResolvedSamplingParameters resolved;
};

} // namespace ninfer::runtime
