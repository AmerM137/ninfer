#include "ninfer/engine.h"

#include "runtime/contract/sampling.h"
#include "runtime/contract/types.h"
#include "runtime/engine/engine_state.h"

#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

namespace ninfer {
namespace {

runtime::ResolvedRequestOptions resolve_request_options(const ModelSamplingDefaults& defaults,
                                                        SamplingMode mode, RequestOptions options) {
    runtime::ResolvedRequestOptions resolved;
    resolved.execution.sampling =
        runtime::resolve_sampling(defaults, mode, options.execution.sampling);
    resolved.execution.requested_output_tokens = options.execution.requested_output_tokens;
    resolved.execution.allow_prefix_reuse      = options.execution.allow_prefix_reuse;
    resolved.stop                              = std::move(options.stop);
    resolved.output                            = options.output;
    return resolved;
}

std::string context_capacity_error(std::uint32_t prompt_tokens, std::uint32_t max_context) {
    return "prepared prompt has " + std::to_string(prompt_tokens) +
           " tokens, exceeding Engine max_context " + std::to_string(max_context);
}

} // namespace

PreparedPrompt::PreparedPrompt() noexcept                            = default;
PreparedPrompt::~PreparedPrompt()                                    = default;
PreparedPrompt::PreparedPrompt(PreparedPrompt&&) noexcept            = default;
PreparedPrompt& PreparedPrompt::operator=(PreparedPrompt&&) noexcept = default;

PreparedPrompt::PreparedPrompt(std::unique_ptr<runtime::PreparedPromptState> state) noexcept
    : state(std::move(state)) {}

PromptSummary PreparedPrompt::summary() const noexcept {
    return state != nullptr ? state->value.summary() : PromptSummary{};
}

PromptPreparationStats PreparedPrompt::preparation_stats() const noexcept {
    return state != nullptr ? state->value.preparation_stats() : PromptPreparationStats{};
}

PreparedPrompt::operator bool() const noexcept { return state != nullptr; }

GenerationHandle::GenerationHandle() noexcept                              = default;
GenerationHandle::~GenerationHandle()                                      = default;
GenerationHandle::GenerationHandle(GenerationHandle&&) noexcept            = default;
GenerationHandle& GenerationHandle::operator=(GenerationHandle&&) noexcept = default;

GenerationHandle::GenerationHandle(
    std::unique_ptr<runtime::GenerationSubmission> submission) noexcept
    : submission(std::move(submission)) {}

GenerationHandle::operator bool() const noexcept { return submission != nullptr; }

const ResolvedSamplingParameters& GenerationHandle::resolved_sampling() const noexcept {
    static const ResolvedSamplingParameters empty;
    return submission != nullptr ? submission->resolved : empty;
}

GenerationResult GenerationHandle::wait(OutputSink* sink, const CancellationView& cancellation) {
    if (submission == nullptr) { throw std::logic_error("GenerationHandle is empty"); }
    std::unique_ptr<runtime::GenerationSubmission> owned_submission = std::move(submission);
    return owned_submission->wait(sink, cancellation);
}

runtime::EngineState::EngineState(EngineOptions engine_options)
    : options(std::move(engine_options)), device(options.device) {
    targets::ConstructedTarget constructed = targets::construct_target(options, device);
    load                                   = std::move(constructed.load);
    sampling_defaults                      = constructed.sampling_defaults;
    std::visit(
        [&](auto& target) {
            using Instance = typename std::remove_reference_t<decltype(target)>::element_type;
            if constexpr (std::is_same_v<Instance, targets::Qwen3_6_27BInstance>) {
                runtime.emplace<runtime::Qwen3_6_27BRuntime>(std::move(target), options);
            } else {
                runtime.emplace<runtime::Qwen3_6_35BA3BRuntime>(std::move(target), options);
            }
        },
        constructed.active);
}

runtime::EngineState::~EngineState() noexcept {
    std::visit(
        [](auto& active) {
            using Runtime = std::remove_cvref_t<decltype(active)>;
            if constexpr (!std::is_same_v<Runtime, std::monostate>) { active.executor.reset(); }
        },
        runtime);
    try {
        device.synchronize();
    } catch (...) {}
}

Engine::Engine(EngineOptions options)
    : state(std::make_shared<runtime::EngineState>(std::move(options))) {}

Engine::~Engine()                            = default;
Engine::Engine(Engine&&) noexcept            = default;
Engine& Engine::operator=(Engine&&) noexcept = default;

PreparedPrompt Engine::prepare(PromptInput input, const PreparationControl& control) const {
    if (state == nullptr) { throw std::logic_error("Engine is moved from"); }
    const SamplingMode sampling_mode =
        input.options.enable_thinking ? SamplingMode::Thinking : SamplingMode::NonThinking;
    return state->with_active([&](const auto& active) {
        auto prepared               = active.instance->frontend.prepare(std::move(input), control);
        const PromptSummary summary = prepared.summary();
        if (summary.prompt_tokens > active.instance->capacity) {
            throw RequestError(
                RequestErrorKind::ContextLengthExceeded,
                context_capacity_error(summary.prompt_tokens, active.instance->capacity));
        }
        return PreparedPrompt(
            std::make_unique<runtime::PreparedPromptState>(sampling_mode, std::move(prepared)));
    });
}

PreparedPrompt Engine::prepare_tokens(std::vector<TokenId> token_ids,
                                      bool allow_prefix_identity) const {
    if (state == nullptr) { throw std::logic_error("Engine is moved from"); }
    return state->with_active([&](const auto& active) {
        auto prepared =
            active.instance->frontend.prepare_tokens(std::move(token_ids), allow_prefix_identity);
        const PromptSummary summary = prepared.summary();
        if (summary.prompt_tokens > active.instance->capacity) {
            throw RequestError(
                RequestErrorKind::ContextLengthExceeded,
                context_capacity_error(summary.prompt_tokens, active.instance->capacity));
        }
        return PreparedPrompt(std::make_unique<runtime::PreparedPromptState>(SamplingMode::Thinking,
                                                                             std::move(prepared)));
    });
}

std::uint32_t Engine::count_tokens(PromptInput input, const PreparationControl& control) const {
    if (state == nullptr) { throw std::logic_error("Engine is moved from"); }
    return state->with_active([&](const auto& active) {
        return active.instance->frontend.count_tokens(std::move(input), control);
    });
}

PromptCapabilities Engine::prompt_capabilities() const {
    if (state == nullptr) { throw std::logic_error("Engine is moved from"); }
    return state->with_active(
        [](const auto& active) { return active.instance->frontend.prompt_capabilities(); });
}

ModelSamplingDefaults Engine::sampling_defaults() const {
    if (state == nullptr) { throw std::logic_error("Engine is moved from"); }
    return state->sampling_defaults;
}

GenerationHandle Engine::submit(PreparedPrompt prompt, RequestOptions options,
                                std::chrono::steady_clock::time_point pending_deadline) {
    if (state == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (prompt.state == nullptr) { throw std::invalid_argument("PreparedPrompt is empty"); }

    runtime::ResolvedRequestOptions resolved_options = resolve_request_options(
        state->sampling_defaults, prompt.state->sampling_mode, std::move(options));
    const ResolvedSamplingParameters resolved_sampling = resolved_options.execution.sampling;

    const PromptSummary prompt_summary = prompt.state->value.summary();
    if (prompt_summary.prompt_tokens > state->options.max_context) {
        throw RequestError(
            RequestErrorKind::ContextLengthExceeded,
            context_capacity_error(prompt_summary.prompt_tokens, state->options.max_context));
    }
    const double prepare_seconds = prompt.state->value.view().prepare.seconds;
    if (resolved_options.execution.requested_output_tokens == 0) {
        runtime::ImmediateSubmission immediate;

        immediate.result.prompt                  = prompt_summary;
        immediate.result.finish_reason           = FinishReason::OutputLimit;
        immediate.result.timings.prepare_seconds = prepare_seconds;
        immediate.result.timings.total_seconds   = prepare_seconds;
        prompt.state.reset();
        return GenerationHandle(std::make_unique<runtime::GenerationSubmission>(
            state, std::move(immediate), resolved_sampling));
    }

    return state->with_active([&](auto& active) {
        auto submission =
            active.executor->submit(std::move(prompt.state->value), prompt_summary, prepare_seconds,
                                    std::move(resolved_options), pending_deadline);
        return GenerationHandle(std::make_unique<runtime::GenerationSubmission>(
            state, std::move(submission), resolved_sampling));
    });
}

GenerationResult Engine::generate(PreparedPrompt prompt, RequestOptions options, OutputSink* sink,
                                  const CancellationView& cancellation) {
    return submit(std::move(prompt), std::move(options)).wait(sink, cancellation);
}

const EngineOptions& Engine::options() const {
    if (state == nullptr) { throw std::logic_error("Engine is moved from"); }
    return state->options;
}

LoadSummary Engine::load_summary() const {
    if (state == nullptr) { throw std::logic_error("Engine is moved from"); }
    return state->load;
}

MemorySummary Engine::memory_summary() const {
    if (state == nullptr) { throw std::logic_error("Engine is moved from"); }
    return state->with_active([](const auto& active) { return active.executor->memory_summary(); });
}

MediaCacheSummary Engine::media_cache_summary() const {
    if (state == nullptr) { throw std::logic_error("Engine is moved from"); }
    return state->with_active(
        [](const auto& active) { return active.instance->frontend.media_cache_summary(); });
}

RuntimeStats Engine::runtime_stats() const {
    if (state == nullptr) { throw std::logic_error("Engine is moved from"); }
    return state->with_active([](const auto& active) { return active.executor->runtime_stats(); });
}

void Engine::reset_memory_peaks() noexcept {
    if (state == nullptr) { return; }
    std::visit(
        [](auto& active) {
            using Runtime = std::remove_cvref_t<decltype(active)>;
            if constexpr (!std::is_same_v<Runtime, std::monostate>) {
                active.executor->reset_memory_peaks();
            }
        },
        state->runtime);
}

} // namespace ninfer
