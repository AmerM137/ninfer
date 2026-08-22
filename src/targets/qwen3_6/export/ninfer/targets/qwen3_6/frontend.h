#pragma once

#include "ninfer/types.h"
#include "runtime/contract/types.h"

#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ninfer::targets::qwen3_6 {

inline constexpr std::size_t kTokenDomain = 248077;

struct FrontendOptions {
    bool vision_enabled                    = true;
    std::uint32_t max_context              = 2'048;
    std::size_t media_cache_bytes          = kDefaultMediaCacheBytes;
    std::size_t media_live_bytes           = kDefaultMediaLiveBytes;
    std::uint32_t media_preprocess_threads = 0;
};

struct FrontendResources;
class Frontend;
class FrontendTestAccess;

namespace frontend_internal {
class Tokenizer;
}

struct OutputDecoderState {
    std::string utf8_pending;
    std::string think_marker_pending;
    std::array<std::string, 2> stop_pending;
    bool in_reasoning              = false;
    bool strip_content_leading     = false;
    bool terminal                  = false;
    std::uint64_t decoded_bytes    = 0;
    std::uint32_t reasoning_tokens = 0;
};

class PublishedOutput {
public:
    using iterator       = std::array<OutputDelta, 2>::iterator;
    using const_iterator = std::array<OutputDelta, 2>::const_iterator;

    PublishedOutput()                                  = default;
    PublishedOutput(const PublishedOutput&)            = default;
    PublishedOutput& operator=(const PublishedOutput&) = default;
    PublishedOutput(PublishedOutput&& other) noexcept;
    PublishedOutput& operator=(PublishedOutput&& other) noexcept;

    [[nodiscard]] bool empty() const noexcept { return count == 0; }

    [[nodiscard]] std::size_t size() const noexcept { return count; }

    [[nodiscard]] iterator begin() noexcept { return values.begin(); }

    [[nodiscard]] const_iterator begin() const noexcept { return values.begin(); }

    [[nodiscard]] iterator end() noexcept { return values.begin() + count; }

    [[nodiscard]] const_iterator end() const noexcept { return values.begin() + count; }

    [[nodiscard]] OutputDelta& back() noexcept { return values[count - 1]; }

    [[nodiscard]] const OutputDelta& back() const noexcept { return values[count - 1]; }

    void clear() noexcept;
    void push_back(OutputDelta value);

private:
    std::array<OutputDelta, 2> values{};
    std::size_t count = 0;
};

class OutputSession {
public:
    ~OutputSession()                        = default;
    OutputSession(OutputSession&&) noexcept = default;

    OutputSession()                                = delete;
    OutputSession(const OutputSession&)            = delete;
    OutputSession& operator=(const OutputSession&) = delete;
    OutputSession& operator=(OutputSession&&)      = delete;

    [[nodiscard]] runtime::OutputDecision preview(std::span<const TokenId> tokens,
                                                  std::uint32_t budget_remaining,
                                                  FinishReason limit_reason);
    [[nodiscard]] runtime::OutputDecision preview_terminal(FinishReason reason);
    [[nodiscard]] PublishedOutput commit_preview() noexcept;
    [[nodiscard]] std::uint32_t reasoning_tokens() const noexcept;

private:
    OutputSession(std::shared_ptr<const frontend_internal::Tokenizer> tokenizer, StopPolicy policy,
                  OutputOptions output, bool starts_in_reasoning) noexcept;

    std::shared_ptr<const frontend_internal::Tokenizer> tokenizer;
    StopPolicy policy;
    bool preserve_special = false;
    OutputDecoderState decoder;
    OutputDecoderState preview_decoder;
    PublishedOutput preview_output;
    bool preview_ready = false;

    friend class Frontend;
};

class Frontend {
public:
    Frontend(const Frontend&);
    Frontend& operator=(const Frontend&);
    Frontend(Frontend&&) noexcept;
    Frontend& operator=(Frontend&&) noexcept;
    ~Frontend();

    [[nodiscard]] PreparedPrompt prepare(PromptInput input,
                                         const PreparationControl& control = {}) const;
    [[nodiscard]] std::uint32_t count_tokens(PromptInput input,
                                             const PreparationControl& control = {}) const;
    [[nodiscard]] PreparedPrompt prepare_tokens(std::vector<TokenId> token_ids,
                                                bool allow_prefix_identity = true) const;
    [[nodiscard]] PromptCapabilities prompt_capabilities() const noexcept;
    [[nodiscard]] MediaCacheSummary media_cache_summary() const;
    [[nodiscard]] OutputSession make_output_session(const PreparedPrompt& prompt,
                                                    const StopPolicy& caller_stop,
                                                    const OutputOptions& output = {}) const;
    [[nodiscard]] const StopPolicy& default_stop_policy() const noexcept;

private:
    struct FrontendState;
    explicit Frontend(std::shared_ptr<const FrontendState> state) noexcept;
    std::shared_ptr<const FrontendState> state;

    friend class FrontendTestAccess;
    friend Frontend make_frontend(const FrontendResources& resources, FrontendOptions options);
};

[[nodiscard]] Frontend make_frontend(const FrontendResources& resources, FrontendOptions options);

} // namespace ninfer::targets::qwen3_6
