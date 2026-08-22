#pragma once

// OpenAI Responses wire-format layer. It owns request validation, typed Item
// translation, terminal response objects, and semantic SSE events. Engine and
// transport concerns remain outside this file.

#include "serve/request.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ninfer::serve {

struct GenerationOutcome;

struct ResponsesRequest {
    GenerationRequest generation;

    // Current request input only. The HTTP layer prepends instructions and a
    // resolved previous_response_id context to generation.messages before
    // submitting it to GenerationService.
    std::vector<ChatTurn> input_turns;
    std::vector<nlohmann::json> input_items;

    std::optional<std::string> instructions;
    std::optional<std::string> previous_response_id;
    nlohmann::json metadata    = nlohmann::json::object();
    nlohmann::json tools       = nlohmann::json::array();
    nlohmann::json tool_choice = "auto";
    bool store                 = true;
    bool stream                = false;
};

struct ResponsesRuntimeValues {
    float temperature       = 1.0F;
    float top_p             = 1.0F;
    int cached_input_tokens = 0;
};

struct BuiltResponse {
    nlohmann::json body;
    std::vector<nlohmann::json> output_items;
    std::vector<ChatTurn> output_history;
};

// Parse POST /v1/responses. Only NInfer Responses Core capabilities are
// accepted; recognized but unsupported OpenAI fields fail explicitly.
ResponsesRequest parse_responses_request(const nlohmann::json& body, const RequestLimits& limits);

// Parse POST /v1/responses/input_tokens. The current OpenAI endpoint accepts
// model + input; the result still uses GenerationRequest for shared translation.
ResponsesRequest parse_response_input_tokens_request(const nlohmann::json& body,
                                                     const RequestLimits& limits);

// Compose the stateless GenerationRequest submitted to Engine. Current
// instructions are first, followed by the stored response context and current
// input; previous instructions are intentionally absent from stored context.
void compose_responses_generation_messages(ResponsesRequest& request,
                                           const std::vector<ChatTurn>& previous_context);

void inherit_responses_preserve_thinking(ResponsesRequest& request, bool parent_value);

BuiltResponse make_response_object(const std::string& id, std::int64_t created_at,
                                   const ResponsesRequest& request,
                                   const ResponsesRuntimeValues& runtime,
                                   const GenerationOutcome& outcome);

std::string make_response_input_tokens_body(int input_tokens);

struct ResponsesStreamFinish {
    BuiltResponse response;
    std::vector<std::string> events_before_terminal;
};

struct ResponsesItemIds {
    std::string reasoning;
    std::string message;
    std::vector<std::string> function_calls;
};

// Stateful semantic-event encoder for one Responses SSE stream. It assigns
// stable Item IDs and monotonically increasing sequence_number values.
class ResponsesEventStream {
public:
    ResponsesEventStream(std::string response_id, std::int64_t created_at, ResponsesRequest request,
                         ResponsesRuntimeValues runtime);
    ResponsesEventStream(const ResponsesEventStream&)            = delete;
    ResponsesEventStream& operator=(const ResponsesEventStream&) = delete;
    ResponsesEventStream(ResponsesEventStream&&)                 = delete;
    ResponsesEventStream& operator=(ResponsesEventStream&&)      = delete;

    std::vector<std::string> start();
    std::vector<std::string> reasoning_delta(const std::string& text);
    std::vector<std::string> content_delta(const std::string& text);
    ResponsesStreamFinish finish(const GenerationOutcome& outcome);
    std::string terminal(const BuiltResponse& response);
    std::string failed(const ApiError& error);

private:
    nlohmann::json event(std::string type, nlohmann::json fields = nlohmann::json::object());
    std::vector<std::string> ensure_reasoning();
    std::vector<std::string> close_reasoning(const std::string& final_text,
                                             const char* item_status = "completed");
    std::vector<std::string> ensure_message();
    std::vector<std::string> close_message(const std::string& final_text,
                                           const char* item_status = "completed");

    std::string id;
    std::int64_t created_at = 0;
    ResponsesRequest request;
    ResponsesRuntimeValues runtime;
    std::uint64_t sequence = 0;
    int next_output_index  = 0;
    int reasoning_index    = -1;
    int message_index      = -1;
    bool started           = false;
    bool reasoning_started = false;
    bool reasoning_done    = false;
    bool message_started   = false;
    bool message_done      = false;
    bool finish_built      = false;
    bool terminal_emitted  = false;
    std::string reasoning_text;
    std::string content_text;
    ResponsesItemIds ids;
};

std::string new_response_id();
std::string new_response_item_id(const char* prefix);

} // namespace ninfer::serve
