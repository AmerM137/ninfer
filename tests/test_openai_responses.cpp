// Host-only contract tests for the OpenAI Responses adapter. These tests keep request parsing,
// previous_response_id state reconstruction, output encoding, and SSE sequencing independent of
// an Engine instance.

#include "serve/generation_service.h"
#include "serve/openai_responses.h"
#include "serve/translate.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using Json = nlohmann::json;
using namespace ninfer::serve;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

RequestLimits limits() {
    RequestLimits value;
    value.default_max_tokens = 256;
    return value;
}

std::string api_code(const std::function<void()>& action) {
    try {
        action();
    } catch (const ApiException& exception) { return exception.error().code; } catch (...) {
        return "wrong_exception";
    }
    return {};
}

Json parse_event(const std::string& wire) {
    const std::size_t newline = wire.find('\n');
    if (!wire.starts_with("event: ") || newline == std::string::npos || !wire.ends_with("\n\n")) {
        throw std::runtime_error("invalid SSE framing");
    }
    const std::string type   = wire.substr(7, newline - 7);
    const std::string prefix = "data: ";
    if (wire.compare(newline + 1, prefix.size(), prefix) != 0) {
        throw std::runtime_error("missing SSE data field");
    }
    const std::size_t begin = newline + 1 + prefix.size();
    Json payload            = Json::parse(wire.substr(begin, wire.size() - begin - 2));
    if (payload.at("type") != type) { throw std::runtime_error("SSE type mismatch"); }
    return payload;
}

ChatTurn text_turn(ninfer::ChatRole role, std::string text) {
    ChatTurn turn;
    turn.role = role;
    ContentPart part;
    part.kind     = ContentKind::Text;
    part.type_raw = role == ninfer::ChatRole::Assistant ? "output_text" : "input_text";
    part.text     = std::move(text);
    turn.content.push_back(std::move(part));
    return turn;
}

ChatTurn call_turn(std::initializer_list<std::pair<const char*, const char*>> calls) {
    ChatTurn turn;
    turn.role = ninfer::ChatRole::Assistant;
    for (const auto& [id, name] : calls) {
        turn.tool_calls.push_back(
            ToolCall{.id = id, .name = name, .arguments_json = R"({"value":1})"});
    }
    return turn;
}

StoredOpenAIResponse stored_parent(OpenAIResponseContext context) {
    StoredOpenAIResponse record;
    record.id                = "resp_parent";
    record.session_key       = "responses-session";
    record.response          = Json{{"id", record.id}, {"object", "response"}};
    record.context           = std::move(context);
    record.preserve_thinking = true;
    return record;
}

GenerationOutcome sample_outcome() {
    GenerationOutcome outcome;
    outcome.text                            = "answer";
    outcome.reasoning                       = "thought";
    outcome.prompt_tokens                   = 11;
    outcome.completion_tokens               = 7;
    outcome.reasoning_tokens                = 3;
    outcome.finish_reason                   = ninfer::FinishReason::StopToken;
    outcome.metrics.prefix_cache_hit_tokens = 4;
    return outcome;
}

int test_basic_request_and_resolution() {
    const Json body = {{"model", "qwen3.6-27b"},
                       {"input", "hello"},
                       {"instructions", "be concise"},
                       {"max_output_tokens", 64},
                       {"temperature", 0.3},
                       {"top_p", 0.8},
                       {"reasoning", Json{{"effort", "medium"}}},
                       {"metadata", Json{{"trace", "abc"}}}};
    const OpenAIResponsesCreateRequest request =
        parse_openai_responses_create_request(body, limits());

    int failures = 0;
    failures += check(request.prompt.model == "qwen3.6-27b", "model parsed");
    failures += check(request.prompt.input_turns.size() == 1 &&
                          request.prompt.input_turns[0].role == ninfer::ChatRole::User &&
                          request.prompt.input_turns[0].content[0].text == "hello",
                      "string input normalized to a user turn");
    failures +=
        check(request.prompt.input_items.size() == 1 &&
                  request.prompt.input_items[0].at("type") == "message" &&
                  request.prompt.input_items[0].at("content")[0].at("type") == "input_text" &&
                  request.prompt.input_items[0].contains("id"),
              "canonical input Item built");
    failures += check(request.prompt.instructions && *request.prompt.instructions == "be concise",
                      "instructions retained outside current input state");
    failures += check(request.requested_max_output_tokens == 64 &&
                          request.prompt.generation.max_tokens == 64,
                      "explicit output budget reaches generation request");
    failures +=
        check(request.prompt.generation.reasoning_effort == RequestedReasoningEffort::Medium,
              "reasoning effort parsed");
    failures += check(request.store && !request.stream && request.parallel_tool_calls,
                      "Responses defaults applied");

    OpenAIResponsesStore store(8, 1ULL << 20);
    const OpenAIResponsesResolvedPrompt resolved =
        resolve_openai_responses_prompt(request.prompt, store, "resp_current", true);
    failures += check(resolved.generation.messages.size() == 2 &&
                          resolved.generation.messages[0].role == ninfer::ChatRole::Developer &&
                          resolved.generation.messages[0].content[0].text == "be concise" &&
                          resolved.generation.messages[1].content[0].text == "hello",
                      "instructions and current input composed in model order");
    failures +=
        check(resolved.session_key == "resp_current" &&
                  resolved.cache_hints.session_key == "resp_current" &&
                  resolved.cache_hints.retention == ninfer::CacheRetentionHint::LiveSession &&
                  resolved.cache_hints.update_session_index,
              "stored root response receives one live Engine session");
    return failures;
}

int test_budgets_and_nonsemantic_hints() {
    const Json base = {{"model", "m"}, {"input", "hello"}};
    int failures    = 0;

    const OpenAIResponsesCreateRequest omitted =
        parse_openai_responses_create_request(base, limits());
    failures +=
        check(!omitted.requested_max_output_tokens && omitted.prompt.generation.max_tokens == 256,
              "omitted output budget uses the server default without changing its echo");
    for (const int budget : {0, 1, 15}) {
        Json body                 = base;
        body["max_output_tokens"] = budget;
        const OpenAIResponsesCreateRequest parsed =
            parse_openai_responses_create_request(body, limits());
        failures += check(parsed.requested_max_output_tokens == budget &&
                              parsed.prompt.generation.max_tokens == budget,
                          "non-negative output budget accepted");
    }

    Json hints = base;
    hints.update({{"background", false},
                  {"include", Json::array()},
                  {"max_tool_calls", 0},
                  {"prompt_cache_key", "stable-prefix"},
                  {"prompt_cache_retention", "24h"},
                  {"prompt_cache_options", Json{{"mode", "explicit"}, {"ttl", "30m"}}},
                  {"safety_identifier", "local-user"},
                  {"service_tier", "auto"},
                  {"stream_options", Json{{"include_obfuscation", false}}},
                  {"text", Json{{"format", Json{{"type", "text"}}}, {"verbosity", "medium"}}},
                  {"top_logprobs", 0},
                  {"truncation", "disabled"},
                  {"user", "legacy-user"}});
    const OpenAIResponsesCreateRequest accepted =
        parse_openai_responses_create_request(hints, limits());
    failures += check(accepted.max_tool_calls == 0,
                      "typed nonsemantic hints are accepted without changing generation");
    return failures;
}

int test_typed_items_and_cache_markers() {
    const Json body = {
        {"model", "m"},
        {"input",
         Json::array(
             {Json{{"id", "rs_1"},
                   {"type", "reasoning"},
                   {"summary", Json::array()},
                   {"content",
                    Json::array({Json{{"type", "reasoning_text"}, {"text", "use tools"}}})}},
              Json{{"id", "fc_1"},
                   {"type", "function_call"},
                   {"call_id", "call_1"},
                   {"name", "weather"},
                   {"arguments", R"({"city":"Paris"})"}},
              Json{{"id", "fco_1"},
                   {"type", "function_call_output"},
                   {"call_id", "call_1"},
                   {"output",
                    Json::array({Json{{"type", "input_text"},
                                      {"text", "20C"},
                                      {"prompt_cache_breakpoint", Json{{"mode", "explicit"}}}},
                                 Json{{"type", "input_image"},
                                      {"image_url", "data:image/png;base64,AA=="},
                                      {"detail", "auto"}}})}},
              Json{{"id", "msg_refusal"},
                   {"type", "message"},
                   {"role", "assistant"},
                   {"status", "incomplete"},
                   {"phase", "commentary"},
                   {"content",
                    Json::array({Json{{"type", "refusal"}, {"refusal", "cannot answer that"}}})}},
              Json{{"id", "msg_1"},
                   {"type", "message"},
                   {"role", "user"},
                   {"content", Json::array({Json{
                                   {"type", "input_text"},
                                   {"text", "describe"},
                                   {"prompt_cache_breakpoint", Json{{"mode", "explicit"}}}}})}}})}};

    const OpenAIResponsesCreateRequest request =
        parse_openai_responses_create_request(body, limits());
    int failures = 0;
    failures += check(request.prompt.input_turns.size() == 4 &&
                          request.prompt.input_turns[0].role == ninfer::ChatRole::Assistant &&
                          request.prompt.input_turns[0].reasoning_content == "use tools" &&
                          request.prompt.input_turns[0].tool_calls.size() == 1,
                      "reasoning and function call form one assistant turn");
    failures += check(request.prompt.input_turns[1].role == ninfer::ChatRole::Tool &&
                          request.prompt.input_turns[1].content.size() == 2 &&
                          request.prompt.input_turns[1].content[0].cache_boundary_after ==
                              ninfer::PromptCacheMarkerKind::SharedStablePrefix &&
                          request.prompt.input_turns[1].content[1].kind == ContentKind::Image,
                      "typed multimodal tool output and explicit cache marker preserved");
    failures += check(request.prompt.input_turns[2].role == ninfer::ChatRole::Assistant &&
                          request.prompt.input_turns[2].content[0].text == "cannot answer that" &&
                          request.prompt.input_items[3].at("status") == "incomplete" &&
                          request.prompt.input_items[3].at("phase") == "commentary",
                      "refusal text and harmless assistant metadata are accepted");
    failures += check(request.prompt.input_turns[3].content[0].cache_boundary_after ==
                          ninfer::PromptCacheMarkerKind::SharedStablePrefix,
                      "message cache marker preserved");

    OpenAIResponsesStore store(8, 1ULL << 20);
    const OpenAIResponsesResolvedPrompt resolved =
        resolve_openai_responses_prompt(request.prompt, store, "resp_typed", true);
    failures += check(resolved.generation.messages.size() == 4 &&
                          resolved.generation.messages[1].tool_call_id == "call_1",
                      "typed Items survive call-graph normalization");
    const ninfer::PromptInput translated = to_prompt_input(
        resolved.generation, ResolvedPromptSemantics{}, [](const ContentPart& part) {
            ninfer::OwnedMedia media;
            media.kind  = part.kind == ContentKind::Image ? ninfer::MediaKind::Image
                                                          : ninfer::MediaKind::Video;
            media.bytes = {1};
            return media;
        });
    failures += check(translated.context_cache.markers.size() == 2 &&
                          translated.context_cache.markers[0].kind ==
                              ninfer::PromptCacheMarkerKind::SharedStablePrefix &&
                          translated.context_cache.markers[0].location ==
                              ninfer::PromptCacheMarkerLocation::MessagePartBoundary &&
                          translated.context_cache.markers[1].kind ==
                              ninfer::PromptCacheMarkerKind::SharedStablePrefix,
                      "Responses breakpoints become shared Engine part boundaries");
    return failures;
}

int test_tools_and_effective_subset() {
    const Json weather = {{"type", "function"},
                          {"name", "weather"},
                          {"description", "Get weather"},
                          {"parameters", Json{{"type", "object"}}},
                          {"strict", false}};
    const Json clock   = {{"type", "function"},
                          {"name", "clock"},
                          {"allowed_callers", Json::array({"direct"})},
                          {"defer_loading", false}};
    const Json body    = {
        {"model", "m"},
        {"input", "time"},
        {"tools", Json::array({weather, clock})},
        {"tool_choice",
            Json{{"type", "allowed_tools"},
                 {"mode", "auto"},
                 {"tools", Json::array({Json{{"type", "function"}, {"name", "clock"}}})}}}};
    const OpenAIResponsesCreateRequest request =
        parse_openai_responses_create_request(body, limits());
    int failures = 0;
    failures += check(request.tools.size() == 2 && request.prompt.generation.tools.size() == 1 &&
                          request.prompt.generation.tools[0].name == "clock",
                      "wire tool list and effective callable subset remain distinct");

    Json none           = body;
    none["tool_choice"] = "none";
    const OpenAIResponsesCreateRequest disabled =
        parse_openai_responses_create_request(none, limits());
    failures += check(disabled.prompt.generation.tool_choice.mode == ToolChoiceMode::None &&
                          !disabled.prompt.generation.uses_tools(),
                      "tool_choice none disables generation tools without deleting their echo");
    return failures;
}

int test_explicit_rejections() {
    const Json base = {{"model", "m"}, {"input", "hello"}};
    int failures    = 0;

    Json value     = base;
    value["tools"] = Json::array({Json{{"type", "function"}, {"name", "f"}, {"strict", true}}});
    failures += check(api_code([&] {
                          (void)parse_openai_responses_create_request(value, limits());
                      }) == "strict_tools_not_supported",
                      "strict function schema is rejected explicitly");

    value         = base;
    value["text"] = Json{{"format", Json{{"type", "json_schema"}}}};
    failures += check(api_code([&] {
                          (void)parse_openai_responses_create_request(value, limits());
                      }) == "structured_outputs_not_supported",
                      "structured output is rejected explicitly");

    value               = base;
    value["background"] = true;
    failures += check(api_code([&] {
                          (void)parse_openai_responses_create_request(value, limits());
                      }) == "background_not_supported",
                      "background execution is rejected explicitly");

    value                 = base;
    value["conversation"] = "conv_1";
    failures += check(api_code([&] {
                          (void)parse_openai_responses_create_request(value, limits());
                      }) == "conversations_not_supported",
                      "unavailable platform state is rejected explicitly");

    value               = base;
    value["truncation"] = "auto";
    failures += check(api_code([&] {
                          (void)parse_openai_responses_create_request(value, limits());
                      }) == "truncation_not_supported",
                      "lossy server truncation is rejected explicitly");

    value                  = base;
    value["made_up_field"] = 1;
    failures += check(api_code([&] {
                          (void)parse_openai_responses_create_request(value, limits());
                      }) == "unknown_parameter",
                      "unknown request parameter is rejected");
    return failures;
}

int test_previous_response_call_graph() {
    OpenAIResponsesStore store(16, 1ULL << 20);
    const OpenAIResponseContext context =
        append_openai_response_context({}, {text_turn(ninfer::ChatRole::User, "run both"),
                                            call_turn({{"call_a", "alpha"}, {"call_b", "beta"}})});
    store.put(stored_parent(context));

    const Json reordered_body = {
        {"model", "m"},
        {"previous_response_id", "resp_parent"},
        {"input",
         Json::array(
             {Json{{"type", "function_call_output"}, {"call_id", "call_b"}, {"output", "B"}},
              Json{{"type", "function_call_output"}, {"call_id", "call_a"}, {"output", "A"}}})}};
    const OpenAIResponsesCreateRequest request =
        parse_openai_responses_create_request(reordered_body, limits());
    const OpenAIResponsesResolvedPrompt resolved =
        resolve_openai_responses_prompt(request.prompt, store, "resp_child", true);
    int failures = 0;
    failures += check(resolved.generation.messages.size() == 4 &&
                          resolved.generation.messages[2].tool_call_id == "call_a" &&
                          resolved.generation.messages[3].tool_call_id == "call_b",
                      "complete tool results are normalized to declaration order");
    failures += check(resolved.session_key == "responses-session" &&
                          resolved.generation.preserve_thinking == true &&
                          !resolved.preserve_thinking_semantic_change,
                      "parent continuation inherits session and prompt semantics");

    const OpenAIResponsesResolvedPrompt disposable =
        resolve_openai_responses_prompt(request.prompt, store, "resp_disposable", false);
    failures +=
        check(disposable.session_key == "responses-session" &&
                  disposable.cache_hints.retention == ninfer::CacheRetentionHint::Disposable &&
                  !disposable.cache_hints.update_session_index,
              "store=false consumes parent session without advancing it");

    Json partial     = reordered_body;
    partial["input"] = Json::array(
        {Json{{"type", "function_call_output"}, {"call_id", "call_b"}, {"output", "B"}}});
    const OpenAIResponsesCreateRequest invalid =
        parse_openai_responses_create_request(partial, limits());
    failures += check(api_code([&] {
                          (void)resolve_openai_responses_prompt(invalid.prompt, store,
                                                                "resp_invalid", true);
                      }) == "invalid_tool_history",
                      "non-prefix partial tool results are rejected");

    Json unknown     = reordered_body;
    unknown["input"] = Json::array(
        {Json{{"type", "function_call_output"}, {"call_id", "call_unknown"}, {"output", "x"}}});
    const OpenAIResponsesCreateRequest unknown_request =
        parse_openai_responses_create_request(unknown, limits());
    failures += check(api_code([&] {
                          (void)resolve_openai_responses_prompt(unknown_request.prompt, store,
                                                                "resp_unknown", true);
                      }) == "invalid_tool_history",
                      "unknown tool result call_id is rejected");

    OpenAIResponsesCreateRequest missing_parent = request;
    missing_parent.prompt.previous_response_id  = "resp_missing";
    failures += check(api_code([&] {
                          (void)resolve_openai_responses_prompt(missing_parent.prompt, store,
                                                                "resp_new", true);
                      }) == "response_not_found",
                      "missing parent response is reported precisely");
    return failures;
}

int test_response_object() {
    const OpenAIResponsesCreateRequest request =
        parse_openai_responses_create_request(Json{{"model", "m"},
                                                   {"input", "hello"},
                                                   {"reasoning", Json{{"effort", "low"}}},
                                                   {"store", false}},
                                              limits());
    OpenAIResponsesRuntimeValues runtime;
    runtime.temperature = 0.6F;
    runtime.top_p       = 0.95F;
    const BuiltOpenAIResponse built =
        make_openai_response_object("resp_test", 123, request, runtime, sample_outcome());
    const Json& response = built.body;
    int failures         = 0;
    failures += check(response.at("object") == "response" && response.at("status") == "completed" &&
                          response.at("completed_at").is_number_integer(),
                      "completed response has a completion timestamp");
    failures += check(response.at("max_output_tokens").is_null(),
                      "omitted output budget remains null in the response");
    failures += check(response.at("output").size() == 2 &&
                          response.at("output")[0].at("type") == "reasoning" &&
                          response.at("output")[1].at("type") == "message",
                      "reasoning and message are emitted as typed output Items");
    failures +=
        check(response.at("usage").at("input_tokens_details").at("cached_tokens") == 4 &&
                  response.at("usage").at("output_tokens_details").at("reasoning_tokens") == 3 &&
                  response.at("usage").at("total_tokens") == 18,
              "usage and cached token details serialized");

    GenerationOutcome incomplete = sample_outcome();
    incomplete.text.clear();
    incomplete.finish_reason = ninfer::FinishReason::OutputLimit;
    const BuiltOpenAIResponse limited =
        make_openai_response_object("resp_limit", 123, request, runtime, incomplete);
    failures += check(limited.body.at("status") == "incomplete" &&
                          limited.body.at("completed_at").is_null() &&
                          limited.body.at("output").size() == 1 &&
                          limited.body.at("output")[0].at("type") == "reasoning",
                      "reasoning-only incomplete output does not invent an empty message");

    GenerationOutcome tools = sample_outcome();
    tools.text.clear();
    tools.reasoning.clear();
    tools.tool_calls.push_back(
        ninfer::GeneratedToolCall{.name = "weather", .arguments_json = R"({"city":"Paris"})"});
    const BuiltOpenAIResponse tool_response =
        make_openai_response_object("resp_tool", 123, request, runtime, tools);
    const Json& item = tool_response.body.at("output").at(0);
    failures += check(item.at("type") == "function_call" &&
                          item.at("call_id").get<std::string>().starts_with("call_") &&
                          tool_response.output_history[0].tool_calls[0].id ==
                              item.at("call_id").get<std::string>(),
                      "wire and continuation history share one stable function call_id");
    return failures;
}

int test_sse_sequence_and_failures() {
    OpenAIResponsesCreateRequest request = parse_openai_responses_create_request(
        Json{{"model", "m"}, {"input", "hello"}, {"stream", true}}, limits());
    OpenAIResponsesEventStream encoder("resp_stream", 123, request, {});
    std::vector<std::string> wire = encoder.start();
    std::vector<std::string> next = encoder.reasoning_delta("thought");
    wire.insert(wire.end(), next.begin(), next.end());
    next = encoder.content_delta("ans");
    wire.insert(wire.end(), next.begin(), next.end());
    OpenAIResponsesStreamFinish finish = encoder.finish(sample_outcome());
    wire.insert(wire.end(), finish.events_before_terminal.begin(),
                finish.events_before_terminal.end());
    wire.push_back(encoder.terminal(finish.response));

    int failures                    = 0;
    std::uint64_t expected_sequence = 0;
    std::string text_deltas;
    for (const std::string& event : wire) {
        failures += check(event.find("[DONE]") == std::string::npos,
                          "Responses stream does not use Chat [DONE]");
        const Json payload = parse_event(event);
        failures += check(payload.at("sequence_number") == expected_sequence++,
                          "SSE sequence numbers are contiguous");
        if (payload.at("type") == "response.output_text.delta") {
            text_deltas += payload.at("delta").get<std::string>();
        }
    }
    failures += check(parse_event(wire.front()).at("type") == "response.created" &&
                          parse_event(wire.back()).at("type") == "response.completed" &&
                          text_deltas == "answer",
                      "SSE starts, reconstructs output, and terminates canonically");

    OpenAIResponsesEventStream failed("resp_failed", 123, std::move(request), {});
    (void)failed.start();
    const Json failure = parse_event(failed.failed(
        ApiError{.status = 500, .type = "server_error", .message = "stopped", .code = "stopped"}));
    failures += check(failure.at("type") == "response.failed" &&
                          failure.at("response").at("status") == "failed" &&
                          failure.at("response").at("completed_at").is_null(),
                      "runtime failure uses response.failed with no completion timestamp");

    OpenAIResponsesCreateRequest cancelled_request = parse_openai_responses_create_request(
        Json{{"model", "m"}, {"input", "hello"}, {"stream", true}}, limits());
    OpenAIResponsesEventStream cancelled("resp_cancelled", 123, cancelled_request, {});
    (void)cancelled.start();
    GenerationOutcome cancelled_outcome;
    cancelled_outcome.finish_reason                    = ninfer::FinishReason::Cancelled;
    const OpenAIResponsesStreamFinish cancelled_finish = cancelled.finish(cancelled_outcome);
    const Json cancelled_terminal = parse_event(cancelled.terminal(cancelled_finish.response));
    failures += check(cancelled_terminal.at("type") == "response.failed" &&
                          cancelled_terminal.at("response").at("status") == "cancelled",
                      "cancelled generation uses the protocol terminal failure event");
    return failures;
}

int test_input_tokens_uses_shared_state_path() {
    OpenAIResponsesStore store(8, 1ULL << 20);
    store.put(stored_parent(
        append_openai_response_context({}, {text_turn(ninfer::ChatRole::User, "parent"),
                                            text_turn(ninfer::ChatRole::Assistant, "answer")})));

    const OpenAIResponsesPromptRequest request =
        parse_openai_responses_input_tokens_request(Json{{"model", "m"},
                                                         {"previous_response_id", "resp_parent"},
                                                         {"instructions", "count this"},
                                                         {"input", "next"}},
                                                    limits());
    const OpenAIResponsesResolvedPrompt resolved =
        resolve_openai_responses_prompt(request, store, std::nullopt, false);
    int failures = 0;
    failures += check(resolved.generation.messages.size() == 4 &&
                          resolved.generation.messages[0].role == ninfer::ChatRole::Developer &&
                          resolved.generation.messages.back().content[0].text == "next",
                      "input token counting resolves instructions, parent, and current input");
    failures += check(Json::parse(make_openai_response_input_tokens_body(9)) ==
                          Json{{"object", "response.input_tokens"}, {"input_tokens", 9}},
                      "input token count response shape");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_basic_request_and_resolution();
    failures += test_budgets_and_nonsemantic_hints();
    failures += test_typed_items_and_cache_markers();
    failures += test_tools_and_effective_subset();
    failures += test_explicit_rejections();
    failures += test_previous_response_call_graph();
    failures += test_response_object();
    failures += test_sse_sequence_and_failures();
    failures += test_input_tokens_uses_shared_state_path();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
