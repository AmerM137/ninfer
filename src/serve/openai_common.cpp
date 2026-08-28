#include "serve/openai_common.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <utility>

namespace ninfer::serve {

using Json = nlohmann::json;

std::string make_models_list(const std::string& model_id, std::int64_t created) {
    const Json payload = {{"object", "list"},
                          {"data", Json::array({Json{{"id", model_id},
                                                     {"object", "model"},
                                                     {"created", created},
                                                     {"owned_by", "ninfer"}}})}};
    return payload.dump();
}

std::string make_model_object(const std::string& model_id, std::int64_t created) {
    const Json payload = {
        {"id", model_id}, {"object", "model"}, {"created", created}, {"owned_by", "ninfer"}};
    return payload.dump();
}

std::string make_error_body(const ApiError& error) {
    Json rendered     = {{"message", error.message}, {"type", error.type}};
    rendered["param"] = error.param.empty() ? Json(nullptr) : Json(error.param);
    rendered["code"]  = error.code.empty() ? Json(nullptr) : Json(error.code);
    return Json{{"error", std::move(rendered)}}.dump();
}

std::int64_t unix_time_now() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

} // namespace ninfer::serve
