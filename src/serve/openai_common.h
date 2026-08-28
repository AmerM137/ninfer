#pragma once

// OpenAI wire objects shared by Chat Completions and Responses HTTP handlers.

#include "serve/request.h"

#include <cstdint>
#include <string>

namespace ninfer::serve {

std::string make_models_list(const std::string& model_id, std::int64_t created);
std::string make_model_object(const std::string& model_id, std::int64_t created);
std::string make_error_body(const ApiError& error);
std::int64_t unix_time_now();

} // namespace ninfer::serve
