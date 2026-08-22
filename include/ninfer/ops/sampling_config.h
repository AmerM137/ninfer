#pragma once

#include <cstdint>

namespace ninfer::ops {

// Device-resident sampling parameters. token_counts is an optional device I32
// [token_domain] occurrence-count array used by both penalties.
struct SamplingConfig {
    float temperature          = 0.0f; // <= 0 => greedy argmax (bit-identical to argmax())
    std::int32_t top_k         = 0;    // clamped to 20: top_k <= 0 or top_k > 20 => 20
    float top_p                = 1.0f; // >= 1 => disabled
    float min_p                = 0.0f; // <= 0 => disabled
    float presence_penalty     = 0.0f;
    float frequency_penalty    = 0.0f;
    unsigned long long seed    = 0;
    std::int32_t* token_counts = nullptr; // device [token_domain] i32, or null
};

} // namespace ninfer::ops
