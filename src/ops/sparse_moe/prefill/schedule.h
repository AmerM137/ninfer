#pragma once
#include "ops/sparse_moe/plan.h"
#include <algorithm>

namespace ninfer::ops::moe_detail {
inline constexpr int kRouterBM      = 16;
inline constexpr int kRouterBN      = 64;
inline constexpr int kRouterBK      = 64;
inline constexpr int kRouterStages  = 2;
inline constexpr int kRouterWarps   = 8;
inline constexpr int kRouterThreads = kRouterWarps * 32;
inline constexpr int kExpertBM      = 64;
inline constexpr int kExpertBN      = 64;
inline constexpr int kExpertBK      = 64;
inline constexpr int kExpertStages  = 2;
inline constexpr int kExpertWarps   = 8;
inline constexpr int kExpertThreads = kExpertWarps * 32;
// Six stages fit the narrow 32-column kernel's 48 KiB shared capacity.
inline constexpr int kGateUpNarrowStages     = 6;
inline constexpr int kPrefillQueuedCtasPerSm = 32;
// Device routing distinguishes sparse expert tiles from repeated use of cached weights.
enum class GateUpRoute { Any, Spread, Packed };
inline constexpr int kGateUpDeepJobsNum = 7;
inline constexpr int kGateUpDeepJobsDen = 4;
inline constexpr int kAdaptiveJobsNum   = 7;
inline constexpr int kAdaptiveJobsDen   = 2;

struct PrefillSlice {
    bool wide;
    bool adaptive;
    int job_columns;
    int gate_blocks;
    int down_blocks;
};

inline PrefillSlice select_prefill_slice(int tokens, QType down, int sm_count) {
    const bool wide         = tokens >= 768;
    const int adaptive_last = down == QType::Q5_G64_FP16 ? 51 : down == QType::Q6_G64_FP16 ? 52 : 0;
    const bool adaptive     = tokens >= 47 && tokens <= adaptive_last;
    const int columns       = wide ? 64 : 32;
    const int max_jobs      = tokens * kTopK / columns + kExperts;
    const std::int64_t max_blocks = static_cast<std::int64_t>(sm_count) * kPrefillQueuedCtasPerSm;
    return {wide, adaptive, columns,
            static_cast<int>(
                std::min<std::int64_t>(max_jobs * (kIntermediate / (kExpertBM / 2)), max_blocks)),
            static_cast<int>(std::min<std::int64_t>(max_jobs * (kHidden / kExpertBM), max_blocks))};
}
} // namespace ninfer::ops::moe_detail
