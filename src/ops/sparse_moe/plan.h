#pragma once

#include "ninfer/ops/sparse_moe.h"
#include <variant>

namespace ninfer::ops::moe_detail {
inline constexpr int kHidden          = 2048;
inline constexpr int kExperts         = 256;
inline constexpr int kRouterRows      = kExperts + 1;
inline constexpr int kTopK            = 8;
inline constexpr int kIntermediate    = 512;
inline constexpr int kSmallMaxTokens  = 46;
inline constexpr int kPrefillSliceMax = 4096;
inline constexpr int kRouteTileTokens = 8;
inline constexpr int kRouterScoreRows = 260;

enum class GateUpSchedule : std::uint8_t {
    Paths1,
    Paths3,
    Paths9,
};

enum class DownSchedule : std::uint8_t {
    Rows1,
    Rows2,
    Rows4,
};

struct DecodePlan {
    std::size_t workspace_bytes = 0;
    // L2 cache hint resolved from the caller's SparseMoeHints. No numeric effect.
    const void* next_weight_prefetch       = nullptr;
    std::size_t next_weight_prefetch_bytes = 0;
};

struct SmallPlan {
    std::int32_t tokens         = 0;
    std::size_t workspace_bytes = 0;
    GateUpSchedule gate_up      = GateUpSchedule::Paths3;
    DownSchedule down           = DownSchedule::Rows1;
};

struct PrefillPlan {
    std::int32_t tokens         = 0;
    std::int32_t slice_tokens   = 0;
    std::size_t workspace_bytes = 0;
};

using Plan = std::variant<DecodePlan, SmallPlan, PrefillPlan>;
int prefill_first(QType gate_up, QType down);
Plan select_plan(int tokens, QType gate_up, QType down, const SparseMoeHints& hints = {});
std::size_t workspace_bytes(const Plan& plan);

struct DecodeWorkspace;
struct SmallWorkspace;
struct PrefillWorkspace;

void launch_decode(const Tensor& x, const SparseMoeWeights& weights, Tensor& destination,
                   const DecodePlan& plan, const DecodeWorkspace& workspace, cudaStream_t stream);

void launch_token_gate_up(const Tensor& x, const SparseMoeWeights& weights, const int* token_ids,
                          float* token_activations, std::int32_t tokens, GateUpSchedule schedule,
                          cudaStream_t stream, const int* adaptive_route_jobs = nullptr);

void launch_token_down(const SparseMoeWeights& weights, Tensor& destination, const int* token_ids,
                       const float* token_alpha, const float* shared_scale,
                       const float* token_activations, std::int32_t tokens, DownSchedule schedule,
                       cudaStream_t stream, const int* adaptive_route_jobs = nullptr);

void launch_small(const Tensor& x, const SparseMoeWeights& weights, Tensor& destination,
                  const SmallPlan& plan, const SmallWorkspace& workspace, cudaStream_t stream);

void launch_prefill(const Tensor& x, const SparseMoeWeights& weights, Tensor& destination,
                    const PrefillPlan& plan, const PrefillWorkspace& workspace,
                    CudaExecutionView execution);

void launch_decode_gate_up(const Tensor& x, const SparseMoeWeights& weights,
                           const DecodeWorkspace& workspace, cudaStream_t stream);
void launch_decode_down(const SparseMoeWeights& weights, Tensor& destination,
                        const DecodeWorkspace& workspace, cudaStream_t stream,
                        const void* prefetch_data, std::size_t prefetch_bytes);

} // namespace ninfer::ops::moe_detail
