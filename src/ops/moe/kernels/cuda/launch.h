#pragma once

#include "ninfer/ops/moe.h"

namespace ninfer::ops::detail {

template <int H, int I, int E, int K>
struct SwiGLUShape {
    static constexpr int hidden = H, intermediate = I, experts = E, top_k = K;
    static constexpr int control_rows = E + 1;
    static constexpr int score_rows   = (E + 4) / 4 * 4;
};

using SwiGLUH2048I512E256K8  = SwiGLUShape<2048, 512, 256, 8>;
using SwiGLUH2560I640E512K10 = SwiGLUShape<2560, 640, 512, 10>;

enum class RouterAlgorithm : std::uint8_t { Vector, TokenTile, Mma };

struct RouterPlan {
    RouterAlgorithm algorithm = RouterAlgorithm::Vector;
    bool programmatic         = false;
    bool count_assignments    = false;
};

struct RouteStorage {
    float* scores    = nullptr;
    int* local_ranks = nullptr;
    int* tile_counts = nullptr;
};

struct GroupedInfo {
    int jobs;
    int direct;
    int spread;
};

struct GroupedStorage {
    int* local_ranks  = nullptr;
    int* tile_counts  = nullptr;
    int* tile_bases   = nullptr;
    int* offsets      = nullptr;
    int* job_experts  = nullptr;
    int* job_columns  = nullptr;
    int* inverse      = nullptr;
    int* tokens       = nullptr;
    GroupedInfo* info = nullptr;
};

struct TokenPlan {
    int paths_per_block     = 1;
    int output_rows         = 1;
    bool single_token       = false;
    bool programmatic_input = false;
};

struct SharedPlan {
    TokenPlan token;
    int columns = 0; // Zero selects SIMT; otherwise the MMA column tile.
};

struct GroupedPlan {
    int columns            = 0;
    bool adaptive          = false;
    int spread_numerator   = 0;
    int spread_denominator = 0;
    int queued_ctas_per_sm = 0;
    int shared_columns     = 0;
};

struct ExpertStorage {
    float* token_activation  = nullptr;
    void* grouped_activation = nullptr;
    void* grouped_io         = nullptr;
    void* shared_activation  = nullptr;
    float* routed_sum        = nullptr;
    GroupedStorage groups;
};

// Shared launch mechanisms; semantic entries own their shape policies and workspace lifetimes.
void launch_route(const Tensor& x, const Weight& control, const ExpertRoutes& routes,
                  const Tensor& shared, const RouterPlan& plan, const RouteStorage& storage,
                  const MoeExecutionHints& hints, CudaExecutionView execution);
void launch_group_count(const ExpertRoutes& routes, ExpertSubset subset,
                        const GroupedStorage& storage, CudaExecutionView execution);
void launch_group_scan(const ExpertRoutes& routes, const GroupedPlan& plan,
                       const GroupedStorage& storage, CudaExecutionView execution);
void launch_group_index(const ExpertRoutes& routes, const GroupedStorage& storage,
                        CudaExecutionView execution);

void launch_token_swiglu(const Tensor& x, const RoutedSwiGLUProfile& profile,
                         const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                         ExpertSubset subset, const ScaledSwiGLUTerm* shared, MoeEpilogue epilogue,
                         Tensor& destination, const TokenPlan& plan, const ExpertStorage& storage,
                         const MoeExecutionHints& hints, CudaExecutionView execution,
                         bool adaptive = false);
void launch_grouped_swiglu(const Tensor& x, const RoutedSwiGLUProfile& profile,
                           const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                           const ScaledSwiGLUTerm* shared, MoeEpilogue epilogue,
                           Tensor& destination, const GroupedPlan& plan,
                           const ExpertStorage& storage, CudaExecutionView execution);
void launch_shared_swiglu(const Tensor& x, const ScaledSwiGLUTerm& shared, MoeEpilogue epilogue,
                          Tensor& destination, const ExpertStorage& storage, const SharedPlan& plan,
                          CudaExecutionView execution);
void launch_zero(Tensor& destination, CudaExecutionView execution);

} // namespace ninfer::ops::detail
