#pragma once

#include "ops/sparse_moe/prefill/schedule.h"
#include "ops/common/numeric/math.cuh"
#include "ops/common/numeric/convert.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/cuda/mma.cuh"
#include "ops/common/quant/integer.cuh"
#include "ops/sparse_moe/kernels/routing.cuh"

namespace ninfer::ops::moe_detail {
using namespace common;

namespace {

__global__ void sparse_moe_prefill_select_count_kernel(const float* __restrict__ scores,
                                                       int* __restrict__ ids,
                                                       float* __restrict__ alpha,
                                                       float* __restrict__ shared_scale,
                                                       int* __restrict__ local_rank,
                                                       int* __restrict__ tile_counts, int tokens) {
    __shared__ int counts[kExperts];
    __shared__ float selected_logits[kRouteTileTokens][kTopK];
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (tid < kExperts) { counts[tid] = 0; }
    __syncthreads();

    const int token = static_cast<int>(blockIdx.x) * kRouteTileTokens + warp;
    if (token < tokens) {
        sparse_moe_select_top8_warp(scores + static_cast<std::int64_t>(token) * kRouterScoreRows,
                                    ids + token * kTopK, alpha + token * kTopK,
                                    shared_scale + token, selected_logits[warp]);
        __syncwarp();
        if (lane == 0) {
#pragma unroll
            for (int rank = 0; rank < kTopK; ++rank) {
                const int assignment   = token * kTopK + rank;
                local_rank[assignment] = atomicAdd(&counts[ids[assignment]], 1);
            }
        }
    }
    __syncthreads();
    if (tid < kExperts) {
        tile_counts[static_cast<std::int64_t>(blockIdx.x) * kExperts + tid] = counts[tid];
    }
}

__global__ void sparse_moe_prefill_scan_kernel(const int* __restrict__ tile_counts,
                                               int* __restrict__ tile_bases,
                                               int* __restrict__ expert_offsets,
                                               int* __restrict__ route_job_experts,
                                               int* __restrict__ route_job_columns,
                                               int* __restrict__ route_job_count, int route_tiles,
                                               int job_bn, int tokens, bool adaptive) {
    __shared__ int scan[kExperts];
    const int expert = static_cast<int>(threadIdx.x);
    int count        = 0;
    for (int tile = 0; tile < route_tiles; ++tile) {
        count += tile_counts[static_cast<std::int64_t>(tile) * kExperts + expert];
    }
    scan[expert] = count;
    __syncthreads();

#pragma unroll
    for (int offset = 1; offset < kExperts; offset <<= 1) {
        const int add = expert >= offset ? scan[expert - offset] : 0;
        __syncthreads();
        scan[expert] += add;
        __syncthreads();
    }

    const int base         = expert == 0 ? 0 : scan[expert - 1];
    expert_offsets[expert] = base;
    if (expert == kExperts - 1) { expert_offsets[kExperts] = scan[expert]; }

    int cursor = base;
    for (int tile = 0; tile < route_tiles; ++tile) {
        const std::int64_t index = static_cast<std::int64_t>(tile) * kExperts + expert;
        tile_bases[index]        = cursor;
        cursor += tile_counts[index];
    }

    // Reuse scan only after every expert has consumed the assignment prefix.
    __syncthreads();
    scan[expert] = (count + job_bn - 1) / job_bn;
    __syncthreads();
#pragma unroll
    for (int offset = 1; offset < kExperts; offset <<= 1) {
        const int add = expert >= offset ? scan[expert - offset] : 0;
        __syncthreads();
        scan[expert] += add;
        __syncthreads();
    }
    const int touched   = __syncthreads_count(count > 0);
    const int job_begin = expert == 0 ? 0 : scan[expert - 1];
    const int jobs      = (count + job_bn - 1) / job_bn;
    for (int job = 0; job < jobs; ++job) {
        route_job_experts[job_begin + job] = expert;
        route_job_columns[job_begin + job] = job * job_bn;
    }
    if (expert == kExperts - 1) {
        const int jobs = scan[expert];
        // Above 3.5 grouped jobs per token, fixed token work wins by avoiding sparse expert tiles.
        route_job_count[0] =
            adaptive && jobs * kAdaptiveJobsDen > kAdaptiveJobsNum * tokens ? -jobs : jobs;
        route_job_count[1] = touched;
    }
}

template <bool Adaptive>
__global__ void
sparse_moe_prefill_gather_kernel(const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids,
                                 const int* __restrict__ local_rank, int* __restrict__ packed_index,
                                 const int* __restrict__ tile_bases,
                                 __nv_bfloat16* __restrict__ gathered,
                                 const int* __restrict__ route_job_count) {
    if constexpr (Adaptive) {
        if (*route_job_count < 0) { return; }
    }
    const int assignment = static_cast<int>(blockIdx.x);
    const int token      = assignment / kTopK;
    const int expert     = ids[assignment];
    const int tile       = token / kRouteTileTokens;
    const int packed =
        tile_bases[static_cast<std::int64_t>(tile) * kExperts + expert] + local_rank[assignment];
    const int k       = static_cast<int>(threadIdx.x) * 8;
    const uint4 value = load_vec<uint4>(x + static_cast<std::int64_t>(token) * kHidden + k);
    store_vec(gathered + static_cast<std::int64_t>(packed) * kHidden + k, value);
    if (threadIdx.x == 0) { packed_index[assignment] = packed; }
}

// Publishes the packed-column map without moving activations. One thread per assignment, so
// unlike the gather it reads the tile-local rank and writes the inverse map from the same
// thread and never has the two live in one buffer.
template <bool Adaptive>
__global__ void
sparse_moe_prefill_index_kernel(const int* __restrict__ ids, const int* __restrict__ local_rank,
                                int* __restrict__ packed_index, const int* __restrict__ tile_bases,
                                int* __restrict__ packed_token, int assignments,
                                const int* __restrict__ route_job_count) {
    if constexpr (Adaptive) {
        if (*route_job_count < 0) { return; }
    }
    const int assignment =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (assignment >= assignments) { return; }
    const int token  = assignment / kTopK;
    const int expert = ids[assignment];
    const int tile   = token / kRouteTileTokens;
    const int packed =
        tile_bases[static_cast<std::int64_t>(tile) * kExperts + expert] + local_rank[assignment];
    packed_index[assignment] = packed;
    packed_token[packed]     = token;
}

} // namespace

} // namespace ninfer::ops::moe_detail
