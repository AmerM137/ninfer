#include "ops/sparse_moe/plan.h"
#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/numeric/math.cuh"
#include "ops/common/numeric/convert.cuh"
#include "ops/common/cuda/memory.cuh"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/quant/integer.cuh"
#include <stdexcept>
#include "ops/sparse_moe/workspace.h"
#include "ops/sparse_moe/kernels/routing.cuh"

namespace ninfer::ops::moe_detail {
using namespace common;

namespace {
constexpr int kRouterWarps = 8;

__device__ __forceinline__ float dot_bf16_eight(const __nv_bfloat16* a, const __nv_bfloat16* b) {
    const uint4 av  = load_vec<uint4>(a);
    const uint4 bv  = load_vec<uint4>(b);
    const float2 a0 = bf16x2_bits_to_float2(av.x);
    const float2 a1 = bf16x2_bits_to_float2(av.y);
    const float2 a2 = bf16x2_bits_to_float2(av.z);
    const float2 a3 = bf16x2_bits_to_float2(av.w);
    const float2 b0 = bf16x2_bits_to_float2(bv.x);
    const float2 b1 = bf16x2_bits_to_float2(bv.y);
    const float2 b2 = bf16x2_bits_to_float2(bv.z);
    const float2 b3 = bf16x2_bits_to_float2(bv.w);
    float sum       = 0.0f;
    sum             = fmaf(a0.x, b0.x, sum);
    sum             = fmaf(a0.y, b0.y, sum);
    sum             = fmaf(a1.x, b1.x, sum);
    sum             = fmaf(a1.y, b1.y, sum);
    sum             = fmaf(a2.x, b2.x, sum);
    sum             = fmaf(a2.y, b2.y, sum);
    sum             = fmaf(a3.x, b3.x, sum);
    sum             = fmaf(a3.y, b3.y, sum);
    return sum;
}

__device__ __forceinline__ float router_row_dot(const __nv_bfloat16* x, const __nv_bfloat16* row) {
    constexpr int kSlice = kHidden / kRouterWarps;
    constexpr int kVecs  = kSlice / (32 * 8);
    const int warp       = static_cast<int>(threadIdx.x) >> 5;
    const int lane       = static_cast<int>(threadIdx.x) & 31;
    float sum            = 0.0f;
#pragma unroll
    for (int vector = 0; vector < kVecs; ++vector) {
        const int k = warp * kSlice + vector * 32 * 8 + lane * 8;
        sum += dot_bf16_eight(row + k, x + k);
    }
    return warp_reduce_sum(sum);
}

__global__ void router_decode_kernel(const __nv_bfloat16* __restrict__ x,
                                     const __nv_bfloat16* __restrict__ router,
                                     float* __restrict__ scores,
                                     const char* __restrict__ shared_down_codes,
                                     unsigned long long shared_down_bytes) {
    __shared__ float partial[kRouterWarps];
    const int row   = static_cast<int>(blockIdx.x);
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const float dot = router_row_dot(x, router + static_cast<std::int64_t>(row) * kHidden);
    if (lane == 0) { partial[warp] = dot; }
    __syncthreads();
    if (warp == 0) {
        float value = lane < kRouterWarps ? partial[lane] : 0.0f;
        value       = warp_reduce_sum<kRouterWarps>(value);
        if (lane == 0) { scores[row] = value; }
    }
    if (shared_down_codes != nullptr) {
        // Warm the shared-expert down codes without introducing a data dependency.
        const unsigned long long offset =
            (static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x) * 128ull;
        if (offset < shared_down_bytes) {
            asm volatile("prefetch.global.L2 [%0];" ::"l"(shared_down_codes + offset));
        }
    }
}

__global__ void select_decode_kernel(const float* __restrict__ scores, int* __restrict__ ids,
                                     float* __restrict__ alpha, float* __restrict__ shared_scale) {
    __shared__ float selected_logits[kTopK];
    if (threadIdx.x == 0) { pdl::trigger_dependents(); }
    sparse_moe_select_top8_warp(scores, ids, alpha, shared_scale, selected_logits);
}

void launch_router(const Tensor& x, const SparseMoeWeights& weights,
                   const DecodeWorkspace& workspace, cudaStream_t stream) {
    router_decode_kernel<<<kRouterRows, kRouterWarps * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(weights.router_shared_gate.qdata),
        static_cast<float*>(workspace.scratch.data),
        static_cast<const char*>(weights.shared_down.qdata),
        static_cast<unsigned long long>(kHidden) * kIntermediate);
    CUDA_CHECK(cudaGetLastError());
}

void launch_select_gate_up(const Tensor& x, const SparseMoeWeights& weights,
                           const DecodeWorkspace& workspace, cudaStream_t stream) {
    const auto* scores = static_cast<const float*>(workspace.scratch.data);
    auto* ids          = static_cast<int*>(workspace.ids.data);
    auto* alpha        = static_cast<float*>(workspace.alpha.data);
    auto* shared_scale = static_cast<float*>(workspace.shared_scale.data);
    select_decode_kernel<<<1, 32, 0, stream>>>(scores, ids, alpha, shared_scale);
    CUDA_CHECK(cudaGetLastError());

    launch_decode_gate_up(x, weights, workspace, stream);
}
} // namespace

void launch_decode(const Tensor& x, const SparseMoeWeights& weights, Tensor& destination,
                   const DecodePlan& plan, const DecodeWorkspace& workspace, cudaStream_t stream) {
    launch_router(x, weights, workspace, stream);
    launch_select_gate_up(x, weights, workspace, stream);
    launch_decode_down(weights, destination, workspace, stream, plan.next_weight_prefetch,
                       plan.next_weight_prefetch_bytes);
}

} // namespace ninfer::ops::moe_detail
