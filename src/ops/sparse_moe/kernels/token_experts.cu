#include "ops/sparse_moe/kernels/token_experts.cuh"
#include "ops/sparse_moe/workspace.h"

namespace ninfer::ops::moe_detail {
using namespace common;

template <class Codec>
void launch_decode_gate_up_codec(const Tensor& x, const SparseMoeWeights& weights,
                                 const DecodeWorkspace& workspace, cudaStream_t stream) {
    const auto* input         = static_cast<const __nv_bfloat16*>(x.data);
    const auto* ids           = static_cast<const int*>(workspace.ids.data);
    auto* act                 = static_cast<float*>(workspace.scratch.data);
    const auto* routed_codes  = static_cast<const std::uint8_t*>(weights.routed_gate_up.qdata);
    const auto* routed_high   = static_cast<const std::uint8_t*>(weights.routed_gate_up.qhigh);
    const auto* routed_scales = static_cast<const std::uint8_t*>(weights.routed_gate_up.scales);
    const auto* shared_codes  = static_cast<const std::uint8_t*>(weights.shared_gate_up.qdata);
    const auto* shared_scales = static_cast<const std::uint8_t*>(weights.shared_gate_up.scales);
    CUDA_CHECK(pdl::launch_dependent({dim3(kIntermediate), dim3(9 * 32), 0, stream},
                                     gate_up_decode_kernel<Codec>, input, ids, routed_codes,
                                     routed_high, routed_scales, shared_codes, shared_scales, act));
}

template <class Codec>
void launch_decode_down_codec(const SparseMoeWeights& weights, Tensor& destination,
                              const DecodeWorkspace& workspace, cudaStream_t stream,
                              const void* prefetch_data, std::size_t prefetch_bytes) {
    const auto* ids           = static_cast<const int*>(workspace.ids.data);
    const auto* alpha         = static_cast<const float*>(workspace.alpha.data);
    const auto* shared_scale  = static_cast<const float*>(workspace.shared_scale.data);
    const auto* act           = static_cast<const float*>(workspace.scratch.data);
    const auto* routed_codes  = static_cast<const std::uint8_t*>(weights.routed_down.qdata);
    const auto* routed_high   = static_cast<const std::uint8_t*>(weights.routed_down.qhigh);
    const auto* routed_scales = static_cast<const std::uint8_t*>(weights.routed_down.scales);
    const auto* shared_codes  = static_cast<const std::uint8_t*>(weights.shared_down.qdata);
    const auto* shared_scales = static_cast<const std::uint8_t*>(weights.shared_down.scales);
    auto* output              = static_cast<__nv_bfloat16*>(destination.data);
    CUDA_CHECK(pdl::launch_dependent({dim3(kHidden), dim3(9 * 32), 0, stream},
                                     down_decode_kernel<Codec, 1>, ids, alpha, shared_scale, act,
                                     routed_codes, routed_high, routed_scales, shared_codes,
                                     shared_scales, output, static_cast<const char*>(prefetch_data),
                                     static_cast<unsigned long long>(prefetch_bytes)));
}

void launch_decode_down(const SparseMoeWeights& weights, Tensor& destination,
                        const DecodeWorkspace& workspace, cudaStream_t stream,
                        const void* prefetch_data, std::size_t prefetch_bytes) {
    switch (weights.routed_down.qtype) {
    case QType::Q5_G64_FP16:
        launch_decode_down_codec<Q5Codec>(weights, destination, workspace, stream, prefetch_data,
                                          prefetch_bytes);
        return;
    case QType::Q6_G64_FP16:
        launch_decode_down_codec<Q6Codec>(weights, destination, workspace, stream, prefetch_data,
                                          prefetch_bytes);
        return;
    case QType::Q8_G32_FP16:
        launch_decode_down_codec<Q8Codec>(weights, destination, workspace, stream, prefetch_data,
                                          prefetch_bytes);
        return;
    default:
        throw std::invalid_argument("sparse_moe: unsupported down/merge codec");
    }
}

template <class Codec, int PathsPerBlock, bool Adaptive>
void launch_token_gate_up_paths(const Tensor& x, const SparseMoeWeights& weights,
                                const int* token_ids, float* token_activations, std::int32_t tokens,
                                cudaStream_t stream, const int* adaptive_route_jobs) {
    constexpr int kPathBlocks = (kTopK + 1) / PathsPerBlock;
    const auto* input         = static_cast<const __nv_bfloat16*>(x.data);
    const auto* routed_codes  = static_cast<const std::uint8_t*>(weights.routed_gate_up.qdata);
    const auto* routed_high   = static_cast<const std::uint8_t*>(weights.routed_gate_up.qhigh);
    const auto* routed_scales = static_cast<const std::uint8_t*>(weights.routed_gate_up.scales);
    const auto* shared_codes  = static_cast<const std::uint8_t*>(weights.shared_gate_up.qdata);
    const auto* shared_scales = static_cast<const std::uint8_t*>(weights.shared_gate_up.scales);
    if constexpr (Adaptive) {
        gate_up_token_kernel<Codec, PathsPerBlock, true>
            <<<kAdaptiveGateUpBlocks, PathsPerBlock * 32, 0, stream>>>(
                input, token_ids, routed_codes, routed_high, routed_scales, shared_codes,
                shared_scales, token_activations, tokens, adaptive_route_jobs);
        CUDA_CHECK(cudaGetLastError());
    } else {
        CUDA_CHECK(pdl::launch_dependent(
            {dim3(kIntermediate, tokens * kPathBlocks), dim3(PathsPerBlock * 32), 0, stream},
            gate_up_token_kernel<Codec, PathsPerBlock, false>, input, token_ids, routed_codes,
            routed_high, routed_scales, shared_codes, shared_scales, token_activations, tokens,
            nullptr));
    }
}

template <class Codec, bool Adaptive>
void launch_token_gate_up_codec(const Tensor& x, const SparseMoeWeights& weights,
                                const int* token_ids, float* token_activations, std::int32_t tokens,
                                GateUpSchedule schedule, cudaStream_t stream,
                                const int* adaptive_route_jobs) {
    switch (schedule) {
    case GateUpSchedule::Paths1:
        launch_token_gate_up_paths<Codec, 1, Adaptive>(x, weights, token_ids, token_activations,
                                                       tokens, stream, adaptive_route_jobs);
        return;
    case GateUpSchedule::Paths3:
        launch_token_gate_up_paths<Codec, 3, Adaptive>(x, weights, token_ids, token_activations,
                                                       tokens, stream, adaptive_route_jobs);
        return;
    case GateUpSchedule::Paths9:
        launch_token_gate_up_paths<Codec, 9, Adaptive>(x, weights, token_ids, token_activations,
                                                       tokens, stream, adaptive_route_jobs);
        return;
    }
    throw std::logic_error("sparse_moe: unknown small-T gate/up schedule");
}

template <class Codec, int Rows, bool Adaptive>
void launch_token_down_rows(const SparseMoeWeights& weights, Tensor& destination,
                            const int* token_ids, const float* token_alpha,
                            const float* shared_scale, const float* token_activations,
                            std::int32_t tokens, cudaStream_t stream,
                            const int* adaptive_route_jobs) {
    const dim3 grid = Adaptive ? dim3(kAdaptiveDownBlocks) : dim3(kHidden / Rows, tokens);
    down_token_kernel<Codec, Rows, Adaptive><<<grid, 9 * 32, 0, stream>>>(
        token_ids, token_alpha, shared_scale, token_activations,
        static_cast<const std::uint8_t*>(weights.routed_down.qdata),
        static_cast<const std::uint8_t*>(weights.routed_down.qhigh),
        static_cast<const std::uint8_t*>(weights.routed_down.scales),
        static_cast<const std::uint8_t*>(weights.shared_down.qdata),
        static_cast<const std::uint8_t*>(weights.shared_down.scales),
        static_cast<__nv_bfloat16*>(destination.data), tokens, adaptive_route_jobs);
    CUDA_CHECK(cudaGetLastError());
}

template <class Codec, bool Adaptive>
void launch_token_down_codec(const SparseMoeWeights& weights, Tensor& destination,
                             const int* token_ids, const float* token_alpha,
                             const float* shared_scale, const float* token_activations,
                             std::int32_t tokens, DownSchedule schedule, cudaStream_t stream,
                             const int* adaptive_route_jobs) {
    switch (schedule) {
    case DownSchedule::Rows1:
        launch_token_down_rows<Codec, 1, Adaptive>(weights, destination, token_ids, token_alpha,
                                                   shared_scale, token_activations, tokens, stream,
                                                   adaptive_route_jobs);
        return;
    case DownSchedule::Rows2:
        launch_token_down_rows<Codec, 2, Adaptive>(weights, destination, token_ids, token_alpha,
                                                   shared_scale, token_activations, tokens, stream,
                                                   adaptive_route_jobs);
        return;
    case DownSchedule::Rows4:
        launch_token_down_rows<Codec, 4, Adaptive>(weights, destination, token_ids, token_alpha,
                                                   shared_scale, token_activations, tokens, stream,
                                                   adaptive_route_jobs);
        return;
    }
    throw std::logic_error("sparse_moe: unknown small-T down/merge schedule");
}

void launch_token_gate_up(const Tensor& x, const SparseMoeWeights& weights, const int* token_ids,
                          float* token_activations, std::int32_t tokens, GateUpSchedule schedule,
                          cudaStream_t stream, const int* adaptive_route_jobs) {
    switch (weights.routed_gate_up.qtype) {
    case QType::Q4_G64_FP16:
        if (adaptive_route_jobs == nullptr) {
            launch_token_gate_up_codec<Q4Codec, false>(x, weights, token_ids, token_activations,
                                                       tokens, schedule, stream, nullptr);
        } else {
            launch_token_gate_up_codec<Q4Codec, true>(x, weights, token_ids, token_activations,
                                                      tokens, schedule, stream,
                                                      adaptive_route_jobs);
        }
        return;
    case QType::Q8_G32_FP16:
        launch_token_gate_up_codec<Q8Codec, false>(x, weights, token_ids, token_activations, tokens,
                                                   schedule, stream, nullptr);
        return;
    default:
        throw std::invalid_argument("sparse_moe: unsupported small-T gate/up codec");
    }
}

void launch_token_down(const SparseMoeWeights& weights, Tensor& destination, const int* token_ids,
                       const float* token_alpha, const float* shared_scale,
                       const float* token_activations, std::int32_t tokens, DownSchedule schedule,
                       cudaStream_t stream, const int* adaptive_route_jobs) {
    switch (weights.routed_down.qtype) {
    case QType::Q5_G64_FP16:
        if (adaptive_route_jobs == nullptr) {
            launch_token_down_codec<Q5Codec, false>(weights, destination, token_ids, token_alpha,
                                                    shared_scale, token_activations, tokens,
                                                    schedule, stream, nullptr);
        } else {
            launch_token_down_codec<Q5Codec, true>(weights, destination, token_ids, token_alpha,
                                                   shared_scale, token_activations, tokens,
                                                   schedule, stream, adaptive_route_jobs);
        }
        return;
    case QType::Q6_G64_FP16:
        if (adaptive_route_jobs == nullptr) {
            launch_token_down_codec<Q6Codec, false>(weights, destination, token_ids, token_alpha,
                                                    shared_scale, token_activations, tokens,
                                                    schedule, stream, nullptr);
        } else {
            launch_token_down_codec<Q6Codec, true>(weights, destination, token_ids, token_alpha,
                                                   shared_scale, token_activations, tokens,
                                                   schedule, stream, adaptive_route_jobs);
        }
        return;
    case QType::Q8_G32_FP16:
        launch_token_down_codec<Q8Codec, false>(weights, destination, token_ids, token_alpha,
                                                shared_scale, token_activations, tokens, schedule,
                                                stream, nullptr);
        return;
    default:
        throw std::invalid_argument("sparse_moe: unsupported small-T down/merge codec");
    }
}

void launch_decode_gate_up(const Tensor& x, const SparseMoeWeights& weights,
                           const DecodeWorkspace& workspace, cudaStream_t stream) {
    switch (weights.routed_gate_up.qtype) {
    case QType::Q4_G64_FP16:
        launch_decode_gate_up_codec<Q4Codec>(x, weights, workspace, stream);
        return;
    case QType::Q8_G32_FP16:
        launch_decode_gate_up_codec<Q8Codec>(x, weights, workspace, stream);
        return;
    default:
        throw std::invalid_argument("sparse_moe: unsupported gate/up codec");
    }
}

} // namespace ninfer::ops::moe_detail
