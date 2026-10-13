#pragma once

#include "ops/moe/kernels/cuda/nvfp4/launch.h"

namespace ninfer::ops::detail::moe_nvfp4 {

template <class Arena>
Storage allocate(Arena& arena, const Plan& plan, int tokens, bool routed, bool shared) {
    Storage out;
    if (routed && plan.vector) {
        out.vector_activation =
            static_cast<float*>(arena.alloc(DType::FP32, {kIntermediate, kTopK * tokens}).data);
        if (shared) {
            out.routed_sum = static_cast<float*>(arena.alloc(DType::FP32, {kHidden, tokens}).data);
        }
    } else if (routed) {
        const int assignments = tokens * kTopK;
        const auto bytes      = [&](std::size_t size) {
            return static_cast<std::uint8_t*>(arena.alloc_bytes(size).data);
        };
        const auto integers = [&](int count) {
            return static_cast<int*>(arena.alloc(DType::I32, {count}).data);
        };
        out.input_codes    = bytes(std::size_t(tokens) * kHidden / 2);
        out.input_scales   = bytes(std::size_t(tokens) * kHidden / 16);
        out.product_codes  = bytes(std::size_t(assignments) * kIntermediate / 2);
        out.product_scales = bytes(std::size_t(assignments) * kIntermediate / 16);
        if (plan.atomic_output) {
            out.routed_sum = static_cast<float*>(arena.alloc(DType::FP32, {kHidden, tokens}).data);
        } else {
            out.contributions =
                static_cast<float*>(arena.alloc(DType::FP32, {kHidden, assignments}).data);
        }
        auto& g = out.groups;
        if (!plan.single_block_grouping) { g.counts = integers(kExperts); }
        g.offsets = integers(kExperts + 1);
        if (!plan.single_block_grouping) { g.cursors = integers(kExperts); }
        const int max_jobs = assignments / plan.job_columns() + kExperts;
        g.job_experts      = integers(max_jobs);
        g.job_columns      = integers(max_jobs);
        g.jobs             = integers(1);
        g.assignments      = integers(assignments);
        if (!plan.atomic_output) { g.inverse = integers(assignments); }
    }
    if (shared) { out.shared_activation = arena.alloc(DType::FP32, {kIntermediate, tokens}).data; }
    return out;
}

} // namespace ninfer::ops::detail::moe_nvfp4
