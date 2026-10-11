#pragma once

#include "core/arena.h"
#include "core/layout.h"
#include "ops/sparse_moe/plan.h"

namespace ninfer::ops::moe_detail {

struct DecodeWorkspace {
    Tensor ids;
    Tensor alpha;
    Tensor shared_scale;
    Tensor scratch;
};

template <class Arena>
DecodeWorkspace allocate_decode_workspace(Arena& arena) {
    DecodeWorkspace out;
    out.ids          = arena.alloc(DType::I32, {8}, 16);
    out.alpha        = arena.alloc(DType::FP32, {8}, 16);
    out.shared_scale = arena.alloc(DType::FP32, {1}, 4);
    // router uses the first 257 values as scores. gate/up then reuses the same lifetime for [9,512]
    // natural FP32 SwiGLU results consumed by down/merge.
    out.scratch = arena.alloc(DType::FP32, {9, 512}, 256);
    return out;
}

struct SmallWorkspace {
    Tensor token_ids;
    Tensor token_alpha;
    Tensor shared_scale;
    Tensor scratch;
};

template <class Arena>
SmallWorkspace allocate_small_workspace(Arena& arena, std::int32_t tokens) {
    SmallWorkspace out;
    const std::int32_t assignments = 8 * tokens;
    out.token_ids                  = arena.alloc(DType::I32, {assignments}, 16);
    out.token_alpha                = arena.alloc(DType::FP32, {assignments}, 16);
    out.shared_scale               = arena.alloc(DType::FP32, {tokens}, 16);
    // router uses [T,257,4] partial router scores. After selection, each token reuses
    // its [9,512] region for eight routed and one shared SwiGLU activations.
    out.scratch = arena.alloc(DType::FP32, {512, 9 * tokens}, 256);
    return out;
}

struct PrefillWorkspace {
    Tensor token_ids;
    Tensor token_alpha;
    // Selection writes one rank local to a routing tile. The gather must keep this source separate
    // from the final inverse map because all threads in an assignment block consume the rank while
    // thread 0 publishes the packed column. The index kernel does both from one thread per
    // assignment, so the separation is there for the gather the Q8 routed codec still takes.
    Tensor local_rank;
    Tensor shared_scale;
    // Scan is the final consumer of tile_counts. Both maps below alias its dead prefix for the
    // remaining lifetime; 256 * ceil(T / 8) is at least 32 * T elements and they take 8 * T each.
    Tensor tile_counts;
    Tensor packed_index;
    // The token each packed column was routed from, so the routed gate/up GEMM can stage its
    // activation tile from `x` instead of from a materialised copy. Sits in the same dead prefix
    // behind packed_index and needs no allocation of its own.
    Tensor packed_token;
    Tensor tile_bases;
    Tensor expert_offsets;
    Tensor route_job_experts;
    Tensor route_job_columns;
    // A negative count selects the token-oriented adaptive route; its magnitude is the unused
    // grouped-route job count. Nonnegative values select the normal grouped route.
    Tensor route_job_count;

    // The three large allocations are lifetime unions:
    //   router scores FP32 <-> shared SwiGLU BF16
    //   gathered X BF16    <-> routed down output BF16 <-> adaptive FP32 activations
    //     (a Q4 routed gate/up stages from x and gathers nothing, so it only ever writes here)
    //   routed SwiGLU BF16 <-> routed FP32 token reduction
    Tensor score_storage;
    Tensor shared_activation;
    Tensor grouped_io;
    Tensor routed_storage;
    Tensor routed_sum;
};

template <class Arena>
PrefillWorkspace allocate_prefill_workspace(Arena& arena, std::int32_t capacity_tokens) {
    PrefillWorkspace out;
    const std::int32_t assignments = 8 * capacity_tokens;
    const std::int32_t route_tiles = (capacity_tokens + kRouteTileTokens - 1) / kRouteTileTokens;

    out.token_ids    = arena.alloc(DType::I32, {assignments}, 256);
    out.token_alpha  = arena.alloc(DType::FP32, {assignments}, 256);
    out.local_rank   = arena.alloc(DType::I32, {assignments}, 256);
    out.shared_scale = arena.alloc(DType::FP32, {capacity_tokens}, 256);
    out.tile_counts  = arena.alloc(DType::I32, {256, route_tiles}, 256);
    // Both maps alias the dead prefix of tile_counts, whose last reader is the scan. The
    // layout builder allocates nothing, so the offset is only formed when there is a pointer.
    auto* const counts = static_cast<std::int32_t*>(out.tile_counts.data);
    out.packed_index   = Tensor(counts, DType::I32, {assignments});
    out.packed_token =
        Tensor(counts != nullptr ? counts + assignments : nullptr, DType::I32, {assignments});
    out.tile_bases     = arena.alloc(DType::I32, {256, route_tiles}, 256);
    out.expert_offsets = arena.alloc(DType::I32, {257}, 256);
    // A route job is one nonempty expert column tile. The bound is for the
    // narrowest prefill tile (32 assignments) and includes every expert tail.
    const std::int32_t max_route_jobs = assignments / 32 + 256;
    out.route_job_experts             = arena.alloc(DType::I32, {max_route_jobs}, 256);
    out.route_job_columns             = arena.alloc(DType::I32, {max_route_jobs}, 256);
    // [0] is the job count, negative when the scan hands the extent to the decode path;
    // [1] is how many experts the route touched, which picks the gate/up pipeline depth.
    out.route_job_count = arena.alloc(DType::I32, {2}, 256);

    out.score_storage     = arena.alloc(DType::FP32, {kRouterScoreRows, capacity_tokens}, 256);
    out.shared_activation = Tensor(out.score_storage.data, DType::BF16, {512, capacity_tokens});

    out.grouped_io = arena.alloc(DType::BF16, {2048, assignments}, 256);

    out.routed_storage = arena.alloc(DType::BF16, {512, assignments}, 256);
    out.routed_sum     = Tensor(out.routed_storage.data, DType::FP32, {2048, capacity_tokens});
    return out;
}

} // namespace ninfer::ops::moe_detail
