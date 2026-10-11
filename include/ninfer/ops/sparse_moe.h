#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

struct SparseMoeWeights {
    Weight router_shared_gate;
    Weight routed_gate_up;
    Weight routed_down;
    Weight shared_gate_up;
    Weight shared_down;
};

enum class SparseMoeEpilogue : std::uint8_t {
    AddResidual,
};

/**
 * Optional per-call execution hints. Every field is a pure cache hint with no numeric effect:
 * the same call with a default-constructed SparseMoeHints produces bit-identical output.
 *
 * next_weight_prefetch names a weight span the next decode-step consumer will stream; the decode
 * computation may prefetch its first bytes into L2. The span is caller-owned
 * and read once, inside the call: the Op keeps no state between calls, and no hidden channel
 * carries it.
 */
struct SparseMoeHints {
    const void* next_weight_prefetch       = nullptr;
    std::size_t next_weight_prefetch_bytes = 0;
};

/**
 * Returns the transient capacity required by SparseMoe for every T in the inclusive
 * [min_tokens,max_tokens] interval. The routed QTypes are the fixed implementation profile.
 * Invalid profiles or intervals throw.
 */
[[nodiscard]] std::size_t sparse_moe_workspace_capacity_bytes(QType routed_gate_up,
                                                              QType routed_down,
                                                              std::int32_t min_tokens,
                                                              std::int32_t max_tokens);

/**
 * Sparse mixture of SwiGLU experts with a shared expert and residual addition.
 *
 * For each contiguous BF16 x[:,t] and destination[:,t], both [2048,T], T>0:
 *   scores = router_shared_gate * x
 *   E = top8(scores[0:256]), breaking ties by lower expert id
 *   alpha = softmax(scores[E]); beta = sigmoid(scores[256])
 *   h_e = silu(gate_e * x) * (up_e * x)
 *   destination' = BF16(destination + sum_e_in_E alpha_e * (down_e * h_e)
 *                                   + beta * (shared_down * h_shared))
 *
 * Stored weights are interpreted through their exact numeric formats. Internal staging,
 * accumulation and scale placement are private; they add no semantic rounding boundaries.
 * destination is the only observable mutation. All inputs remain unchanged.
 *
 * The five weights have the exact registered shapes: BF16 router/shared gate [257,2048], routed
 * gate/up [256*1024,2048], routed down [256*2048,512], shared gate/up [1024,2048], and shared down
 * [2048,512]. Admitted codec profiles are Q4+Q5, Q4+Q6, and Q8+Q8 for the two routed banks; both
 * shared banks are Q8. Expert e directly selects its stored row spans; no selected-weight gather
 * or repack occurs.
 *
 * Every positive T is supported.
 *
 * x, destination, all weight planes, and live workspace must be pairwise non-overlapping.
 * Execution is enqueued on execution.stream() without host synchronization. The positive physical
 * SM count in execution selects the persistent prefill grid. Workspace is caller-owned,
 * graph-stable transient storage and carries no state beyond the call; its capacity does not
 * depend on the grid.
 */
void sparse_moe(const Tensor& x, const SparseMoeWeights& weights, SparseMoeEpilogue epilogue,
                Tensor& destination, MemoryArena& workspace, CudaExecutionView execution);

/**
 * The same Op with caller-supplied execution hints. Semantics, workspace requirement and output
 * are exactly those of the overload above; hints only steer cache warming.
 */
void sparse_moe(const Tensor& x, const SparseMoeWeights& weights, SparseMoeEpilogue epilogue,
                Tensor& destination, const SparseMoeHints& hints, MemoryArena& workspace,
                CudaExecutionView execution);

} // namespace ninfer::ops
