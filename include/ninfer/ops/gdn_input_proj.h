#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/projection_weights.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

struct GdnProjectionOutputs {
    Tensor qkv;
    Tensor z;
};

struct GdnConvOutputs {
    Tensor query, key, value, z;
};

struct GdnConvSourceState {
    Tensor history;      // Read-only BF16 [C,3,S].
    Tensor source_slots; // Read-only device I32 [B], absolute slots in [0,S).
};

struct GdnConvSnapshotState {
    Tensor history;        // BF16 [C,3,S], source and destination pool.
    Tensor source_slots;   // Read-only device I32 [B], absolute slots in [0,S).
    Tensor snapshot_bases; // Read-only device I32 [B], reserves [base,base+T).
};

/**
 * Op: gdn_input_proj
 *
 * Math / indexing:
 *   qkv[:,m] = concat(Wq*x[:,m], Wk*x[:,m], Wv*x[:,m]); z[:,m] = Wz*x[:,m].
 *   The logical row order is [query,key,value,z]. Q and K each have 2048 rows.
 *
 * Supported weights / geometry (H=input rows, V=value/z rows, C=4096+V):
 *   Single Q8_G32_FP16 RowSplit [12288,2048]: H=2048, V=4096.
 *   Single Q8_G32_FP16 RowSplit [16384,2560]: H=2560, V=6144.
 *   Single NVFP4 BlockScaleK16M128x4 or FP8_E4M3FN_ROW_BF16 RowScale
 *     [16384,5120]: H=5120, V=6144.
 *   Paired Q4_G64_FP16 RowSplit [4096,5120] for [q,k] and Q5_G64_FP16
 *     RowSplit [12288,5120] for [v,z]: H=5120, V=6144.
 *   Paired operands use A16. Single-parent policy is an activation-compute permission;
 *   Q8 uses A16, FP8 may use A8 when permitted, and NVFP4 may use A4 when permitted.
 *
 * Logical shapes / numeric:
 *   x BF16 [H,M], qkv BF16 [C,M], z BF16 [V,M], contiguous and 16-byte aligned.
 *   M may be any positive representable column count. Weights are decoded according to
 *   their represented format. The outputs approximate the complete matrix products;
 *   internal precision and staging are private implementation choices.
 *
 * Effects / workspace / execution:
 *   Writes both complete outputs; inputs and weight storage are read-only. Tensor operands
 *   are mutually disjoint and do not overlap either weight parent or this call's scratch.
 *   workspace is caller-owned; only a scoped allocation from its current cursor is used.
 *   Existing allocations preceding that cursor may contain inputs or outputs. A zero
 *   capacity performs no allocation. Calls and capacity queries consume the same prepared
 *   weights. Execution uses stream and supports CUDA Graph capture/replay.
 */
void gdn_input_proj(const Tensor& x, const ProjectionWeights& weights, GdnProjectionOutputs outputs,
                    WorkspaceArena& workspace, cudaStream_t stream);

/** Minimum call-local capacity covering every M in the inclusive positive interval. */
[[nodiscard]] std::size_t gdn_input_proj_workspace_capacity_bytes(const ProjectionWeights& weights,
                                                                  std::int32_t min_columns,
                                                                  std::int32_t max_columns);

/**
 * Op: gdn_input_proj_conv_snapshot
 *
 * Math / indexing:
 *   Let p=concat(Wq*x,Wk*x,Wv*x) and z=Wz*x. For each batch row b, read the width-three
 *   history selected by source_slots[b]. At each valid token t, evaluate the four-tap
 *   causal depthwise convolution over history and p[:,t,b], apply SiLU, and split its
 *   channels to query/key/value. Publish the resulting width-three history at
 *   snapshot_bases[b]+t. Z bypasses convolution.
 *
 * Logical shapes / supported domain:
 *   The weight representations and H/V/C geometries are those of gdn_input_proj.
 *   x BF16 [H,T,B], conv_weight BF16 [C,4], history BF16 [C,3,S],
 *   query/key BF16 [2048,T,B], value/z BF16 [V,T,B]. All are contiguous and 16-byte
 *   aligned. Selectors are contiguous device I32 [B]. B=1 accepts every positive T;
 *   B=2..8 accepts T=1..16. Empty valid_columns means every row is dense; otherwise it is
 *   device I32 [B] with each extent in [1,T]. Projection processes all M=B*T safe columns.
 *
 * Numeric / effects:
 *   Projection, convolution and SiLU form one computation. Private projected values are
 *   not an observable BF16 boundary. Outputs and published histories are BF16.
 *   Query/key/value are exact zero in invalid tails; z is written for every physical
 *   column. Only valid destination history slots are written. The caller reserves disjoint
 *   complete [base,base+T) intervals, prevents a row from overwriting another row's source,
 *   and may overlap its own source with its destination after the initial history is loaded.
 *   Other slots are unchanged. Tensor operands are disjoint except that the two read-only
 *   state selector tensors may alias. Weight parents and call-local scratch overlap none
 *   of the tensors. Device selector values and cross-row slot relationships are caller promises.
 *
 * Workspace / execution:
 *   Uses scoped caller storage as in gdn_input_proj. The capacity query includes the complete
 *   computation. Capture/replay uses the supplied stream and consumes device metadata at replay.
 */
void gdn_input_proj_conv_snapshot(const Tensor& x, const ProjectionWeights& weights,
                                  const Tensor& conv_weight, GdnConvSnapshotState state,
                                  const Tensor& valid_columns, GdnConvOutputs outputs,
                                  WorkspaceArena& workspace, cudaStream_t stream);

[[nodiscard]] std::size_t
gdn_input_proj_conv_snapshot_workspace_capacity_bytes(const ProjectionWeights& weights,
                                                      std::int32_t batch, std::int32_t min_tokens,
                                                      std::int32_t max_tokens);

/**
 * Op: gdn_input_proj_conv_record
 *
 * Evaluates the same projection/convolution/SiLU computation from a read-only source history.
 * Supported weights, tensor layouts, valid prefixes and output effects match snapshot above;
 * the record domain is B=1..8, T=2..16. conv_record is contiguous, 16-byte-aligned BF16 [C,T,B].
 * Only its valid prefix is defined. It receives each newest projected history column.
 *
 * Query/key/value/z and valid record columns are bit-identical to snapshot from the same
 * initial history, represented inputs, physical T/B and activation permission. Records equal
 * the newest column of each corresponding snapshot. Source history is never modified.
 * The record's BF16 publication does not require rounding private convolution intermediates.
 *
 * All tensor operands are disjoint from one another, the weight parents and call-local scratch.
 * Materialized computation may write projection directly to conv_record. Workspace queries
 * cover the complete Op and the supplied inclusive T interval. Capture/replay is supported.
 */
void gdn_input_proj_conv_record(const Tensor& x, const ProjectionWeights& weights,
                                const Tensor& conv_weight, GdnConvSourceState state,
                                const Tensor& valid_columns, GdnConvOutputs outputs,
                                Tensor& conv_record, WorkspaceArena& workspace,
                                cudaStream_t stream);

[[nodiscard]] std::size_t
gdn_input_proj_conv_record_workspace_capacity_bytes(const ProjectionWeights& weights,
                                                    std::int32_t batch, std::int32_t min_tokens,
                                                    std::int32_t max_tokens);

} // namespace ninfer::ops
