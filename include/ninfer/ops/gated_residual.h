#pragma once

#include "core/device.h"
#include "core/tensor.h"

namespace ninfer::ops {

/**
 * Op: four-stream residual expansion and grouped offset RMSNorm.
 *
 * Math / indexing:
 *   residual[d,s,m] = embedding[d,m] (word-preserving copy)
 *   inv[m] = 1 / sqrt(sum_d embedding[d,m]^2 / 2560 + epsilon)
 *   normalized_ideal[d,s,m] = embedding[d,m] * inv[m] * (1 + norm_weight[d,s]).
 *
 * Supported domain:
 *   Contiguous BF16 embedding [2560,M], norm_weight [2560,4], and both outputs
 *   [2560,4,M], with 16-byte-aligned backing. M is positive within represented views
 *   and available storage. A [T,B] block is flattened with M=T*B, column=t+T*b.
 *   epsilon is positive and finite; execution supplies a stream and positive physical SM count.
 *
 * Numeric / effects:
 *   The offset gain is not first rounded to BF16. Normalized BF16 outputs are promoted
 *   and compared directly with the complete independent FP64 formula over represented inputs;
 *   the oracle does not reproduce private reduction/staging or normalized storage rounding.
 *   Residual preserves every embedding word, including signed zero. Both outputs are fully
 *   written, mutually disjoint and disjoint from all inputs. Read-only inputs may share backing.
 *   Inputs remain unchanged. No workspace, hidden allocation or persistent state effect.
 *   Capture/replay reads current contents at the bound addresses.
 */
void hc_expand_norm(const Tensor& embedding, const Tensor& norm_weight, float epsilon,
                    Tensor& residual, Tensor& normalized, DeviceExecutionView execution);

/**
 * Op: grouped offset RMSNorm of four independent residual streams.
 *
 * Math / indexing:
 *   inv[s,m] = 1 / sqrt(sum_d residual[d,s,m]^2 / 2560 + epsilon)
 *   normalized_ideal[d,s,m] = residual[d,s,m] * inv[s,m] * (1 + norm_weight[d,s]).
 *
 * Supported domain / numeric:
 *   Contiguous BF16 residual/normalized [2560,4,M] and norm_weight [2560,4].
 *   Alignment, positive M/epsilon, execution and independent FP64 comparison follow
 *   hc_expand_norm. Each stream has its own statistics and affine, without a BF16 gain table.
 *
 * Effects:
 *   Normalized is fully written and disjoint from all inputs; inputs remain unchanged.
 *   Read-only inputs may share backing. No workspace or persistent state effect.
 *   Capture/replay reads current operand contents.
 */
void hc_grouped_rmsnorm(const Tensor& residual, const Tensor& norm_weight, float epsilon,
                        Tensor& normalized, DeviceExecutionView execution);

/**
 * Op: four-stream residual injection.
 *
 * Math / indexing:
 *   updated_ideal[d,s,m] = residual[d,s,m] + injection[s,m] * block_output[d,m].
 *
 * Supported domain:
 *   Contiguous BF16 residual/updated_residual [2560,4,M], block_output [2560,M],
 *   and FP32 injection [4,M]; all backing is 16-byte aligned. M is positive.
 *   Injection contains finite, already activated coefficients in [0,2]; these device
 *   values are a caller precondition, not host-synchronized validation. The coefficient
 *   is applied directly, without another sigmoid or scaling. Execution follows hc_expand_norm.
 *
 * Numeric / effects:
 *   The BF16 updated output is compared directly with updated_ideal evaluated independently
 *   in FP64 from represented inputs. Updated residual may exactly alias the residual view,
 *   or be disjoint; partial overlap is invalid. All other output/input overlap is invalid,
 *   including block_output/injection overlapping an in-place residual. Read-only inputs may
 *   share backing. The entire output is written; all non-aliased inputs remain unchanged.
 *   No workspace, hidden allocation or persistent state effect. Replay reads current inputs.
 */
void hc_combine(const Tensor& residual, const Tensor& block_output, const Tensor& injection,
                Tensor& updated_residual, DeviceExecutionView execution);

/**
 * Op: residual injection followed by grouped offset RMSNorm.
 *
 * Math / observable representation:
 *   updated_ideal = residual + injection * broadcast(block_output)
 *   residual_reference = BF16_RNE(updated_ideal)
 *   normalized_ideal = GroupedOffsetNorm_FP64(residual_reference, next_norm_weight, epsilon).
 *   The complete independent oracle preserves this BF16 residual seam. Updated output is
 *   compared with updated_ideal; normalized output is compared with normalized_ideal.
 *   Normalization must consume the actual produced BF16 updated residual words, which are
 *   additionally qualified against the independent grouped-norm formula. Private arithmetic
 *   and reduction organization are implementation choices, not additional public casts.
 *
 * Supported domain / effects:
 *   Combine operands/rules follow hc_combine. next_norm_weight is BF16 [2560,4],
 *   normalized is BF16 [2560,4,M], and both are 16-byte aligned. epsilon is positive and finite.
 *   Normalized is fully written and disjoint from updated_residual and all inputs. Norm weight
 *   cannot overlap an in-place residual. Updated residual allows only exact residual alias.
 *   Both outputs are available after the call; non-aliased inputs remain unchanged.
 *   No workspace or persistent state effect; capture/replay reads current operand contents.
 */
void hc_combine_norm(const Tensor& residual, const Tensor& block_output, const Tensor& injection,
                     const Tensor& next_norm_weight, float epsilon, Tensor& updated_residual,
                     Tensor& normalized, DeviceExecutionView execution);

/**
 * Unit-weight broadcast-add and grouped offset RMSNorm. The complete formula, BF16 residual
 * seam, shapes, effects and alias rules follow the injection overload with coefficient 1.
 * This overload neither reads nor creates an injection tensor.
 */
void hc_combine_norm(const Tensor& residual, const Tensor& block_output,
                     const Tensor& next_norm_weight, float epsilon, Tensor& updated_residual,
                     Tensor& normalized, DeviceExecutionView execution);

} // namespace ninfer::ops
