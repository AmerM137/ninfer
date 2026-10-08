#pragma once

#include "ops/gdn_input_proj/operands.h"

namespace ninfer::ops::detail::gdn_input {

enum class Compute { A16, A8, A4 };
enum class ConvKernel {
    Materialized,
    Q4Q5Fused,
    Q8Decode,
    Q8SlicedK,
    Q8Window,
    Q8Aggregate,
    Fp8Fused,
    Nvfp4Decode,
    Nvfp4Window,
};
enum class ProjectedConvKernel { Sequential, TokenParallel };

struct ProjectionPlan {
    Compute compute = Compute::A16;
};

struct ConvPlan {
    ProjectionPlan projection;
    ConvKernel kernel        = ConvKernel::Materialized;
    ProjectedConvKernel post = ProjectedConvKernel::Sequential;

    [[nodiscard]] bool materialized() const { return kernel == ConvKernel::Materialized; }
};

ProjectionPlan resolve_projection_plan(const WeightOperands& weights, std::int32_t columns);
// Publication does not select arithmetic. Snapshot and record consume the same physical block.
ConvPlan resolve_conv_plan(const WeightOperands& weights, Batch batch);
std::size_t projection_capacity(const WeightOperands& weights, std::int32_t first,
                                std::int32_t last);
std::size_t convolution_capacity(const WeightOperands& weights, Publication publication,
                                 std::int32_t batch, std::int32_t first, std::int32_t last);

} // namespace ninfer::ops::detail::gdn_input
