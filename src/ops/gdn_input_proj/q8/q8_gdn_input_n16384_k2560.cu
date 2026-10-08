#include "ops/gdn_input_proj/q8/kernels.h"
#include "ops/linear/q8/q8_gemv_launch.cuh"
#include "ops/linear/q8/q8_grouped_sliced_k_launch.cuh"
#include "ops/linear/q8/q8_mma_launch.cuh"
#include "ops/linear/q8/q8_sliced_k_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Output = LinearBf16SegmentedOutput<10240, 6144>;
using Launch = void (*)(const Tensor&, const Weight&, Tensor&, Tensor&, cudaStream_t);

using GemvR8K2   = Q8A16GemvSchedule<8, 1, 2, 2560, 8>;
using SlicedK4M8 = Q8A16SlicedKMmaSchedule<8, 4, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                           Q8ActivationStage::PaddedZero, 2560>;
using GroupedK2M16 =
    Q8A16GroupedSlicedKMmaSchedule<16, 2, 2, 2, 1, 2560, Cache::cg, Cache::ca, true>;
using SlicedK4M24  = Q8A16SlicedKMmaSchedule<24, 4, 1, 1, Q8ScaleAccess::Shared, Cache::ca,
                                             Cache::cg, Q8ActivationStage::PaddedZero, 2560>;
using SlicedK4M32  = Q8A16SlicedKMmaSchedule<32, 4, 2, 1, Q8ScaleAccess::Shared, Cache::ca,
                                             Cache::cg, Q8ActivationStage::PaddedZero, 2560>;
using SlicedK4M40  = Q8A16SlicedKMmaSchedule<40, 4, 2, 1, Q8ScaleAccess::Shared, Cache::ca,
                                             Cache::cg, Q8ActivationStage::PaddedZero, 2560>;
using Mma64x48     = Q8A16MmaSchedule<64, 48, 128, 16, 24, 1, 2>;
using Mma32x64     = Q8A16MmaSchedule<32, 64, 64, 32, 16, 2, 3>;
using Mma64x64K128 = Q8A16MmaSchedule<64, 64, 128, 32, 16, 1, 2>;
using Mma64x96     = Q8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>;
using Mma64x128    = Q8A16MmaSchedule<64, 128, 64, 32, 32, 2, 2, Q8MmaFragmentPipeline::Serial,
                                      Cache::cg, Cache::cg, Cache::ca, false, 2560, 20>;
using Mma64x64K64  = Q8A16MmaSchedule<64, 64, 64, 32, 16, 2, 3, Q8MmaFragmentPipeline::Serial,
                                      Cache::cg, Cache::cg, Cache::ca, false, 2560, 20>;

template <class Schedule>
void gemv(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z, cudaStream_t stream) {
    launch_q8_a16_gemv<Schedule>(
        q8_linear_operands(x, weight),
        Output{static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)},
        LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void grouped(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z, cudaStream_t stream) {
    launch_q8_a16_grouped_sliced_k_mma<Schedule>(
        q8_linear_operands(x, weight),
        Output{static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)},
        LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void mma(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z, cudaStream_t stream) {
    launch_q8_a16_mma<Schedule>(
        q8_linear_operands(x, weight),
        Output{static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)},
        LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void sliced(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z, cudaStream_t stream) {
    launch_q8_a16_sliced_k_mma<Schedule>(
        q8_linear_operands(x, weight),
        Output{static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)},
        LinearIdentityEpilogue{}, stream);
}

Launch select_projection(std::int32_t tokens) {
    if (tokens <= 1) return gemv<GemvR8K2>;
    if (tokens <= 8) return sliced<SlicedK4M8>;
    if (tokens <= 16) return grouped<GroupedK2M16>;
    if (tokens <= 24) return sliced<SlicedK4M24>;
    if (tokens <= 32) return sliced<SlicedK4M32>;
    if (tokens <= 40) return sliced<SlicedK4M40>;
    if (tokens <= 48) return mma<Mma64x48>;
    if (tokens <= 63) return mma<Mma32x64>;
    if (tokens <= 64) return mma<Mma64x64K128>;
    if (tokens <= 96) return mma<Mma64x96>;
    if (tokens <= 128) return mma<Mma64x128>;
    if (tokens <= 512) return mma<Mma64x64K64>;
    return mma<Mma64x128>;
}
} // namespace

void q8_gdn_input_n16384_k2560_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                      cudaStream_t stream) {
    select_projection(x.ne[1])(x, weight, qkv, z, stream);
}
} // namespace ninfer::ops::detail
