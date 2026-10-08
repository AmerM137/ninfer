#include "ops/gdn_input_proj/q4_q5/kernels.h"

namespace ninfer::ops::detail {

void q4_q5_gdn_input_dispatch(const Tensor& x, const Weight& qk_weight,
                              const Weight& value_z_weight, Tensor& qkv, Tensor& z,
                              cudaStream_t stream) {
    if (x.ne[1] <= 12) {
        Tensor qk    = qkv.slice(0, 0, 4096);
        Tensor value = qkv.slice(0, 4096, 6144);
        q4_q5_gdn_input_independent_launch(x, qk_weight, value_z_weight, qk, value, z, stream);
        return;
    }
    const auto schedule = x.ne[1] <= 32   ? Q4Q5GdnInputScheduleId::Mma32x32
                          : x.ne[1] <= 64 ? Q4Q5GdnInputScheduleId::Mma32x64
                                          : Q4Q5GdnInputScheduleId::Mma64x128;
    q4_q5_gdn_input_grouped_mma_launch(x, qk_weight, value_z_weight, qkv, z, schedule, stream);
}

} // namespace ninfer::ops::detail
