#include "ops/gdn_input_proj/q8/kernels.h"

namespace ninfer::ops::detail {

void q8_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                           cudaStream_t stream) {
    if (weight.k == 2560) {
        q8_gdn_input_n16384_k2560_launch(x, weight, qkv, z, stream);
    } else if (x.ne[1] == 1) {
        q8_gdn_input_decode_launch(x, weight, qkv, z, stream);
    } else if (x.ne[1] <= 96) {
        q8_gdn_input_splitk_mma_launch(x, weight, qkv, z, stream);
    } else {
        q8_gdn_input_mma_r64_c128_launch(x, weight, qkv, z, stream);
    }
}

} // namespace ninfer::ops::detail
