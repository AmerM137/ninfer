#include "ops/moe/kernels/cpu/avx512.h"
#include "ops/moe/kernels/cpu/compute.h"

namespace ninfer::ops::detail::moe_cpu {
void run_avx512(void* argument, const CpuWorkerContext& worker) noexcept {
    if (static_cast<Work*>(argument)->down_bits == 6)
        compute<Avx512, 6>(argument, worker);
    else
        compute<Avx512, 5>(argument, worker);
}
} // namespace ninfer::ops::detail::moe_cpu
