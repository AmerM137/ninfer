#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

template <int Pairs>
struct alignas(Pairs* static_cast<int>(sizeof(__nv_bfloat162))) Bf16PairPack {
    static_assert(Pairs == 1 || Pairs == 2 || Pairs == 4);
    __nv_bfloat162 pair[Pairs];
};

using Bf16x4Pack = Bf16PairPack<2>;
using Bf16x8Pack = Bf16PairPack<4>;

static_assert(sizeof(Bf16x4Pack) == 8);
static_assert(sizeof(Bf16x8Pack) == 16);

} // namespace ninfer::ops
