#include "ops/moe/kernels/cuda/nvfp4/mma.cuh"
#include "ops/moe/kernels/cuda/nvfp4/output.cuh"
#include "ops/moe/kernels/cuda/nvfp4/vector.cuh"

#include <algorithm>

namespace ninfer::ops::detail::moe_nvfp4 {
namespace {

__global__ void quantize_input(const __nv_bfloat16* x, std::uint8_t* codes, std::uint8_t* scales,
                               float divisor, int tokens, int* counts, float* routed_sum,
                               bool clear_counts) {
    const int group = blockIdx.x * blockDim.x + threadIdx.x;
    if (routed_sum) {
        for (int i = group; i < tokens * kHidden; i += gridDim.x * blockDim.x) {
            routed_sum[i] = 0;
        }
    }
    if (counts && clear_counts && group < kExperts) { counts[group] = 0; }
    if (group < tokens * (kHidden / 16)) {
        const auto packed = quantize_nvfp4_k16(x + std::int64_t(group) * 16, divisor);
        *reinterpret_cast<uint2*>(codes + std::int64_t(group) * 8) =
            make_uint2(packed.codes_lo, packed.codes_hi);
        scales[group] = packed.scale;
    }
}

__device__ __forceinline__ bool selected(ExpertSubsetKind kind, const int* mask, int e) {
    return kind == ExpertSubsetKind::All || mask[e] == (kind == ExpertSubsetKind::MaskOnes);
}

__global__ void count_assignments(const int* ids, ExpertSubsetKind kind, const int* mask,
                                  int* counts, int assignments) {
    __shared__ int histogram[kExperts];
    for (int e = threadIdx.x; e < kExperts; e += blockDim.x) { histogram[e] = 0; }
    __syncthreads();
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a < assignments) {
        const int e = ids[a];
        if (selected(kind, mask, e)) { atomicAdd(histogram + e, 1); }
    }
    __syncthreads();
    for (int e = threadIdx.x; e < kExperts; e += blockDim.x) {
        if (histogram[e]) { atomicAdd(counts + e, histogram[e]); }
    }
}

__global__ void scan_assignments(Groups g, int columns) {
    __shared__ int scan[kExperts];
    const int e     = threadIdx.x;
    const int count = g.counts[e];
    scan[e]         = count;
    __syncthreads();
#pragma unroll
    for (int delta = 1; delta < kExperts; delta *= 2) {
        const int preceding = e >= delta ? scan[e - delta] : 0;
        __syncthreads();
        scan[e] += preceding;
        __syncthreads();
    }
    g.offsets[e] = e ? scan[e - 1] : 0;
    g.cursors[e] = g.offsets[e];
    if (e == kExperts - 1) { g.offsets[kExperts] = scan[e]; }
    __syncthreads();
    const int jobs = (count + columns - 1) / columns;
    scan[e]        = jobs;
    __syncthreads();
#pragma unroll
    for (int delta = 1; delta < kExperts; delta *= 2) {
        const int preceding = e >= delta ? scan[e - delta] : 0;
        __syncthreads();
        scan[e] += preceding;
        __syncthreads();
    }
    const int first = e ? scan[e - 1] : 0;
    for (int j = 0; j < jobs; ++j) {
        g.job_experts[first + j] = e;
        g.job_columns[first + j] = j * columns;
    }
    if (e == kExperts - 1) { *g.jobs = scan[e]; }
}

__global__ void index_assignments(const int* ids, ExpertSubsetKind kind, const int* mask, Groups g,
                                  int assignments) {
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= assignments) { return; }
    const int expert = ids[a];
    if (!selected(kind, mask, expert)) {
        if (g.inverse) { g.inverse[a] = -1; }
        return;
    }
    const int packed      = atomicAdd(g.cursors + expert, 1);
    g.assignments[packed] = a;
    if (g.inverse) { g.inverse[a] = packed; }
}

__global__ void group_small(const int* ids, ExpertSubsetKind kind, const int* mask, Groups g,
                            int assignments, int columns) {
    __shared__ int counts[kExperts], ranks[640], scan[kExperts];
    const int tid = threadIdx.x;
    counts[tid]   = 0;
    __syncthreads();
    for (int a = tid; a < assignments; a += kExperts) {
        const int e = ids[a];
        ranks[a]    = selected(kind, mask, e) ? atomicAdd(counts + e, 1) : -1;
    }
    __syncthreads();
    const int count = counts[tid];
    scan[tid]       = count;
    __syncthreads();
#pragma unroll
    for (int delta = 1; delta < kExperts; delta *= 2) {
        const int preceding = tid >= delta ? scan[tid - delta] : 0;
        __syncthreads();
        scan[tid] += preceding;
        __syncthreads();
    }
    const int begin = tid ? scan[tid - 1] : 0;
    g.offsets[tid]  = begin;
    if (tid == kExperts - 1) { g.offsets[kExperts] = scan[tid]; }
    const int jobs = (count + columns - 1) / columns;
    __syncthreads();
    scan[tid] = jobs;
    __syncthreads();
#pragma unroll
    for (int delta = 1; delta < kExperts; delta *= 2) {
        const int preceding = tid >= delta ? scan[tid - delta] : 0;
        __syncthreads();
        scan[tid] += preceding;
        __syncthreads();
    }
    const int first = tid ? scan[tid - 1] : 0;
    for (int j = 0; j < jobs; ++j) {
        g.job_experts[first + j] = tid;
        g.job_columns[first + j] = j * columns;
    }
    if (tid == kExperts - 1) { *g.jobs = scan[tid]; }
    for (int a = tid; a < assignments; a += kExperts) {
        const int packed = ranks[a] >= 0 ? g.offsets[ids[a]] + ranks[a] : -1;
        if (g.inverse) { g.inverse[a] = packed; }
        if (packed >= 0) { g.assignments[packed] = a; }
    }
}

template <class Output>
__global__ void reduce(Storage storage, const float* coefficients, Output output, int elements) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < elements) {
        output.store(i, output.initial(i) +
                            routed_value(i / kHidden, i % kHidden, storage, coefficients));
    }
}

template <bool Gate, int Columns, int Stages, bool Atomic = false>
void launch_mma(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                const Storage& s, int job_columns, CudaExecutionView execution) {
    using S            = Schedule<Columns, Stages, Gate>;
    constexpr int rows = Gate ? kIntermediate : kHidden;
    const int subtiles = job_columns / Columns;
    const int max_jobs = (kTopK * x.ne[1] / job_columns + kExperts) * subtiles;
    const int blocks = std::min(max_jobs * (rows / S::rows), execution.multiprocessor_count() * 6);
    grouped_mma<Columns, Stages, Gate, Atomic><<<blocks, S::threads, 0, execution.stream()>>>(
        Gate ? s.input_codes : s.product_codes, Gate ? s.input_scales : s.product_scales,
        Gate ? experts.bank.gate_up : experts.bank.down,
        Gate ? *experts.activation_divisors.gate_up : *experts.activation_divisors.down,
        *experts.activation_divisors.down, s.groups, s.product_codes, s.product_scales,
        s.contributions, static_cast<const float*>(routes.coefficients.data), s.routed_sum,
        subtiles);
    CUDA_CHECK(cudaGetLastError());
}

template <bool Gate, class Function>
void with_columns(int columns, Function&& function) {
    switch (columns) {
    case 8:
        function.template operator()<8>();
        break;
    case 32:
        function.template operator()<32>();
        break;
    case 64:
        function.template operator()<64>();
        break;
    case 128:
        if constexpr (Gate) {
            function.template operator()<128>();
            break;
        }
        [[fallthrough]];
    default:
        throw std::logic_error("invalid NVFP4 MoE column tile");
    }
}
} // namespace

void launch(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
            ExpertSubset subset, const ScaledSwiGLUTerm* shared, MoeEpilogue mode,
            Tensor& destination, const Plan& plan, const Storage& s, CudaExecutionView execution,
            bool shared_prepared) {
    const auto stream = execution.stream();
    if (subset.kind != ExpertSubsetKind::None && plan.vector) {
        const auto* ids          = static_cast<const int*>(routes.ids.data);
        const auto* mask         = static_cast<const int*>(subset.mask.data);
        const auto* coefficients = static_cast<const float*>(routes.coefficients.data);
        vector_gate<<<dim3(kIntermediate / 4, x.ne[1] * kTopK), 128, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), experts.bank.gate_up, ids, subset.kind, mask,
            s.vector_activation);
        CUDA_CHECK(cudaGetLastError());
        const auto down = [&](auto output) {
            vector_down<4><<<dim3(kHidden / 4, x.ne[1]), kTopK * 32, 0, stream>>>(
                s.vector_activation, experts.bank.down, ids, coefficients, subset.kind, mask,
                output);
            CUDA_CHECK(cudaGetLastError());
        };
        if (shared) {
            down(FloatContribution{s.routed_sum});
        } else {
            with_output(destination, mode, down);
            return;
        }
    } else if (subset.kind != ExpertSubsetKind::None) {
        const int assignments = x.ne[1] * kTopK;
        const auto* ids       = static_cast<const int*>(routes.ids.data);
        const auto* mask      = static_cast<const int*>(subset.mask.data);
        quantize_input<<<(std::max(x.ne[1] * (kHidden / 16), kExperts) + 255) / 256, 256, 0,
                         stream>>>(static_cast<const __nv_bfloat16*>(x.data), s.input_codes,
                                   s.input_scales, *experts.activation_divisors.gate_up, x.ne[1],
                                   s.groups.counts, s.routed_sum, !shared_prepared);
        CUDA_CHECK(cudaGetLastError());
        if (plan.single_block_grouping) {
            group_small<<<1, kExperts, 0, stream>>>(ids, subset.kind, mask, s.groups, assignments,
                                                    plan.job_columns());
            CUDA_CHECK(cudaGetLastError());
        } else {
            if (!shared_prepared) {
                count_assignments<<<(assignments + 255) / 256, 256, 0, stream>>>(
                    ids, subset.kind, mask, s.groups.counts, assignments);
                CUDA_CHECK(cudaGetLastError());
            }
            scan_assignments<<<1, kExperts, 0, stream>>>(s.groups, plan.job_columns());
            CUDA_CHECK(cudaGetLastError());
            index_assignments<<<(assignments + 255) / 256, 256, 0, stream>>>(ids, subset.kind, mask,
                                                                             s.groups, assignments);
            CUDA_CHECK(cudaGetLastError());
        }
        with_columns<true>(plan.gate_columns, [&]<int C>() {
            launch_mma<true, C, 2>(x, experts, routes, s, plan.job_columns(), execution);
        });
        with_columns<false>(plan.down_columns, [&]<int C>() {
            if (plan.atomic_output) {
                launch_mma<false, C, 2, true>(x, experts, routes, s, plan.job_columns(), execution);
            } else {
                launch_mma<false, C, 2>(x, experts, routes, s, plan.job_columns(), execution);
            }
        });
    }
    if (shared) {
        launch_shared(x, *shared, routes, mode, destination, s, execution, shared_prepared);
    } else {
        with_output(destination, mode, [&](auto output) {
            const int elements = x.ne[1] * kHidden;
            reduce<<<(elements + 255) / 256, 256, 0, stream>>>(
                s, static_cast<const float*>(routes.coefficients.data), output, elements);
        });
        CUDA_CHECK(cudaGetLastError());
    }
}
} // namespace ninfer::ops::detail::moe_nvfp4
