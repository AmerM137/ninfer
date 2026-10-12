#include "ops/common/cpu/avx2.h"
#include "ops/moe/kernels/cpu/compute.h"

namespace ninfer::ops::detail::moe_cpu {
namespace {
struct Avx2 {
    using Float                = __m256;
    static constexpr int lanes = 8, rows = 2, max_columns = 2;
    static constexpr bool bf16_gate = false;

    static Float zero() noexcept { return _mm256_setzero_ps(); }

    static Float load(const float* p) noexcept { return _mm256_loadu_ps(p); }

    static void store(float* p, Float v) noexcept { _mm256_storeu_ps(p, v); }

    static Float bf16(const std::uint16_t* p) noexcept {
        return ::ninfer::ops::common::cpu::avx2::load_bf16(p);
    }

    static Float fma(Float a, Float b, Float c) noexcept { return _mm256_fmadd_ps(a, b, c); }

    static Float scale(std::uint16_t bits) noexcept { return _mm256_set1_ps(_cvtsh_ss(bits)); }

    static float sum(Float v) noexcept { return ::ninfer::ops::common::cpu::avx2::reduce_sum(v); }

    template <int Bits>
    static Float decode(const std::uint8_t* code, const std::uint8_t* high) noexcept {
        return common::cpu::avx2::decode_signed_q<Bits>(code, high);
    }
};

// Retain the paired collective on AVX2. Four-column specialization and the
// BF16-input shortcut belong to the AVX-512 implementations.
template <int DownBits>
void compute_paired(void* argument, const CpuWorkerContext& worker) noexcept {
    auto& work = *static_cast<Work*>(argument);
    for (int begin = 0; begin < work.tokens;) {
        const int columns = std::min(work.token_block, work.tokens - begin);
        if (worker.rank() == 0) prepare_chunk(work, begin, columns);
        const auto input_range = worker.range(hidden * columns / Avx2::lanes);
        for (auto v = input_range.begin; v < input_range.end; ++v) {
            const auto index = v * Avx2::lanes;
            Avx2::store(work.scratch.input + index,
                        Avx2::bf16(work.input + static_cast<std::size_t>(begin) * hidden + index));
        }
        worker.barrier();
        constexpr int blocks = intermediate / channel_block;
        const auto tasks     = worker.range(work.task_count * blocks);
        for (auto index = tasks.begin; index < tasks.end; ++index) {
            const auto& task  = work.scratch.tasks[index / blocks];
            const auto& group = work.scratch.groups[task.group];
            const int channel = (index % blocks) * channel_block;
            if (task.count == 2)
                gate_up<Avx2, 2>(work, group, task.first, channel, begin, 0);
            else
                gate_up<Avx2, 1>(work, group, task.first, channel, begin, 0);
        }
        worker.barrier();
        const auto outputs = worker.range(hidden / channel_block);
        for (auto block = outputs.begin; block < outputs.end; ++block) {
            const int channel = block * channel_block;
            float accumulated[max_token_block][channel_block]{};
            for (int g = 0; g < work.group_count; ++g) {
                const auto& group = work.scratch.groups[g];
                for (int first = 0; first < group.count; first += 2) {
                    if (first + 1 < group.count)
                        down<Avx2, DownBits, 2>(work, group, first, channel, accumulated);
                    else
                        down<Avx2, DownBits, 1>(work, group, first, channel, accumulated);
                }
            }
            for (int t = 0; t < columns; ++t) {
                auto* output = work.output + static_cast<std::size_t>(begin + t) * hidden + channel;
                for (int i = 0; i < channel_block; ++i) output[i] = accumulated[t][i];
            }
        }
        begin += columns;
        if (begin < work.tokens) worker.barrier();
    }
}
} // namespace

void run_avx2(void* argument, const CpuWorkerContext& worker) noexcept {
    if (static_cast<Work*>(argument)->down_bits == 6)
        compute_paired<6>(argument, worker);
    else
        compute_paired<5>(argument, worker);
}
} // namespace ninfer::ops::detail::moe_cpu
