#include "core/nvtx.h"
#include "runtime/contract/types.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string_view>

namespace {

using Clock = std::chrono::steady_clock;

std::uint64_t parse_iterations(int argc, char** argv) {
    if (argc == 1) { return 1000000; }
    if (argc != 3 || std::string_view(argv[1]) != "--iterations") {
        throw std::invalid_argument("usage: ninfer_host_timing_bench [--iterations N]");
    }
    char* end                      = nullptr;
    const unsigned long long value = std::strtoull(argv[2], &end, 10);
    if (end == argv[2] || *end != '\0' || value == 0) {
        throw std::invalid_argument("--iterations must be a positive integer");
    }
    return static_cast<std::uint64_t>(value);
}

inline void black_box(std::uint64_t& value) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : "+r"(value) : : "memory");
#else
    (void)value;
#endif
}

std::uint64_t baseline(std::uint64_t iterations) {
    std::uint64_t sink = 0;
    const auto started = Clock::now();
    for (std::uint64_t round = 0; round < iterations; ++round) {
        sink += round & 1U;
        black_box(sink);
    }
    const auto finished = Clock::now();
    black_box(sink);
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(finished - started).count());
}

// Mirrors one steady ordinary-decode observation: boundary, Program submit/wait/post, Engine
// commit with a Program post segment, then maintenance/stats detail. It performs no model, CUDA
// allocation, synchronization, lock, heap allocation, or string formatting inside the loop.
std::uint64_t instrumented(std::uint64_t iterations) {
    std::uint64_t sink = 0;
    const auto started = Clock::now();
    for (std::uint64_t round = 0; round < iterations; ++round) {
        const auto boundary_started = Clock::now();
        {
            ninfer::nvtx::ScopedRange range(ninfer::nvtx::Name::EngineBoundary,
                                            ninfer::nvtx::Category::Runtime);
            black_box(sink);
        }
        sink += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - boundary_started)
                .count());

        ninfer::runtime::ExecutionTimingRecorder decode;
        decode.begin_wait();
        decode.end_wait();
        const ninfer::runtime::ExecutionTiming decode_timing = decode.finish();
        sink += decode_timing.elapsed_ns();

        const auto commit_started = Clock::now();
        std::optional<ninfer::nvtx::ScopedRange> commit_range;
        commit_range.emplace(ninfer::nvtx::Name::EngineCommitOutput,
                             ninfer::nvtx::Category::Runtime);
        commit_range.reset();
        ninfer::runtime::ExecutionTimingRecorder commit(
            ninfer::runtime::ExecutionTimingPhase::Post);
        const ninfer::runtime::ExecutionTiming commit_timing = commit.finish();
        commit_range.emplace(ninfer::nvtx::Name::EngineCommitOutput,
                             ninfer::nvtx::Category::Runtime);
        commit_range.reset();
        sink += commit_timing.elapsed_ns();
        sink += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - commit_started)
                .count());

        const auto maintenance_started = Clock::now();
        {
            ninfer::nvtx::ScopedRange maintenance(ninfer::nvtx::Name::EngineMaintenance,
                                                  ninfer::nvtx::Category::Runtime);
            ninfer::nvtx::ScopedRange detail(ninfer::nvtx::Name::StatsPublication,
                                             ninfer::nvtx::Category::Control);
            black_box(sink);
        }
        sink += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - maintenance_started)
                .count());
        black_box(sink);
    }
    const auto finished = Clock::now();
    black_box(sink);
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(finished - started).count());
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::uint64_t iterations = parse_iterations(argc, argv);

        // Register the fixed NVTX domain/messages before the measured steady-state interval.
        (void)instrumented(1);

        const std::uint64_t baseline_ns     = baseline(iterations);
        const std::uint64_t instrumented_ns = instrumented(iterations);
        const std::uint64_t incremental_ns =
            instrumented_ns > baseline_ns ? instrumented_ns - baseline_ns : 0;
        const double nanoseconds_per_round =
            static_cast<double>(incremental_ns) / static_cast<double>(iterations);
        constexpr double kBudgetNanosecondsPerRound = 500.0;

        std::cout << std::fixed << std::setprecision(2) << "iterations=" << iterations
                  << " baseline_ns=" << baseline_ns << " instrumented_ns=" << instrumented_ns
                  << " incremental_ns_per_round=" << nanoseconds_per_round
                  << " budget_ns_per_round=" << kBudgetNanosecondsPerRound << " budget_met="
                  << (nanoseconds_per_round <= kBudgetNanosecondsPerRound ? "true" : "false")
                  << '\n';
        return nanoseconds_per_round <= kBudgetNanosecondsPerRound ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
