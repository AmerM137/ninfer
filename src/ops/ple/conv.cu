#include "ops/ple/launch.h"
#include "ops/ple/conv.cuh"

namespace ninfer::ops::detail::ple {
namespace {
constexpr int kChannelTiles = kChannels / 256;

void tiled_conv(const ConvArguments& a, cudaStream_t stream) {
    const auto tiles = (std::int64_t(a.tokens) + 7) / 8;
    conv_add_tiled_kernel<128, 2, 8>
        <<<conv_grid(tiles * a.batch * kChannelTiles), 128, 0, stream>>>(a);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace

void launch_conv(const Tensor& residual, const Tensor& gated, const Tensor& kernel,
                 const PleBatchView& batch, const PleConvStateView& before,
                 const PleConvRecordWindowView& current, Tensor& updated, const Tensor* norm_weight,
                 float epsilon, Tensor* normalized, DeviceExecutionView execution) {
    const auto a      = conv_arguments(residual, gated, kernel, batch, before, current, updated,
                                       norm_weight, epsilon, normalized);
    const bool paired = pair_aligned(kernel) && pair_aligned(before.history) &&
                        pair_aligned(current.values) &&
                        (!norm_weight || pair_aligned(*norm_weight));
    const auto columns = std::int64_t(a.tokens) * a.batch;
    if (normalized) {
        const auto groups = columns * kStreams;
        const auto blocks = conv_grid(groups);
        if (!paired) {
            conv_norm_kernel<256, 1, true><<<blocks, 256, 0, execution.stream>>>(a);
        } else if (groups <= 2LL * execution.multiprocessor_count) {
            // The wider CTA reduces short-window latency while this grid fits two CTAs/SM.
            conv_norm_kernel<512, 2, true><<<blocks, 512, 0, execution.stream>>>(a);
        } else if (columns >= 1024 && columns <= 2304) {
            // In this measured extent range, token reuse offsets the extra norm launch.
            // Y is an existing public output and supplies the norm's BF16 seam.
            tiled_conv(a, execution.stream);
            conv_norm_kernel<256, 2, false><<<blocks, 256, 0, execution.stream>>>(a);
        } else {
            conv_norm_kernel<256, 2, true><<<blocks, 256, 0, execution.stream>>>(a);
        }
    } else {
        if (paired && a.tokens >= 128) {
            tiled_conv(a, execution.stream);
        } else if (paired) {
            conv_add_kernel<128, 2>
                <<<conv_grid(columns * kChannelTiles), 128, 0, execution.stream>>>(a);
        } else {
            conv_add_kernel<256, 1>
                <<<conv_grid(columns * kChannelTiles), 256, 0, execution.stream>>>(a);
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail::ple
