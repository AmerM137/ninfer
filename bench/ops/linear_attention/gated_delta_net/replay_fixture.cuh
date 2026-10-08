#pragma once
// ReplaySSM recurrent-Record and all-layer Fold benchmark.
//
// Each timed GPU body is exactly one public Op call. Row-control construction, buffer
// initialization, and L2 eviction are outside the timed interval. Both graph and eager launches
// report host submission time separately from CUDA-event GPU latency.

#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gdn_replay.h"

#include "core/gdn_replay_records.h"
#include "core/layout.h"
#include "core/linear_attention_state.h"
#include "op_measurement.cuh"
#include "replay_options.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::bench::gdn_replay {
constexpr int kQkHeads = 16, kStateDim = 128, kRecordCapacity = 8, kStateSlots = 11;

__global__ void initialize_record_gates(float* gates, std::size_t pairs) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= pairs) return;
    gates[2 * i]     = bench::fixture::uniform(i, 311U, -1.5F, -.1F);
    gates[2 * i + 1] = bench::fixture::uniform(i, 313U, .1F, .9F);
}

class FoldFixture {
public:
    FoldFixture(const Profile& profile, std::int32_t width)
        : profile_(profile), width_(width), record_storage_(record_bytes(profile, width)),
          records_({record_storage_.p, record_storage_.bytes}, record_layout_),
          state_storage_(state_bytes(profile)),
          states_({state_storage_.p, state_storage_.bytes}, state_layout_) {
        record_storage_.fill(0); // alignment gaps only; represented planes are filled below.
        state_storage_.fill(0);
        CUDA_CHECK(bench::fixture::fill_values(static_cast<__nv_bfloat16*>(records_.conv.data),
                                               records_.conv.numel(), 301U, -.5F, .5F));
        CUDA_CHECK(bench::fixture::fill_values(static_cast<__nv_bfloat16*>(records_.key.data),
                                               records_.key.numel(), 307U, -.5F, .5F));
        CUDA_CHECK(bench::fixture::fill_values(static_cast<__nv_bfloat16*>(records_.value.data),
                                               records_.value.numel(), 309U, -.5F, .5F));
        const auto pairs = records_.gate.numel() / 2;
        initialize_record_gates<<<(pairs + 255) / 256, 256>>>(
            static_cast<float*>(records_.gate.data), pairs);
        CUDA_CHECK(cudaGetLastError());
        for (int layer = 0; layer < profile.layers; ++layer) {
            const auto view = states_.layer_view(layer);
            CUDA_CHECK(bench::fixture::fill_values(static_cast<__nv_bfloat16*>(view.conv.data),
                                                   view.conv.numel(), 401U + layer, -.5F, .5F));
            CUDA_CHECK(bench::fixture::fill_values(static_cast<float*>(view.recurrent.data),
                                                   view.recurrent.numel(), 501U + layer, -.02F,
                                                   .02F));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        initial_ = std::make_unique<bench::SavedBuffer>(state_storage_);
    }

    void restore(cudaStream_t stream) const { initial_->restore(stream); }

    [[nodiscard]] const ops::GdnReplayFoldPlan& fold_plan() const { return fold_plan_; }

private:
    static GdnReplayRecordLayout make_record_layout(const Profile& profile, std::int32_t width,
                                                    LayoutBuilder& builder) {
        return plan_gdn_replay_records(builder, {.layers          = profile.layers,
                                                 .record_capacity = kRecordCapacity,
                                                 .width           = width,
                                                 .conv_channels   = profile.conv_channels,
                                                 .qk_heads        = kQkHeads,
                                                 .value_heads     = profile.value_heads,
                                                 .key_dim         = kStateDim,
                                                 .value_dim       = kStateDim});
    }

    static LinearAttentionStatePoolLayout make_state_layout(const Profile& profile,
                                                            LayoutBuilder& builder) {
        return plan_linear_attention_state_pool(
            builder, {.layers         = static_cast<std::uint32_t>(profile.layers),
                      .conv_channels  = profile.conv_channels,
                      .conv_width     = 3,
                      .value_heads    = profile.value_heads,
                      .value_head_dim = kStateDim,
                      .key_head_dim   = kStateDim,
                      .slot_count     = kStateSlots,
                      .conv_dtype     = DType::BF16});
    }

    static std::size_t record_bytes(const Profile& profile, std::int32_t width) {
        LayoutBuilder builder;
        (void)make_record_layout(profile, width, builder);
        return builder.finish(256);
    }

    static std::size_t state_bytes(const Profile& profile) {
        LayoutBuilder builder;
        (void)make_state_layout(profile, builder);
        return builder.finish(256);
    }

    Profile profile_;
    std::int32_t width_;
    LayoutBuilder record_builder_;
    GdnReplayRecordLayout record_layout_ = make_record_layout(profile_, width_, record_builder_);
    LayoutBuilder state_builder_;
    LinearAttentionStatePoolLayout state_layout_ = make_state_layout(profile_, state_builder_);
    DeviceBuffer record_storage_;
    GdnReplayRecords records_;
    DeviceBuffer state_storage_;
    LinearAttentionStatePool states_;
    std::unique_ptr<bench::SavedBuffer> initial_;
    ops::GdnReplayFoldPlan fold_plan_{records_, states_.all_layers_view()};
};

using measurement::make_i32;

class RecordFixture {
public:
    RecordFixture(const Profile& profile, std::int32_t width, std::int32_t batch,
                  std::vector<int> valid)
        : profile_(profile), width_(width), batch_(batch), valid_policy_(valid),
          q_(bench::make_bf16(qk_elements(), 101U)), k_(bench::make_bf16(qk_elements(), 103U)),
          v_(bench::make_bf16(value_elements(), 105U)),
          g_(bench::make_f32(gate_elements(), 211U, -1.5F, -.1F)),
          beta_(bench::make_f32(gate_elements(), 213U, .1F, .9F)),
          record_states_(record_state_elements() * sizeof(float)),
          record_initial_(make_i32(record_initial_slots())), valid_(make_valid()),
          record_out_(value_elements() * sizeof(std::uint16_t)),
          key_record_(qk_elements() * sizeof(std::uint16_t)),
          value_record_(value_elements() * sizeof(std::uint16_t)),
          gate_record_(gate_elements() * 2 * sizeof(float)) {
        CUDA_CHECK(bench::fixture::fill_values(static_cast<float*>(record_states_.p),
                                               record_state_elements(), 217U, -.02F, .02F));
        CUDA_CHECK(cudaDeviceSynchronize());
        record_out_.fill(0);
        key_record_.fill(0);
        value_record_.fill(0);
        gate_record_.fill(0);
    }

    void launch_record(cudaStream_t stream) {
        Tensor q(q_.p, DType::BF16, {kStateDim, kQkHeads, width_, batch_});
        Tensor k(k_.p, DType::BF16, {kStateDim, kQkHeads, width_, batch_});
        Tensor v(v_.p, DType::BF16, {kStateDim, profile_.value_heads, width_, batch_});
        Tensor g(g_.p, DType::FP32, {profile_.value_heads, width_, batch_});
        Tensor beta(beta_.p, DType::FP32, {profile_.value_heads, width_, batch_});
        Tensor states(record_states_.p, DType::FP32,
                      {kStateDim, kStateDim, profile_.value_heads, batch_});
        Tensor valid = valid_tensor();
        Tensor initial(record_initial_.p, DType::I32, {batch_});
        Tensor key_record(key_record_.p, DType::BF16, {kStateDim, kQkHeads, width_, batch_});
        Tensor value_record(value_record_.p, DType::BF16,
                            {kStateDim, profile_.value_heads, width_, batch_});
        Tensor gate_record(gate_record_.p, DType::FP32, {2, profile_.value_heads, width_, batch_});
        Tensor out(record_out_.p, DType::BF16, {kStateDim, profile_.value_heads, width_, batch_});
        ops::gated_delta_net_replay_record(q, k, v, g, beta, scale(), states, valid, initial,
                                           {key_record, value_record, gate_record}, out, stream);
    }

private:
    [[nodiscard]] static float scale() { return 1.0F / std::sqrt(128.0F); }

    [[nodiscard]] std::size_t qk_elements() const {
        return static_cast<std::size_t>(kStateDim) * kQkHeads * width_ * batch_;
    }

    [[nodiscard]] std::size_t value_elements() const {
        return static_cast<std::size_t>(kStateDim) * profile_.value_heads * width_ * batch_;
    }

    [[nodiscard]] std::size_t gate_elements() const {
        return static_cast<std::size_t>(profile_.value_heads) * width_ * batch_;
    }

    [[nodiscard]] std::size_t state_slot_elements() const {
        return static_cast<std::size_t>(kStateDim) * kStateDim * profile_.value_heads;
    }

    [[nodiscard]] std::size_t record_state_elements() const {
        return state_slot_elements() * static_cast<std::size_t>(batch_);
    }

    [[nodiscard]] std::vector<std::int32_t> record_initial_slots() const {
        std::vector<std::int32_t> slots(static_cast<std::size_t>(batch_));
        for (std::int32_t row = 0; row < batch_; ++row) { slots[row] = row; }
        return slots;
    }

    [[nodiscard]] DeviceBuffer make_valid() const {
        return valid_policy_.empty() ? DeviceBuffer{} : make_i32(valid_policy_);
    }

    [[nodiscard]] Tensor valid_tensor() const {
        if (valid_.p == nullptr) return {};
        return Tensor(valid_.p, DType::I32, {batch_});
    }

    Profile profile_;
    std::int32_t width_;
    std::int32_t batch_;
    std::vector<int> valid_policy_;
    DeviceBuffer q_;
    DeviceBuffer k_;
    DeviceBuffer v_;
    DeviceBuffer g_;
    DeviceBuffer beta_;
    DeviceBuffer record_states_;
    DeviceBuffer record_initial_;
    DeviceBuffer valid_;
    DeviceBuffer record_out_;
    DeviceBuffer key_record_;
    DeviceBuffer value_record_;
    DeviceBuffer gate_record_;
};

} // namespace ninfer::bench::gdn_replay
