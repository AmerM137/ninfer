#pragma once

#include "state_fixture.cuh"
#include "ninfer/ops/qsa.h"

#include <memory>
#include <numeric>
#include <random>

namespace ninfer::bench::qsa {

__global__ void initialize_sparse_cache(std::uint16_t* words, std::size_t count, bool value) {
    const unsigned seed = value ? 1311u : 701u;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += std::size_t(gridDim.x) * blockDim.x) {
        unsigned x = unsigned(i) + seed;
        x ^= x >> 16;
        x *= 0x7feb352dU;
        x ^= x >> 15;
        x *= 0x846ca68bU;
        x ^= x >> 16;
        const float f = (int(x % 10001) - 5000) / 3000.0f;
        words[i]      = value ? __half_as_ushort(__float2half_rn(f))
                              : __bfloat16_as_ushort(__float2bfloat16_rn(f));
    }
}

struct SparseGqaFixture {
    DeviceBuffer query_data, gate_data, output_data, key_data, value_data, positions, valid,
        table_rows, tables, ids, counts;
    std::vector<int> contexts, lengths;
    Tensor query, gate, out;
    PagedKVBatchLayerView cache;
    ops::QsaBatchView batch;
    ops::QsaBlockSelection selected;
    ops::QsaSparseGqaExecutionEnvelope envelope;
    std::unique_ptr<WorkspaceArena> workspace;
    std::size_t workspace_bytes, selected_tokens = 0, unique_tokens = 0;

    SparseGqaFixture(const Options& o, int tokens, DeviceExecutionView execution)
        : contexts(broadcast(o.contexts, o.batch)),
          lengths(o.valid == "dense" ? std::vector<int>(o.batch, tokens)
                                     : broadcast(measurement::integers(o.valid, 0), o.batch)) {
        if (o.storage != KvCacheStorage::BFloat16)
            throw std::invalid_argument("sparse-gqa supports bf16 KV");
        envelope.max_visible_tokens = o.context_capacity_explicit
                                          ? o.context_capacity
                                          : *std::max_element(contexts.begin(), contexts.end());
        const int B                 = o.batch,
                  logical =
                      std::max(1, (*std::max_element(contexts.begin(), contexts.end()) + 63) / 64);
        int pages = 2;
        for (int b = 0; b < B; ++b) {
            if (contexts[b] > int(envelope.max_visible_tokens) || lengths[b] > tokens ||
                lengths[b] > contexts[b])
                throw std::invalid_argument("invalid sparse context/valid length");
            pages += (contexts[b] + 63) / 64;
        }
        query_data          = make_bf16(std::size_t(6144) * tokens * B, 101, -1.5f, 1.5f);
        gate_data           = make_bf16(std::size_t(6144) * tokens * B, 103, -4.f, 4.f);
        output_data         = make_zeros(query_data.bytes);
        const int leading   = o.layout == "unaligned" ? 1 : 0;
        const auto allocate = [&](DeviceBuffer& storage, bool value) {
            const int row  = 256 + (leading ? (value ? 3 : 1) : 0);
            const int head = 64 * row, page = 2 * head * (o.layout == "layered" ? 2 : 1);
            storage = DeviceBuffer((std::size_t(page) * pages + leading) * 2);
            initialize_sparse_cache<<<1024, 256, 0, execution.stream>>>(
                static_cast<std::uint16_t*>(storage.p), storage.bytes / 2, value);
            Tensor view(static_cast<std::uint16_t*>(storage.p) + leading,
                        value ? DType::FP16 : DType::BF16, {256, 64, 2, pages});
            view.nb[1] = row * 2;
            view.nb[2] = head * 2;
            view.nb[3] = page * 2;
            return view;
        };
        const auto key = allocate(key_data, false), value = allocate(value_data, true);
        std::vector<int> pos(tokens * B, -1), selectors(B, -1), table(logical * B, -1),
            physical(pages), block_ids(std::size_t(512) * tokens * B, -1), block_counts(tokens * B);
        std::iota(physical.begin(), physical.end(), 0);
        std::mt19937 random(731);
        std::shuffle(physical.begin(), physical.end(), random);
        int next = 0;
        for (int b = 0; b < B; ++b) {
            selectors[b] = lengths[b] ? B - 1 - b : -1;
            for (int p = 0; p < (contexts[b] + 63) / 64; ++p)
                table[(B - 1 - b) * logical + p] = physical[next++];
            std::vector<int> order(contexts[b] / 4);
            std::iota(order.begin(), order.end(), 0);
            std::shuffle(order.begin(), order.end(), random);
            std::vector<bool> seen(contexts[b]);
            for (int t = 0; t < lengths[b]; ++t) {
                const int m = b * tokens + t, p = contexts[b] - lengths[b] + t, C = (p + 1) / 4,
                          c     = std::min(o.selected_blocks, C);
                pos[m]          = p;
                block_counts[m] = c;
                std::vector<int> chosen;
                if (o.selection == "shared") {
                    for (int g : order) {
                        if (g < C && int(chosen.size()) < c) chosen.push_back(g);
                        if (int(chosen.size()) == c) break;
                    }
                } else {
                    int step = C ? int(random() % C) | 1 : 1;
                    while (C && std::gcd(step, C) != 1) step += 2;
                    const int offset = C ? random() % C : 0;
                    for (int i = 0; i < c; ++i)
                        chosen.push_back(o.selection == "contiguous"
                                             ? C - 1 - i
                                             : int((std::int64_t(step) * i + offset) % C));
                }
                for (int i = 0; i < c; ++i) {
                    block_ids[std::size_t(m) * 512 + i] = chosen[i];
                    for (int j = 0; j < 4; ++j) seen[chosen[i] * 4 + j] = true;
                }
                for (int j = C * 4; j <= p; ++j) seen[j] = true;
                selected_tokens += c * 4 + (p + 1) % 4;
            }
            unique_tokens += std::count(seen.begin(), seen.end(), true);
        }
        CUDA_CHECK(cudaStreamSynchronize(execution.stream));
        positions  = measurement::make_i32(pos);
        valid      = measurement::make_i32(lengths);
        table_rows = measurement::make_i32(selectors);
        tables     = measurement::make_i32(table);
        ids        = measurement::make_i32(block_ids);
        counts     = measurement::make_i32(block_counts);
        query      = Tensor(query_data.p, DType::BF16, {256, 24, tokens, B});
        gate       = Tensor(gate_data.p, DType::BF16, {256, 24, tokens, B});
        out        = Tensor(output_data.p, DType::BF16, {256, 24, tokens, B});
        cache      = {key,
                      value,
                      {},
                      {},
                      Tensor(tables.p, DType::I32, {logical, B}),
                      256,
                      2,
                      KvCacheStorage::BFloat16};
        batch    = {Tensor(positions.p, DType::I32, {tokens, B}), Tensor(valid.p, DType::I32, {B}),
                    Tensor(table_rows.p, DType::I32, {B})};
        selected = {Tensor(ids.p, DType::I32, {512, tokens, B}),
                    Tensor(counts.p, DType::I32, {tokens, B})};
        workspace_bytes = ops::qsa_sparse_gqa_workspace_capacity_bytes(
            cache.storage, tokens, tokens, B, envelope, execution);
        workspace = std::make_unique<WorkspaceArena>(std::max<std::size_t>(256, workspace_bytes));
    }

    void launch(DeviceExecutionView execution) {
        ops::qsa_sparse_gqa(query, gate, cache, batch, selected, envelope, *workspace, out,
                            execution);
    }
};
} // namespace ninfer::bench::qsa
