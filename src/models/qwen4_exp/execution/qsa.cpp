#include "models/qwen4_exp/execution/qsa.h"
#include "models/qwen4_exp/execution/workspace.h"
#include "ninfer/ops/linear_split.h"

namespace ninfer::models::qwen4_exp::execution {

void qsa(const Tensor& input, const QsaConfig& c, const QsaParameters& p, const BatchView& batch,
         const ops::QsaTailStateView& before, ops::QsaTailStateView after, ops::QsaCacheView cache,
         ops::QsaRawRecordWriteView records, Tensor& output, MemoryArena& arena,
         CudaExecutionView cuda) {
    const auto d = dimension(c.head_dim), qh = dimension(c.query_heads), kh = dimension(c.kv_heads);
    const auto id = dimension(c.index_dim), ih = dimension(c.index_heads);
    const auto qw = dimension(c.query_width()), kw = dimension(c.kv_width());
    const auto iq = dimension(c.index_query_width()), ik = dimension(c.index_key_width());
    auto scope   = arena.scope();
    const auto t = batch.tokens, b = batch.batch, m = t * b;
    auto buffers = workspace::qsa(arena, c, m, !records.raw_keys.data);
    if (!records.raw_keys.data)
        records = {buffers.raw_records.view({ik, t, b}), buffers.coordinate_records.view({3, t, b}),
                   batch.record_offsets};
    {
        auto scratch = arena.scope();
        if (const auto* main = std::get_if<MainQsaProjection>(&p.input)) {
            const std::array qg{buffers.query.view({qw, m}), buffers.gate.view({qw, m})};
            ops::linear_split(input, main->query_gate, qg, arena, cuda);
        } else {
            const std::array all{buffers.query.view({qw, m}),       buffers.key.view({kw, m}),
                                 buffers.gate.view({qw, m}),        buffers.value.view({kw, m}),
                                 buffers.index_query.view({iq, m}), buffers.index_key};
            ops::linear_split(input, std::get<LinearParameters>(p.input), all, arena, cuda);
        }
    }
    if (const auto* main = std::get_if<MainQsaProjection>(&p.input)) {
        auto scratch = arena.scope();
        const std::array kv{buffers.key.view({kw, m}), buffers.value.view({kw, m}),
                            buffers.index_query.view({iq, m}), buffers.index_key};
        ops::linear_split(input, main->kv_index, kv, arena, cuda);
    }
    auto query       = buffers.query.view({d, qh, t, b});
    auto index_query = buffers.index_query.view({id, ih, t, b});
    const ops::QsaBatchView metadata{batch.positions, batch.valid_columns, batch.table_rows};
    ops::qsa_prepare({query, buffers.key.view({d, kh, t, b}), buffers.value.view({d, kh, t, b}),
                      index_query, buffers.index_key.view({ik, t, b})},
                     p.prepare, metadata, batch.rope_positions, before, cache, records, query,
                     index_query, cuda);
    ops::QsaBlockSelection selected{buffers.selection.view({dimension(c.block_budget), t, b}),
                                    buffers.counts.view({t, b})};
    {
        auto scratch = arena.scope();
        ops::qsa_indexer(index_query, cache.index, metadata, {batch.max_visible_tokens}, arena,
                         selected, cuda);
    }
    {
        auto scratch   = arena.scope();
        auto attention = buffers.output.view({d, qh, t, b});
        ops::qsa_sparse_gqa(query, buffers.gate.view({d, qh, t, b}), cache.main, metadata, selected,
                            {batch.max_visible_tokens}, arena, attention, cuda);
    }
    ops::linear(buffers.output.view({qw, m}), p.output.weight, output, p.output.policy, arena,
                cuda);
    ops::qsa_tail_extract(
        before,
        {records.raw_keys, records.rope_positions, batch.base_positions, batch.valid_columns},
        batch.valid_columns, after, cuda);
}

} // namespace ninfer::models::qwen4_exp::execution
