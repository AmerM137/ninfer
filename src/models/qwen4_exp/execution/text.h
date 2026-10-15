#pragma once

#include "models/qwen4_exp/execution/batch.h"
#include "models/qwen4_exp/execution/workspace.h"
#include "models/qwen4_exp/execution/parameters.h"
#include "models/qwen4_exp/state/state_image.h"
#include "models/qwen4_exp/state/kv_cache.h"
#include "models/common/moe/resources.h"

namespace ninfer::models::qwen4_exp::execution {

struct TextExecution {
    MemoryArena& workspace;
    const StateViews& states;
    const QsaCache& cache;
    moe::Resources& experts;
    Tensor qsa_raw_records, qsa_coordinate_records, ple_records;
};

// Activations stay live across the Host PLE handoff. All borrowed views follow the Program's
// compute stream; canonical publication belongs to Program.
class TextRun {
public:
    TextRun(const Parameters&, TextExecution, BatchView, std::optional<SequenceSlots>,
            CudaExecutionView);
    void prefix();
    void consume_ple(const Tensor& embedding);
    void suffix(TextOutputs);

private:
    void layer(std::uint32_t);
    const Parameters& parameters_;
    TextExecution execution_;
    BatchView batch_;
    std::optional<SequenceSlots> sequence_;
    CudaExecutionView cuda_;
    MemoryArena::Scope scope_;
    workspace::Roots roots_;
    std::optional<CpuExecutionView::ActiveScope> cpu_activity_;
};

void project_head(const Tensor& sample, const LinearParameters&, Tensor& logits, MemoryArena&,
                  CudaExecutionView);

} // namespace ninfer::models::qwen4_exp::execution
