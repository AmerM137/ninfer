#include "ops/moe/kernels/cpu/work.h"

#include <algorithm>
#include <array>

namespace ninfer::ops::detail::moe_cpu {

void prepare_chunk(Work& work, int token_begin, int columns) noexcept {
    std::array<int, experts> indexes;
    indexes.fill(-1);
    work.group_count = work.task_count = 0;
    work.has_four_columns              = false;
    for (int t = 0; t < columns; ++t) {
        const auto offset = static_cast<std::size_t>(token_begin + t) * top_k;
        for (int j = 0; j < top_k; ++j) {
            const int expert = work.ids[offset + j];
            if (!participates(work.subset, work.mask, expert)) { continue; }
            auto& index = indexes[expert];
            if (index < 0) {
                index         = work.group_count++;
                auto& group   = work.scratch.groups[index];
                group.gate_up = work.gate_up.matrix(expert);
                group.down    = work.down.matrix(expert);
                group.count   = 0;
            }
            auto& group                = work.scratch.groups[index];
            const int member           = group.count++;
            group.tokens[member]       = t;
            group.slots[member]        = t * top_k + j;
            group.coefficients[member] = work.coefficients[offset + j];
        }
    }
    for (int g = 0; g < work.group_count; ++g) {
        for (int first = 0; first < work.scratch.groups[g].count;) {
            const int remaining = work.scratch.groups[g].count - first;
            const int tile      = remaining >= 4 ? 4 : remaining >= 2 ? 2 : 1;
            const int count     = std::min(work.max_columns, tile);

            work.scratch.tasks[work.task_count++] = {g, first, count};
            work.has_four_columns |= count == 4;
            first += count;
        }
    }
}

} // namespace ninfer::ops::detail::moe_cpu
