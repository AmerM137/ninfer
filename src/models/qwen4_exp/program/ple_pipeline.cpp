#include "models/qwen4_exp/program/ple_pipeline.h"
#include "core/nvtx.h"

#include <algorithm>
#include <thread>

namespace ninfer::models::qwen4_exp {

PlePipeline::PlePipeline(Resources& r, const DeviceContext& device)
    : resources_(r), cuda_(device.execution_view()), completed_(device) {}

PlePipeline::~PlePipeline() {
    try {
        drain();
    } catch (...) { std::terminate(); }
}

bool PlePipeline::matches(const Window& w, std::uint32_t tokens,
                          std::span<const ple::InputRow> rows) const {
    if (w.cancelled || w.consumed || w.tokens != tokens || w.rows.size() != rows.size())
        return false;
    std::size_t offset = 0;
    for (std::size_t b = 0; b < rows.size(); ++b) {
        const auto& a = w.rows[b];
        const auto& c = rows[b].identity;
        if (a.sequence != c.sequence || a.epoch != c.epoch || a.frontier != c.frontier ||
            a.raw_tail != c.raw_tail || rows[b].ids.size() != tokens ||
            !std::equal(rows[b].ids.begin(), rows[b].ids.end(), w.ids.begin() + offset))
            return false;
        offset += tokens;
    }
    return true;
}

std::optional<ple::Ticket>
PlePipeline::prepare(std::uint32_t tokens, std::span<const ple::InputRow> rows, bool required) {
    nvtx::ScopedRange prepare(nvtx::Name::PlePrepare, nvtx::Category::Runtime,
                              tokens * rows.size());
    progress();
    for (auto& w : windows_) {
        if (w && matches(*w, tokens, rows)) {
            if (required) resources_.ple_prepare->prioritize(w->ticket);
            return w->ticket;
        }
    }
    auto ticket = resources_.ple_prepare->prepare(tokens, rows, required);
    if (!ticket) return {};
    auto& w = windows_.at(ticket->slot);
    if (w) throw std::logic_error("PLE producer reused a live window");
    w.emplace();
    w->ticket = *ticket;
    w->tokens = tokens;
    std::array<std::uint32_t, kMaximumConcurrency> valid{};
    for (std::size_t b = 0; b < rows.size(); ++b) {
        w->rows.push_back(rows[b].identity);
        w->ids.insert(w->ids.end(), rows[b].ids.begin(), rows[b].ids.end());
        valid[b] = rows[b].ids.size();
    }
    resources_.ple_prepare->seal(*ticket, {valid.data(), rows.size()});
    return ticket;
}

ple::Ticket PlePipeline::require(std::uint32_t tokens, std::span<const ple::InputRow> rows) {
    for (;;) {
        if (auto ticket = prepare(tokens, rows, true)) return *ticket;
        for (auto& w : windows_) {
            if (w && !w->consumed && !w->cancelled) {
                resources_.ple_prepare->cancel(w->ticket);
                w->cancelled = true;
            }
        }
        std::this_thread::yield();
    }
}

void PlePipeline::prefetch(std::uint32_t tokens, std::span<const ple::InputRow> rows) {
    (void)prepare(tokens, rows, false);
}

PlePipeline::Window& PlePipeline::window(ple::Ticket ticket) {
    auto& w = windows_.at(ticket.slot);
    if (!w || w->ticket != ticket) throw std::logic_error("stale Program PLE ticket");
    return *w;
}

void PlePipeline::progress() {
    for (auto& entry : windows_) {
        if (!entry) continue;
        auto& w           = *entry;
        const auto status = resources_.ple_prepare->poll(w.ticket);
        if (status.error) w.error = status.error;
        if (!w.cancelled && !w.error && !w.uploaded &&
            status.status == ple::disk::PrepareStatus::HostReady) {
            resources_.ple_upload->upload(resources_.ple_prepare->embedding(w.ticket),
                                          resources_.ple_transfer());
            w.uploaded = true;
        }
        if ((w.consumed || w.cancelled) && status.drained &&
            resources_.ple_upload->try_release(w.ticket)) {
            resources_.ple_prepare->release(w.ticket);
            entry.reset();
        }
    }
}

Tensor PlePipeline::consume(ple::Ticket ticket) {
    nvtx::ScopedRange wait(nvtx::Name::PleWait, nvtx::Category::Runtime);
    for (;;) {
        progress();
        auto& w = window(ticket);
        if (w.error) std::rethrow_exception(w.error);
        if (w.cancelled) throw std::logic_error("consuming a cancelled PLE window");
        if (w.uploaded) {
            auto value = resources_.ple_upload->embedding(ticket, cuda_);
            w.acquired = true;
            return value;
        }
        std::this_thread::yield();
    }
}

void PlePipeline::consumed(ple::Ticket ticket) {
    auto& w = window(ticket);
    resources_.ple_upload->record_consumed(ticket, cuda_);
    w.consumed = true;
}

void PlePipeline::wait_compute() {
    completed_.record(cuda_.stream());
    while (!completed_.ready()) {
        progress();
        std::this_thread::yield();
    }
    progress();
}

void PlePipeline::cancel(std::uint64_t sequence) {
    for (auto& w : windows_) {
        if (w && !w->cancelled && std::any_of(w->rows.begin(), w->rows.end(), [&](const auto& row) {
                return row.sequence == sequence;
            })) {
            if (w->acquired && !w->consumed) consumed(w->ticket);
            resources_.ple_prepare->cancel(w->ticket);
            w->cancelled = true;
        }
    }
    progress();
}

void PlePipeline::drain() {
    for (auto& w : windows_) {
        if (w && !w->cancelled) {
            if (w->acquired && !w->consumed) consumed(w->ticket);
            resources_.ple_prepare->cancel(w->ticket);
            w->cancelled = true;
        }
    }
    while (std::any_of(windows_.begin(), windows_.end(), [](const auto& w) { return bool(w); })) {
        progress();
        std::this_thread::yield();
    }
}

} // namespace ninfer::models::qwen4_exp
