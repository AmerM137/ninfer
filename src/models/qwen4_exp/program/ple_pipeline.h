#pragma once

#include "models/qwen4_exp/program/resources.h"

namespace ninfer::models::qwen4_exp {

class PlePipeline {
public:
    PlePipeline(Resources&, const DeviceContext&);
    ~PlePipeline();
    [[nodiscard]] ple::Ticket require(std::uint32_t tokens, std::span<const ple::InputRow>);
    void prefetch(std::uint32_t tokens, std::span<const ple::InputRow>);
    [[nodiscard]] Tensor consume(ple::Ticket);
    void consumed(ple::Ticket);
    void progress();
    void wait_compute();
    void cancel(std::uint64_t sequence);
    void drain();
private:
    struct Window {
        ple::Ticket ticket;
        std::uint32_t tokens;
        std::vector<ple::SequenceIdentity> rows;
        std::vector<std::int32_t> ids;
        bool uploaded = false, acquired = false, consumed = false, cancelled = false;
        std::exception_ptr error;
    };

    [[nodiscard]] bool matches(const Window&, std::uint32_t, std::span<const ple::InputRow>) const;
    [[nodiscard]] std::optional<ple::Ticket> prepare(std::uint32_t, std::span<const ple::InputRow>,
                                                     bool);
    Window& window(ple::Ticket);
    Resources& resources_;
    CudaExecutionView cuda_;
    CudaCompletionEvent completed_;
    std::array<std::optional<Window>, 2> windows_;
};

} // namespace ninfer::models::qwen4_exp
