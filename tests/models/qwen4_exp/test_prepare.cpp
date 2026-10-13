#include "models/qwen4_exp/ple/gpu.h"
#include "models/qwen4_exp/ple/hash.h"
#include "models/qwen4_exp/ple/row_reader.h"
#include "core/buffer.h"
#include "core/decode_graph.h"
#include "core/transfer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

namespace {
using namespace ninfer;
using namespace ninfer::models::qwen4_exp;
bool fail_event_record = false;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

template <class F>
void rejects(F fn) {
    try {
        fn();
    } catch (const std::logic_error&) { return; }
    throw std::runtime_error("invalid PLE lifecycle operation accepted");
}

template <class F>
void until(F ready) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!ready()) {
        require(std::chrono::steady_clock::now() < end, "PLE completion timed out");
        std::this_thread::yield();
    }
}

template <class T>
T read() {
    T value;
    std::cin.read(reinterpret_cast<char*>(&value), sizeof(value));
    require(bool(std::cin), "truncated test input");
    return value;
}

void emit(const void* data, std::size_t bytes) {
    std::cout.write(static_cast<const char*>(data), bytes);
}

struct Input {
    std::uint32_t tokens = read<std::uint32_t>(), batch = read<std::uint32_t>();
    std::int32_t eos = read<std::int32_t>();
    std::array<std::vector<std::int32_t>, 8> ids;
    std::array<PleInputRow, 8> rows;
    std::array<std::uint32_t, 8> valid{};

    Input() {
        require(batch && batch <= 8 && tokens && tokens <= 8, "invalid test shape");
        for (std::uint32_t b = 0; b < batch; ++b) {
            valid[b]         = read<std::uint32_t>();
            const auto older = read<std::int32_t>(), previous = read<std::int32_t>();
            ids[b].resize(tokens);
            for (auto& id : ids[b]) id = read<std::int32_t>();
            rows[b] = {{100 + b, 7, 31 + b, {older, previous}}, ids[b]};
        }
    }

    auto inputs() const { return std::span(rows).first(batch); }

    auto lengths() const { return std::span(valid).first(batch); }

    std::size_t bytes() const { return std::size_t(tokens) * batch * 5120; }
};

PleTicket prepare(PlePrepareService& service, const Input& input) {
    std::optional<PleTicket> ticket;
    until([&] {
        ticket = service.prepare(input.tokens, input.inputs());
        return ticket.has_value();
    });
    return *ticket;
}

void ready(PlePrepareService& service, PleTicket ticket) {
    until([&] {
        const auto s = service.poll(ticket);
        if (s.error) std::rethrow_exception(s.error);
        return s.status == PlePrepareStatus::HostReady && s.drained;
    });
}

void row_reads(const PleTableSource& source) {
    std::array<std::uint16_t, 160 * 8> a, b{};
    a.fill(0xbeef);
    ple_detail::RowReader reader(source, 2, 8, 4, 0);
    reader.begin(0, a, false);
    reader.begin(1, b, true);
    reader.request(0, 0, 44);
    reader.request(0, 1, 44);
    reader.request(1, 0, 44);
    reader.cancel(0, 0, 8);
    until([&] {
        reader.pump();
        return reader.drained(0) && reader.drained(1);
    });
    require(std::all_of(a.begin(), a.end(), [](auto v) { return v == 0xbeef; }),
            "cancelled row reader wrote output");
    require(!reader.error(1), "shared row failed");
    const auto reads = reader.stats().reads;
    reader.release(0);
    reader.request(1, 1, 44);
    require(reader.stats().reads == reads &&
                std::equal(b.begin(), b.begin() + 160, b.begin() + 160),
            "releasing a cancelled owner lost a shared row");
    emit(b.data(), 320);
    reader.release(1);

    ple_detail::RowReader cancellation(source, 2, 8, 1, 0);
    cancellation.begin(0, a, false);
    cancellation.begin(1, b, true);
    cancellation.request(0, 0, 0);
    cancellation.request(1, 0, 46);
    cancellation.cancel(0, 0, 8);
    cancellation.pump();
    require(cancellation.drained(0), "queued cancellation waited for another owner's I/O");
    until([&] {
        cancellation.pump();
        return cancellation.drained(1);
    });
    cancellation.release(0);
    cancellation.release(1);

    ple_detail::RowReader cached(source, 1, 8, 4, 512);
    const auto one = [&] {
        cached.begin(0, b, true);
        cached.request(0, 0, 44);
        until([&] {
            cached.pump();
            return cached.drained(0);
        });
        require(!cached.error(0), "cached row failed");
        cached.release(0);
    };
    one();
    const auto first = cached.stats().reads;
    one();
    require(cached.stats().reads == first, "row cache hit performed I/O");
    cached.begin(0, a, true);
    for (std::uint32_t i = 0; i < 5; ++i) cached.request(0, i, i);
    until([&] {
        cached.pump();
        return cached.drained(0);
    });
    cached.release(0);
    const auto before = cached.stats().reads;
    one();
    require(cached.stats().reads > before, "bounded cache did not evict");
    emit(b.data(), 320);
}

void hashes(const PleTableSource& source, const Input& input) {
    for (const auto& row : input.inputs()) {
        std::vector<std::uint32_t> full(input.tokens * 16), split(full.size());
        ple_hash(row.ids, row.identity.raw_tail, input.eos, source.controls(), full);
        auto tail = row.identity.raw_tail;
        for (std::uint32_t t = 0; t < input.tokens; ++t) {
            ple_hash(row.ids.subspan(t, 1), tail, input.eos, source.controls(),
                     std::span(split).subspan(t * 16, 16));
            tail = {tail[1], row.ids[t]};
        }
        require(full == split, "chunked hash changed row IDs");
        emit(full.data(), full.size() * sizeof(full[0]));
    }
}

void host(const PleTableSource& source, const Input& input, bool failure) {
    HostBuffer first(input.bytes()), second(input.bytes());
    const std::array outputs{first.span(), second.span()};
    PlePrepareService service(source, input.eos,
                              {.max_positions = input.tokens * input.batch,
                               .windows       = 2,
                               .io_depth      = 4,
                               .cache_bytes   = 512},
                              outputs);
    if (!failure) {
        const std::array empty{PleInputRow{input.rows[0].identity, {}}};
        const auto cancelled = service.prepare(1, empty);
        require(cancelled.has_value(), "initial window admission failed");
        service.cancel(*cancelled);
        require(!service.prepare(1, empty, false), "prefetch consumed the last required slot");
        rejects([&] { service.prioritize(*cancelled); });
        until([&] { return service.poll(*cancelled).drained; });
        service.release(*cancelled);
    }
    const auto full = prepare(service, input);
    if (failure) {
        service.seal(full, input.lengths());
        until([&] {
            const auto s = service.poll(full);
            return s.status == PlePrepareStatus::Failed && s.drained;
        });
        require(bool(service.poll(full).error), "failed window lost its error");
        rejects([&] { (void)service.embedding(full); });
        service.release(full);
        return;
    }
    auto incremental = input.rows;
    for (std::uint32_t b = 0; b < input.batch; ++b)
        incremental[b].ids = incremental[b].ids.first(1);
    std::optional<PleTicket> next;
    until([&] {
        next = service.prepare(input.tokens, std::span(incremental).first(input.batch), false);
        return next.has_value();
    });
    require(!service.prepare(input.tokens, input.inputs()), "unbounded PLE window admission");
    rejects([&] { service.release(full); });
    rejects([&] { (void)service.embedding(full); });
    service.prioritize(*next);
    for (std::uint32_t t = 1; t < input.tokens; ++t) {
        std::array<PleAppend, 8> append;
        for (std::uint32_t b = 0; b < input.batch; ++b)
            append[b] = {b, input.rows[b].ids.subspan(t, 1)};
        service.append(*next, std::span(append).first(input.batch));
    }
    require(service.poll(*next).status == PlePrepareStatus::Preparing,
            "unsealed window published E");
    service.seal(full, input.lengths());
    service.seal(*next, input.lengths());
    rejects([&] { service.append(*next, {}); });
    ready(service, full);
    ready(service, *next);
    for (auto ticket : {full, *next}) {
        const auto e = service.embedding(ticket);
        require(e.window.tokens == input.tokens && e.window.batch == input.batch &&
                    e.window.identities[0].sequence == 100 && e.window.identities[0].frontier == 31,
                "window identity/shape changed");
        emit(e.values.data, e.values.bytes);
    }
    service.release(full);
    rejects([&] { (void)service.poll(full); });
    const auto cancelled = prepare(service, input);
    require(cancelled.slot == full.slot && cancelled.generation != full.generation,
            "slot reused without a new generation");
    service.cancel(cancelled);
    until([&] {
        const auto s = service.poll(cancelled);
        return s.status == PlePrepareStatus::Cancelled && s.drained;
    });
    service.cancel(cancelled);
    require(service.poll(cancelled).drained, "repeated cancel restarted retirement");
    service.release(cancelled);
    service.release(*next);

    const auto reads = service.stats().reads;
    const std::array empty{PleInputRow{input.rows[0].identity, {}}};
    std::optional<PleTicket> zero;
    until([&] {
        zero = service.prepare(1, empty);
        return zero.has_value();
    });
    service.seal(*zero, std::array<std::uint32_t, 1>{0});
    ready(service, *zero);
    const auto embedding = service.embedding(*zero);
    const auto* words    = static_cast<const std::uint16_t*>(embedding.values.data);
    require(embedding.values.bytes == 5120 &&
                std::all_of(words, words + 2560, [](auto w) { return w == 0; }),
            "empty reshaped window retained old E");
    require(service.stats().reads == reads, "empty window read the table");
    service.release(*zero);
}

class StreamGate {
public:
    explicit StreamGate(cudaStream_t stream) : stream_(stream) {
        CUDA_CHECK(cudaLaunchHostFunc(
            stream,
            [](void* argument) {
                auto& gate = *static_cast<StreamGate*>(argument);
                while (!gate.open_.load(std::memory_order_acquire)) std::this_thread::yield();
            },
            this));
    }

    ~StreamGate() {
        open();
        (void)cudaStreamSynchronize(stream_);
    }

    void open() { open_.store(true, std::memory_order_release); }
private:
    cudaStream_t stream_;
    std::atomic<bool> open_{false};
};

void gpu(const PleTableSource& source, const Input& input) {
    DeviceContext context;
    PleGpuPreparation preparation(source, input.eos, context,
                                  {.max_positions = input.tokens * input.batch,
                                   .windows       = 2,
                                   .io_depth      = 4,
                                   .cache_bytes   = 512});
    auto& service = preparation.host();
    DeviceBuffer copied(context.device(), input.bytes());
    std::vector<std::uint16_t> actual(input.bytes() / 2);
    const auto ticket = prepare(service, input);
    service.seal(ticket, input.lengths());
    ready(service, ticket);
    {
        StreamGate transfer(context.transfer_stream()), consumer(context.stream());
        preparation.upload(ticket, context.transfer_view());
        const auto e = preparation.embedding(ticket, context.execution_view());
        copy_bytes_async(copied.span(), {e.data, e.bytes(), e.device}, context.execution_view());
        preparation.record_consumed(ticket, context.execution_view());
        service.cancel(ticket);
        until([&] {
            const auto s = service.poll(ticket);
            return s.status == PlePrepareStatus::Cancelled && s.drained;
        });
        require(!preparation.try_release(ticket), "pending H2D released its pinned source");
        rejects([&] { (void)preparation.embedding(ticket, context.execution_view()); });
        transfer.open();
        until([&] { return preparation.upload_complete(ticket); });
        require(!preparation.try_release(ticket), "pending consumer released Device E");
        consumer.open();
        until([&] { return preparation.try_release(ticket); });
    }
    copied.copy_to_host(actual.data(), input.bytes());
    emit(actual.data(), input.bytes());

    for (bool consumer_failure : {false, true}) {
        const auto failed = prepare(service, input);
        service.seal(failed, input.lengths());
        ready(service, failed);
        bool rejected = false;
        try {
            fail_event_record = !consumer_failure;
            preparation.upload(failed, context.transfer_view());
            const auto e = preparation.embedding(failed, context.execution_view());
            copy_bytes_async(copied.span(), {e.data, e.bytes(), e.device},
                             context.execution_view());
            fail_event_record = true;
            preparation.record_consumed(failed, context.execution_view());
        } catch (const std::runtime_error&) { rejected = true; }
        require(rejected && !fail_event_record, "CUDA event failure was not reported");
        until([&] { return preparation.try_release(failed); });
    }

    const auto again = prepare(service, input);
    service.seal(again, input.lengths());
    const auto fixed = preparation.device_slot(again.slot, input.tokens, input.batch);
    DecodeGraphDefinition definition;
    definition.capture(context.stream(), [&] {
        copy_bytes_async(copied.span(), {fixed.data, fixed.bytes(), fixed.device},
                         context.execution_view());
    });
    DecodeGraphExecutable graph;
    graph.instantiate(definition);
    ready(service, again);
    preparation.upload(again, context.transfer_view());
    const auto e = preparation.embedding(again, context.execution_view());
    require(e.data == fixed.data, "Device E changed its captured address");
    graph.launch(context.stream());
    preparation.record_consumed(again, context.execution_view());
    until([&] { return preparation.try_release(again); });
    copied.copy_to_host(actual.data(), input.bytes());
    emit(actual.data(), input.bytes());
}
} // namespace

extern "C" cudaError_t __real_cudaEventRecord(cudaEvent_t, cudaStream_t);

extern "C" cudaError_t __wrap_cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    if (fail_event_record) {
        fail_event_record = false;
        return cudaErrorInvalidResourceHandle;
    }
    return __real_cudaEventRecord(event, stream);
}

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::invalid_argument("expected ARTIFACT --host|--gpu|--failure");
        const std::string_view mode = argv[2];
        if (mode == "--gpu") {
            int count         = 0;
            const auto status = cudaGetDeviceCount(&count);
            if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
                (status == cudaSuccess && !count))
                return 77;
            CUDA_CHECK(status);
        }
        Input input;
        auto source = [&] {
            artifact::Reader reader(argv[1]);
            return PleTableSource(
                reader, reader.directory().bindings.at("text/layers/1/ple/table"),
                reader.directory().component("text").resources.at("ple_hash_controls"));
        }();
        if (mode == "--failure")
            host(source, input, true);
        else if (mode == "--host") {
            row_reads(source);
            hashes(source, input);
            host(source, input, false);
        } else if (mode == "--gpu")
            gpu(source, input);
        else
            throw std::invalid_argument("unknown test mode");
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
