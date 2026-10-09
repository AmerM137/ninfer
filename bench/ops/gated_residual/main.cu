#include "fixture.cuh"
#include "conditioning.cuh"

using namespace ninfer;
using namespace ninfer::bench;

int main(int argc, char** argv) {
    try {
        const auto options = hc::parse(argc, argv);
        measurement::hardware();
        DeviceContext device;
        L2FlushBuffer flush(options.timing.flush_bytes);
        measurement::Csv csv(options.timing.csv_out, "alias,condition_us");
        for (int tokens : options.tokens) {
            hc::Fixture f(options, tokens);
            for (const auto& execution :
                 measurement::selections(options.timing.execution, "graph", "eager"))
                for (const auto& cache :
                     measurement::selections(options.timing.cache, "cold", "warm")) {
                    const auto result = measurement::measure(
                        [&](auto stream) { f.launch(device.execution_view().on_stream(stream)); },
                        [&](auto stream) {
                            hc::condition(device.execution_view().on_stream(stream),
                                          options.condition_us);
                            f.restore(stream);
                        },
                        flush, device.stream, options.timing, execution, cache);
                    measurement::Row row;
                    row.op          = hc::name(options.op);
                    row.format      = "bf16";
                    row.n           = hc::kWidth;
                    row.value_heads = hc::kStreams;
                    row.tokens      = tokens;
                    row.batch       = options.batch;
                    csv.write(row, result, execution, cache,
                              std::string(options.in_place ? "inplace," : "disjoint,") +
                                  std::to_string(options.condition_us));
                }
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
