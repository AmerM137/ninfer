#include "fixture.cuh"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/qsa.h"

using namespace ninfer;
using namespace ninfer::bench;

namespace {
void launch_split(qsa::Projection projection, qsa::Fixture& f, const Weight& weight,
                  cudaStream_t stream) {
    auto& out = f.split;
    switch (projection) {
    case qsa::Projection::QueryGate:
        ops::qsa_input_proj(f.x, weight, out[0], out[1], stream);
        return;
    case qsa::Projection::KeyValueIndex:
        ops::qsa_input_proj(f.x, weight, out[0], out[1], out[2], out[3], stream);
        return;
    case qsa::Projection::Joint:
        ops::qsa_input_proj(f.x, weight, out[0], out[1], out[2], out[3], out[4], out[5], stream);
        return;
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = qsa::parse_options(argc, argv);
        measurement::hardware();
        DeviceContext device;
        L2FlushBuffer flush(options.timing.flush_bytes);
        measurement::Csv csv(options.timing.csv_out, "projection,round");
        for (auto projection : options.projections) {
            qsa::Weights weights(projection);
            for (int tokens : options.tokens) {
                qsa::Fixture fixture(weights, tokens);
                measurement::Row row;
                row.format = weights.weight.qtype == QType::BF16 ? "bf16" : "q8";
                row.policy = "a16";
                row.n      = weights.parent_rows;
                row.k      = qsa::kInputRows;
                row.tokens = tokens;
                for (int round = 0; round < options.rounds; ++round) {
                    auto entries = measurement::selections(options.entry, "linear", "split");
                    if (round % 2) std::reverse(entries.begin(), entries.end());
                    for (const auto& execution :
                         measurement::selections(options.timing.execution, "graph", "eager"))
                        for (const auto& cache :
                             measurement::selections(options.timing.cache, "cold", "warm")) {
                            for (const auto& entry : entries) {
                                const bool baseline = entry == "linear";
                                row.op              = baseline ? "linear" : "qsa_input_proj";
                                const auto result   = measurement::measure(
                                    [&](auto stream) {
                                        if (baseline)
                                            ops::linear(fixture.x, weights.weight, fixture.packed,
                                                          stream);
                                        else
                                            launch_split(projection, fixture, weights.weight,
                                                           stream);
                                    },
                                    [](auto) {}, flush, device.stream, options.timing, execution,
                                    cache);
                                csv.write(row, result, execution, cache,
                                          std::string(qsa::name(projection)) + "," +
                                              std::to_string(round));
                            }
                        }
                }
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
