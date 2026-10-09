#pragma once

#include "op_measurement.cuh"

namespace ninfer::bench::qsa {

enum class Projection { QueryGate, KeyValueIndex, Joint };

inline const char* name(Projection p) {
    switch (p) {
    case Projection::QueryGate:
        return "qg";
    case Projection::KeyValueIndex:
        return "kv-index";
    case Projection::Joint:
        return "qkgv-index";
    }
    throw std::invalid_argument("invalid QSA projection");
}

inline std::vector<int> segments(Projection p) {
    switch (p) {
    case Projection::QueryGate:
        return {6144, 6144};
    case Projection::KeyValueIndex:
        return {512, 512, 512, 128};
    case Projection::Joint:
        return {6144, 512, 6144, 512, 512, 128};
    }
    throw std::invalid_argument("invalid QSA projection");
}

struct Options {
    measurement::Options timing;
    std::string entry = "both";
    std::vector<Projection> projections{Projection::QueryGate, Projection::KeyValueIndex,
                                        Projection::Joint};
    std::vector<int> tokens{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    int rounds = 1;
};

inline Options parse_options(int argc, char** argv) {
    Options o;
    bool explicit_tokens = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--help" || flag == "-h") {
            std::cout << "QSA input projection benchmark (A16)\n"
                         "  --projection qg|kv-index|qkgv-index|all\n"
                         "  --entry split|linear|both\n"
                         "  --tokens M[,M...] or --sweep START:END[:STEP] --rounds N\n";
            measurement::help();
            std::exit(0);
        }
        const auto value = measurement::take(argc, argv, i);
        if (flag == "--entry") {
            if (value != "split" && value != "linear" && value != "both")
                throw std::invalid_argument("entry: split|linear|both");
            o.entry = value;
        } else if (flag == "--projection") {
            if (value == "all")
                o.projections = {Projection::QueryGate, Projection::KeyValueIndex,
                                 Projection::Joint};
            else if (value == "qg")
                o.projections = {Projection::QueryGate};
            else if (value == "kv-index")
                o.projections = {Projection::KeyValueIndex};
            else if (value == "qkgv-index")
                o.projections = {Projection::Joint};
            else
                throw std::invalid_argument("projection: qg|kv-index|qkgv-index|all");
        } else if (flag == "--tokens" || flag == "--sweep") {
            if (explicit_tokens) throw std::invalid_argument("specify one token list or sweep");
            explicit_tokens = true;
            o.tokens =
                flag == "--tokens" ? measurement::integers(value) : measurement::sweep(value);
        } else if (flag == "--rounds")
            o.rounds = measurement::integer(value);
        else if (!measurement::parse(o.timing, flag, value))
            throw std::invalid_argument("unknown option: " + std::string(flag));
    }
    return o;
}

} // namespace ninfer::bench::qsa
