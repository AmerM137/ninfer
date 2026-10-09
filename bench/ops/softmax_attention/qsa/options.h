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
    std::string op    = "input-proj";
    std::string entry = "both";
    std::vector<Projection> projections{Projection::QueryGate, Projection::KeyValueIndex,
                                        Projection::Joint};
    std::vector<int> tokens{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    int rounds = 1;
    std::vector<int> capacities{16}, bases{6};
    std::string valid = "dense", accepted = "dense", layout = "contiguous";
    int batch    = 1;
    bool inplace = false;
};

inline Options parse_options(int argc, char** argv) {
    Options o;
    bool explicit_tokens    = false;
    bool projection_options = false, tail_options = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--help" || flag == "-h") {
            std::cout << "QSA Op benchmark\n"
                         "  --op input-proj|tail-extract (default input-proj)\n"
                         "  --projection qg|kv-index|qkgv-index|all\n"
                         "  --entry split|linear|both\n"
                         "  --tokens M[,M...] or --sweep START:END[:STEP] --rounds N\n"
                         "Tail extraction:\n"
                         "  --capacity R[,R...] --batch B --base-positions F0[,F0...]\n"
                         "  --valid dense|N[,N...] --accepted dense|mixed|N[,N...]\n"
                         "  --layout contiguous|layered|unaligned --inplace\n"
                         "A single F0/count broadcasts to all B rows. B=1..8.\n";
            measurement::help();
            std::exit(0);
        }
        if (flag == "--inplace") {
            o.inplace    = true;
            tail_options = true;
            continue;
        }
        projection_options |=
            flag == "--projection" || flag == "--entry" || flag == "--tokens" || flag == "--sweep";
        tail_options |= flag == "--capacity" || flag == "--batch" || flag == "--base-positions" ||
                        flag == "--valid" || flag == "--accepted" || flag == "--layout";
        const auto value = measurement::take(argc, argv, i);
        if (flag == "--op")
            o.op = value;
        else if (flag == "--capacity")
            o.capacities = measurement::integers(value);
        else if (flag == "--batch")
            o.batch = measurement::integer(value);
        else if (flag == "--base-positions")
            o.bases = measurement::integers(value, 0);
        else if (flag == "--valid")
            o.valid = value;
        else if (flag == "--accepted")
            o.accepted = value;
        else if (flag == "--layout")
            o.layout = value;
        else if (flag == "--entry") {
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
    if (o.op != "input-proj" && o.op != "tail-extract")
        throw std::invalid_argument("op: input-proj|tail-extract");
    if ((o.op == "input-proj" && tail_options) || (o.op == "tail-extract" && projection_options))
        throw std::invalid_argument("options must belong to the selected QSA Op");
    if (o.batch > 8 ||
        (o.layout != "contiguous" && o.layout != "layered" && o.layout != "unaligned"))
        throw std::invalid_argument("invalid tail batch/layout");
    return o;
}

} // namespace ninfer::bench::qsa
