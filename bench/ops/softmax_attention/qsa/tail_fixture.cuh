#pragma once

#include "options.h"
#include "ninfer/ops/qsa.h"

#include <limits>
#include <memory>
#include <numeric>

namespace ninfer::bench::qsa {

inline std::vector<int> broadcast(std::vector<int> values, int batch) {
    if (values.size() == 1) values.resize(batch, values[0]);
    if (values.size() != std::size_t(batch)) throw std::invalid_argument("metadata must match B");
    return values;
}

inline std::string csv_list(const std::vector<int>& values) {
    std::string result;
    for (int value : values) {
        if (!result.empty()) result += ':';
        result += std::to_string(value);
    }
    return result;
}

template <class Word>
struct TailPlane {
    DeviceBuffer storage;
    Tensor view;
    std::vector<Word> host;
    std::size_t leading = 0;

    TailPlane(int features, int count, int outer, DType dtype, const std::string& layout,
              bool state_coordinates = false)
        : view(nullptr, dtype, {features, count, outer}) {
        if (layout == "layered") {
            const int axis = state_coordinates ? 1 : 2;
            view.nb[axis] *= 12;
        } else if (layout == "unaligned" && dtype == DType::BF16) {
            leading    = 1;
            view.nb[1] = (features + 1) * sizeof(Word);
            view.nb[2] = ((features + 1) * count + 1) * sizeof(Word);
        }
        const auto words = leading + features + (count - 1) * (view.nb[1] / sizeof(Word)) +
                           (outer - 1) * (view.nb[2] / sizeof(Word));
        host.resize(words);
        for (std::size_t i = 0; i < words; ++i) host[i] = Word((i * 37 + 101) & 0x7fff);
        storage   = DeviceBuffer(words * sizeof(Word));
        view.data = static_cast<Word*>(storage.p) + leading;
    }

    Word& at(int feature, int column, int outer = 0) {
        return host[leading + feature + std::size_t(column) * (view.nb[1] / sizeof(Word)) +
                    std::size_t(outer) * (view.nb[2] / sizeof(Word))];
    }

    void upload() { storage.copy_from_host(host.data(), host.size() * sizeof(Word)); }
};

struct TailFixture {
    TailPlane<std::uint16_t> raw_before, raw_after, raw_records;
    TailPlane<std::int32_t> rope_before, rope_after, rope_records;
    std::vector<int> bases, counts, commits;
    DeviceBuffer slots, base, valid, accepted;
    std::unique_ptr<SavedBuffer> saved_raw, saved_rope;
    ops::QsaTailStateView before, after;
    ops::QsaRawRecordReadView records;
    Tensor count;

    TailFixture(const Options& o, int capacity)
        : raw_before(128, 3, o.batch, DType::BF16, o.layout),
          raw_after(128, 3, o.batch, DType::BF16, o.layout),
          raw_records(128, capacity, o.batch, DType::BF16, o.layout),
          rope_before(3, o.batch, 1, DType::I32, o.layout, true),
          rope_after(3, o.batch, 1, DType::I32, o.layout, true),
          rope_records(3, capacity, o.batch, DType::I32, o.layout),
          bases(broadcast(o.bases, o.batch)) {
        counts = o.valid == "dense" ? std::vector<int>(o.batch, capacity)
                                    : broadcast(measurement::integers(o.valid, 0), o.batch);
        if (o.accepted == "dense")
            commits = counts;
        else if (o.accepted == "mixed") {
            for (int b = 0; b < o.batch; ++b)
                commits.push_back(b == 0 ? 0 : std::max(0, counts[b] - b % 4));
        } else
            commits = broadcast(measurement::integers(o.accepted, 0), o.batch);
        for (int b = 0; b < o.batch; ++b) {
            if (commits[b] > counts[b] || counts[b] > capacity ||
                bases[b] > std::numeric_limits<int>::max() - counts[b])
                throw std::invalid_argument("invalid tail frontier/prefix");
            for (int j = bases[b] % 4; j < 3; ++j)
                for (int c = 0; c < 128; ++c) raw_before.at(c, j, b) = 0;
            if (bases[b] % 4 == 0)
                for (int axis = 0; axis < 3; ++axis) rope_before.at(axis, b) = 0;
        }
        raw_before.upload();
        raw_after.upload();
        raw_records.upload();
        rope_before.upload();
        rope_after.upload();
        rope_records.upload();
        std::vector<int> indices(o.batch);
        std::iota(indices.begin(), indices.end(), 0);
        slots    = measurement::make_i32(indices);
        base     = measurement::make_i32(bases);
        valid    = measurement::make_i32(counts);
        accepted = measurement::make_i32(commits);
        const Tensor selected(slots.p, DType::I32, {o.batch});
        before = {raw_before.view, rope_before.view, selected};
        after  = {o.inplace ? raw_before.view : raw_after.view,
                 o.inplace ? rope_before.view : rope_after.view, selected};
        records = {raw_records.view, rope_records.view, Tensor(base.p, DType::I32, {o.batch}),
                   Tensor(valid.p, DType::I32, {o.batch})};
        count   = Tensor(accepted.p, DType::I32, {o.batch});
        if (o.inplace) {
            saved_raw  = std::make_unique<SavedBuffer>(raw_before.storage);
            saved_rope = std::make_unique<SavedBuffer>(rope_before.storage);
        }
    }

    void restore(cudaStream_t stream) const {
        if (saved_raw) {
            saved_raw->restore(stream);
            saved_rope->restore(stream);
        }
    }

    void launch(DeviceExecutionView execution) {
        ops::qsa_tail_extract(before, records, count, after, execution);
    }
};

} // namespace ninfer::bench::qsa
