#include "models/qwen4_exp/ple/table.h"

#include <algorithm>
#include <bit>
#include <limits>

namespace ninfer::models::qwen4_exp {

struct PleTableSource::Impl {
    struct Shard {
        artifact::FileObjectView object;
        WeightGeometry geometry;
        Nf4Bf16Lut lut;
        std::uint64_t row_end;
    };

    artifact::FileBacking backing;
    std::vector<Shard> shards;
    PleHashControls controls;
    std::uint64_t rows       = 0;
    std::size_t max_segments = 0;
};

PleTableSource::PleTableSource(const artifact::Reader& reader, const artifact::Binding& table,
                               artifact::ObjectHandle controls)
    : impl_(std::make_unique<Impl>()) {
    using namespace artifact;
    if (table.parts.empty()) throw ArtifactError("PLE table has no shards");
    std::vector<ObjectHandle> objects;
    objects.reserve(table.parts.size());
    for (const auto& part : table.parts) {
        const auto& g = reader.geometry(part.object);
        if (g.format != QType::NF4_G16_E4M3FN || g.layout != QuantLayout::RowPagedG16_4K ||
            g.shape.size() != 2 || g.shape[1] != 160 || part.begin || part.end != g.elements)
            throw ArtifactError("PLE requires complete NF4 shards with width 160");
        impl_->rows = checked_add(impl_->rows, g.shape[0], "PLE rows");
        objects.push_back(part.object);
    }
    if (impl_->rows > std::numeric_limits<std::uint32_t>::max() ||
        checked_mul(impl_->rows, 160, "PLE elements") != table.elements)
        throw ArtifactError("PLE table logical extent is inconsistent");
    impl_->backing = prepare_file_backing(reader, objects);
    impl_->shards.reserve(objects.size());
    std::uint64_t row_end = 0;
    for (const auto object : objects) {
        const auto& g = impl_->backing.geometry(object);
        std::array<std::byte, 4> word{};
        reader.read_into(checked_add(reader.directory().tensor(object).offset, g.divisor_offset,
                                     "PLE divisor offset"),
                         word);
        const auto divisor  = std::bit_cast<float>(read_u32_le(word.data()));
        const auto view     = impl_->backing.object(object);
        impl_->max_segments = std::max(impl_->max_segments, view.segments.size());
        row_end += g.shape[0];
        try {
            impl_->shards.push_back({view, g, make_nf4_bf16_lut(divisor), row_end});
        } catch (const std::invalid_argument& e) {
            throw ArtifactError(reader.directory().tensor(object).id + ": " + e.what());
        }
    }
    const auto& resource = reader.directory().object(controls);
    if (!std::holds_alternative<ResourceObject>(resource) || object_bytes(resource) != 280)
        throw ArtifactError("invalid PLE hash controls resource");
    const auto bytes = reader.read_object(controls);
    auto word        = [&](std::size_t i) {
        return std::bit_cast<std::int64_t>(read_u64_le(bytes.data() + 8 * i));
    };
    auto& c = impl_->controls;
    for (std::size_t i = 0; i < 3; ++i) c.multipliers[i] = word(i);
    for (std::size_t i = 0; i < 16; ++i) {
        c.vocab_sizes[i] = word(3 + i);
        c.offsets[i]     = word(19 + i);
        if (c.vocab_sizes[i] <= 0 || c.offsets[i] < 0 ||
            std::uint64_t(c.offsets[i]) > impl_->rows ||
            std::uint64_t(c.vocab_sizes[i]) > impl_->rows - c.offsets[i])
            throw ArtifactError("PLE hash head exceeds table rows");
    }
}

PleTableSource::~PleTableSource()                                    = default;
PleTableSource::PleTableSource(PleTableSource&&) noexcept            = default;
PleTableSource& PleTableSource::operator=(PleTableSource&&) noexcept = default;

PleRowSource PleTableSource::row(std::uint32_t global_row) const {
    if (global_row >= impl_->rows) throw std::out_of_range("PLE global row exceeds table");
    const auto it = std::upper_bound(
        impl_->shards.begin(), impl_->shards.end(), global_row,
        [](std::uint32_t row, const Impl::Shard& shard) { return row < shard.row_end; });
    const auto local = global_row - (it->row_end - it->geometry.shape[0]);
    return {it->object, weight_paged_row_offset(it->geometry, local), &it->lut};
}

std::uint64_t PleTableSource::rows() const noexcept { return impl_->rows; }

const PleHashControls& PleTableSource::controls() const noexcept { return impl_->controls; }

std::span<const io::FileView> PleTableSource::files() const noexcept {
    return impl_->backing.files();
}

std::size_t PleTableSource::max_row_segments() const noexcept { return impl_->max_segments; }

} // namespace ninfer::models::qwen4_exp
