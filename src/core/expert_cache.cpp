#include "core/expert_cache.h"

#include "core/buffer.h"
#include "core/transfer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <stdexcept>

namespace ninfer {
namespace {

constexpr auto absent = kExpertAbsentOffset;

void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
}

void outside_capture(CudaExecutionView execution) {
    if (!execution.stream()) {
        throw std::invalid_argument("expert cache handoffs require an explicit CUDA stream");
    }
    cudaStreamCaptureStatus status;
    cuda_check(cudaStreamIsCapturing(execution.stream(), &status));
    if (status != cudaStreamCaptureStatusNone) {
        throw std::logic_error("expert cache handoffs run outside CUDA Graph capture");
    }
}

ExpertMatrixEntryLayout add_matrix(LayoutBuilder& builder, const WeightBankLayout& matrix) {
    const auto add = [&](const WeightBankPlane& plane) {
        return plane.expert_bytes ? builder.add(plane.expert_bytes, 16, "expert plane")
                                  : LayoutRegion{};
    };
    return {add(matrix.codes), add(matrix.high), add(matrix.scales)};
}

} // namespace

ExpertCacheRequirements describe_expert_cache(std::span<const ExpertBankSource> sources,
                                              const ExpertCacheConfig& config) {
    if (!config.max_batch_bytes || !config.max_batch_copies || !config.upload_slots ||
        !config.max_in_flight_uses || config.capacity_bytes % 256) {
        throw std::invalid_argument(
            "expert cache requires aligned capacity and bounded uploads/uses");
    }
    ExpertCacheRequirements out;
    out.config = config;
    LayoutBuilder offsets, divisors;
    for (const auto& bank : sources) {
        for (const auto* matrix : {&bank.gate_up, &bank.down}) {
            if (!matrix->parent) { throw std::invalid_argument("expert bank has no source"); }
            (void)bind_weight_bank(*matrix->parent, matrix->layout);
        }
        if (bank.gate_up.parent->device != bank.down.parent->device ||
            bank.gate_up.layout.region.count != bank.down.layout.region.count) {
            throw std::invalid_argument("expert matrices must share count and source device");
        }
        ExpertBankCacheLayout layout;
        layout.count    = bank.gate_up.layout.region.count;
        layout.resident = bank.gate_up.parent->device.type() == DeviceType::CUDA;
        if (!layout.resident && bank.gate_up.parent->device != Device::cpu()) {
            throw std::invalid_argument("expert source requires CPU or CUDA backing");
        }
        LayoutBuilder entry;
        layout.gate_up     = add_matrix(entry, bank.gate_up.layout);
        layout.down        = add_matrix(entry, bank.down.layout);
        layout.entry_bytes = entry.finish(256, "expert entry");
        for (const auto* matrix : {&bank.gate_up, &bank.down}) {
            layout.payload_bytes += matrix->layout.codes.expert_bytes +
                                    matrix->layout.high.expert_bytes +
                                    matrix->layout.scales.expert_bytes;
        }
        if (!layout.resident) {
            const auto validate_upload = [](const WeightBankView& matrix, CudaHostView upload) {
                validate_memory(upload.host);
                if (upload.host.data != matrix.parent->data ||
                    upload.host.bytes < matrix.parent->geometry.bytes || !upload.host.host_pinned ||
                    upload.host.device != Device::cpu() || !upload.device_address ||
                    upload.cuda_device.type() != DeviceType::CUDA) {
                    throw std::invalid_argument(
                        "Host expert source requires registered transfer backing");
                }
            };
            validate_upload(bank.gate_up, bank.gate_up_upload);
            validate_upload(bank.down, bank.down_upload);
            if (bank.gate_up_upload.cuda_device != bank.down_upload.cuda_device) {
                throw std::invalid_argument("expert upload mappings belong to different devices");
            }
        }
        layout.first_entry = out.entry_count;
        if (!layout.resident) {
            layout.offsets = offsets.add(layout.count * sizeof(std::uint64_t), 8, "expert offsets");
            if (bank.gate_up.layout.has_divisors) {
                layout.gate_up_divisors =
                    divisors.add(layout.count * sizeof(float), 4, "GU divisors");
            }
            if (bank.down.layout.has_divisors) {
                layout.down_divisors =
                    divisors.add(layout.count * sizeof(float), 4, "down divisors");
            }
            out.entry_count += layout.count;
        }
        out.banks.push_back(layout);
    }
    out.offset_bytes  = offsets.finish();
    out.divisor_bytes = divisors.finish();
    if (out.entry_count) {
        LayoutBuilder ring;
        for (std::uint32_t i = 0; i < config.upload_slots; ++i) {
            (void)ring.add(out.offset_bytes, 1, "expert index snapshot");
        }
        out.pinned_bytes = ring.finish();
    }
    if (!out.entry_count && config.capacity_bytes) {
        throw std::invalid_argument("resident expert banks require no cache pool");
    }
    return out;
}

struct ExpertCache::Impl {
    enum class State { Absent, Reserved, Published };

    struct Entry {
        ExpertKey key;
        State state                    = State::Absent;
        std::uint64_t offset           = absent;
        std::size_t copied             = 0;
        std::uint32_t pins             = 0;
        std::size_t ready_slot         = 0;
        std::uint64_t ready_generation = 0;
        std::uint64_t use_epoch        = 0;
        bool required = false, wanted = false, selected = false, dirty = false;
        std::size_t previous = SIZE_MAX, next = SIZE_MAX;
    };

    struct FreeRange {
        std::size_t offset, bytes;
    };

    struct Slot {
        PinnedHostBuffer index_snapshot;
        CudaHostView index_source;
        CudaHostCopyBatch data_copies, index_copies;
        CudaCompletionEvent done;
        std::size_t payload_bytes = 0;
        bool pending              = false;
        std::uint64_t generation = 0, wait_epoch = 0;

        Slot(const ExpertCacheRequirements& req, const DeviceContext& ctx)
            : index_snapshot(req.offset_bytes),
              index_source(index_snapshot.cuda_view(ctx.device())),
              data_copies(ctx.device(), req.config.max_batch_copies),
              index_copies(ctx.device(), req.entry_count), done(ctx) {}
    };

    struct Use {
        CudaCompletionEvent done;
        std::vector<std::size_t> entries;
        std::optional<CudaExecutionView> execution;
        std::uint64_t generation = 0;
        bool released            = false;

        Use(const DeviceContext& ctx, std::size_t count) : done(ctx) { entries.reserve(count); }
    };

    std::vector<ExpertBankSource> sources;
    ExpertCacheRequirements requirements;
    CudaExecutionView transfer;
    DeviceBuffer pool, offsets, divisors;

    std::vector<ExpertBankDeviceView> views;
    std::vector<Entry> entries;
    std::vector<FreeRange> free, prefetch_free;
    std::vector<std::unique_ptr<Slot>> slots;
    std::vector<std::unique_ptr<Use>> uses;
    std::vector<std::size_t> required, prefetch, dirty, completed, selection;
    std::size_t oldest = SIZE_MAX, newest = SIZE_MAX;
    std::uint64_t acquire_epoch = 0;
    std::size_t next_slot       = 0;
    ExpertCacheUsage counters;
    bool failed = false;

    Impl(std::span<const ExpertBankSource> input, const ExpertCacheRequirements& req,
         const DeviceContext& ctx, CudaExecutionView upload)
        : sources(input.begin(), input.end()),
          requirements(describe_expert_cache(input, req.config)), transfer(upload),
          pool(ctx.device(), requirements.config.capacity_bytes),
          offsets(ctx.device(), requirements.offset_bytes),
          divisors(ctx.device(), requirements.divisor_bytes) {
        // The description is a capacity promise; execution recomputes it from the actual sources.
        if (req != requirements) {
            throw std::invalid_argument("expert cache requirements differ from sources");
        }
        if (transfer.device() != ctx.device()) {
            throw std::invalid_argument("expert upload device differs from its backing");
        }
        outside_capture(transfer);
        offsets.fill(0xff);
        entries.reserve(requirements.entry_count);
        free.reserve(requirements.entry_count + 1);
        prefetch_free.reserve(requirements.entry_count + 1);
        free.push_back({0, pool.size()});
        for (auto* vector : {&required, &prefetch, &dirty, &completed, &selection}) {
            vector->reserve(requirements.entry_count);
        }
        for (std::size_t b = 0; b < sources.size(); ++b) {
            const auto& source = sources[b];
            const auto& layout = requirements.banks[b];
            if (!layout.resident && source.gate_up_upload.cuda_device != ctx.device()) {
                throw std::invalid_argument("expert upload mapping belongs to another GPU");
            }
            const auto bind = [&](const WeightBankView& matrix,
                                  const ExpertMatrixEntryLayout& image,
                                  const LayoutRegion& divisor_region) {
                if (layout.resident) {
                    if (matrix.parent->device != ctx.device()) {
                        throw std::invalid_argument("resident expert belongs to another GPU");
                    }
                    return device_weight_bank(matrix);
                }
                ExpertMatrixDeviceView view;
                const auto plane = [&](const LayoutRegion& region) -> const std::byte* {
                    return pool.data() && region.bytes
                               ? static_cast<const std::byte*>(pool.data()) + region.offset
                               : nullptr;
                };
                view.codes           = plane(image.codes);
                view.high            = plane(image.high);
                view.scales          = plane(image.scales);
                view.indexed_storage = pool.span();
                view.offsets =
                    static_cast<const std::uint64_t*>(layout.offsets.bind(offsets.span()).data);
                if (divisor_region.bytes) {
                    std::vector<float> values(layout.count);
                    for (std::uint32_t e = 0; e < layout.count; ++e) {
                        values[e] = matrix.divisor(e);
                    }
                    divisors.copy_from_host(values.data(), divisor_region.bytes,
                                            divisor_region.offset);
                    view.divisors =
                        static_cast<const float*>(divisor_region.bind(divisors.span()).data);
                    view.divisor_stride = 1;
                }
                return view;
            };
            views.push_back({bind(source.gate_up, layout.gate_up, layout.gate_up_divisors),
                             bind(source.down, layout.down, layout.down_divisors), layout.count,
                             ctx.device()});
            if (!layout.resident) {
                for (std::uint32_t e = 0; e < layout.count; ++e) {
                    entries.push_back({ExpertKey{ExpertBankId{static_cast<std::uint32_t>(b)}, e}});
                }
            }
        }
        if (requirements.entry_count) {
            for (std::uint32_t i = 0; i < requirements.config.upload_slots; ++i) {
                slots.push_back(std::make_unique<Slot>(requirements, ctx));
            }
        }
        for (std::uint32_t i = 0; i < requirements.config.max_in_flight_uses; ++i) {
            uses.push_back(std::make_unique<Use>(ctx, entries.size()));
        }
    }

    ~Impl() {
        // Also covers a failed event record after a successful DMA or consumer submission.
        (void)cudaStreamSynchronize(transfer.stream());
        for (const auto& use : uses) {
            if (use->execution) { (void)cudaStreamSynchronize(use->execution->stream()); }
        }
    }

    void healthy() const {
        if (failed) { throw std::runtime_error("expert cache failed during CUDA submission"); }
    }

    std::optional<std::size_t> index(ExpertKey key) const {
        if (key.bank.index >= requirements.banks.size()) { throw std::out_of_range("expert bank"); }
        const auto& bank = requirements.banks[key.bank.index];
        if (key.expert >= bank.count) { throw std::out_of_range("logical expert"); }
        if (bank.resident) { return {}; }
        return bank.first_entry + key.expert;
    }

    std::size_t size(std::size_t i) const {
        return requirements.banks[entries[i].key.bank.index].entry_bytes;
    }

    std::size_t payload_size(std::size_t i) const {
        return requirements.banks[entries[i].key.bank.index].payload_bytes;
    }

    void reap() {
        for (auto& slot : slots) {
            if (slot->pending && slot->done.ready()) { slot->pending = false; }
        }
        for (auto& use : uses) {
            if (use->execution && use->released && use->done.ready()) {
                for (const auto i : use->entries) { --entries[i].pins; }
                use->entries.clear();
                use->execution.reset();
            }
        }
    }

    std::optional<std::size_t> allocate(std::size_t bytes) {
        for (auto it = free.begin(); it != free.end(); ++it) {
            if (it->bytes < bytes) { continue; }
            const auto offset = it->offset;
            it->offset += bytes;
            it->bytes -= bytes;
            if (!it->bytes) { free.erase(it); }
            counters.used_bytes += bytes;
            return offset;
        }
        return {};
    }

    void unlink(std::size_t i) {
        auto& entry = entries[i];
        if (entry.previous == SIZE_MAX) {
            oldest = entry.next;
        } else {
            entries[entry.previous].next = entry.next;
        }
        if (entry.next == SIZE_MAX) {
            newest = entry.previous;
        } else {
            entries[entry.next].previous = entry.previous;
        }
        entry.previous = entry.next = SIZE_MAX;
    }

    void touch(std::size_t i, bool linked = true) {
        if (linked) { unlink(i); }
        auto& entry    = entries[i];
        entry.previous = newest;
        if (newest == SIZE_MAX) {
            oldest = i;
        } else {
            entries[newest].next = i;
        }
        newest = i;
    }

    void mark_dirty(std::size_t i) {
        if (!entries[i].dirty) {
            dirty.push_back(i);
            entries[i].dirty = true;
        }
    }

    static std::size_t insert_free(std::vector<FreeRange>& ranges, std::size_t offset,
                                   std::size_t bytes) {
        auto position =
            std::lower_bound(ranges.begin(), ranges.end(), offset,
                             [](const auto& range, auto offset) { return range.offset < offset; });
        if (position != ranges.begin() &&
            (position - 1)->offset + (position - 1)->bytes == offset) {
            --position;
            position->bytes += bytes;
        } else {
            position = ranges.insert(position, {offset, bytes});
        }
        if (position + 1 != ranges.end() &&
            position->offset + position->bytes == (position + 1)->offset) {
            position->bytes += (position + 1)->bytes;
            ranges.erase(position + 1);
        }
        return position->bytes;
    }

    void discard(std::size_t i) {
        auto& entry = entries[i];
        if (entry.state == State::Absent) { return; }
        unlink(i);
        insert_free(free, entry.offset, size(i));
        counters.used_bytes -= size(i);
        entry.state  = State::Absent;
        entry.offset = absent;
        entry.copied = 0;
        mark_dirty(i);
    }

    bool can_prefetch(std::size_t bytes) {
        if (bytes > pool.size()) { return false; }
        prefetch_free.assign(free.begin(), free.end());
        for (auto i = oldest; i != SIZE_MAX; i = entries[i].next) {
            const auto& entry = entries[i];
            if (entry.pins || entry.required || entry.wanted) { continue; }
            if (insert_free(prefetch_free, entry.offset, size(i)) >= bytes) { return true; }
        }
        return false;
    }

    bool reserve(std::size_t i, bool necessary) {
        auto& entry = entries[i];
        if (entry.state != State::Absent) { return true; }
        auto address = allocate(size(i));
        // A failed optional placement must not discard reusable copies without loading a new one.
        if (!address && !necessary && !can_prefetch(size(i))) { return false; }
        while (!address) {
            std::optional<std::size_t> victim;
            for (auto j = oldest; j != SIZE_MAX; j = entries[j].next) {
                const auto& candidate = entries[j];
                if (candidate.pins || candidate.required || (!necessary && candidate.wanted)) {
                    continue;
                }
                victim = j;
                break;
            }
            if (!victim) { return false; }
            discard(*victim);
            address = allocate(size(i));
        }
        entry.offset = *address;
        entry.state  = State::Reserved;
        touch(i, false);
        return true;
    }

    bool reserve_required() {
        for (const auto i : required) {
            if (reserve(i, true)) { continue; }
            if (std::any_of(entries.begin(), entries.end(), [](const auto& e) { return e.pins; })) {
                return false;
            }
            // No consumer can retain an address. Reload immutable sources into a compact pool;
            // previous uploads are ordered before the new writes on the same transfer stream.
            while (oldest != SIZE_MAX) {
                const auto j = oldest;
                unlink(j);
                entries[j].state  = State::Absent;
                entries[j].offset = absent;
                entries[j].copied = 0;
                mark_dirty(j);
            }
            free.clear();
            free.push_back({0, pool.size()});
            counters.used_bytes = 0;
            ++counters.rebuilds;
            for (const auto j : required) {
                if (!reserve(j, true)) {
                    throw std::logic_error("required set did not fit its checked capacity");
                }
            }
            break;
        }
        return true;
    }

    template <typename F>
    void planes(std::size_t i, F&& visit) const {
        const auto& entry  = entries[i];
        const auto& source = sources[entry.key.bank.index];
        const auto& layout = requirements.banks[entry.key.bank.index];
        const auto matrix  = [&](const WeightBankView& bank, CudaHostView upload,
                                const ExpertMatrixEntryLayout& image) {
            const std::array input{bank.layout.codes, bank.layout.high, bank.layout.scales};
            const std::array output{image.codes, image.high, image.scales};
            for (std::size_t p = 0; p < input.size(); ++p) {
                if (!input[p].expert_bytes) { continue; }
                const auto offset = input[p].offset + entry.key.expert * input[p].expert_stride;
                visit(upload, offset, input[p].expert_bytes, input[p].expert_stride,
                       output[p].offset);
            }
        };
        matrix(source.gate_up, source.gate_up_upload, layout.gate_up);
        matrix(source.down, source.down_upload, layout.down);
    }

    void append_copy(Slot& slot, std::size_t i, std::size_t limit) {
        auto& entry             = entries[i];
        std::size_t plane_begin = 0;
        planes(i, [&](CudaHostView source, std::size_t offset, std::size_t bytes, std::size_t,
                      std::size_t target) {
            const auto plane_end = plane_begin + bytes;
            if (entry.copied < plane_end && slot.payload_bytes < limit &&
                slot.data_copies.size() < slot.data_copies.capacity()) {
                const auto from  = entry.copied - plane_begin;
                const auto count = std::min(bytes - from, limit - slot.payload_bytes);
                slot.data_copies.append(pool.span().subspan(entry.offset + target + from, count),
                                        source.subspan(offset + from, count));
                entry.copied += count;
                slot.payload_bytes += count;
            }
            plane_begin = plane_end;
        });
    }

    bool affine_batch() const {
        if (completed.size() < 64) { return false; }
        const auto& first = entries[completed.front()];
        for (std::size_t n = 0; n < completed.size(); ++n) {
            const auto& entry = entries[completed[n]];
            if (entry.key.bank != first.key.bank || entry.key.expert != first.key.expert + n ||
                entry.offset != first.offset + n * size(completed.front())) {
                return false;
            }
        }
        return true;
    }

    void upload_affine() {
        const auto first = completed.front();
        const auto rows  = completed.size();
        planes(first, [&](CudaHostView source, std::size_t offset, std::size_t bytes,
                          std::size_t stride, std::size_t target) {
            const auto source_bytes      = (rows - 1) * stride + bytes;
            const auto destination_bytes = (rows - 1) * size(first) + bytes;
            copy_host_2d_async(
                pool.span().subspan(entries[first].offset + target, destination_bytes), size(first),
                source.subspan(offset, source_bytes), stride, bytes, rows, transfer);
            ++counters.data_copy_calls;
        });
    }

    void publish(Slot& slot) {
        if (dirty.empty()) { return; }
        std::sort(dirty.begin(), dirty.end());
        auto* words = static_cast<std::uint64_t*>(slot.index_snapshot.data());
        for (std::size_t j = 0; j < dirty.size(); ++j) {
            const auto& entry = entries[dirty[j]];
            words[j] = entry.state != State::Absent && entry.copied == payload_size(dirty[j])
                           ? entry.offset
                           : absent;
        }
        slot.index_copies.clear();
        for (std::size_t first = 0; first < dirty.size();) {
            auto end = first + 1;
            while (end < dirty.size() && dirty[end] == dirty[end - 1] + 1) { ++end; }
            const auto bytes = (end - first) * sizeof(std::uint64_t);
            slot.index_copies.append(
                offsets.span().subspan(dirty[first] * sizeof(std::uint64_t), bytes),
                slot.index_source.subspan(first * sizeof(std::uint64_t), bytes));
            counters.index_bytes += bytes;
            first = end;
        }
        slot.index_copies.submit(transfer);
        ++counters.index_copy_calls;
        for (const auto i : dirty) { entries[i].dirty = false; }
        dirty.clear();
    }

    ExpertCacheProgress progress(ExpertUploadBudget budget) {
        reap();
        ExpertCacheProgress result;
        if (slots.empty()) {
            result.required_published = true;
            return result;
        }
        if (budget.max_batches && budget.max_bytes) {
            result.waiting_for_consumers = !reserve_required();
        }
        for (std::size_t batch = 0;
             batch < budget.max_batches && result.submitted_bytes < budget.max_bytes; ++batch) {
            auto& slot = *slots[next_slot % slots.size()];
            if (slot.pending) {
                result.waiting_for_uploads = true;
                break;
            }
            slot.payload_bytes = 0;
            slot.data_copies.clear();
            const auto limit          = std::min(requirements.config.max_batch_bytes,
                                                 budget.max_bytes - result.submitted_bytes);
            const auto planning_start = std::chrono::steady_clock::now();
            completed.clear();
            bool whole_experts = true;
            const auto append  = [&](std::size_t i) {
                auto& entry = entries[i];
                if (entry.state != State::Reserved || entry.copied == payload_size(i)) {
                    return true;
                }
                const auto remaining = payload_size(i) - entry.copied;
                if (slot.payload_bytes && remaining > limit - slot.payload_bytes) { return false; }
                if (entry.copied) { whole_experts = false; }
                append_copy(slot, i, limit);
                if (entry.copied == payload_size(i)) {
                    completed.push_back(i);
                } else {
                    whole_experts = false;
                    return false;
                }
                return slot.payload_bytes < limit &&
                       slot.data_copies.size() < slot.data_copies.capacity();
            };
            bool space = true;
            for (const auto i : required) {
                if (!(space = append(i))) { break; }
            }
            // A required set blocked by live consumers must not lose free space to prefetch.
            if (space && !result.waiting_for_consumers) {
                for (const auto i : prefetch) {
                    if (!reserve(i, false)) { continue; }
                    if (!append(i)) { break; }
                }
            }
            const bool affine = whole_experts && affine_batch();
            counters.planning_nanoseconds += std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                 std::chrono::steady_clock::now() - planning_start)
                                                 .count();
            if (!slot.payload_bytes && dirty.empty()) { break; }
            const auto submit_start = std::chrono::steady_clock::now();
            if (slot.payload_bytes) {
                if (affine) {
                    upload_affine();
                } else {
                    slot.data_copies.submit(transfer);
                    ++counters.data_copy_calls;
                }
                counters.data_bytes += slot.payload_bytes;
            }
            for (const auto i : completed) { mark_dirty(i); }
            publish(slot);
            slot.done.record(transfer.stream());
            slot.pending = true;
            ++slot.generation;
            for (const auto i : completed) {
                entries[i].state            = State::Published;
                entries[i].ready_slot       = next_slot % slots.size();
                entries[i].ready_generation = slot.generation;
            }
            counters.submit_nanoseconds += std::chrono::duration_cast<std::chrono::nanoseconds>(
                                               std::chrono::steady_clock::now() - submit_start)
                                               .count();
            ++result.submitted_batches;
            result.submitted_bytes += slot.payload_bytes;
            ++next_slot;
        }
        result.required_published = std::all_of(required.begin(), required.end(), [&](auto i) {
            return entries[i].state == State::Published;
        });
        return result;
    }
};

ExpertCache::ExpertCache(std::span<const ExpertBankSource> sources,
                         const ExpertCacheRequirements& requirements, const DeviceContext& device,
                         CudaExecutionView upload)
    : impl_(std::make_unique<Impl>(sources, requirements, device, upload)) {}

ExpertCache::~ExpertCache() = default;

void ExpertCache::prepare(std::span<const ExpertKey> required,
                          std::span<const ExpertKey> prefetch) {
    auto& state = *impl_;
    state.healthy();
    // Validate before replacing a pending request.
    std::size_t bytes = 0;
    auto& selected    = state.selection;
    for (const auto i : selected) { state.entries[i].selected = false; }
    selected.clear();
    for (const auto key : required) {
        if (auto i = state.index(key); i && !state.entries[*i].selected) {
            selected.push_back(*i);
            state.entries[*i].selected = true;
        }
    }
    std::sort(selected.begin(), selected.end());
    for (const auto i : selected) {
        if (state.size(i) > state.pool.size() - bytes) {
            throw std::invalid_argument("required expert set exceeds the cache capacity");
        }
        bytes += state.size(i);
    }
    for (const auto key : prefetch) { (void)state.index(key); }
    for (const auto i : state.required) {
        state.entries[i].required = state.entries[i].wanted = false;
    }
    for (const auto i : state.prefetch) { state.entries[i].wanted = false; }
    state.required.assign(selected.begin(), selected.end());
    state.prefetch.clear();
    for (const auto i : state.required) {
        state.entries[i].required = state.entries[i].wanted = true;
    }
    for (const auto key : prefetch) {
        if (auto i = state.index(key); i && !state.entries[*i].wanted) {
            state.prefetch.push_back(*i);
            state.entries[*i].wanted = true;
        }
    }
}

ExpertCacheProgress ExpertCache::progress(ExpertUploadBudget budget) {
    impl_->healthy();
    outside_capture(impl_->transfer);
    try {
        return impl_->progress(budget);
    } catch (...) {
        impl_->failed = true;
        throw;
    }
}

std::optional<ExpertCacheUse> ExpertCache::acquire(std::span<const ExpertKey> required,
                                                   CudaExecutionView execution) {
    auto& state = *impl_;
    state.healthy();
    if (execution.device() != state.transfer.device()) {
        throw std::invalid_argument("expert consumer device differs");
    }
    outside_capture(execution);
    state.reap();
    for (const auto key : required) {
        if (auto i = state.index(key)) {
            if (state.entries[*i].state != Impl::State::Published) { return {}; }
        }
    }
    for (std::size_t u = 0; u < state.uses.size(); ++u) {
        auto& use = *state.uses[u];
        if (use.execution) { continue; }
        ++state.acquire_epoch;
        use.execution = execution;
        use.released  = false;
        ++use.generation;
        try {
            for (const auto key : required) {
                if (const auto i = state.index(key)) {
                    const auto& entry = state.entries[*i];
                    auto& slot        = *state.slots[entry.ready_slot];
                    if (entry.ready_generation == slot.generation &&
                        slot.wait_epoch != state.acquire_epoch) {
                        slot.done.wait(execution.stream());
                        slot.wait_epoch = state.acquire_epoch;
                    }
                }
            }
        } catch (...) {
            state.failed = true;
            throw;
        }
        for (const auto key : required) {
            if (auto i = state.index(key)) {
                auto& entry = state.entries[*i];
                if (entry.use_epoch == state.acquire_epoch) { continue; }
                entry.use_epoch = state.acquire_epoch;
                ++entry.pins;
                state.touch(*i);
                use.entries.push_back(*i);
            }
        }
        return ExpertCacheUse{this, static_cast<std::uint32_t>(u), use.generation};
    }
    throw std::logic_error("expert cache concurrent use capacity exhausted");
}

void ExpertCache::release(ExpertCacheUse token, CudaExecutionView execution) {
    auto& state = *impl_;
    state.healthy();
    if (token.owner != this) { throw std::invalid_argument("expert use belongs to another cache"); }
    auto& use = *state.uses.at(token.slot);
    if (!use.execution || use.generation != token.generation || use.released ||
        execution.device() != use.execution->device() ||
        execution.stream() != use.execution->stream()) {
        throw std::invalid_argument("expert use must be released once on its consumer stream");
    }
    outside_capture(execution);
    try {
        use.done.record(execution.stream());
    } catch (...) {
        state.failed = true;
        throw;
    }
    use.released = true;
}

const ExpertBankDeviceView& ExpertCache::view(ExpertBankId bank) const {
    return impl_->views.at(bank.index);
}

ExpertCacheUsage ExpertCache::usage() const {
    auto out = impl_->counters;
    for (const auto& slot : impl_->slots) {
        if (slot->pending && !slot->done.ready()) { out.in_flight_bytes += slot->payload_bytes; }
    }
    for (const auto range : impl_->free) {
        out.largest_free_bytes = std::max(out.largest_free_bytes, range.bytes);
    }
    for (const auto& entry : impl_->entries) {
        out.published_experts += entry.state == Impl::State::Published;
    }
    return out;
}

} // namespace ninfer
