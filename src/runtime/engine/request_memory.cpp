#include "runtime/engine/request_memory.h"

#include "core/arena.h"
#include "core/device.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>

namespace ninfer::runtime {
namespace {

bool is_power_of_two(std::size_t value) noexcept {
    return value != 0 && (value & (value - 1)) == 0;
}

} // namespace

RequestMemory::RequestMemory(DeviceContext& context, std::size_t frozen_capacity_bytes)
    : device(context.device) {
    if (frozen_capacity_bytes != 0) {
        CUDA_CHECK(cudaSetDevice(device));
        arena = std::make_unique<DeviceArena>(frozen_capacity_bytes);
    }
}

RequestMemory::~RequestMemory() {
    if (arena != nullptr) {
        (void)cudaSetDevice(device);
        arena.reset();
    }
}

void RequestMemory::activate(std::size_t bytes, std::size_t alignment) {
    if (bytes == 0) {
        if (alignment != 1) {
            throw std::invalid_argument("an empty transient region must use alignment one");
        }
        deactivate();
        return;
    }

    if (!is_power_of_two(alignment) || alignment > kDeviceAllocationAlignment) {
        throw std::invalid_argument("unsupported transient region alignment");
    }

    if (arena == nullptr || bytes > arena->capacity()) {
        throw std::invalid_argument("request transient exceeds its frozen startup capacity");
    }

    active_bytes     = bytes;
    active_alignment = alignment;
    peak_bytes       = std::max(peak_bytes, bytes);
}

void RequestMemory::deactivate() noexcept {
    active_bytes     = 0;
    active_alignment = 1;
}

TransientRegion RequestMemory::region() const noexcept {
    if (active_bytes == 0) { return {}; }
    return {static_cast<std::byte*>(arena->base()), active_bytes, active_alignment};
}

ArenaMemorySummary RequestMemory::summary() const noexcept {
    return ArenaMemorySummary{
        arena != nullptr ? arena->capacity() : 0,
        active_bytes,
        peak_bytes,
    };
}

void RequestMemory::reset_peak() noexcept { peak_bytes = active_bytes; }

} // namespace ninfer::runtime
