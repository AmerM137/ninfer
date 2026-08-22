#pragma once

#include "ninfer/types.h"
#include "runtime/contract/transient_region.h"

#include <cstddef>
#include <memory>

namespace ninfer {

struct DeviceContext;
class DeviceArena;

namespace runtime {

// Owns the startup-frozen request transient allocation. Requests activate only a prefix; no
// request-time device allocation or replacement is permitted.
class RequestMemory {
public:
    static constexpr std::size_t kDeviceAllocationAlignment = 256;

    RequestMemory(DeviceContext& device, std::size_t frozen_capacity_bytes);
    ~RequestMemory();

    RequestMemory(const RequestMemory&)            = delete;
    RequestMemory& operator=(const RequestMemory&) = delete;
    RequestMemory(RequestMemory&&)                 = delete;
    RequestMemory& operator=(RequestMemory&&)      = delete;

    void activate(std::size_t bytes, std::size_t alignment);
    void deactivate() noexcept;

    [[nodiscard]] TransientRegion region() const noexcept;
    [[nodiscard]] ArenaMemorySummary summary() const noexcept;
    void reset_peak() noexcept;

private:
    int device = 0;
    std::unique_ptr<DeviceArena> arena;
    std::size_t active_bytes     = 0;
    std::size_t active_alignment = 1;
    std::size_t peak_bytes       = 0;
};

} // namespace runtime
} // namespace ninfer
