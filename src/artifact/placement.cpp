#include "artifact/placement.h"

#include "artifact/reader.h"

#include <algorithm>

namespace ninfer::artifact {

MaterializationPlan plan_materialization(const Reader& reader,
                                         std::span<const PlacementRequest> requests,
                                         PreparedResources resources) {
    if (resources.source && resources.source != &reader) {
        throw ArtifactError("prepared resources belong to another load session");
    }
    MaterializationPlan out;
    out.source       = &reader;
    out.object_count = reader.directory().objects.size();
    out.resources    = std::move(resources);
    std::vector<bool> host(out.object_count), registered(out.object_count),
        device(out.object_count);
    bool has_device = false;
    for (const auto& request : requests) {
        (void)reader.geometry(request.object);
        if (request.device.type() == DeviceType::CPU) {
            host.at(request.object.index) = true;
            if (request.host_access == HostAccess::CudaTransfer) {
                registered.at(request.object.index) = true;
            }
        } else {
            if (request.device.type() != DeviceType::CUDA ||
                (has_device && request.device != out.device) ||
                request.host_access != HostAccess::CpuOnly) {
                throw ArtifactError("materialization requires one CUDA device");
            }
            out.device                      = request.device;
            has_device                      = true;
            device.at(request.object.index) = true;
        }
    }
    LayoutBuilder host_layout, registered_layout, device_layout;
    for (std::size_t i = 0; i < out.object_count; ++i) {
        if (!host[i] && !device[i]) { continue; }
        const ObjectHandle handle{i};
        const auto& g    = reader.geometry(handle);
        const auto& name = reader.directory().tensor(handle).id;
        if (host[i]) {
            auto& layout = registered[i] ? registered_layout : host_layout;
            auto& placements =
                registered[i] ? out.registered_host_objects : out.ordinary_host_objects;
            placements.push_back({handle, layout.add(g.bytes, g.alignment, name)});
            out.host_alignment =
                std::max(out.host_alignment, static_cast<std::size_t>(g.alignment));
        }
        if (device[i]) {
            out.device_objects.push_back({handle, device_layout.add(g.bytes, g.alignment, name)});
        }
    }
    if (!out.registered_host_objects.empty()) {
        out.host_alignment        = std::max(out.host_alignment, std::size_t{4096});
        out.registered_host_bytes = registered_layout.finish(out.host_alignment);
    }
    LayoutBuilder host_storage;
    if (out.registered_host_bytes) {
        (void)host_storage.add(out.registered_host_bytes, out.host_alignment,
                               "registered Host weights");
    }
    if (host_layout.finish()) {
        (void)host_storage.add(host_layout.finish(), out.host_alignment, "ordinary Host weights");
    }
    out.host_capacity_bytes   = host_storage.finish();
    out.device_capacity_bytes = device_layout.finish();
    return out;
}

} // namespace ninfer::artifact
