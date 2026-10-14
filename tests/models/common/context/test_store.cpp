#include "core/device.h"
#include "models/qwen3_5/state/state_image.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace q36   = ninfer::models::qwen3_5;
namespace store = ninfer::models::context;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

void test_state_store(ninfer::DeviceContext& device) {
    q36::StateImageSpec spec{
        .slot_count = 4,
        .linear =
            {
                .layers         = 1,
                .conv_channels  = 8,
                .conv_width     = 3,
                .value_heads    = 2,
                .value_head_dim = 4,
                .key_head_dim   = 4,
                .conv_dtype     = ninfer::DType::BF16,
            },
        .hidden = 8,
        .dflash_local =
            q36::DFlashLocalStateSpec{.layers = 1, .capacity = 8, .kv_heads = 2, .head_dim = 4},
    };
    ninfer::LayoutBuilder builder;
    const q36::StateImageDeviceLayout layout = q36::plan_state_image_device_pool(builder, spec);
    ninfer::DeviceBuffer arena_storage(device.device(), builder.finish(256));
    ninfer::MemoryArena arena(arena_storage.span());
    q36::StateImageViews physical(arena.storage(), layout);
    ninfer::HostContextArena host_backing(layout.images.image_stride * 2,
                                          layout.images.image_stride);
    ninfer::models::qwen3_5::StateImageStore images(physical.storage(), &host_backing, 6);

    const auto initialize = [&] {
        auto handle = images.reserve_destination();
        if (handle) {
            physical.storage().clear(images.physical_slot(*handle), device.execution_view());
            images.complete_initialization(*handle);
        }
        return handle;
    };
    const auto source = initialize();
    expect(source.has_value(), "state source allocation");
    const std::int32_t original_slot = images.physical_slot(*source);
    const auto source_epoch          = images.content_epoch(*source);
    images.freeze(*source);
    expect(images.content_epoch(*source) == source_epoch, "freezing retains content identity");
    images.make_mutable(*source);
    expect(images.physical_slot(*source) == original_slot &&
               images.role(*source) == store::StateImageRole::ActiveMutable,
           "private Move preserves the logical image and physical slot");
    images.freeze(*source);
    const auto destination = images.reserve_destination();
    expect(destination.has_value(), "state destination reservation");
    const store::StateImageSelectors selectors = images.begin_write(*source, *destination);
    expect(selectors.source >= 0 && selectors.destination >= 0 &&
               selectors.source != selectors.destination,
           "fork resolves distinct physical selectors");
    expect(images.role(*destination) == store::StateImageRole::Writing,
           "pending write is not readable active content");
    expect(!images.release(*source) && !images.release(*destination),
           "fork pins both logical images");
    const auto concurrent_destination = images.reserve_destination();
    expect(concurrent_destination.has_value(), "concurrent fork destination reservation");
    (void)images.begin_write(*source, *concurrent_destination);
    expect(images.source_pins(*source) == 2 && !images.release(*source),
           "independent forks each retain the source");
    images.abort_write(*source, *concurrent_destination);
    expect(images.release(*concurrent_destination), "concurrent fork destination release");
    expect(images.source_pins(*source) == 1 && !images.release(*source) &&
               !images.release(*destination),
           "aborting one reader preserves the remaining fork's pins");
    images.complete_write(*source, *destination);
    expect(images.role(*source) == store::StateImageRole::CheckpointImmutable &&
               images.role(*destination) == store::StateImageRole::ActiveMutable &&
               images.content_epoch(*destination) != source_epoch,
           "fork commit preserves source and activates destination");
    images.retain_checkpoint_reference(*source);
    expect(!images.release(*source), "checkpoint owner keeps the source alive after fork commit");
    expect(images.release_checkpoint_owner(*source) && images.release(*destination) &&
               images.occupied() == 0,
           "checkpoint and active state ownership close after release");
    expect(!images.valid(*source), "released state generation becomes stale");

    const auto reused = initialize();
    expect(reused.has_value() && *reused != *source, "state descriptor reuse advances generation");
    expect(images.release(*reused), "reused state image releases");

    for (const bool cancel_last_reader : {false, true}) {
        const auto shared       = initialize();
        const auto first_reader = images.reserve_destination();
        const auto last_reader  = images.reserve_destination();
        expect(shared && first_reader && last_reader, "shared State reader fixture allocates");
        const auto source_slot = images.physical_slot(*shared);
        const std::array<std::uint16_t, 8> content{1, 3, 5, 7, 9, 11, 13, 15};
        const auto hidden = physical.continuation_hidden_slot(source_slot);
        CUDA_CHECK(cudaMemcpyAsync(hidden.data, content.data(), sizeof(content),
                                   cudaMemcpyHostToDevice, device.stream()));
        images.freeze(*shared);
        images.retain_checkpoint_reference(*shared); // Cache checkpoint.
        images.retain_checkpoint_reference(*shared); // Paused request snapshot.
        (void)images.begin_write(*shared, *first_reader);
        (void)images.begin_write(*shared, *last_reader);
        expect(images.release_checkpoint_owner(*shared) &&
                   images.release_checkpoint_owner(*shared) && images.valid(*shared) &&
                   images.checkpoint_references(*shared) == 0 && images.source_pins(*shared) == 2,
               "cache eviction and paused cancellation leave execution readers alive");
        expect(!images.can_release_after_checkpoint_references(*shared, 0) &&
                   images.device_occupied() == 3,
               "retired content owner does not advertise pinned State capacity as free");
        physical.storage().copy(source_slot, images.physical_slot(*first_reader),
                                device.execution_view());
        device.synchronize();
        images.complete_write(*shared, *first_reader);
        expect(images.valid(*shared) && images.source_pins(*shared) == 1,
               "one completed reader does not release the remaining reader's State");
        physical.storage().copy(source_slot, images.physical_slot(*last_reader),
                                device.execution_view());
        std::array<std::uint16_t, 8> observed{};
        const auto last_hidden =
            physical.continuation_hidden_slot(images.physical_slot(*last_reader));
        CUDA_CHECK(cudaMemcpyAsync(observed.data(), last_hidden.data, sizeof(observed),
                                   cudaMemcpyDeviceToHost, device.stream()));
        device.synchronize();
        expect(observed == content, "remaining State reader retains the exact source content");
        if (cancel_last_reader) {
            images.abort_write(*shared, *last_reader);
        } else {
            images.complete_write(*shared, *last_reader);
        }
        expect(!images.valid(*shared) && images.device_occupied() == 2,
               "last reader retirement releases the orphaned State automatically");
        const auto replacement = initialize();
        expect(replacement && *replacement != *shared &&
                   images.physical_slot(*replacement) == source_slot,
               "retired State storage is reusable under a new generation");
        expect(images.release(*replacement) && images.release(*first_reader) &&
                   images.release(*last_reader) && images.occupied() == 0,
               "retired checkpoint and execution readers leave no State allocation behind");
    }

    {
        const auto shared = initialize();
        const auto reader = images.reserve_destination();
        images.freeze(*shared);
        images.retain_checkpoint_reference(*shared);
        (void)images.begin_write(*shared, *reader);
        expect(images.release_checkpoint_owner(*shared), "read lease outlives original checkpoint");
        // Pausing an unwritten fork converts its read lease into a new snapshot owner.
        images.retain_checkpoint_reference(*shared);
        images.abort_write(*shared, *reader);
        expect(images.valid(*shared) && images.checkpoint_references(*shared) == 1 &&
                   images.source_pins(*shared) == 0,
               "new snapshot ownership is installed before its read lease retires");
        expect(images.release(*reader) && images.release_checkpoint_owner(*shared) &&
                   images.occupied() == 0,
               "reowned snapshot retires without leaking its previous read lease");
    }

    for (const bool cancel_transfer : {false, true}) {
        const auto shared = initialize();
        device.synchronize();
        images.freeze(*shared);
        images.retain_checkpoint_reference(*shared);
        auto transfer = images.begin_device_to_host(*shared, device.transfer_view());
        expect(transfer && images.release_checkpoint_owner(*shared) && images.valid(*shared),
               "submitted State transfer retains a retired checkpoint source");
        CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream()));
        if (cancel_transfer) {
            images.abort_transfer(std::move(*transfer));
        } else {
            images.publish_transfer(std::move(*transfer), true);
        }
        expect(!images.valid(*shared) && images.occupied() == 0 &&
                   host_backing.occupied_bytes() == 0,
               "last transfer lease retirement frees both State replicas and reservations");
    }

    {
        const auto first  = initialize();
        const auto second = initialize();
        images.freeze(*first);
        images.freeze(*second);
        const auto first_writer  = images.reserve_destination();
        const auto second_writer = images.reserve_destination();
        (void)images.begin_write(*first, *first_writer);
        (void)images.begin_write(*second, *second_writer);
        bool rejected = false;
        try {
            images.complete_write(*first, *second_writer);
        } catch (const std::logic_error&) { rejected = true; }
        expect(rejected && images.source_pins(*first) == 1 && images.source_pins(*second) == 1,
               "write completion must name its reserved source");
        images.abort_write(*first, *first_writer);
        images.abort_write(*second, *second_writer);
        expect(images.release(*first) && images.release(*second) && images.release(*first_writer) &&
                   images.release(*second_writer),
               "independent writers release their own pins");
    }

    const auto host_source = initialize();
    expect(host_source.has_value(), "Host state source allocation");
    device.synchronize();
    images.freeze(*host_source);
    {
        auto allocation     = host_backing.allocate(layout.images.image_stride);
        const auto occupied = host_backing.occupied_bytes();
        auto cancelled      = images.reserve_device_to_host(*host_source, std::move(*allocation));
        expect(host_backing.occupied_bytes() == occupied && host_backing.allocation_count() == 1,
               "transfer consumes the existing Host reservation without reallocating");
        expect(cancelled && host_backing.reserved_bytes() == layout.images.image_stride &&
                   host_backing.live_bytes() == 0 && images.source_pins(*host_source) == 1 &&
                   !images.release(*host_source),
               "State transfer reservation charges destination while its source is pinned");
    }
    expect(host_backing.occupied_bytes() == 0 && images.source_pins(*host_source) == 0 &&
               images.residency(*host_source) == store::StateReplicaResidency::DeviceOnly,
           "cancelled State transfer releases its actual Host extent and retains source state");
    auto d2h = images.begin_device_to_host(*host_source, device.transfer_view());
    expect(d2h.has_value() &&
               images.residency(*host_source) == store::StateReplicaResidency::DeviceOnly,
           "incomplete State D2H does not publish a Host replica");
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream()));
    images.publish_transfer(std::move(*d2h), true);
    expect(images.residency(*host_source) == store::StateReplicaResidency::Both &&
               host_backing.live_bytes() == layout.images.image_stride &&
               host_backing.reserved_bytes() == 0,
           "State D2H backup publishes stable Both residency");

    const auto moved_device = images.reserve_logical_destination();
    expect(moved_device.has_value(), "State replica split destination allocation");
    images.split_device_replica_identity(*host_source, *moved_device);
    expect(images.role(*moved_device) == store::StateImageRole::ReservedDestination,
           "replica handover does not publish a writer");
    images.complete_initialization(*moved_device);
    expect(images.residency(*host_source) == store::StateReplicaResidency::HostOnly &&
               images.residency(*moved_device) == store::StateReplicaResidency::DeviceOnly &&
               images.content_epoch(*moved_device) == images.content_epoch(*host_source),
           "State replica identity split keeps old Host content and moves Device ownership");

    const auto fork_one = images.reserve_logical_destination();
    const auto fork_two = images.reserve_logical_destination();
    expect(fork_one && fork_two, "concurrent Host State forks reserve logical destinations");
    auto h2d_one = images.begin_host_to_device(*host_source, *fork_one, device.transfer_view());
    auto h2d_two = images.begin_host_to_device(*host_source, *fork_two, device.transfer_view());
    expect(h2d_one && h2d_two && images.source_pins(*host_source) == 2,
           "immutable Host State source supports multiple fork pins");
    auto h2d_source = images.begin_host_to_device(*host_source, device.transfer_view());
    expect(h2d_source && images.source_pins(*host_source) == 3,
           "Host replica remains shared while restoring its own Device replica");
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream()));
    bool rejected = false;
    try {
        images.publish_transfer(std::move(*h2d_source), false);
    } catch (const std::logic_error&) { rejected = true; }
    expect(rejected && h2d_source->valid() && images.host_resident(*host_source) &&
               !images.device_resident(*host_source) && images.source_pins(*host_source) == 3,
           "restoration cannot retire Host storage retained by other transfers");
    images.publish_transfer(std::move(*h2d_one), true);
    images.publish_transfer(std::move(*h2d_two), true);
    images.publish_transfer(std::move(*h2d_source), false);
    expect(images.role(*fork_one) == store::StateImageRole::ReservedDestination,
           "Host materialization is separate from native adoption");
    images.complete_initialization(*fork_one);
    images.complete_initialization(*fork_two);
    expect(images.source_pins(*host_source) == 0 &&
               images.residency(*fork_one) == store::StateReplicaResidency::DeviceOnly &&
               images.residency(*fork_two) == store::StateReplicaResidency::DeviceOnly &&
               images.content_epoch(*fork_one) == images.content_epoch(*host_source) &&
               images.content_epoch(*fork_two) == images.content_epoch(*host_source),
           "Host State forks publish independent Device destinations");
    expect(images.release(*host_source) && images.release(*moved_device) &&
               images.release(*fork_one) && images.release(*fork_two) &&
               images.host_occupied() == 0,
           "State Host/Device replica ownership closes without leaked slots");
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || count == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    CUDA_CHECK(count_err);

    try {
        ninfer::DeviceContext device(0);
        test_state_store(device);
        device.synchronize();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
        return 1;
    }
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
