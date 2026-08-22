#include "core/device.h"
#include "targets/qwen3_6/impl/runtime/logical_kv_store.h"
#include "targets/qwen3_6/impl/runtime/state_image_store.h"

#include <ninfer/targets/qwen3_6/state_image.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <exception>
#include <iostream>
#include <string_view>

namespace {

namespace q36   = ninfer::targets::qwen3_6;
namespace store = ninfer::targets::qwen3_6::detail;

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
        .linear =
            {
                .layers         = 1,
                .conv_channels  = 8,
                .conv_width     = 3,
                .value_heads    = 2,
                .value_head_dim = 4,
                .key_head_dim   = 4,
                .slot_count     = 4,
                .conv_dtype     = ninfer::DType::BF16,
            },
        .hidden = 8,
        .dflash_local =
            q36::DFlashLocalStateSpec{.layers = 1, .capacity = 8, .kv_heads = 2, .head_dim = 4},
    };
    ninfer::LayoutBuilder builder;
    const q36::StateImageDeviceLayout layout = q36::plan_state_image_device_pool(builder, spec);
    ninfer::DeviceArena arena(builder.finish(256));
    q36::StateImageDevicePool physical({arena.base(), arena.capacity()}, layout);
    store::StateImageStore images(physical);

    const auto source = images.reserve_reset(device.stream);
    expect(source.has_value(), "state source allocation");
    const std::int32_t original_slot = images.physical_slot(*source);
    images.freeze(*source);
    images.move_checkpoint_to_active(*source);
    expect(images.physical_slot(*source) == original_slot &&
               images.role(*source) == store::StateImageRole::ActiveMutable,
           "private Move preserves the logical image and physical slot");
    images.freeze(*source);
    const auto destination = images.reserve_destination();
    expect(destination.has_value(), "state destination reservation");
    const store::StateImageSelectors selectors = images.begin_fork(*source, *destination);
    expect(selectors.source >= 0 && selectors.destination >= 0 &&
               selectors.source != selectors.destination,
           "fork resolves distinct physical selectors");
    expect(!images.release(*source) && !images.release(*destination),
           "fork pins both logical images");
    images.commit_fork(*source, *destination);
    expect(images.role(*source) == store::StateImageRole::CheckpointImmutable &&
               images.role(*destination) == store::StateImageRole::ActiveMutable,
           "fork commit preserves source and activates destination");
    expect(images.release(*source), "superseded rewrite image releases before rotation");
    images.freeze(*destination);
    const auto rotated = images.reserve_destination();
    expect(rotated.has_value(), "rewrite rotation reuses the second-image entitlement");
    const auto rotated_selectors = images.begin_fork(*destination, *rotated);
    expect(rotated_selectors.source != rotated_selectors.destination && images.occupied() == 2,
           "rewrite rotation remains within two StateImages");
    images.commit_fork(*destination, *rotated);
    expect(images.release(*destination) && images.release(*rotated) && images.occupied() == 0,
           "rotated state image ownership closes after release");
    expect(!images.valid(*source), "released state generation becomes stale");

    const auto reused = images.reserve_reset(device.stream);
    expect(reused.has_value() && *reused != *source, "state descriptor reuse advances generation");
    expect(images.release(*reused), "reused state image releases");
}

void test_kv_store(ninfer::DeviceContext& device) {
    ninfer::LayoutBuilder builder;
    ninfer::DeviceKVPagePoolSpec page_spec{
        .page_group_count = 8,
        .geometry =
            {
                .page_tokens        = static_cast<std::uint32_t>(ninfer::kPagedKVPageSize),
                .device_plane_order = ninfer::PagedKVPlaneOrder::PageMajor,
                .planes = {{.dtype = ninfer::DType::BF16, .leading_extent = 8, .head_extent = 2}},
            },
    };
    const ninfer::DeviceKVPagePoolLayout page_layout =
        ninfer::plan_device_kv_page_pool(builder, page_spec);
    const ninfer::KVExecutionTableLayout table_layout =
        ninfer::plan_kv_execution_tables(builder, {.logical_page_capacity = 4, .table_rows = 2});
    ninfer::DeviceArena arena(builder.finish(256));
    const ninfer::DeviceSpan backing{arena.base(), arena.capacity()};
    ninfer::DeviceKVPagePool physical_pages(backing, page_layout);
    ninfer::KVExecutionTablePool physical_tables(backing, table_layout, physical_pages);
    store::LogicalKVPageStore pages(physical_pages);
    store::KVAddressSpaceStore addresses(pages, physical_tables, 4, 4);

    const auto address = addresses.create_active(3, 0);
    expect(address.has_value(), "active KV address allocation");
    addresses.materialize_to_tokens(*address, 65, device.stream);
    expect(addresses.mapped_pages(*address) == 2 && addresses.committed_frontier(*address) == 0,
           "physical KV append does not publish canonical coverage before commit");
    addresses.commit_frontier(*address, 65);
    expect(addresses.mapped_pages(*address) == 2 && addresses.entitlement(*address) == 3 &&
               addresses.committed_frontier(*address) == 65 && addresses.bound_row(*address) == 0,
           "KV address tracks mapped pages, entitlement, frontier, and execution row");
    addresses.set_checkpoint_requirements(*address, 65, 32);
    expect(pages.occupied() == 2 && addresses.mapped_pages(*address) == 2,
           "endpoint and rewrite requirements share one ordered page mapping");
    const std::uint64_t old_epoch = addresses.content_epoch(*address, 0);

    addresses.deactivate(*address);
    expect(addresses.bound_row(*address) == -1 && addresses.entitlement(*address) == 2,
           "catalogued KV address owns no execution row or growth reservation");
    auto activation = addresses.prepare_activation(*address, 3, 1);
    expect(addresses.bound_row(*address) == -1 && addresses.entitlement(*address) == 2 &&
               physical_pages.reserved_pages() == 1,
           "prepared KV activation preserves the catalogued mapping until publication");
    addresses.commit_activation(std::move(activation), device.stream);
    addresses.destructive_truncate(*address, 32);
    expect(addresses.mapped_pages(*address) == 1 && addresses.entitlement(*address) == 3 &&
               addresses.committed_frontier(*address) == 32 &&
               addresses.content_epoch(*address, 0) != old_epoch,
           "destructive rewrite truncates coverage and advances content epoch");
    addresses.deactivate(*address);
    expect(addresses.release(*address), "catalogued KV address releases");
    expect(!addresses.valid(*address) && pages.occupied() == 0 &&
               physical_pages.allocated_pages() == 0 && physical_pages.reserved_pages() == 0,
           "KV release invalidates generations and closes physical ownership");
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
        test_kv_store(device);
        device.synchronize();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
        return 1;
    }
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
