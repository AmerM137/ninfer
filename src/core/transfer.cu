#include "core/transfer.h"
#include "core/device.h"

#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

cudaMemcpyKind validate_copy(MemorySpan destination, ConstMemorySpan source,
                             CudaExecutionView execution, bool asynchronous) {
    validate_memory(destination);
    validate_memory(source);
    if (destination.bytes != source.bytes) {
        throw std::invalid_argument("copy spans must have equal byte lengths");
    }
    for (const auto memory : {static_cast<ConstMemorySpan>(destination), source}) {
        if (memory.device != Device::cpu() && memory.device != execution.device()) {
            throw std::invalid_argument("copy memory does not belong to the execution device");
        }
        if (asynchronous && memory.device == Device::cpu() && !memory.host_pinned) {
            throw std::invalid_argument("asynchronous copy requires pinned host memory");
        }
    }
    if (source.device == Device::cpu() && destination.device == Device::cpu()) {
        throw std::invalid_argument("use CPU copy for host-to-host transfer");
    }
    if (source.device == destination.device) {
        const auto src = reinterpret_cast<std::uintptr_t>(source.data);
        const auto dst = reinterpret_cast<std::uintptr_t>(destination.data);
        if (src != dst && (src < dst ? dst - src : src - dst) < source.bytes) {
            throw std::invalid_argument("CUDA copy spans must not partially overlap");
        }
        return cudaMemcpyDeviceToDevice;
    }
    return source.device == Device::cpu() ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
}

void enqueue(MemorySpan destination, ConstMemorySpan source, cudaMemcpyKind kind,
             CudaExecutionView execution) {
    if (source.bytes == 0 || source.data == destination.data) { return; }
    const auto error =
        cudaMemcpyAsync(destination.data, source.data, source.bytes, kind, execution.stream());
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA copy: ") + cudaGetErrorString(error));
    }
}

} // namespace

void copy_bytes_async(MemorySpan destination, ConstMemorySpan source, CudaExecutionView execution) {
    const auto kind = validate_copy(destination, source, execution, true);
    CudaDeviceScope scope(execution.device());
    enqueue(destination, source, kind, execution);
}

void copy_bytes(MemorySpan destination, ConstMemorySpan source, CudaExecutionView execution) {
    const auto kind = validate_copy(destination, source, execution, false);
    CudaDeviceScope scope(execution.device());
    enqueue(destination, source, kind, execution);
    const auto error = cudaStreamSynchronize(execution.stream());
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA copy completion: ") + cudaGetErrorString(error));
    }
}

} // namespace ninfer
