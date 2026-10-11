#pragma once

#include "core/memory.h"

namespace ninfer {

class CudaExecutionView;

// Exact byte copies between equally sized spans. CPU copies allow overlapping ranges.
// CUDA copies reject partial overlap. The synchronous overload waits for its execution stream.
void copy_bytes(MemorySpan destination, ConstMemorySpan source);
void copy_bytes(MemorySpan destination, ConstMemorySpan source, CudaExecutionView execution);

// Host endpoints must be pinned; CUDA endpoints must belong to execution.device().
// The caller keeps both spans alive and orders all consumers through the stream or an event.
void copy_bytes_async(MemorySpan destination, ConstMemorySpan source, CudaExecutionView execution);

} // namespace ninfer
