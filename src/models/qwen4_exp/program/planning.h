#pragma once

#include "models/qwen4_exp/execution/workspace.h"
#include "models/qwen4_exp/state/state_image.h"
#include "models/qwen4_exp/state/kv_cache.h"
#include "models/qwen4_exp/program/buffers.h"
#include "models/qwen4_exp/ple/disk/prepare.h"
#include "models/qwen4_exp/ple/upload.h"
#include "runtime/engine/kv_capacity.h"

namespace ninfer::models::qwen4_exp {
struct PersistentLayout {
    StateLayout state;
    QsaCacheLayout main;
    std::optional<QsaCacheLayout> mtp;
    BufferLayout buffers;
    std::size_t bytes;
};

struct ExecutionShape {
    std::uint32_t max_context, max_concurrency, prefill_chunk, draft_tokens;
    KvCacheStorage kv_storage;
    EnginePurpose purpose;
};

struct ExecutionPlan {
    std::shared_ptr<const execution::ModelProfile> profile;
    ExecutionShape shape;
    PersistentLayout persistent;
    execution::WorkspacePlan workspace;
    std::shared_ptr<const moe::Plan> moe;
    ple::disk::PrepareOptions ple;
    ple::UploadMemory ple_upload{};
    std::size_t pinned_output_bytes = 0;
    std::size_t ple_prepare_bytes   = 0;
    std::size_t host_context_bytes  = 0;
    std::size_t pinned_io_bytes     = 0;
    runtime::KvCapacityResolution capacity;

    [[nodiscard]] std::size_t device_bytes() const;
};

// Available bytes exclude immutable GPU weights and the caller's startup headroom.
// Ordinary Host, registered expert backing, and pinned context capacity are independent budgets.
[[nodiscard]] std::optional<ExecutionPlan>
plan_execution(std::shared_ptr<const execution::ModelProfile>, const ple::disk::Table&,
               const EngineOptions&, DeviceContext&, std::size_t available_bytes);
} // namespace ninfer::models::qwen4_exp
