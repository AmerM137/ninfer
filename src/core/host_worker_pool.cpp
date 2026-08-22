#include "core/host_worker_pool.h"

#include <chrono>
#include <stdexcept>
#include <utility>

namespace ninfer {

HostWorkerPool::HostWorkerPool(std::uint32_t threads, std::size_t queue_capacity)
    : thread_count(threads), queue_capacity(queue_capacity) {
    if (thread_count == 0) { throw std::invalid_argument("host worker count must be nonzero"); }
    if (queue_capacity == 0) {
        throw std::invalid_argument("host worker queue capacity must be nonzero");
    }
    workers.reserve(thread_count);
    try {
        for (std::uint32_t index = 0; index < thread_count; ++index) {
            workers.emplace_back([this] { worker_loop(); });
        }
    } catch (...) {
        {
            std::lock_guard lock(mutex);
            stopping = true;
        }
        work_available.notify_all();
        for (std::thread& worker : workers) {
            if (worker.joinable()) { worker.join(); }
        }
        throw;
    }
}

HostWorkerPool::~HostWorkerPool() {
    {
        std::lock_guard lock(mutex);
        stopping = true;
    }
    work_available.notify_all();
    queue_space.notify_all();
    for (std::thread& worker : workers) {
        if (worker.joinable()) { worker.join(); }
    }
}

void HostWorkerPool::worker_loop() noexcept {
    for (;;) {
        std::function<void()> task;
        {
            std::unique_lock lock(mutex);
            work_available.wait(lock, [this] { return stopping || !queue.empty(); });
            if (stopping && queue.empty()) { return; }
            task = std::move(queue.front());
            queue.pop_front();
            ++active;
        }
        queue_space.notify_one();
        try {
            task();
        } catch (...) {
            // submit() wraps work in packaged_task, so an escaping exception means the
            // executor contract itself was violated.
            std::terminate();
        }
        {
            std::lock_guard lock(mutex);
            if (active == 0) { std::terminate(); }
            --active;
        }
    }
}

void HostWorkerPool::enqueue(std::function<void()> task, Checkpoint checkpoint) {
    if (!task) { throw std::invalid_argument("host worker task must not be empty"); }
    for (;;) {
        if (checkpoint) { checkpoint(); }
        std::unique_lock lock(mutex);
        if (stopping) { throw std::runtime_error("host worker pool is stopping"); }
        if (queue.size() < queue_capacity) {
            queue.push_back(std::move(task));
            lock.unlock();
            work_available.notify_one();
            return;
        }
        queue_space.wait_for(lock, std::chrono::milliseconds(10));
    }
}

HostWorkerPool::Snapshot HostWorkerPool::snapshot() const {
    std::lock_guard lock(mutex);
    return Snapshot{.threads = thread_count, .queued = queue.size(), .active = active};
}

} // namespace ninfer
