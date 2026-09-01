#pragma once

#include <memory>
#include <string>

namespace spdlog {
class logger;
}

namespace ninfer::product {

enum class LogLevel {
    Trace,
    Debug,
    Info,
    Warning,
    Error,
    Critical,
    Off,
};

enum class LogColorMode {
    Auto,
    Always,
    Never,
};

struct LoggingOptions {
    std::string logger_name;
    LogLevel level     = LogLevel::Info;
    LogColorMode color = LogColorMode::Auto;
};

// Application-owned operational logger lifetime. Construction does not mutate spdlog's global
// default logger or registry; producers receive and retain the returned explicit shared handle.
class LoggingRuntime {
public:
    explicit LoggingRuntime(LoggingOptions options);
    ~LoggingRuntime();

    LoggingRuntime(const LoggingRuntime&)            = delete;
    LoggingRuntime& operator=(const LoggingRuntime&) = delete;
    LoggingRuntime(LoggingRuntime&&)                 = delete;
    LoggingRuntime& operator=(LoggingRuntime&&)      = delete;

    [[nodiscard]] std::shared_ptr<spdlog::logger> logger() const noexcept;
    void flush() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::product
