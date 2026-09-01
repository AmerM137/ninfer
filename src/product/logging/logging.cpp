#include "product/logging/logging.h"

#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <atomic>
#include <cstdio>
#include <stdexcept>
#include <utility>

namespace ninfer::product {
namespace {

constexpr const char* kOperationalPattern = "[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%n] %v";

spdlog::level::level_enum to_spdlog_level(LogLevel level) {
    switch (level) {
    case LogLevel::Trace:
        return spdlog::level::trace;
    case LogLevel::Debug:
        return spdlog::level::debug;
    case LogLevel::Info:
        return spdlog::level::info;
    case LogLevel::Warning:
        return spdlog::level::warn;
    case LogLevel::Error:
        return spdlog::level::err;
    case LogLevel::Critical:
        return spdlog::level::critical;
    case LogLevel::Off:
        return spdlog::level::off;
    }
    throw std::invalid_argument("LoggingOptions level is invalid");
}

spdlog::color_mode to_spdlog_color_mode(LogColorMode mode) {
    switch (mode) {
    case LogColorMode::Auto:
        return spdlog::color_mode::automatic;
    case LogColorMode::Always:
        return spdlog::color_mode::always;
    case LogColorMode::Never:
        return spdlog::color_mode::never;
    }
    throw std::invalid_argument("LoggingOptions color mode is invalid");
}

void report_logging_error(const std::string& message) noexcept {
    static std::atomic_flag reported = ATOMIC_FLAG_INIT;
    if (reported.test_and_set(std::memory_order_relaxed)) { return; }
    std::fprintf(stderr, "ninfer logging failure: %s\n", message.c_str());
    std::fflush(stderr);
}

} // namespace

struct LoggingRuntime::Impl {
    explicit Impl(LoggingOptions options) {
        if (options.logger_name.empty()) {
            throw std::invalid_argument("LoggingOptions logger_name must not be empty");
        }
        auto sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>(
            to_spdlog_color_mode(options.color));
        logger = std::make_shared<spdlog::logger>(std::move(options.logger_name), std::move(sink));
        logger->set_pattern(kOperationalPattern);
        logger->set_level(to_spdlog_level(options.level));
        logger->flush_on(spdlog::level::warn);
        logger->set_error_handler(report_logging_error);
    }

    std::shared_ptr<spdlog::logger> logger;
};

LoggingRuntime::LoggingRuntime(LoggingOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}

LoggingRuntime::~LoggingRuntime() { flush(); }

std::shared_ptr<spdlog::logger> LoggingRuntime::logger() const noexcept {
    return impl_ == nullptr ? nullptr : impl_->logger;
}

void LoggingRuntime::flush() noexcept {
    if (impl_ == nullptr || impl_->logger == nullptr) { return; }
    try {
        impl_->logger->flush();
    } catch (const std::exception& exception) {
        report_logging_error(exception.what());
    } catch (...) { report_logging_error("unknown flush error"); }
}

} // namespace ninfer::product
