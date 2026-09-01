#include "product/logging/logging.h"

#include <spdlog/logger.h>
#include <spdlog/spdlog.h>

#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

class StderrCapture {
public:
    StderrCapture() {
        if (::pipe(pipe_) != 0) { throw std::runtime_error(std::strerror(errno)); }
        saved_ = ::dup(STDERR_FILENO);
        if (saved_ < 0 || ::dup2(pipe_[1], STDERR_FILENO) < 0) {
            throw std::runtime_error(std::strerror(errno));
        }
        ::close(pipe_[1]);
        pipe_[1] = -1;
    }

    ~StderrCapture() {
        if (!finished_) {
            if (saved_ >= 0) {
                (void)::dup2(saved_, STDERR_FILENO);
                ::close(saved_);
            }
            if (pipe_[0] >= 0) { ::close(pipe_[0]); }
        }
    }

    StderrCapture(const StderrCapture&)            = delete;
    StderrCapture& operator=(const StderrCapture&) = delete;

    std::string finish() {
        if (finished_) { throw std::logic_error("stderr capture already finished"); }
        std::fflush(stderr);
        if (::dup2(saved_, STDERR_FILENO) < 0) { throw std::runtime_error(std::strerror(errno)); }
        ::close(saved_);
        saved_ = -1;

        std::string output;
        std::array<char, 4096> buffer{};
        for (;;) {
            const ssize_t count = ::read(pipe_[0], buffer.data(), buffer.size());
            if (count == 0) { break; }
            if (count < 0) {
                if (errno == EINTR) { continue; }
                throw std::runtime_error(std::strerror(errno));
            }
            output.append(buffer.data(), static_cast<std::size_t>(count));
        }
        ::close(pipe_[0]);
        pipe_[0]  = -1;
        finished_ = true;
        return output;
    }

private:
    int pipe_[2]{-1, -1};
    int saved_     = -1;
    bool finished_ = false;
};

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

std::size_t line_count(const std::string& value) {
    std::size_t count = 0;
    for (const char ch : value) {
        if (ch == '\n') { ++count; }
    }
    return count;
}

std::size_t occurrence_count(const std::string& value, const std::string& needle) {
    std::size_t count    = 0;
    std::size_t position = 0;
    while ((position = value.find(needle, position)) != std::string::npos) {
        ++count;
        position += needle.size();
    }
    return count;
}

} // namespace

int main() {
    constexpr int kThreads        = 4;
    constexpr int kLinesPerThread = 5;
    int failures                  = 0;
    bool empty_name_rejected      = false;
    try {
        ninfer::product::LoggingRuntime invalid({.logger_name = ""});
    } catch (const std::invalid_argument&) { empty_name_rejected = true; }
    failures += check(empty_name_rejected, "empty logger name was accepted");

    const std::string logger_name = "ninfer-product-logging-test";
    failures += check(spdlog::get(logger_name) == nullptr,
                      "test logger unexpectedly existed in the global registry");

    StderrCapture capture;
    {
        ninfer::product::LoggingRuntime runtime({
            .logger_name = logger_name,
            .level       = ninfer::product::LogLevel::Info,
            .color       = ninfer::product::LogColorMode::Never,
        });
        const std::shared_ptr<spdlog::logger> logger = runtime.logger();
        logger->debug("event=hidden");
        logger->info("event=visible value={}", 7);

        std::vector<std::thread> workers;
        workers.reserve(kThreads);
        for (int thread = 0; thread < kThreads; ++thread) {
            workers.emplace_back([logger, thread] {
                for (int item = 0; item < kLinesPerThread; ++item) {
                    logger->info("event=parallel thread={} item={}", thread, item);
                }
            });
        }
        for (std::thread& worker : workers) { worker.join(); }
        runtime.flush();
    }
    const std::string output = capture.finish();

    failures += check(output.find("event=hidden") == std::string::npos,
                      "logger did not filter debug records");
    const std::size_t first_line_end = output.find('\n');
    const std::string first_line     = output.substr(0, first_line_end);
    const std::regex operational_pattern(
        R"(^\[[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}\] \[info\] \[ninfer-product-logging-test\] event=visible value=7$)");
    failures += check(std::regex_match(first_line, operational_pattern),
                      "logger did not emit the fixed operational record format");
    failures += check(output.find("[info] [ninfer-product-logging-test] event=visible value=7") !=
                          std::string::npos,
                      "logger did not apply the operational pattern");
    failures += check(output.find("\x1b[") == std::string::npos,
                      "disabled color mode emitted ANSI escapes");
    failures += check(line_count(output) == 21, "concurrent logger did not emit complete records");
    for (int thread = 0; thread < kThreads; ++thread) {
        for (int item = 0; item < kLinesPerThread; ++item) {
            const std::string record =
                "event=parallel thread=" + std::to_string(thread) + " item=" + std::to_string(item);
            failures += check(occurrence_count(output, record) == 1,
                              "concurrent logger lost, duplicated, or interleaved a record");
        }
    }
    failures +=
        check(spdlog::get(logger_name) == nullptr, "LoggingRuntime registered a global logger");
    return failures == 0 ? 0 : 1;
}
