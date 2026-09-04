#include "product/logging/logging.h"
#include "product/logging/startup_log.h"

#include <spdlog/logger.h>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#else
#include <unistd.h>
#endif

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>

namespace {

class StderrCapture {
public:
    StderrCapture() {
        if (make_pipe(pipe_) != 0) { throw std::runtime_error(std::strerror(errno)); }
        saved_ = duplicate(stderr_fd());
        if (saved_ < 0 || duplicate_to(pipe_[1], stderr_fd()) < 0) {
            throw std::runtime_error(std::strerror(errno));
        }
        close_fd(pipe_[1]);
        pipe_[1] = -1;
    }

    ~StderrCapture() {
        if (saved_ >= 0) {
            (void)duplicate_to(saved_, stderr_fd());
            close_fd(saved_);
        }
        if (pipe_[0] >= 0) { close_fd(pipe_[0]); }
    }

    std::string finish() {
        std::fflush(stderr);
        if (duplicate_to(saved_, stderr_fd()) < 0) {
            throw std::runtime_error(std::strerror(errno));
        }
        close_fd(saved_);
        saved_ = -1;

        std::string output;
        std::array<char, 4096> buffer{};
        for (;;) {
            const auto count = read_fd(pipe_[0], buffer.data(), buffer.size());
            if (count == 0) { break; }
            if (count < 0) {
                if (errno == EINTR) { continue; }
                throw std::runtime_error(std::strerror(errno));
            }
            output.append(buffer.data(), static_cast<std::size_t>(count));
        }
        close_fd(pipe_[0]);
        pipe_[0] = -1;
        return output;
    }

private:
#ifdef _WIN32
    static int make_pipe(int (&pipe)[2]) { return ::_pipe(pipe, 4096, _O_BINARY); }
    static int stderr_fd() { return ::_fileno(stderr); }
    static int duplicate(int fd) { return ::_dup(fd); }
    static int duplicate_to(int source, int destination) { return ::_dup2(source, destination); }
    static int close_fd(int fd) { return ::_close(fd); }
    static int read_fd(int fd, char* data, std::size_t size) {
        return ::_read(fd, data, static_cast<unsigned int>(size));
    }
#else
    static int make_pipe(int (&pipe)[2]) { return ::pipe(pipe); }
    static int stderr_fd() { return STDERR_FILENO; }
    static int duplicate(int fd) { return ::dup(fd); }
    static int duplicate_to(int source, int destination) { return ::dup2(source, destination); }
    static int close_fd(int fd) { return ::close(fd); }
    static ssize_t read_fd(int fd, char* data, std::size_t size) { return ::read(fd, data, size); }
#endif

    int pipe_[2]{-1, -1};
    int saved_ = -1;
};

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

std::size_t line_count(std::string_view value) {
    return static_cast<std::size_t>(std::count(value.begin(), value.end(), '\n'));
}

} // namespace

int main() {
    int failures = 0;
    std::string service_output;
    {
        StderrCapture capture;
        {
            ninfer::product::LoggingRuntime logging(
                {.logger_name  = "ninfer-serve",
                 .color        = ninfer::product::LogColorMode::Auto,
                 .presentation = ninfer::product::LogPresentation::Service});
            logging.logger()->info("throughput | sample");
            logging.flush();
        }
        service_output = capture.finish();
    }
    failures += check(
        std::regex_match(
            service_output,
            std::regex(
                R"(^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}  INFO  throughput \| sample\n$)")),
        "service pretty prefix mismatch");
    failures += check(service_output.find("ninfer-serve") == std::string::npos,
                      "service pretty output repeated the executable name");
    failures += check(service_output.find("\x1b[") == std::string::npos,
                      "redirected service output contains ANSI escapes");

    std::string startup_output;
    {
        StderrCapture capture;
        {
            ninfer::product::LoggingRuntime logging(
                {.logger_name  = "ninfer-serve",
                 .color        = ninfer::product::LogColorMode::Never,
                 .presentation = ninfer::product::LogPresentation::Service});
            ninfer::product::StartupLogRenderer startup(logging);
            ninfer::StartupObserver observer = startup.observer();
            observer.callback({.phase  = ninfer::StartupPhase::EngineStartup,
                               .status = ninfer::StartupStatus::Begin});
            observer.callback({.phase  = ninfer::StartupPhase::CudaInitialize,
                               .status = ninfer::StartupStatus::Begin});
            observer.callback({.phase      = ninfer::StartupPhase::CudaInitialize,
                               .status     = ninfer::StartupStatus::Complete,
                               .elapsed_ns = 1'000'000'000});
            observer.callback({.phase         = ninfer::StartupPhase::WeightsMaterialize,
                               .status        = ninfer::StartupStatus::Begin,
                               .progress_unit = ninfer::StartupProgressUnit::Bytes,
                               .total         = 16ULL << 30});
            observer.callback({.phase         = ninfer::StartupPhase::WeightsMaterialize,
                               .status        = ninfer::StartupStatus::Complete,
                               .progress_unit = ninfer::StartupProgressUnit::Bytes,
                               .current       = 16ULL << 30,
                               .total         = 16ULL << 30,
                               .elapsed_ns    = 2'000'000'000});
            observer.callback({.phase      = ninfer::StartupPhase::EngineStartup,
                               .status     = ninfer::StartupStatus::Complete,
                               .elapsed_ns = 3'000'000'000});
            startup.engine_ready({.model_id             = "qwen3.6-27b",
                                  .weights_id           = "groupwise-int",
                                  .host_to_device_bytes = 16ULL << 30});
            logging.flush();
        }
        startup_output = capture.finish();
    }
    failures +=
        check(line_count(startup_output) == 4 &&
                  startup_output.find("starting engine") != std::string::npos &&
                  startup_output.find("loading weights | 16.0 GiB") != std::string::npos &&
                  startup_output.find("weights ready | 16.0 GiB | 2.0s | 8.00 GiB/s") !=
                      std::string::npos &&
                  startup_output.find("engine ready | qwen3.6-27b/groupwise-int | total 3.0s") !=
                      std::string::npos &&
                  startup_output.find("CUDA initialized") == std::string::npos,
              "normal startup pretty output is noisy or incomplete");

    std::string tool_output;
    {
        StderrCapture capture;
        {
            ninfer::product::LoggingRuntime logging(
                {.logger_name  = "ninfer",
                 .color        = ninfer::product::LogColorMode::Never,
                 .presentation = ninfer::product::LogPresentation::Tool});
            logging.logger()->info("engine ready");
            logging.logger()->error("failed");
            logging.flush();
        }
        tool_output = capture.finish();
    }
    failures +=
        check(tool_output == "engine ready\nerror: failed\n", "tool pretty prefix mismatch");
    return failures == 0 ? 0 : 1;
}
