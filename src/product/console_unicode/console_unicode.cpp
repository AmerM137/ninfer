#include "product/console_unicode/console_unicode.h"

#ifdef _WIN32

#include <windows.h>

#include <shellapi.h>

#include <cstddef>

namespace ninfer::product {

namespace {

std::string narrow_utf8(const wchar_t* wide) {
    const int size =
        ::WideCharToMultiByte(CP_UTF8, 0, wide, /*cchWideChar=*/-1, nullptr, 0, nullptr, nullptr);
    if (size <= 1) { return {}; }
    std::string out(static_cast<std::size_t>(size - 1), '\0');
    ::WideCharToMultiByte(CP_UTF8, 0, wide, -1, out.data(), size, nullptr, nullptr);
    return out;
}

} // namespace

ConsoleUtf8Scope::ConsoleUtf8Scope(int argc, char** argv) : argc_(argc), argv_(argv) {
    // GetConsoleCP() returns 0 when no console is attached (piped/service
    // use); the Set calls then fail harmlessly and restore is skipped.
    previous_input_cp_  = ::GetConsoleCP();
    previous_output_cp_ = ::GetConsoleOutputCP();
    ::SetConsoleCP(CP_UTF8);
    ::SetConsoleOutputCP(CP_UTF8);

    int wide_count      = 0;
    wchar_t** wide_args = ::CommandLineToArgvW(::GetCommandLineW(), &wide_count);
    if (wide_args == nullptr || wide_count <= 0) { return; } // keep the ANSI argv
    storage_.reserve(static_cast<std::size_t>(wide_count));
    for (int i = 0; i < wide_count; ++i) { storage_.push_back(narrow_utf8(wide_args[i])); }
    ::LocalFree(wide_args);
    pointers_.reserve(storage_.size() + 1);
    for (std::string& argument : storage_) { pointers_.push_back(argument.data()); }
    pointers_.push_back(nullptr);
    argc_ = wide_count;
    argv_ = pointers_.data();
}

ConsoleUtf8Scope::~ConsoleUtf8Scope() {
    if (previous_input_cp_ != 0) { ::SetConsoleCP(previous_input_cp_); }
    if (previous_output_cp_ != 0) { ::SetConsoleOutputCP(previous_output_cp_); }
}

} // namespace ninfer::product

#else

namespace ninfer::product {

ConsoleUtf8Scope::ConsoleUtf8Scope(int argc, char** argv) : argc_(argc), argv_(argv) {}

ConsoleUtf8Scope::~ConsoleUtf8Scope() = default;

} // namespace ninfer::product

#endif
