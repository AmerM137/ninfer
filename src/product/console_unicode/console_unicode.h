#pragma once

#ifdef _WIN32
#include <string>
#include <vector>
#endif

namespace ninfer::product {

// UTF-8 console text for product entry points.
//
// On Windows the `char** argv` main() receives is in the process ANSI code
// page (non-ASCII arguments are lossy), and console I/O uses the console's
// legacy code page (UTF-8 output renders as mojibake). This scope rebuilds
// argv as UTF-8 from the wide command line and switches both console code
// pages to UTF-8, restoring them on destruction. Byte-identical file and
// pipe output is unaffected either way. On other platforms the scope is a
// passthrough: argc()/argv() return the constructor arguments unchanged.
//
// Known limitation: with a UTF-8 input code page, legacy conhost may still
// deliver `?` for non-ASCII characters typed interactively; arguments,
// files, and pipes are unaffected.
class ConsoleUtf8Scope {
public:
    ConsoleUtf8Scope(int argc, char** argv);
    ~ConsoleUtf8Scope();

    ConsoleUtf8Scope(const ConsoleUtf8Scope&)            = delete;
    ConsoleUtf8Scope& operator=(const ConsoleUtf8Scope&) = delete;

    int argc() const { return argc_; }
    char** argv() const { return argv_; }

private:
    int argc_;
    char** argv_;
#ifdef _WIN32
    std::vector<std::string> storage_;
    std::vector<char*> pointers_;
    unsigned int previous_input_cp_  = 0;
    unsigned int previous_output_cp_ = 0;
#endif
};

} // namespace ninfer::product
