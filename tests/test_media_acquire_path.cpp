#include "product/media_acquire/acquire.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer::product::media_acquire;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

void write_file(const std::filesystem::path& path, const std::string& payload) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << payload;
    if (!output) { throw std::runtime_error("failed to write fixture: " + path.string()); }
}

std::vector<std::uint8_t> read_under_root(const std::filesystem::path& file,
                                          const std::filesystem::path& root) {
    Policy policy;
    policy.media_root = root;
    Source source;
    source.kind  = SourceKind::Path;
    source.value = file.string();
    return acquire_bytes(source, policy);
}

bool rejected_outside_root(const std::filesystem::path& file, const std::filesystem::path& root) {
    try {
        (void)read_under_root(file, root);
        return false;
    } catch (const std::invalid_argument& error) {
        return std::string(error.what()).find("outside configured media root") !=
               std::string::npos;
    }
}

bool payload_is(const std::vector<std::uint8_t>& bytes, const std::string& payload) {
    return std::string(bytes.begin(), bytes.end()) == payload;
}

} // namespace

int main() try {
    const std::filesystem::path base =
        std::filesystem::temp_directory_path() / "ninfer-media-acquire-path-test";
    std::filesystem::remove_all(base);
    const std::filesystem::path root = base / "media-root";
    std::filesystem::create_directories(root / "sub");

    // The media-root containment check must compare path elements, not a
    // native-string ".." prefix: a real entry named "..hidden" is inside the
    // root, while anything the relative path reaches through a leading ".."
    // element is outside it.
    write_file(root / "inside.bin", "inside-payload");
    write_file(root / "sub" / "nested.bin", "nested-payload");
    write_file(root / "..hidden", "dot-dot-name-payload");
    write_file(base / "outside.bin", "outside-payload");

    int failures = 0;
    failures += check(payload_is(read_under_root(root / "inside.bin", root), "inside-payload"),
                      "file directly under the media root was not readable");
    failures += check(
        payload_is(read_under_root(root / "sub" / "nested.bin", root), "nested-payload"),
        "file nested under the media root was not readable");
    failures += check(
        payload_is(read_under_root(root / "..hidden", root), "dot-dot-name-payload"),
        "entry named '..hidden' inside the media root was falsely rejected");
    failures += check(rejected_outside_root(base / "outside.bin", root),
                      "file outside the media root was not rejected");
    failures += check(
        rejected_outside_root(root / "sub" / ".." / ".." / "outside.bin", root),
        "dot-dot escape through the media root was not rejected");

    std::filesystem::remove_all(base);
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr << "uncaught exception: " << error.what() << std::endl;
    return 1;
}
