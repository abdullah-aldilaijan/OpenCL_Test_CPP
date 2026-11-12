// Shared helpers: text conversion, paths, files, CRC-32, SHA-256, output.
#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

struct Failure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// All program output goes through say(); the CLI prints it, the GUI appends it to the log.
using Sink = std::function<void(const std::string&)>;
void set_sink(Sink sink);
void say(const std::string& line);

std::string utf8(const std::wstring& text);
std::wstring wide(const std::string& text);
std::string commas(uint64_t value);
std::string percent(double fraction);
std::string win_error(DWORD code);
double seconds_now();

// Paths are absolute, backslash-separated wide strings. long_path() adds the \\?\ prefix
// so Win32 calls work beyond MAX_PATH.
std::wstring absolute_path(const std::wstring& path);
std::wstring long_path(const std::wstring& path);
std::wstring parent_of(const std::wstring& path);
std::wstring name_of(const std::wstring& path);
std::wstring join(const std::wstring& a, const std::wstring& b);
std::wstring fold_case(const std::wstring& text);
DWORD attributes(const std::wstring& path);  // INVALID_FILE_ATTRIBUTES when missing
bool exists(const std::wstring& path);
bool is_within(const std::wstring& child, const std::wstring& parent);
void make_directory(const std::wstring& path);  // ok if it already exists as a directory
void move_new(const std::wstring& from, const std::wstring& to);  // refuses to replace

class File {
public:
    static File open_read(const std::wstring& path);
    static File create_new(const std::wstring& path);  // fails if it exists
    File() = default;
    File(File&& other) noexcept;
    File& operator=(File&& other) noexcept;
    File(const File&) = delete;
    File& operator=(const File&) = delete;
    ~File();

    size_t read(void* data, size_t size);  // returns fewer bytes only at end of file
    void read_exact(void* data, size_t size);
    void write(const void* data, size_t size);
    void flush();
    void seek(uint64_t position);
    uint64_t tell();
    uint64_t size();
    void close();

private:
    HANDLE handle_ = INVALID_HANDLE_VALUE;
    std::wstring path_;
    std::vector<uint8_t> pending_;
};

// Uniquely named directory that is deleted (with contents) when this object is destroyed.
class TempDir {
public:
    TempDir(const std::wstring& parent, const wchar_t* prefix);
    ~TempDir();
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;
    const std::wstring& path() const { return path_; }

private:
    std::wstring path_;
};

uint32_t crc32(uint32_t crc, const void* data, size_t size);

class Sha256 {
public:
    Sha256();
    ~Sha256();
    Sha256(const Sha256&) = delete;
    Sha256& operator=(const Sha256&) = delete;
    void update(const void* data, size_t size);
    std::array<uint8_t, 32> digest();

private:
    void* algorithm_ = nullptr;
    void* hash_ = nullptr;
};

inline void put16(std::vector<uint8_t>& b, uint32_t v) {
    b.push_back(uint8_t(v));
    b.push_back(uint8_t(v >> 8));
}
inline void put32(std::vector<uint8_t>& b, uint32_t v) {
    for (int i = 0; i < 4; i++) b.push_back(uint8_t(v >> (8 * i)));
}
inline void put64(std::vector<uint8_t>& b, uint64_t v) {
    for (int i = 0; i < 8; i++) b.push_back(uint8_t(v >> (8 * i)));
}
inline uint16_t get16(const uint8_t* p) { return uint16_t(p[0] | (p[1] << 8)); }
inline uint32_t get32(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
inline uint64_t get64(const uint8_t* p) { return uint64_t(get32(p)) | (uint64_t(get32(p + 4)) << 32); }

// ---- Operations (one per original Python tool) ----

// hardware.py: labels shown in the device dropdown, mapped to the value passed as --device.
struct DeviceOption {
    std::string label;
    std::string value;  // "CPU" or the OpenCL GPU name
};
std::vector<DeviceOption> discover_devices(std::vector<std::string>& notices);

// gpu_zip.py: standard ZIP/ZIP64 using fixed-Huffman DEFLATE. device "CPU" runs the same
// encoder on CPU threads; anything else selects an OpenCL GPU by name substring.
void create_zip(const std::wstring& source, const std::wstring& target, const std::string& device);

// folder_archive.py restore(): safe ZIP extraction (stored + DEFLATE) into a new folder.
void restore_zip(const std::wstring& source, const std::wstring& target);

// opencl_lz.py / opencl_compress.py / gpu_folder.py decompression of legacy .oclr archives.
void restore_legacy(const std::wstring& source, const std::wstring& target, const std::string& device);

// Picks the right restore path from the archive's first bytes and name.
void restore_any(const std::wstring& source, const std::wstring& target, const std::string& device);
