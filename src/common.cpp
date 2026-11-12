#include "common.h"

#include <bcrypt.h>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <mutex>

namespace {
Sink g_sink;
std::mutex g_sink_lock;
}  // namespace

void set_sink(Sink sink) {
    std::lock_guard<std::mutex> lock(g_sink_lock);
    g_sink = std::move(sink);
}

void say(const std::string& line) {
    std::lock_guard<std::mutex> lock(g_sink_lock);
    if (g_sink) {
        g_sink(line);
    } else {
        std::fwrite(line.data(), 1, line.size(), stdout);
        std::fputc('\n', stdout);
        std::fflush(stdout);
    }
}

std::string utf8(const std::wstring& text) {
    if (text.empty()) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, text.data(), int(text.size()), nullptr, 0, nullptr, nullptr);
    std::string out(size_t(n), '\0');
    WideCharToMultiByte(CP_UTF8, 0, text.data(), int(text.size()), out.data(), n, nullptr, nullptr);
    return out;
}

std::wstring wide(const std::string& text) {
    if (text.empty()) return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, text.data(), int(text.size()), nullptr, 0);
    std::wstring out(size_t(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.data(), int(text.size()), out.data(), n);
    return out;
}

std::string commas(uint64_t value) {
    std::string digits = std::to_string(value), out;
    for (size_t i = 0; i < digits.size(); i++) {
        if (i && (digits.size() - i) % 3 == 0) out.push_back(',');
        out.push_back(digits[i]);
    }
    return out;
}

std::string percent(double fraction) {
    char buffer[64];
    std::snprintf(buffer, sizeof buffer, "%.2f%%", fraction * 100.0);
    return buffer;
}

std::string win_error(DWORD code) {
    wchar_t* text = nullptr;
    FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                   nullptr, code, 0, reinterpret_cast<wchar_t*>(&text), 0, nullptr);
    std::wstring message = text ? text : L"Unknown error";
    if (text) LocalFree(text);
    while (!message.empty() && (message.back() == L'\n' || message.back() == L'\r' || message.back() == L' '))
        message.pop_back();
    return utf8(message);
}

double seconds_now() {
    static LARGE_INTEGER frequency = [] {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        return f;
    }();
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return double(t.QuadPart) / double(frequency.QuadPart);
}

std::wstring absolute_path(const std::wstring& path) {
    if (path.empty()) throw Failure("Empty path");
    DWORD n = GetFullPathNameW(path.c_str(), 0, nullptr, nullptr);
    if (!n) throw Failure("Invalid path: " + utf8(path));
    std::wstring out(n, L'\0');
    n = GetFullPathNameW(path.c_str(), n, out.data(), nullptr);
    out.resize(n);
    // Drop a trailing separator except for drive roots such as C:\ .
    while (out.size() > 3 && (out.back() == L'\\' || out.back() == L'/')) out.pop_back();
    if (out.rfind(L"\\\\?\\", 0) == 0) {
        out = out.substr(4);
        if (out.rfind(L"UNC\\", 0) == 0) out = L"\\" + out.substr(3);
    }
    return out;
}

std::wstring long_path(const std::wstring& path) {
    if (path.rfind(L"\\\\?\\", 0) == 0) return path;
    if (path.rfind(L"\\\\", 0) == 0) return L"\\\\?\\UNC\\" + path.substr(2);
    return L"\\\\?\\" + path;
}

std::wstring parent_of(const std::wstring& path) {
    size_t slash = path.find_last_of(L"\\/");
    if (slash == std::wstring::npos) return L".";
    if (slash == 2 && path.size() > 1 && path[1] == L':') return path.substr(0, 3);
    return path.substr(0, slash);
}

std::wstring name_of(const std::wstring& path) {
    size_t slash = path.find_last_of(L"\\/");
    return slash == std::wstring::npos ? path : path.substr(slash + 1);
}

std::wstring join(const std::wstring& a, const std::wstring& b) {
    if (a.empty()) return b;
    if (a.back() == L'\\' || a.back() == L'/') return a + b;
    return a + L"\\" + b;
}

std::wstring fold_case(const std::wstring& text) {
    std::wstring out = text;
    if (!out.empty()) CharLowerBuffW(out.data(), DWORD(out.size()));
    return out;
}

DWORD attributes(const std::wstring& path) { return GetFileAttributesW(long_path(path).c_str()); }

bool exists(const std::wstring& path) { return attributes(path) != INVALID_FILE_ATTRIBUTES; }

static std::wstring resolved(const std::wstring& path) {
    std::error_code ec;
    auto p = std::filesystem::weakly_canonical(std::filesystem::path(long_path(path)), ec);
    if (ec) return fold_case(path);
    return fold_case(absolute_path(p.wstring()));
}

bool is_within(const std::wstring& child, const std::wstring& parent) {
    std::wstring c = resolved(child), p = resolved(parent);
    if (c == p) return true;
    if (!p.empty() && p.back() != L'\\') p.push_back(L'\\');
    return c.rfind(p, 0) == 0;
}

void make_directory(const std::wstring& path) {
    if (CreateDirectoryW(long_path(path).c_str(), nullptr)) return;
    DWORD error = GetLastError();
    DWORD attr = attributes(path);
    if (error == ERROR_ALREADY_EXISTS && attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY)) return;
    throw Failure("Cannot create folder " + utf8(path) + ": " + win_error(error));
}

void move_new(const std::wstring& from, const std::wstring& to) {
    if (!MoveFileExW(long_path(from).c_str(), long_path(to).c_str(), 0))
        throw Failure("Cannot move output to " + utf8(to) + ": " + win_error(GetLastError()));
}

// ---------------- File ----------------

static constexpr size_t kWriteBuffer = 1 << 20;

File File::open_read(const std::wstring& path) {
    File f;
    f.path_ = path;
    f.handle_ = CreateFileW(long_path(path).c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                            FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (f.handle_ == INVALID_HANDLE_VALUE)
        throw Failure("Cannot open " + utf8(path) + ": " + win_error(GetLastError()));
    return f;
}

File File::create_new(const std::wstring& path) {
    File f;
    f.path_ = path;
    f.handle_ = CreateFileW(long_path(path).c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                            FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f.handle_ == INVALID_HANDLE_VALUE)
        throw Failure("Cannot create " + utf8(path) + ": " + win_error(GetLastError()));
    f.pending_.reserve(kWriteBuffer);
    return f;
}

File::File(File&& other) noexcept { *this = std::move(other); }

File& File::operator=(File&& other) noexcept {
    if (this != &other) {
        if (handle_ != INVALID_HANDLE_VALUE) CloseHandle(handle_);
        handle_ = other.handle_;
        path_ = std::move(other.path_);
        pending_ = std::move(other.pending_);
        other.handle_ = INVALID_HANDLE_VALUE;
    }
    return *this;
}

File::~File() {
    if (handle_ != INVALID_HANDLE_VALUE) {
        try {
            flush();
        } catch (...) {
        }
        CloseHandle(handle_);
    }
}

size_t File::read(void* data, size_t size) {
    flush();
    auto* p = static_cast<uint8_t*>(data);
    size_t done = 0;
    while (done < size) {
        DWORD want = DWORD(std::min<size_t>(size - done, 1u << 30)), got = 0;
        if (!ReadFile(handle_, p + done, want, &got, nullptr))
            throw Failure("Cannot read " + utf8(path_) + ": " + win_error(GetLastError()));
        if (!got) break;
        done += got;
    }
    return done;
}

void File::read_exact(void* data, size_t size) {
    if (read(data, size) != size) throw Failure("Truncated archive");
}

void File::write(const void* data, size_t size) {
    auto* p = static_cast<const uint8_t*>(data);
    if (pending_.size() + size > kWriteBuffer) flush();
    if (size >= kWriteBuffer) {
        DWORD wrote = 0;
        size_t done = 0;
        while (done < size) {
            DWORD want = DWORD(std::min<size_t>(size - done, 1u << 30));
            if (!WriteFile(handle_, p + done, want, &wrote, nullptr) || wrote != want)
                throw Failure("Cannot write " + utf8(path_) + ": " + win_error(GetLastError()));
            done += wrote;
        }
        return;
    }
    pending_.insert(pending_.end(), p, p + size);
}

void File::flush() {
    if (pending_.empty()) return;
    DWORD wrote = 0;
    if (!WriteFile(handle_, pending_.data(), DWORD(pending_.size()), &wrote, nullptr) || wrote != pending_.size())
        throw Failure("Cannot write " + utf8(path_) + ": " + win_error(GetLastError()));
    pending_.clear();
}

void File::seek(uint64_t position) {
    flush();
    LARGE_INTEGER to;
    to.QuadPart = LONGLONG(position);
    if (!SetFilePointerEx(handle_, to, nullptr, FILE_BEGIN)) throw Failure("Seek failed: " + win_error(GetLastError()));
}

uint64_t File::tell() {
    LARGE_INTEGER zero{}, at{};
    SetFilePointerEx(handle_, zero, &at, FILE_CURRENT);
    return uint64_t(at.QuadPart) + pending_.size();
}

uint64_t File::size() {
    flush();
    LARGE_INTEGER s{};
    if (!GetFileSizeEx(handle_, &s)) throw Failure("Cannot get size: " + win_error(GetLastError()));
    return uint64_t(s.QuadPart);
}

void File::close() {
    if (handle_ == INVALID_HANDLE_VALUE) return;
    flush();
    CloseHandle(handle_);
    handle_ = INVALID_HANDLE_VALUE;
}

// ---------------- TempDir ----------------

TempDir::TempDir(const std::wstring& parent, const wchar_t* prefix) {
    static unsigned counter = 0;
    for (int attempt = 0; attempt < 1000; attempt++) {
        wchar_t suffix[32];
        swprintf(suffix, 32, L"%08x%04x", unsigned(GetTickCount64() ^ (GetCurrentProcessId() << 12)),
                 (++counter + attempt) & 0xffff);
        std::wstring candidate = join(parent, std::wstring(prefix) + suffix);
        if (CreateDirectoryW(long_path(candidate).c_str(), nullptr)) {
            path_ = candidate;
            return;
        }
        DWORD error = GetLastError();
        if (error != ERROR_ALREADY_EXISTS)
            throw Failure("Cannot create temporary folder in " + utf8(parent) + ": " + win_error(error));
    }
    throw Failure("Cannot create temporary folder");
}

TempDir::~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(std::filesystem::path(long_path(path_)), ec);
}

// ---------------- CRC-32 (slicing-by-8) ----------------

namespace {
struct CrcTables {
    uint32_t t[8][256];
    CrcTables() {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            t[0][i] = c;
        }
        for (uint32_t i = 0; i < 256; i++)
            for (int s = 1; s < 8; s++) t[s][i] = (t[s - 1][i] >> 8) ^ t[0][t[s - 1][i] & 0xff];
    }
};
const CrcTables kCrc;
}  // namespace

uint32_t crc32(uint32_t crc, const void* data, size_t size) {
    auto* p = static_cast<const uint8_t*>(data);
    crc = ~crc;
    while (size >= 8) {
        uint32_t a = crc ^ get32(p), b = get32(p + 4);
        crc = kCrc.t[7][a & 0xff] ^ kCrc.t[6][(a >> 8) & 0xff] ^ kCrc.t[5][(a >> 16) & 0xff] ^ kCrc.t[4][a >> 24] ^
              kCrc.t[3][b & 0xff] ^ kCrc.t[2][(b >> 8) & 0xff] ^ kCrc.t[1][(b >> 16) & 0xff] ^ kCrc.t[0][b >> 24];
        p += 8;
        size -= 8;
    }
    while (size--) crc = kCrc.t[0][(crc ^ *p++) & 0xff] ^ (crc >> 8);
    return ~crc;
}

// ---------------- SHA-256 (Windows CNG) ----------------

Sha256::Sha256() {
    BCRYPT_ALG_HANDLE alg = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    if (BCryptOpenAlgorithmProvider(&alg, BCRYPT_SHA256_ALGORITHM, nullptr, 0) != 0) throw Failure("SHA-256 unavailable");
    if (BCryptCreateHash(alg, &hash, nullptr, 0, nullptr, 0, 0) != 0) {
        BCryptCloseAlgorithmProvider(alg, 0);
        throw Failure("SHA-256 unavailable");
    }
    algorithm_ = alg;
    hash_ = hash;
}

Sha256::~Sha256() {
    if (hash_) BCryptDestroyHash(hash_);
    if (algorithm_) BCryptCloseAlgorithmProvider(algorithm_, 0);
}

void Sha256::update(const void* data, size_t size) {
    auto* p = static_cast<const uint8_t*>(data);
    while (size) {
        ULONG n = ULONG(std::min<size_t>(size, 1u << 30));
        BCryptHashData(hash_, const_cast<PUCHAR>(p), n, 0);
        p += n;
        size -= n;
    }
}

std::array<uint8_t, 32> Sha256::digest() {
    std::array<uint8_t, 32> out{};
    BCryptFinishHash(hash_, out.data(), ULONG(out.size()), 0);
    return out;
}

void restore_any(const std::wstring& source, const std::wstring& target, const std::string& device) {
    uint8_t magic[8] = {};
    {
        File f = File::open_read(absolute_path(source));
        f.read(magic, sizeof magic);
    }
    if (!std::memcmp(magic, "OCLLZ001", 8) || !std::memcmp(magic, "OCLRLE01", 8)) {
        restore_legacy(source, target, device);
    } else {
        say("Standard ZIP extraction uses CPU decoding.");
        restore_zip(source, target);
    }
}
