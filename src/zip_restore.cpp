// Port of folder_archive.restore(): safe ZIP extraction into a NEW folder, with CRC checks.
// Includes a streaming RFC 1951 inflater (stored, fixed and dynamic Huffman blocks), so it
// also extracts ZIPs made by Windows, 7-Zip, Python, etc.
#include <cstring>
#include <set>

#include "common.h"

namespace {

// ---------------- Inflate ----------------

constexpr int kFastBits = 10;
constexpr size_t kWindow = 32768;
constexpr size_t kFlushAt = 4u << 20;

struct Huffman {
    uint16_t fast[1 << kFastBits];  // (symbol << 4) | length, 0 = use slow path
    uint16_t count[16];
    uint16_t symbol[288];
};

// Returns 0 for a complete code, >0 incomplete, <0 over-subscribed (as in zlib's puff.c).
int build(Huffman& h, const uint8_t* lengths, int n) {
    std::memset(&h, 0, sizeof h);
    for (int s = 0; s < n; s++) h.count[lengths[s]]++;
    if (h.count[0] == n) return 0;
    int left = 1;
    for (int len = 1; len < 16; len++) {
        left = (left << 1) - h.count[len];
        if (left < 0) return left;
    }
    uint16_t offs[16] = {};
    for (int len = 1; len < 15; len++) offs[len + 1] = uint16_t(offs[len] + h.count[len]);
    for (int s = 0; s < n; s++)
        if (lengths[s]) h.symbol[offs[lengths[s]]++] = uint16_t(s);
    uint32_t code = 0, next[16] = {};
    for (int len = 1; len < 16; len++) {  // canonical first code for each length
        next[len] = code;
        code = (code + h.count[len]) << 1;
    }
    for (int s = 0; s < n; s++) {
        int len = lengths[s];
        if (!len) continue;
        uint32_t c = next[len]++;
        if (len > kFastBits) continue;
        uint32_t r = 0;
        for (int i = 0; i < len; i++) r |= ((c >> i) & 1) << (len - 1 - i);
        for (uint32_t k = r; k < (1u << kFastBits); k += 1u << len) h.fast[k] = uint16_t((s << 4) | len);
    }
    return left;
}

constexpr uint16_t kLenBase[29] = {3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
                                   31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
constexpr uint8_t kLenExtra[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
constexpr uint16_t kDistBase[30] = {1,   2,   3,   4,   5,   7,    9,    13,   17,   25,   33,   49,   65,    97,    129,
                                    193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
constexpr uint8_t kDistExtra[30] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

class Inflater {
public:
    Inflater(File& in, uint64_t compressed, File& out, uint64_t expected)
        : in_(in), left_(compressed), out_(out), expected_(expected), input_(1 << 16) {
        window_.reserve(kFlushAt + 65536);
    }

    uint32_t run() {
        static Huffman fixed_len, fixed_dist;
        static bool ready = [] {
            uint8_t l[288];
            for (int i = 0; i < 288; i++) l[i] = i < 144 ? 8 : i < 256 ? 9 : i < 280 ? 7 : 8;
            build(fixed_len, l, 288);
            uint8_t d[30];
            std::memset(d, 5, sizeof d);
            build(fixed_dist, d, 30);
            return true;
        }();
        (void)ready;
        bool last = false;
        while (!last) {
            last = bits(1) != 0;
            uint32_t type = bits(2);
            if (type == 0) stored();
            else if (type == 1) codes(fixed_len, fixed_dist);
            else if (type == 2) dynamic();
            else throw Failure("Invalid DEFLATE block type");
        }
        flush(true);
        if (produced_ != expected_) throw Failure("Decompressed size does not match the ZIP header");
        return crc_;
    }

private:
    int next_byte() {
        if (pos_ == len_) {
            if (!left_) return -1;
            size_t want = size_t(std::min<uint64_t>(left_, input_.size()));
            len_ = in_.read(input_.data(), want);
            if (len_ != want) throw Failure("Truncated archive");
            left_ -= len_;
            pos_ = 0;
        }
        return input_[pos_++];
    }
    void refill() {
        while (count_ <= 56) {
            int c = next_byte();
            if (c < 0) break;
            buffer_ |= uint64_t(c) << count_;
            count_ += 8;
        }
    }
    void consume(int n) {
        if (n > count_) throw Failure("Truncated DEFLATE data");
        buffer_ >>= n;
        count_ -= n;
    }
    uint32_t bits(int n) {
        if (!n) return 0;
        if (count_ < n) refill();
        uint32_t v = uint32_t(buffer_ & ((1ull << n) - 1));
        consume(n);
        return v;
    }
    int decode(const Huffman& h) {
        if (count_ < 15) refill();
        uint32_t v = uint32_t(buffer_);
        uint16_t e = h.fast[v & ((1u << kFastBits) - 1)];
        if (e) {
            consume(e & 15);
            return e >> 4;
        }
        int code = 0, first = 0, index = 0;
        for (int len = 1; len < 16; len++) {
            code |= (v >> (len - 1)) & 1;
            int c = h.count[len];
            if (code - c < first) {
                consume(len);
                return h.symbol[index + (code - first)];
            }
            index += c;
            first = (first + c) << 1;
            code <<= 1;
        }
        throw Failure("Invalid DEFLATE code");
    }
    void emit(uint8_t b) {
        if (++produced_ > expected_) throw Failure("Decompressed data is larger than the ZIP header says");
        window_.push_back(b);
    }
    void flush(bool final) {
        size_t n = window_.size() - pending_;
        if (n) {
            out_.write(window_.data() + pending_, n);
            crc_ = crc32(crc_, window_.data() + pending_, n);
        }
        pending_ = window_.size();
        if (!final && window_.size() > kWindow) {
            std::memmove(window_.data(), window_.data() + window_.size() - kWindow, kWindow);
            window_.resize(kWindow);
            pending_ = kWindow;
        }
    }
    void stored() {
        consume(count_ & 7);  // skip to byte boundary
        uint32_t len = bits(16), nlen = bits(16);
        if (len != (~nlen & 0xffff)) throw Failure("Invalid stored DEFLATE block");
        for (uint32_t i = 0; i < len; i++) emit(uint8_t(bits(8)));
        if (window_.size() >= kFlushAt) flush(false);
    }
    void codes(const Huffman& lens, const Huffman& dists) {
        for (;;) {
            int s = decode(lens);
            if (s < 256) {
                emit(uint8_t(s));
            } else if (s == 256) {
                break;
            } else {
                s -= 257;
                if (s >= 29) throw Failure("Invalid DEFLATE length code");
                uint32_t len = kLenBase[s] + bits(kLenExtra[s]);
                int d = decode(dists);
                if (d >= 30) throw Failure("Invalid DEFLATE distance code");
                uint32_t dist = kDistBase[d] + bits(kDistExtra[d]);
                if (dist > window_.size()) throw Failure("Invalid DEFLATE distance");
                size_t from = window_.size() - dist;
                for (uint32_t i = 0; i < len; i++) emit(window_[from + i]);
            }
            if (window_.size() >= kFlushAt) flush(false);
        }
    }
    void dynamic() {
        static const uint8_t order[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
        int nlen = int(bits(5)) + 257, ndist = int(bits(5)) + 1, ncode = int(bits(4)) + 4;
        if (nlen > 286 || ndist > 30) throw Failure("Invalid DEFLATE header counts");
        uint8_t lengths[320] = {};
        for (int i = 0; i < ncode; i++) lengths[order[i]] = uint8_t(bits(3));
        Huffman code_lengths;
        if (build(code_lengths, lengths, 19) != 0) throw Failure("Invalid DEFLATE code lengths");
        int index = 0;
        while (index < nlen + ndist) {
            int s = decode(code_lengths);
            if (s < 16) {
                lengths[index++] = uint8_t(s);
                continue;
            }
            uint8_t value = 0;
            int repeat;
            if (s == 16) {
                if (!index) throw Failure("Invalid DEFLATE repeat");
                value = lengths[index - 1];
                repeat = 3 + int(bits(2));
            } else if (s == 17) {
                repeat = 3 + int(bits(3));
            } else {
                repeat = 11 + int(bits(7));
            }
            if (index + repeat > nlen + ndist) throw Failure("Too many DEFLATE lengths");
            while (repeat--) lengths[index++] = value;
        }
        if (!lengths[256]) throw Failure("DEFLATE block has no end code");
        Huffman lens, dists;
        int err = build(lens, lengths, nlen);
        if (err < 0 || (err > 0 && nlen - lens.count[0] != 1)) throw Failure("Invalid DEFLATE literal/length code");
        err = build(dists, lengths + nlen, ndist);
        if (err < 0 || (err > 0 && ndist - dists.count[0] != 1)) throw Failure("Invalid DEFLATE distance code");
        codes(lens, dists);
    }

    File& in_;
    uint64_t left_;
    File& out_;
    uint64_t expected_, produced_ = 0;
    std::vector<uint8_t> input_;
    size_t pos_ = 0, len_ = 0;
    uint64_t buffer_ = 0;
    int count_ = 0;
    std::vector<uint8_t> window_;
    size_t pending_ = 0;
    uint32_t crc_ = 0;
};

// ---------------- ZIP central directory ----------------

struct ZipEntry {
    std::string name;  // UTF-8, '/' separated
    uint16_t flags, method;
    uint32_t crc, external;
    uint64_t compressed, size, offset;
};

std::string decode_name(const uint8_t* p, size_t n, bool is_utf8) {
    std::string raw(reinterpret_cast<const char*>(p), n);
    if (is_utf8) {
        if (n && !MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, raw.data(), int(n), nullptr, 0))
            throw Failure("Invalid UTF-8 file name in archive");
        return raw;
    }
    int w = MultiByteToWideChar(437, 0, raw.data(), int(n), nullptr, 0);
    std::wstring text(size_t(w), L'\0');
    MultiByteToWideChar(437, 0, raw.data(), int(n), text.data(), w);
    return utf8(text);
}

std::vector<ZipEntry> read_directory(File& z) {
    uint64_t file_size = z.size();
    size_t tail = size_t(std::min<uint64_t>(file_size, 65535 + 22));
    std::vector<uint8_t> buf(tail);
    z.seek(file_size - tail);
    z.read_exact(buf.data(), tail);
    size_t at = std::string::npos;
    for (size_t i = tail >= 22 ? tail - 22 + 1 : 0; i-- > 0;)
        if (get32(&buf[i]) == 0x06054b50) {
            at = i;
            break;
        }
    if (at == std::string::npos) throw Failure("File is not a zip file");
    const uint8_t* e = &buf[at];
    uint64_t entries = get16(e + 10), cd_size = get32(e + 12), cd_offset = get32(e + 16);
    uint64_t eocd_pos = file_size - tail + at;
    if (entries == 0xffff || cd_size == 0xffffffff || cd_offset == 0xffffffff) {
        if (eocd_pos < 20) throw Failure("Bad ZIP64 locator");
        uint8_t loc[20];
        z.seek(eocd_pos - 20);
        z.read_exact(loc, 20);
        if (get32(loc) != 0x07064b50) throw Failure("Bad ZIP64 locator");
        uint8_t rec[56];
        z.seek(get64(loc + 8));
        z.read_exact(rec, 56);
        if (get32(rec) != 0x06064b50) throw Failure("Bad ZIP64 end record");
        entries = get64(rec + 32);
        cd_size = get64(rec + 40);
        cd_offset = get64(rec + 48);
    }
    if (cd_offset + cd_size > file_size || cd_size > (1ull << 32)) throw Failure("Bad central directory");
    std::vector<uint8_t> cd(static_cast<size_t>(cd_size));
    z.seek(cd_offset);
    z.read_exact(cd.data(), cd.size());
    std::vector<ZipEntry> out;
    size_t p = 0;
    for (uint64_t i = 0; i < entries; i++) {
        if (p + 46 > cd.size() || get32(&cd[p]) != 0x02014b50) throw Failure("Bad central directory entry");
        const uint8_t* c = &cd[p];
        ZipEntry entry;
        entry.flags = get16(c + 8);
        entry.method = get16(c + 10);
        entry.crc = get32(c + 16);
        entry.compressed = get32(c + 20);
        entry.size = get32(c + 24);
        size_t nlen = get16(c + 28), elen = get16(c + 30), clen = get16(c + 32);
        entry.external = get32(c + 38);
        entry.offset = get32(c + 42);
        if (p + 46 + nlen + elen + clen > cd.size()) throw Failure("Bad central directory entry");
        entry.name = decode_name(c + 46, nlen, (entry.flags & 0x800) != 0);
        for (auto& ch : entry.name)
            if (ch == '\\') ch = '/';  // same as Python's zipfile on Windows
        const uint8_t* x = c + 46 + nlen;
        for (size_t k = 0; k + 4 <= elen;) {
            uint16_t id = get16(x + k), sz = get16(x + k + 2);
            if (k + 4 + sz > elen) break;
            if (id == 1) {
                const uint8_t* v = x + k + 4;
                size_t q = 0;
                auto take = [&](uint64_t& field) {
                    if (field == 0xffffffff) {
                        if (q + 8 > sz) throw Failure("Corrupt ZIP64 extra field");
                        field = get64(v + q);
                        q += 8;
                    }
                };
                take(entry.size);
                take(entry.compressed);
                take(entry.offset);
            }
            k += 4 + sz;
        }
        out.push_back(std::move(entry));
        p += 46 + nlen + elen + clen;
    }
    return out;
}

bool reserved_name(const std::wstring& part) {
    std::wstring stem = part.substr(0, part.find(L'.'));
    CharUpperBuffW(stem.data(), DWORD(stem.size()));
    static const wchar_t* fixed[] = {L"CON", L"PRN", L"AUX", L"NUL"};
    for (auto* f : fixed)
        if (stem == f) return true;
    if (stem.size() == 4 && (stem.rfind(L"COM", 0) == 0 || stem.rfind(L"LPT", 0) == 0) && stem[3] >= L'1' &&
        stem[3] <= L'9')
        return true;
    return false;
}

std::vector<std::wstring> safe_parts(const std::string& name) {
    if (name.empty() || name[0] == '/') throw Failure("Unsafe archive path");
    std::vector<std::wstring> parts;
    std::wstring w = wide(name);
    size_t start = 0;
    while (start <= w.size()) {
        size_t slash = w.find(L'/', start);
        std::wstring part = w.substr(start, slash == std::wstring::npos ? std::wstring::npos : slash - start);
        if (!part.empty()) {
            if (part == L"." || part == L".." || part.find(L':') != std::wstring::npos ||
                part.back() == L' ' || part.back() == L'.')
                throw Failure("Unsafe archive path");
            for (wchar_t ch : part)
                if (ch < 32 || wcschr(L"<>\"|?*", ch)) throw Failure("Unsafe archive path");
            parts.push_back(part);
        }
        if (slash == std::wstring::npos) break;
        start = slash + 1;
    }
    if (parts.empty()) throw Failure("Unsafe archive path");
    for (auto& part : parts)
        if (reserved_name(part)) throw Failure("Unsupported Windows filename");
    return parts;
}

}  // namespace

void restore_zip(const std::wstring& source_in, const std::wstring& target_in) {
    std::wstring source = absolute_path(source_in), target = absolute_path(target_in);
    if (exists(target)) throw Failure("Choose a new destination folder");
    File z = File::open_read(source);
    std::vector<ZipEntry> entries = read_directory(z);
    {
        TempDir temp(parent_of(target), L".zip-restore-");
        std::wstring root = join(temp.path(), L"restored");
        make_directory(root);
        std::set<std::wstring> names;
        for (auto& e : entries) {
            std::vector<std::wstring> parts = safe_parts(e.name);
            std::wstring key;
            for (auto& part : parts) key += (key.empty() ? L"" : L"/") + part;
            if (!names.insert(fold_case(key)).second) throw Failure("Duplicate archive path");
            if (((e.external >> 16) & 0170000) == 0120000) throw Failure("Archive links are not supported");
            if (e.flags & 1) throw Failure("Encrypted archive entries are not supported");
            bool directory = e.name.back() == '/';
            std::wstring path = root;
            size_t folders = directory ? parts.size() : parts.size() - 1;
            for (size_t i = 0; i < folders; i++) {
                path = join(path, parts[i]);
                make_directory(path);
            }
            if (directory) continue;
            path = join(path, parts.back());

            uint8_t local[30];
            z.seek(e.offset);
            z.read_exact(local, 30);
            if (get32(local) != 0x04034b50) throw Failure("Bad local file header: " + e.name);
            z.seek(e.offset + 30 + get16(local + 26) + get16(local + 28));
            File out = File::create_new(path);
            uint32_t crc = 0;
            if (e.method == 0) {
                if (e.compressed != e.size) throw Failure("Bad stored entry: " + e.name);
                std::vector<uint8_t> buf(1 << 20);
                for (uint64_t left = e.size; left;) {
                    size_t n = size_t(std::min<uint64_t>(left, buf.size()));
                    z.read_exact(buf.data(), n);
                    crc = crc32(crc, buf.data(), n);
                    out.write(buf.data(), n);
                    left -= n;
                }
            } else if (e.method == 8) {
                crc = Inflater(z, e.compressed, out, e.size).run();
            } else {
                throw Failure("Unsupported compression method " + std::to_string(e.method) + ": " + e.name);
            }
            out.close();
            if (crc != e.crc) throw Failure("Bad CRC-32 for file " + e.name);
        }
        move_new(root, target);
    }
    say("Restored to: " + utf8(target));
    say("ZIP CRC integrity checks: PASS");
}
