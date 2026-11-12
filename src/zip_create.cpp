// Port of gpu_zip.py: standard ZIP/ZIP64 with OpenCL fixed-Huffman DEFLATE encoding.
//
// Work runs as a three-stage pipeline so the GPU never waits on the disk:
//   reader thread  -> fills a batch with 32 KiB chunks, packing chunks from MANY files together
//   encoder thread -> runs one OpenCL dispatch per batch (one work-item per chunk, thousands at once)
//   main thread    -> writes ZIP headers + compressed chunks in order, patches CRC/sizes
// Three batches circulate, so reading, GPU encoding and writing all happen at the same time.
// Each file is still split into chunks from its own start, so the ZIP bytes are unchanged.
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <thread>

#include "common.h"
#include "kernels.h"
#include "ocl.h"

namespace {

constexpr uint32_t BLOCK = 32768;
constexpr uint32_t TABLE = 4096;
constexpr uint32_t DEFAULT_BATCH_CHUNKS = 1024;  // 32 MiB of input per GPU dispatch (smaller = better overlap)
constexpr int PIPELINE_DEPTH = 3;               // batches in flight: reading, encoding, writing

// A run of consecutive chunks in a batch that belong to one archive entry.
struct Segment {
    uint32_t entry;      // index into the entry list
    uint32_t first;      // first chunk slot in the batch
    uint32_t chunks;     // number of chunk slots (0 for folders / empty files / end marker)
    bool begins, ends;   // first / last segment of this entry
    uint64_t file_size;  // bytes read so far (final value when ends)
    uint32_t crc;        // CRC-32 so far (final value when ends)
};

struct Batch {
    std::vector<uint8_t> input, output;  // chunk i: input+i*BLOCK in, output+offsets[i] (sizes[i] bytes) out
    std::vector<uint32_t> lengths, sizes;
    std::vector<size_t> offsets;
    std::vector<Segment> segments;
    uint32_t chunks = 0;
    uint64_t seq = 0;  // order in the archive (batches can finish out of order with several devices)
    bool last = false;
    explicit Batch(uint32_t capacity)
        : input(size_t(capacity) * BLOCK), output(size_t(capacity) * BLOCK * 2), lengths(capacity), sizes(capacity),
          offsets(capacity) {}
    void slot_offsets() {  // output laid out in fixed 64 KiB slots
        for (uint32_t i = 0; i < chunks; i++) offsets[i] = size_t(i) * BLOCK * 2;
    }
    void reset() {
        chunks = 0;
        segments.clear();
        last = false;
    }
};

struct Cancelled {};

// Blocking hand-off between pipeline threads. close() makes every wait return false.
class Channel {
public:
    bool push(Batch* b) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_) return false;
            queue_.push_back(b);
        }
        ready_.notify_one();
        return true;
    }
    bool pop(Batch*& b) {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [&] { return closed_ || finished_ || !queue_.empty(); });
        if (closed_ || queue_.empty()) return false;
        b = queue_.front();
        queue_.pop_front();
        return true;
    }
    void close() {  // error: wake everyone, drop the queue
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        ready_.notify_all();
    }
    void finish() {  // no more batches: pop() returns false once the queue is empty
        {
            std::lock_guard<std::mutex> lock(mutex_);
            finished_ = true;
        }
        ready_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<Batch*> queue_;
    bool closed_ = false, finished_ = false;
};

class Encoder {
public:
    virtual ~Encoder() = default;
    // Compresses b.chunks chunks: reads b.input/b.lengths, fills b.output/b.sizes.
    virtual void encode(Batch& b) = 0;
    std::string name;
    uint32_t capacity = DEFAULT_BATCH_CHUNKS;  // chunks per batch
    size_t dispatches = 0;
    double busy_seconds = 0;  // time spent inside encode() (upload + kernel + download for the GPU)
    uint64_t chunks_done = 0;
};

uint32_t env_chunks(const char* name, uint32_t fallback) {
    char v[32];
    DWORD len = GetEnvironmentVariableA(name, v, sizeof v);
    if (len > 0 && len < sizeof v) {
        long n = std::strtol(v, nullptr, 10);
        if (n >= 1 && n <= 65536) return uint32_t(n);
    }
    return fallback;
}

// OCLZIP_BATCH_CHUNKS lets you tune the batch size (e.g. for GPUs with little memory).
uint32_t batch_chunks_setting() { return env_chunks("OCLZIP_BATCH_CHUNKS", DEFAULT_BATCH_CHUNKS); }

uint32_t deflate_chunk(const uint8_t* src, uint32_t n, uint8_t* out, int32_t* table);  // CPU version, below

uint32_t reverse_bits(uint32_t v, uint32_t n) {
    uint32_t r = 0;
    for (uint32_t i = 0; i < n; i++, v >>= 1) r = (r << 1) | (v & 1);
    return r;
}

// Lookup tables shared by the optimised kernel and the CPU encoder.
struct DeflateTables {
    uint32_t lit[288];    // bit-reversed fixed Huffman code | (code length << 16)
    uint8_t lcode[259];   // match length 3..258 -> length code 0..28
    uint8_t dcode[512];   // distance -> distance code (zlib's two-level scheme)
    uint32_t drev[30];    // bit-reversed 5-bit distance codes
    DeflateTables() {
        static const uint32_t lb[29] = {3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
                                        31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
        static const uint32_t db[30] = {1,    2,    3,    4,    5,    7,     9,     13,    17,  25,
                                        33,   49,   65,   97,   129,  193,   257,   385,   513, 769,
                                        1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
        for (uint32_t s = 0; s < 288; s++) {
            uint32_t c, n;
            if (s <= 143) { c = 48 + s; n = 8; }
            else if (s <= 255) { c = 400 + s - 144; n = 9; }
            else if (s <= 279) { c = s - 256; n = 7; }
            else { c = 192 + s - 280; n = 8; }
            lit[s] = reverse_bits(c, n) | (n << 16);
        }
        std::memset(lcode, 0, sizeof lcode);
        for (uint32_t len = 3; len <= 258; len++) {
            uint32_t lc = 0;
            while (lc < 28 && lb[lc + 1] <= len) lc++;
            lcode[len] = uint8_t(lc);
        }
        for (uint32_t d = 1; d <= 32768; d++) {
            uint32_t dc = 0;
            while (dc < 29 && db[dc + 1] <= d) dc++;
            dcode[d <= 256 ? d - 1 : 256 + ((d - 1) >> 7)] = uint8_t(dc);
        }
        for (uint32_t dc = 0; dc < 30; dc++) drev[dc] = reverse_bits(dc, 5);
    }
};
const DeflateTables kTables;

bool use_original_kernel() {
    char v[32];
    DWORD len = GetEnvironmentVariableA("OCLZIP_KERNEL", v, sizeof v);
    return len > 0 && len < sizeof v && _stricmp(v, "original") == 0;
}

// The optimised kernel with its lookup tables prepended as OpenCL constants.
std::string kernel_source() {
    if (use_original_kernel()) return std::string(kDeflateKernel) + kCompactKernel;
    auto list = [](const char* type, const char* name, const auto* values, size_t count) {
        std::string s = std::string("__constant ") + type + " " + name + "[" + std::to_string(count) + "]={";
        for (size_t i = 0; i < count; i++) s += (i ? "," : "") + std::to_string(uint32_t(values[i]));
        return s + "};\n";
    };
    return list("uint", "LIT", kTables.lit, 288) + list("uchar", "LCODE", kTables.lcode, 259) +
           list("uchar", "DCODE", kTables.dcode, 512) + list("uint", "DREV", kTables.drev, 30) + kDeflateKernelFast + kCompactKernel;
}

class GpuEncoder : public Encoder {
public:
    explicit GpuEncoder(const ocl::Device& device)
        : runtime_(device, kernel_source().c_str()), table_entry_(use_original_kernel() ? 4 : 2) {
        name = device.name + (table_entry_ == 4 ? " [original kernel]" : "");
        capacity = batch_chunks_setting();
        // Stay within the device's single-allocation limit and a quarter of its memory.
        uint64_t per_chunk_out = uint64_t(BLOCK) * 2;
        uint64_t per_chunk_total = uint64_t(BLOCK) * 3 + uint64_t(TABLE) * table_entry_ + 8;
        if (device.max_alloc)
            capacity = uint32_t(std::min<uint64_t>(capacity, device.max_alloc / per_chunk_out));
        if (device.global_mem)
            capacity = uint32_t(std::min<uint64_t>(capacity, device.global_mem / 4 / per_chunk_total));
        capacity = std::max(capacity, 16u);
        src_ = std::make_unique<ocl::Buffer>(runtime_, ocl::CL_MEM_READ_ONLY, size_t(BLOCK) * capacity);
        dst_ = std::make_unique<ocl::Buffer>(runtime_, ocl::CL_MEM_WRITE_ONLY, size_t(BLOCK) * 2 * capacity);
        lengths_ = std::make_unique<ocl::Buffer>(runtime_, ocl::CL_MEM_READ_ONLY, sizeof(uint32_t) * capacity);
        sizes_ = std::make_unique<ocl::Buffer>(runtime_, ocl::CL_MEM_READ_WRITE, sizeof(uint32_t) * capacity);
        // Integrated GPUs get shorter launches so the driver's timeout is never reached.
        launch_ = std::min(capacity, env_chunks("OCLZIP_LAUNCH_CHUNKS", device.unified ? 512u : 4096u));
        tables_ = std::make_unique<ocl::Buffer>(runtime_, ocl::CL_MEM_READ_WRITE, size_t(table_entry_) * TABLE * capacity);
        kernel_ = runtime_.kernel("deflate_blocks");
        runtime_.set_arg(kernel_, 0, *src_);
        runtime_.set_arg(kernel_, 1, *dst_);
        runtime_.set_arg(kernel_, 2, *lengths_);
        runtime_.set_arg(kernel_, 3, *sizes_);
        runtime_.set_arg(kernel_, 4, *tables_);
        runtime_.set_arg(kernel_, 5, BLOCK);
        // Packing stage: compressed chunks are moved end to end on the GPU before download.
        offsets_ = std::make_unique<ocl::Buffer>(runtime_, ocl::CL_MEM_READ_ONLY, sizeof(uint32_t) * capacity);
        packed_ = std::make_unique<ocl::Buffer>(runtime_, ocl::CL_MEM_WRITE_ONLY, size_t(BLOCK + 8) * capacity);
        compact_ = runtime_.kernel("compact");
        runtime_.set_arg(compact_, 0, *dst_);
        runtime_.set_arg(compact_, 1, *sizes_);
        runtime_.set_arg(compact_, 2, *offsets_);
        runtime_.set_arg(compact_, 3, *packed_);
        runtime_.set_arg(compact_, 4, BLOCK * 2);
        packed_offsets_.resize(capacity);
    }

    void encode(Batch& b) override {
        uint32_t n = b.chunks;
        if (gpu_failed_) {  // the driver already lost a launch in this run; don't trust the GPU again
            encode_on_cpu(b, n, false);
            return;
        }
        runtime_.write(*src_, b.input.data(), size_t(BLOCK) * n);
        runtime_.write(*lengths_, b.lengths.data(), sizeof(uint32_t) * n);
        // Some drivers (seen on Intel Iris Xe) silently drop a long-running kernel launch and
        // leave the previous results in the buffers. So: launch in slices, pre-fill the sizes
        // with a marker the kernel always overwrites, and check every chunk afterwards.
        for (uint32_t slice = launch_, attempt = 0;; attempt++) {
            std::fill(b.sizes.begin(), b.sizes.begin() + n, kNotWritten);
            runtime_.write(*sizes_, b.sizes.data(), sizeof(uint32_t) * n);
            for (uint32_t first = 0; first < n; first += slice)
                runtime_.run(kernel_, std::min(slice, n - first), first);  // one work-item per chunk
            runtime_.read(*sizes_, b.sizes.data(), sizeof(uint32_t) * n);
            uint32_t missing = 0;
            for (uint32_t i = 0; i < n; i++)
                if (!valid(b.sizes[i])) missing++;
            if (!missing) break;
            if (attempt == 0 && slice > 64) {
                slice = std::max(64u, slice / 8);
                launch_ = slice;  // keep the smaller launch size for the rest of the run
                say("Warning: the GPU driver dropped " + commas(missing) + " chunks; retrying with " +
                    commas(slice) + "-chunk launches.");
                continue;
            }
            runtime_.read(*dst_, b.output.data(), size_t(BLOCK) * 2 * n);
            b.slot_offsets();
            gpu_failed_ = true;
            say("Warning: the GPU did not finish " + commas(missing) +
                " chunks; using the CPU for the rest of this archive (same output bytes).");
            encode_on_cpu(b, n, true);
            return;
        }
        // Download only the real compressed bytes: pack the chunks end to end on the GPU first.
        uint32_t total = 0;
        for (uint32_t i = 0; i < n; i++) {
            packed_offsets_[i] = total;
            b.offsets[i] = total;
            total += b.sizes[i];
        }
        runtime_.write(*offsets_, packed_offsets_.data(), sizeof(uint32_t) * n);
        runtime_.run(compact_, size_t(n) * kCompactGroup, 0, kCompactGroup);  // one work-group per chunk
        runtime_.read(*packed_, b.output.data(), total);
        dispatches++;
    }

private:
    // Encodes chunks on all CPU threads; with only_missing, keeps the chunks the GPU did finish.
    static bool valid(uint32_t size) { return size != kNotWritten && size != 0 && size <= BLOCK + 5; }

    void encode_on_cpu(Batch& b, uint32_t n, bool only_missing) {
        b.slot_offsets();
        std::atomic<uint32_t> next{0};
        auto work = [&] {
            std::vector<int32_t> table(TABLE);
            for (uint32_t i; (i = next.fetch_add(1)) < n;) {
                if (only_missing && valid(b.sizes[i])) continue;
                b.sizes[i] = deflate_chunk(b.input.data() + size_t(i) * BLOCK, b.lengths[i],
                                           b.output.data() + size_t(i) * BLOCK * 2, table.data());
            }
        };
        std::vector<std::thread> pool;
        for (unsigned t = 1; t < std::max(1u, std::thread::hardware_concurrency()); t++) pool.emplace_back(work);
        work();
        for (auto& t : pool) t.join();
        dispatches++;
    }

    bool gpu_failed_ = false;
    static constexpr uint32_t kNotWritten = 0xFFFFFFFFu;
    uint32_t launch_ = 0;  // chunks per kernel launch
    ocl::Runtime runtime_;
    uint32_t table_entry_;  // hash table entry size: 2 bytes (optimised kernel) or 4 (original)
    static constexpr size_t kCompactGroup = 128;
    std::unique_ptr<ocl::Buffer> src_, dst_, lengths_, sizes_, tables_, offsets_, packed_;
    ocl::cl_kernel kernel_ = nullptr, compact_ = nullptr;
    std::vector<uint32_t> packed_offsets_;
};

// ---- CPU version of deflate_blocks: same algorithm, byte-identical output ----

constexpr uint32_t LB[29] = {3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
                             31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
constexpr uint32_t LE[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
constexpr uint32_t DB[30] = {1,   2,   3,   4,   5,   7,    9,    13,   17,   25,   33,   49,   65,    97,    129,
                             193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
constexpr uint32_t DE[30] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

uint32_t rev(uint32_t v, uint32_t n) {
    uint32_t r = 0;
    for (uint32_t i = 0; i < n; i++) {
        r = (r << 1) | (v & 1);
        v >>= 1;
    }
    return r;
}

struct BitWriter {
    uint8_t* out;
    size_t bytes = 0;
    uint64_t acc = 0;
    int count = 0;
    void put(uint32_t value, uint32_t n) {
        acc |= uint64_t(value & ((1ull << n) - 1)) << count;
        count += int(n);
        while (count >= 8) {
            out[bytes++] = uint8_t(acc);
            acc >>= 8;
            count -= 8;
        }
    }
    void align() {
        if (count) {
            out[bytes++] = uint8_t(acc);
            acc = 0;
            count = 0;
        }
    }
    void symbol(uint32_t s) {
        uint32_t c, n;
        if (s <= 143) { c = 48 + s; n = 8; }
        else if (s <= 255) { c = 400 + s - 144; n = 9; }
        else if (s <= 279) { c = s - 256; n = 7; }
        else { c = 192 + s - 280; n = 8; }
        put(rev(c, n), n);
    }
};

uint32_t deflate_chunk(const uint8_t* src, uint32_t n, uint8_t* out, int32_t* table) {
    std::fill(table, table + TABLE, -1);
    BitWriter w{out};
    w.put(2, 3);  // BFINAL=0, fixed Huffman
    uint32_t i = 0;
    while (i < n) {
        uint32_t len = 0, distance = 0;
        if (i + 3 <= n) {
            uint32_t h = ((uint32_t(src[i]) * 251u + src[i + 1]) * 251u + src[i + 2]) & 4095u;
            int32_t prev = table[h];
            table[h] = int32_t(i);
            if (prev >= 0 && i - uint32_t(prev) <= 32768) {
                uint32_t limit = std::min(258u, n - i);
                while (len < limit && src[uint32_t(prev) + len] == src[i + len]) len++;
                distance = i - uint32_t(prev);
            }
        }
        if (len >= 3) {
            uint32_t lc = 0;
            while (lc < 28 && LB[lc + 1] <= len) lc++;
            w.symbol(257 + lc);
            w.put(len - LB[lc], LE[lc]);
            uint32_t dc = 0;
            while (dc < 29 && DB[dc + 1] <= distance) dc++;
            w.put(rev(dc, 5), 5);
            w.put(distance - DB[dc], DE[dc]);
            i += len;
        } else {
            w.symbol(src[i]);
            i++;
        }
    }
    w.symbol(256);
    w.put(0, 3);  // non-final empty stored block aligns the next chunk
    w.align();
    w.put(0, 16);
    w.put(65535, 16);
    uint32_t size = uint32_t(w.bytes);
    if (size >= n + 5) {  // unhelpful chunk: standard uncompressed block
        out[0] = 0;
        out[1] = uint8_t(n);
        out[2] = uint8_t(n >> 8);
        uint32_t inverse = (~n) & 65535;
        out[3] = uint8_t(inverse);
        out[4] = uint8_t(inverse >> 8);
        std::memcpy(out + 5, src, n);
        size = n + 5;
    }
    return size;
}


class CpuEncoder : public Encoder {
public:
    explicit CpuEncoder(unsigned threads = 0) {
        threads_ = threads ? threads : std::max(1u, std::thread::hardware_concurrency());
        name = "CPU (" + std::to_string(threads_) + " threads)";
        capacity = env_chunks("OCLZIP_BATCH_CHUNKS", 256);  // small batches keep the CPU pipeline overlapped
        tables_.resize(size_t(TABLE) * threads_);
    }

    void encode(Batch& b) override {
        uint32_t n = b.chunks;
        std::atomic<uint32_t> next{0};
        auto work = [&](unsigned t) {
            int32_t* table = tables_.data() + size_t(t) * TABLE;
            for (uint32_t i; (i = next.fetch_add(1)) < n;)
                b.sizes[i] = deflate_chunk(b.input.data() + size_t(i) * BLOCK, b.lengths[i],
                                           b.output.data() + size_t(i) * BLOCK * 2, table);
        };
        unsigned count = std::min<unsigned>(threads_, n);
        std::vector<std::thread> pool;
        for (unsigned t = 1; t < count; t++) pool.emplace_back(work, t);
        work(0);
        for (auto& t : pool) t.join();
        b.slot_offsets();
        dispatches++;
    }

private:
    unsigned threads_;
    std::vector<int32_t> tables_;
};

// "ALL": the CPU plus every usable GPU work on the same archive at once.
std::vector<std::unique_ptr<Encoder>> make_all_encoders() {
    std::vector<std::unique_ptr<Encoder>> engines;
    for (auto& d : ocl::devices(ocl::CL_DEVICE_TYPE_GPU)) {
        if (!d.available || !d.compiler) continue;
        try {
            engines.push_back(std::make_unique<GpuEncoder>(d));
        } catch (const std::exception& e) {
            say("Warning: skipping " + d.name + ": " + e.what());
        }
    }
    // Leave threads free for reading, writing and feeding each GPU.
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    unsigned reserved = 2 + unsigned(engines.size());
    engines.push_back(std::make_unique<CpuEncoder>(hw > reserved + 1 ? hw - reserved : 1));
    return engines;
}

std::unique_ptr<Encoder> make_encoder(const std::string& device) {
    if (device == "CPU") return std::make_unique<CpuEncoder>();
    std::string want = utf8(fold_case(wide(device)));
    const ocl::Device* best = nullptr;
    auto all = ocl::devices(ocl::CL_DEVICE_TYPE_GPU);
    for (auto& d : all) {
        if (!d.available || !d.compiler) continue;
        if (utf8(fold_case(wide(d.name))).find(want) == std::string::npos) continue;
        // Prefer discrete GPUs, then the one with the most memory.
        if (!best || (best->unified && !d.unified) || (best->unified == d.unified && d.global_mem > best->global_mem))
            best = &d;
    }
    if (!best) throw Failure("Selected OpenCL GPU is not available");
    return std::make_unique<GpuEncoder>(*best);
}

struct Entry {
    std::wstring path;
    std::string name;  // UTF-8 archive name
    bool directory;
    FILETIME modified;
    uint64_t size = 0;  // file size when the folder was listed
};

void check_link(const std::wstring& path, DWORD attr) {
    if (attr & FILE_ATTRIBUTE_REPARSE_POINT)
        throw Failure("Links and junctions are not supported: " + utf8(path));
}

// Same order as os.walk(top-down) with sorted(dirs)+sorted(files) per folder.
void walk(const std::wstring& folder, const std::string& prefix, std::vector<Entry>& entries) {
    WIN32_FIND_DATAW data;
    HANDLE find = FindFirstFileExW(long_path(join(folder, L"*")).c_str(), FindExInfoBasic, &data,
                                   FindExSearchNameMatch, nullptr, FIND_FIRST_EX_LARGE_FETCH);
    if (find == INVALID_HANDLE_VALUE) {
        DWORD error = GetLastError();
        if (error == ERROR_FILE_NOT_FOUND) return;
        throw Failure("Cannot list " + utf8(folder) + ": " + win_error(error));
    }
    std::vector<Entry> dirs, files;
    do {
        std::wstring name = data.cFileName;
        if (name == L"." || name == L"..") continue;
        std::wstring path = join(folder, name);
        check_link(path, data.dwFileAttributes);
        bool directory = (data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
        Entry e{path, prefix + utf8(name), directory, data.ftLastWriteTime,
                (uint64_t(data.nFileSizeHigh) << 32) | data.nFileSizeLow};
        (directory ? dirs : files).push_back(std::move(e));
    } while (FindNextFileW(find, &data));
    DWORD error = GetLastError();
    FindClose(find);
    if (error != ERROR_NO_MORE_FILES) throw Failure("Cannot list " + utf8(folder) + ": " + win_error(error));
    auto by_name = [](const Entry& a, const Entry& b) { return name_of(a.path) < name_of(b.path); };
    std::sort(dirs.begin(), dirs.end(), by_name);
    std::sort(files.begin(), files.end(), by_name);
    for (auto& d : dirs) entries.push_back({d.path, d.name + "/", true, d.modified});
    for (auto& f : files) entries.push_back(f);
    for (auto& d : dirs) walk(d.path, d.name + "/", entries);
}

void dos_time(FILETIME ft, uint16_t& clock, uint16_t& date) {
    SYSTEMTIME utc{}, local{};
    FileTimeToSystemTime(&ft, &utc);
    if (!SystemTimeToTzSpecificLocalTime(nullptr, &utc, &local)) local = utc;
    int year = std::clamp(int(local.wYear), 1980, 2107);
    date = uint16_t(((year - 1980) << 9) | (local.wMonth << 5) | local.wDay);
    clock = uint16_t((local.wHour << 11) | (local.wMinute << 5) | (local.wSecond / 2));
}

struct Record {
    std::string name;
    uint16_t method, clock, date;
    uint32_t crc;
    uint64_t size, compressed, offset;
    bool directory;
};


// Shared state of one compression run; the first error stops every stage.
class Pipeline {
public:
    Pipeline(uint32_t capacity, int depth) {
        for (int i = 0; i < depth; i++) {
            pool_.push_back(std::make_unique<Batch>(capacity));
            free.push(pool_.back().get());
        }
    }
    void fail(std::exception_ptr e) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!error_) error_ = e;
        }
        free.close();
        filled.close();
        encoded.close();
    }
    void rethrow() {
        if (error_) std::rethrow_exception(error_);
    }
    Channel free, filled, encoded;

private:
    std::vector<std::unique_ptr<Batch>> pool_;
    std::mutex mutex_;
    std::exception_ptr error_;
};

// Time spent by the stages that are not encoders, to show where a run is limited.
struct StageTimes {
    double read_busy = 0;          // reader: copying file data into batches
    double read_wait_files = 0;    // reader: waiting for the read threads (disk)
    double read_wait_buffers = 0;  // reader: waiting for a free batch (devices or writer are behind)
    double write_busy = 0;         // writer: writing the ZIP
    double write_wait = 0;         // writer: waiting for the next compressed batch
};

// CRC-32 of A followed by B, from crc(A), crc(B) and len(B) (zlib's crc32_combine method).
uint32_t crc_multiply(uint32_t a, uint32_t b) {
    uint32_t m = 1u << 31, p = 0;
    for (;;) {
        if (a & m) {
            p ^= b;
            if ((a & (m - 1)) == 0) break;
        }
        m >>= 1;
        b = (b & 1) ? (b >> 1) ^ 0xEDB88320u : b >> 1;
    }
    return p;
}

uint32_t crc32_combine(uint32_t crc_a, uint32_t crc_b, uint64_t len_b) {
    static const auto powers = [] {  // x^(2^k) mod p, for k = 0..63 (bits)
        std::array<uint32_t, 64> t{};
        uint32_t p = 1u << 30;  // x^1
        for (auto& v : t) {
            v = p;
            p = crc_multiply(p, p);
        }
        return t;
    }();
    uint32_t x = 1u << 31;  // x^0
    unsigned k = 3;         // len_b is in bytes = 2^3 bits
    for (uint64_t n = len_b; n; n >>= 1, k++)
        if (n & 1) x = crc_multiply(powers[k & 63], x);
    return crc_multiply(x, crc_a) ^ crc_b;
}

// Parallel reading: every file is split into pieces of up to 8 MiB. Several threads read and
// checksum pieces at the same time (many disk requests in flight, CRC on many cores), while the
// reader takes the pieces strictly in archive order and only copies them into batches.
constexpr uint64_t kPieceSize = 8ull << 20;
constexpr uint64_t kReadAheadBudget = 512ull << 20;  // memory for pieces read ahead
constexpr uint32_t kReadAheadPieces = 16384;          // how far ahead (in pieces)

class ReadAhead {
public:
    struct Piece {
        uint32_t entry;
        uint64_t offset, size;  // planned size (from the folder listing)
        bool last;              // last piece of its file: read to end of file
    };
    struct Loaded {
        std::vector<uint8_t> data;
        uint32_t crc = 0;
        std::exception_ptr error;
        bool ready = false;
    };

    ReadAhead(const std::vector<Entry>& entries, unsigned threads) : entries_(entries) {
        first_.resize(entries.size() + 1);
        for (uint32_t i = 0; i < entries.size(); i++) {
            first_[i] = uint32_t(pieces_.size());
            if (entries[i].directory) continue;
            uint64_t size = entries[i].size;
            for (uint64_t at = 0;; at += kPieceSize) {
                bool last = size - at <= kPieceSize;
                pieces_.push_back({i, at, last ? size - at : kPieceSize, last});
                if (last) break;
            }
        }
        first_[entries.size()] = uint32_t(pieces_.size());
        slots_.resize(pieces_.size());
        for (unsigned t = 0; t < std::max(1u, threads); t++) threads_.emplace_back([this] { run(); });
    }
    ~ReadAhead() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        changed_.notify_all();
        for (auto& t : threads_) t.join();
    }
    uint32_t first_piece(uint32_t entry) const { return first_[entry]; }
    uint32_t piece_count(uint32_t entry) const { return first_[entry + 1] - first_[entry]; }

    // Waits for a piece (in order) and takes its data.
    Loaded take(uint32_t index) {
        std::unique_lock<std::mutex> lock(mutex_);
        changed_.wait(lock, [&] { return slots_[index].ready || stopping_; });
        if (!slots_[index].ready) throw Cancelled{};
        Loaded out = std::move(slots_[index]);
        slots_[index] = Loaded{};
        in_flight_ -= out.data.size();
        consumed_ = index + 1;
        lock.unlock();
        changed_.notify_all();
        return out;
    }

private:
    void run() {
        for (;;) {
            uint32_t index;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                changed_.wait(lock, [&] {
                    return stopping_ || next_ >= pieces_.size() ||
                           (next_ - consumed_ < kReadAheadPieces && in_flight_ < kReadAheadBudget);
                });
                if (stopping_ || next_ >= pieces_.size()) return;
                index = next_++;
                in_flight_ += pieces_[index].size;
            }
            const Piece& piece = pieces_[index];
            const Entry& entry = entries_[piece.entry];
            Loaded result;
            try {
                File f = File::open_read(entry.path);
                if (piece.offset) f.seek(piece.offset);
                // The last piece reads one byte extra to notice a file that grew since listing.
                result.data.resize(size_t(piece.size) + (piece.last ? 1 : 0));
                size_t got = f.read(result.data.data(), result.data.size());
                while (piece.last && got == result.data.size()) {
                    result.data.resize(result.data.size() * 2 + 4096);
                    got += f.read(result.data.data() + got, result.data.size() - got);
                }
                if (!piece.last && got != piece.size)
                    throw Failure("File changed while it was being archived: " + utf8(entry.path));
                result.data.resize(got);
                result.crc = crc32(0, result.data.data(), got);
            } catch (...) {
                result.error = std::current_exception();
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                in_flight_ += result.data.size();
                in_flight_ -= piece.size;
                result.ready = true;
                slots_[index] = std::move(result);
            }
            changed_.notify_all();
        }
    }

    const std::vector<Entry>& entries_;
    std::vector<Piece> pieces_;
    std::vector<uint32_t> first_;
    std::vector<Loaded> slots_;
    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::condition_variable changed_;
    uint32_t next_ = 0, consumed_ = 0;
    uint64_t in_flight_ = 0;
    bool stopping_ = false;
};

// Stage 1: put every file's data into chunk slots, many files per batch.
void read_stage(Pipeline& p, const std::vector<Entry>& entries, uint32_t capacity, ReadAhead& ahead,
                StageTimes& times) {
    double begin = seconds_now();
    Batch* b = nullptr;
    uint64_t seq = 0;
    auto next_batch = [&](bool first) {
        double t = seconds_now();
        if (!first && !p.filled.push(b)) throw Cancelled{};
        if (!p.free.pop(b)) throw Cancelled{};
        b->seq = seq++;
        times.read_wait_buffers += seconds_now() - t;
    };
    next_batch(true);
    for (uint32_t index = 0; index < entries.size(); index++) {
        if (entries[index].directory) {
            b->segments.push_back({index, b->chunks, 0, true, true, 0, 0});
            continue;
        }
        uint32_t crc = 0;
        uint64_t size = 0;
        bool started = false;
        uint32_t first = ahead.first_piece(index), count = ahead.piece_count(index);
        for (uint32_t k = 0; k < count; k++) {
            double t = seconds_now();
            ReadAhead::Loaded piece = ahead.take(first + k);
            times.read_wait_files += seconds_now() - t;
            if (piece.error) std::rethrow_exception(piece.error);
            crc = k ? crc32_combine(crc, piece.crc, piece.data.size()) : piece.crc;
            size_t total = piece.data.size(), done = 0;
            bool last_piece = k + 1 == count;
            // A piece is 8 MiB; a chunk boundary every 32 KiB from the file start is kept because
            // pieces are multiples of 32 KiB, so copying pieces back to back gives the same chunks.
            do {
                if (b->chunks == capacity) next_batch(false);
                size_t space = size_t(capacity - b->chunks) * BLOCK, got = std::min(space, total - done);
                std::memcpy(b->input.data() + size_t(b->chunks) * BLOCK, piece.data.data() + done, got);
                done += got;
                size += got;
                bool eof = last_piece && done == total;
                uint32_t n = uint32_t((got + BLOCK - 1) / BLOCK);
                for (uint32_t j = 0; j < n; j++)
                    b->lengths[b->chunks + j] = uint32_t(std::min<size_t>(BLOCK, got - size_t(j) * BLOCK));
                if (n || eof) b->segments.push_back({index, b->chunks, n, !started, eof, size, eof ? crc : 0});
                if (n || eof) started = true;
                b->chunks += n;
            } while (done < total);
        }
    }
    b->last = true;
    double t = seconds_now();
    if (!p.filled.push(b)) throw Cancelled{};
    times.read_wait_buffers += seconds_now() - t;
    p.filled.finish();  // lets every encoder thread stop once the queue is drained
    times.read_busy = seconds_now() - begin - times.read_wait_files - times.read_wait_buffers;
}

// Stage 2: one OpenCL dispatch (or CPU thread pool run) per batch.
// With several devices, each runs this loop on its own thread and takes the next batch as
// soon as it is free, so faster devices automatically do more of the work.
void encode_stage(Pipeline& p, Encoder& engine) {
    Batch* b = nullptr;
    while (p.filled.pop(b)) {
        if (b->chunks) {
            double t0 = seconds_now();
            engine.encode(*b);
            engine.busy_seconds += seconds_now() - t0;
            engine.chunks_done += b->chunks;
        }
        if (!p.encoded.push(b)) throw Cancelled{};
    }
}

template <class Fn>
std::thread stage_thread(Pipeline& p, Fn fn) {
    return std::thread([&p, fn] {
        try {
            fn();
        } catch (const Cancelled&) {
        } catch (...) {
            p.fail(std::current_exception());
        }
    });
}

}  // namespace

void create_zip(const std::wstring& source_in, const std::wstring& target_in, const std::string& device) {
    std::wstring source = absolute_path(source_in), target = absolute_path(target_in);
    WIN32_FILE_ATTRIBUTE_DATA info;
    if (!GetFileAttributesExW(long_path(source).c_str(), GetFileExInfoStandard, &info))
        throw Failure("Source not found: " + utf8(source));
    bool source_is_dir = (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
    if (exists(target)) throw Failure("Choose a new archive name");
    if (source_is_dir && is_within(target, source)) throw Failure("Save the archive outside the source folder");

    bool all = device == "ALL", cpu = device == "CPU";
    std::vector<std::unique_ptr<Encoder>> engines;
    if (all) engines = make_all_encoders();
    else engines.push_back(make_encoder(device));
    uint32_t capacity = engines[0]->capacity;
    if (all) {
        // Shared batch size: small enough that every device gets work, within each GPU's buffers.
        capacity = env_chunks("OCLZIP_BATCH_CHUNKS", 512);
        for (auto& e : engines)
            if (e->name.rfind("CPU", 0) != 0) capacity = std::min(capacity, e->capacity);
        std::string names;
        for (auto& e : engines) names += (names.empty() ? "" : " + ") + e->name;
        say("Standard ZIP / ALL DEVICES: " + names);
        say("Every device takes the next batch as soon as it is free; batches are written back in order.");
    } else if (cpu) {
        say("Standard ZIP / CPU DEFLATE: " + engines[0]->name);
        say("CPU: same fixed-Huffman encoder as the GPU kernel, plus file I/O, CRC and ZIP headers.");
    } else {
        say("Standard ZIP / GPU DEFLATE: " + engines[0]->name);
        say("GPU: match search and Huffman encoding. CPU: file I/O, CRC and ZIP headers.");
    }
    say("Batches of up to " + commas(capacity) + " chunks (" + commas(capacity / 32) +
        " MiB) packed across files; reading, encoding and writing run in parallel.");

    check_link(source, info.dwFileAttributes);
    std::vector<Entry> entries;
    std::string root = utf8(name_of(source));
    entries.push_back({source, root + (source_is_dir ? "/" : ""), source_is_dir, info.ftLastWriteTime,
                       (uint64_t(info.nFileSizeHigh) << 32) | info.nFileSizeLow});
    if (source_is_dir) walk(source, root + "/", entries);

    std::vector<Record> records;
    StageTimes times;
    uint64_t total = 0, count = 0;
    double start = seconds_now();
    {
        TempDir temp(parent_of(target), L".gpu-zip-");
        std::wstring staged = join(temp.path(), L"archive.zip");
        {
            File out = File::create_new(staged);
            // One batch being read, one being written, and up to two queued per encoder.
            int depth = engines.size() == 1 ? PIPELINE_DEPTH : 2 + 2 * int(engines.size());
            Pipeline pipeline(capacity, depth);
            ReadAhead ahead(entries, env_chunks("OCLZIP_READ_THREADS", 8));
            std::thread reader =
                stage_thread(pipeline, [&] { read_stage(pipeline, entries, capacity, ahead, times); });
            std::vector<std::thread> encoders;
            for (auto& e : engines) {
                Encoder* engine = e.get();
                encoders.push_back(stage_thread(pipeline, [&pipeline, engine] { encode_stage(pipeline, *engine); }));
            }

            // Stage 3 (this thread): write entries in order as their compressed chunks arrive.
            try {
                uint64_t offset = 0, payload_start = 0;
                uint16_t clock = 0, date = 0;
                std::map<uint64_t, Batch*> waiting;  // finished early, waiting for their turn
                uint64_t next = 0;
                double write_begin = seconds_now();
                for (bool last = false; !last;) {
                    Batch* b = nullptr;
                    auto ready = waiting.find(next);
                    if (ready != waiting.end()) {
                        b = ready->second;
                        waiting.erase(ready);
                    } else {
                        double t = seconds_now();
                        bool got = pipeline.encoded.pop(b);
                        times.write_wait += seconds_now() - t;
                        if (!got) break;
                        if (b->seq != next) {
                            waiting[b->seq] = b;
                            continue;
                        }
                    }
                    next++;
                    for (const Segment& s : b->segments) {
                        const Entry& e = entries[s.entry];
                        // A whole entry inside this batch (most files): its CRC and sizes are already
                        // known, so the header is written complete and needs no seek back later.
                        bool whole = s.begins && s.ends;
                        uint64_t whole_compressed = 0;
                        if (whole) {
                            for (uint32_t i = s.first; i < s.first + s.chunks; i++) whole_compressed += b->sizes[i];
                            if (!e.directory) whole_compressed += 5;
                        }
                        if (s.begins) {
                            if (e.name.size() > 65535) throw Failure("Archive path is too long");
                            dos_time(e.modified, clock, date);
                            offset = out.tell();
                            std::vector<uint8_t> h;
                            put32(h, 0x04034b50); put16(h, 45); put16(h, 0x800); put16(h, e.directory ? 0 : 8);
                            put16(h, clock); put16(h, date); put32(h, whole ? s.crc : 0); put32(h, 0xffffffff);
                            put32(h, 0xffffffff); put16(h, uint32_t(e.name.size())); put16(h, 20);
                            h.insert(h.end(), e.name.begin(), e.name.end());
                            put16(h, 1); put16(h, 16);  // ZIP64 extra; patched below for multi-batch files
                            put64(h, whole ? s.file_size : 0); put64(h, whole_compressed);
                            out.write(h.data(), h.size());
                            payload_start = out.tell();
                        }
                        for (uint32_t i = s.first; i < s.first + s.chunks; i++)
                            out.write(b->output.data() + b->offsets[i], b->sizes[i]);
                        if (!s.ends) continue;
                        if (!e.directory) {
                            // Final byte-aligned empty stored block terminates this raw DEFLATE stream.
                            static const uint8_t final_block[5] = {1, 0, 0, 0xff, 0xff};
                            out.write(final_block, 5);
                            count++;
                        }
                        uint64_t end = out.tell(), compressed = end - payload_start;
                        if (whole && compressed != whole_compressed) throw Failure("Internal size mismatch");
                        if (!whole) {
                            std::vector<uint8_t> patch;
                            put32(patch, s.crc);
                            out.seek(offset + 14);
                            out.write(patch.data(), 4);
                            patch.clear();
                            put64(patch, s.file_size);
                            put64(patch, compressed);
                            out.seek(offset + 30 + e.name.size() + 4);
                            out.write(patch.data(), 16);
                            out.seek(end);
                        }
                        records.push_back({e.name, uint16_t(e.directory ? 0 : 8), clock, date, s.crc, s.file_size,
                                           compressed, offset, e.directory});
                        total += s.file_size;
                        say("Packed: " + utf8(name_of(e.path)) + " (" + commas(s.file_size) + " bytes)");
                    }
                    last = b->last;
                    b->reset();
                    if (!last && !pipeline.free.push(b)) break;
                }
                times.write_busy = seconds_now() - write_begin - times.write_wait;
            } catch (...) {
                pipeline.fail(std::current_exception());
            }
            reader.join();
            for (auto& t : encoders) t.join();
            pipeline.rethrow();
            if (records.size() != entries.size()) throw Failure("Compression stopped before all files were written");

            uint64_t central_offset = out.tell();
            for (auto& r : records) {
                std::vector<uint8_t> c;
                uint32_t attr = ((r.directory ? 040755u : 0100644u) << 16) | (r.directory ? 16u : 0u);
                put32(c, 0x02014b50); put16(c, 0x032d); put16(c, 45); put16(c, 0x800); put16(c, r.method);
                put16(c, r.clock); put16(c, r.date); put32(c, r.crc); put32(c, 0xffffffff); put32(c, 0xffffffff);
                put16(c, uint32_t(r.name.size())); put16(c, 28); put16(c, 0); put16(c, 0); put16(c, 0);
                put32(c, attr); put32(c, 0xffffffff);
                c.insert(c.end(), r.name.begin(), r.name.end());
                put16(c, 1); put16(c, 24); put64(c, r.size); put64(c, r.compressed); put64(c, r.offset);
                out.write(c.data(), c.size());
            }
            uint64_t end64 = out.tell(), central_size = end64 - central_offset, n = records.size();
            std::vector<uint8_t> t;
            put32(t, 0x06064b50); put64(t, 44); put16(t, 45); put16(t, 45); put32(t, 0); put32(t, 0);
            put64(t, n); put64(t, n); put64(t, central_size); put64(t, central_offset);
            put32(t, 0x07064b50); put32(t, 0); put64(t, end64); put32(t, 1);
            put32(t, 0x06054b50); put16(t, 0); put16(t, 0); put16(t, 0xffff); put16(t, 0xffff);
            put32(t, 0xffffffff); put32(t, 0xffffffff); put16(t, 0);
            out.write(t.data(), t.size());
            out.close();
        }
        move_new(staged, target);
    }
    uint64_t size = File::open_read(target).size();
    say(commas(count) + " files; " + commas(total) + " source bytes -> " + commas(size) + " ZIP bytes");
    if (size >= total)
        say("NO SPACE SAVED: ZIP is " + commas(size - total) + " bytes larger.");
    else
        say("Space saved: " + percent(double(total - size) / double(total)));
    char elapsed[64];
    double wall = seconds_now() - start;
    std::snprintf(elapsed, sizeof elapsed, "%.3fs", wall);
    if (all) {
        say(std::string("Elapsed after setup: ") + elapsed + ". Work split:");
        uint64_t chunks = 0;
        for (auto& e : engines) chunks += e->chunks_done;
        for (auto& e : engines) {
            char line[256];
            std::snprintf(line, sizeof line, "  %-45s %5zu batches, %5.1f%% of chunks, busy %.3fs", e->name.c_str(),
                          e->dispatches, chunks ? 100.0 * double(e->chunks_done) / double(chunks) : 0.0,
                          e->busy_seconds);
            say(line);
        }
    } else {
        Encoder* engine = engines[0].get();
        say(std::string(cpu ? "CPU encoder batches: " : "OpenCL kernel batches: ") +
            std::to_string(engine->dispatches) + "; elapsed after " + (cpu ? "setup" : "GPU setup") + ": " + elapsed);
        char busy[128];
        std::snprintf(busy, sizeof busy, "%s busy (upload + compress + download): %.3fs of %.3fs",
                      cpu ? "Encoder" : "GPU", engine->busy_seconds, wall);
        say(busy);
    }
    // Where did the time go? Each stage runs on its own thread, so these overlap.
    char line[256];
    std::snprintf(line, sizeof line,
                  "Reader: working %.3fs, waiting for disk %.3fs, waiting for free buffers %.3fs",
                  times.read_busy, times.read_wait_files, times.read_wait_buffers);
    say(line);
    std::snprintf(line, sizeof line, "Writer: writing %.3fs, waiting for compressed data %.3fs", times.write_busy,
                  times.write_wait);
    say(line);
    const char* limit;
    if (times.read_wait_buffers > 0.4 * wall)
        limit = times.write_busy > 0.5 * wall ? "writing the ZIP to disk" : "compression (the devices are the bottleneck)";
    else if (times.read_wait_files > 0.3 * wall)
        limit = "reading the source files from disk (the devices were waiting for data)";
    else if (times.read_busy > 0.5 * wall)
        limit = "the reader thread (copying data into batches); the devices were waiting for data";
    else
        limit = "no single stage (well balanced)";
    say(std::string("Limited by: ") + limit);
}
