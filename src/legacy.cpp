// Restores the earlier archive formats:
//   OCLLZ001  (opencl_lz.py, 64 KiB LZ blocks)        file.oclr
//   OCLRLE01  (opencl_compress.py, 4 KiB RLE blocks)  file.oclr
//   *.folder.oclr (gpu_folder.py: stored ZIP wrapped in one of the above) -> folder
// Payloads are validated on the CPU first, then decoded on the selected OpenCL device
// (or on the CPU when the device is "CPU").
#include <algorithm>
#include <cstring>
#include <memory>
#include <tuple>

#include "common.h"
#include "kernels.h"
#include "ocl.h"

namespace {

struct Format {
    bool lz;
    uint32_t block, batch;
};

// Validates an OCLLZ001 payload and optionally writes the decoded bytes to `out`.
void parse_lz(const uint8_t* payload, size_t size, uint32_t length, uint8_t* out) {
    size_t p = 0;
    uint32_t o = 0;
    while (o < length) {
        if (p >= size) throw Failure("Truncated LZ flags");
        uint8_t flags = payload[p++];
        for (int bit = 0; bit < 8 && o < length; bit++) {
            if (flags & (1 << bit)) {
                if (p + 3 > size) throw Failure("Truncated LZ reference");
                uint32_t distance = payload[p] | (uint32_t(payload[p + 1]) << 8), run = payload[p + 2] + 3u;
                p += 3;
                if (distance == 0 || distance > o || o + run > length) throw Failure("Invalid LZ reference");
                if (out)
                    for (uint32_t j = 0; j < run; j++, o++) out[o] = out[o - distance];
                else
                    o += run;
            } else {
                if (p >= size) throw Failure("Truncated LZ literal");
                if (out) out[o] = payload[p];
                p++;
                o++;
            }
        }
    }
    if (p != size) throw Failure("Trailing LZ payload");
}

void check_rle(const uint8_t* payload, size_t size, uint32_t length) {
    if (size % 2) throw Failure("Invalid RLE payload");
    uint64_t sum = 0;
    for (size_t i = 0; i < size; i += 2) {
        if (!payload[i]) throw Failure("Invalid RLE payload");
        sum += payload[i];
    }
    if (sum != length) throw Failure("Invalid RLE payload");
}

void decode_rle(const uint8_t* payload, size_t size, uint8_t* out) {
    for (size_t i = 0, o = 0; i < size; i += 2) {
        std::memset(out + o, payload[i + 1], payload[i]);
        o += payload[i];
    }
}

class Decoder {
public:
    Decoder(const std::string& device, const Format& format) : format_(format) {
        if (device == "CPU") {
            name = "CPU";
            return;
        }
        // "ALL" only applies to ZIP compression; decoding uses the single best device.
        std::string want = device == "ALL" ? std::string() : utf8(fold_case(wide(device)));
        std::vector<ocl::Device> all = ocl::devices(ocl::CL_DEVICE_TYPE_ALL);
        const ocl::Device* best = nullptr;
        for (auto& d : all) {
            if (utf8(fold_case(wide(d.name))).find(want) == std::string::npos) continue;
            auto key = [](const ocl::Device& x) { return std::make_tuple(x.is_gpu(), !x.unified, x.global_mem); };
            if (!best || key(d) > key(*best)) best = &d;
        }
        if (!best) throw Failure("No matching OpenCL device");
        name = best->name;
        runtime_ = std::make_unique<ocl::Runtime>(*best, kLegacyKernels);
        kernel_ = runtime_->kernel(format.lz ? "lz_decode" : "rle_decode");
        src_ = std::make_unique<ocl::Buffer>(*runtime_, ocl::CL_MEM_READ_ONLY, size_t(format.block) * 2 * format.batch);
        dst_ = std::make_unique<ocl::Buffer>(*runtime_, ocl::CL_MEM_WRITE_ONLY, size_t(format.block) * format.batch);
        sizes_ = std::make_unique<ocl::Buffer>(*runtime_, ocl::CL_MEM_READ_ONLY, 4 * format.batch);
        lengths_ = std::make_unique<ocl::Buffer>(*runtime_, ocl::CL_MEM_READ_ONLY, 4 * format.batch);
        runtime_->set_arg(kernel_, 0, *src_);
        runtime_->set_arg(kernel_, 1, *dst_);
        runtime_->set_arg(kernel_, 2, *sizes_);
        runtime_->set_arg(kernel_, 3, *lengths_);
        runtime_->set_arg(kernel_, 4, format.block);
    }

    // data: n slots of block*2 bytes; out: n slots of block bytes.
    void run(const std::vector<uint8_t>& data, const std::vector<uint32_t>& sizes,
             const std::vector<uint32_t>& lengths, std::vector<uint8_t>& out) {
        size_t n = lengths.size(), block = format_.block;
        if (!runtime_) {
            for (size_t i = 0; i < n; i++) {
                const uint8_t* in = data.data() + i * block * 2;
                uint8_t* dst = out.data() + i * block;
                if (sizes[i] == lengths[i]) std::memcpy(dst, in, lengths[i]);
                else if (format_.lz) parse_lz(in, sizes[i], lengths[i], dst);
                else decode_rle(in, sizes[i], dst);
            }
            return;
        }
        runtime_->write(*src_, data.data(), n * block * 2);
        runtime_->write(*sizes_, sizes.data(), 4 * n);
        runtime_->write(*lengths_, lengths.data(), 4 * n);
        runtime_->run(kernel_, n);
        runtime_->read(*dst_, out.data(), n * block);
    }

    std::string name;

private:
    Format format_;
    std::unique_ptr<ocl::Runtime> runtime_;
    ocl::cl_kernel kernel_ = nullptr;
    std::unique_ptr<ocl::Buffer> src_, dst_, sizes_, lengths_;
};

Format read_format(File& src) {
    uint8_t magic[8];
    src.read_exact(magic, 8);
    if (!std::memcmp(magic, "OCLLZ001", 8)) return {true, 65536, 256};
    if (!std::memcmp(magic, "OCLRLE01", 8)) return {false, 4096, 1024};
    throw Failure("Not an OCLLZ001 or OCLRLE01 archive");
}

void decode_file(const std::wstring& source, const std::wstring& target, const std::string& device,
                 const char* device_label) {
    File src = File::open_read(source);
    Format format = read_format(src);
    Decoder decoder(device, format);
    say(std::string(device_label) + decoder.name);
    double start = seconds_now();
    uint8_t head[12];
    src.read_exact(head, 12);
    if (get32(head) != format.block) throw Failure("Unsupported block size");
    uint64_t total = get64(head + 4), seen = 0;
    File dst = File::create_new(target);
    Sha256 digest;
    std::vector<uint8_t> data(size_t(format.block) * 2 * format.batch), out(size_t(format.block) * format.batch);
    std::vector<uint32_t> sizes, lengths;
    for (;;) {
        uint8_t word[8];
        src.read_exact(word, 4);
        uint32_t n = get32(word);
        if (n == 0) break;
        if (n > format.batch) throw Failure("Invalid batch count");
        sizes.assign(n, 0);
        lengths.assign(n, 0);
        for (uint32_t i = 0; i < n; i++) {
            src.read_exact(word, 8);
            uint32_t length = get32(word), size = get32(word + 4);
            if (!(1 <= size && size <= length && length <= format.block) || seen + length > total)
                throw Failure("Invalid block lengths");
            uint8_t* slot = data.data() + size_t(i) * format.block * 2;
            src.read_exact(slot, size);
            if (size < length) {
                if (format.lz) parse_lz(slot, size, length, nullptr);
                else check_rle(slot, size, length);
            }
            lengths[i] = length;
            sizes[i] = size;
            seen += length;
        }
        decoder.run(data, sizes, lengths, out);
        for (uint32_t i = 0; i < n; i++) {
            const uint8_t* piece = out.data() + size_t(i) * format.block;
            digest.update(piece, lengths[i]);
            dst.write(piece, lengths[i]);
        }
    }
    uint8_t expected[32], extra;
    src.read_exact(expected, 32);
    auto actual = digest.digest();
    if (seen != total || std::memcmp(actual.data(), expected, 32) || src.read(&extra, 1))
        throw Failure("Integrity verification failed");
    dst.close();
    char elapsed[64];
    std::snprintf(elapsed, sizeof elapsed, "%.3fs", seconds_now() - start);
    say(commas(src.size()) + " -> " + commas(total) + " bytes");
    say(std::string("Elapsed: ") + elapsed + " (excludes OpenCL setup/compilation)");
    say(format.lz ? "SHA-256 integrity check: PASS" : "Legacy RLE archive restored; SHA-256: PASS");
}

bool ends_with_folder_oclr(const std::wstring& path) {
    std::wstring lower = fold_case(path);
    const std::wstring suffix = L".folder.oclr";
    return lower.size() >= suffix.size() && lower.compare(lower.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}  // namespace

void restore_legacy(const std::wstring& source_in, const std::wstring& target_in, const std::string& device) {
    std::wstring source = absolute_path(source_in), target = absolute_path(target_in);
    if (exists(target)) throw Failure("Choose a new output name");
    TempDir temp(parent_of(target), L".gpu-folder-");
    if (ends_with_folder_oclr(source)) {
        say("CPU handles packaging and file I/O. Archive decoding uses the selected device.");
        std::wstring bundle = join(temp.path(), L"folder.zip");
        decode_file(source, bundle, device, "Folder archive processing device: ");
        say("Decompression SHA-256 check: PASS");
        restore_zip(bundle, target);
    } else {
        std::wstring staged = join(temp.path(), L"restored.bin");
        decode_file(source, staged, device, "Device: ");
        move_new(staged, target);
    }
}
