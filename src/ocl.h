// Minimal OpenCL 1.2 host API, loaded at runtime from OpenCL.dll (the ICD loader that GPU
// drivers install). No OpenCL SDK is needed to build or run.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ocl {

using cl_int = int32_t;
using cl_uint = uint32_t;
using cl_ulong = uint64_t;
using cl_bool = cl_uint;
using cl_bitfield = cl_ulong;
using cl_device_type = cl_bitfield;
using cl_mem_flags = cl_bitfield;
using cl_command_queue_properties = cl_bitfield;
using cl_context_properties = intptr_t;
using cl_platform_id = struct _cl_platform_id*;
using cl_device_id = struct _cl_device_id*;
using cl_context = struct _cl_context*;
using cl_command_queue = struct _cl_command_queue*;
using cl_mem = struct _cl_mem*;
using cl_program = struct _cl_program*;
using cl_kernel = struct _cl_kernel*;
using cl_event = struct _cl_event*;

constexpr cl_int CL_SUCCESS = 0;
constexpr cl_int CL_DEVICE_NOT_FOUND = -1;
constexpr cl_bool CL_TRUE = 1;
constexpr cl_device_type CL_DEVICE_TYPE_GPU = 1 << 2;
constexpr cl_device_type CL_DEVICE_TYPE_ALL = 0xFFFFFFFF;
constexpr cl_uint CL_PLATFORM_NAME = 0x0902;
constexpr cl_uint CL_DEVICE_TYPE = 0x1000;
constexpr cl_uint CL_DEVICE_MAX_MEM_ALLOC_SIZE = 0x1010;
constexpr cl_uint CL_DEVICE_GLOBAL_MEM_SIZE = 0x101F;
constexpr cl_uint CL_DEVICE_AVAILABLE = 0x1027;
constexpr cl_uint CL_DEVICE_COMPILER_AVAILABLE = 0x1028;
constexpr cl_uint CL_DEVICE_NAME = 0x102B;
constexpr cl_uint CL_DEVICE_HOST_UNIFIED_MEMORY = 0x1035;
constexpr cl_uint CL_PROGRAM_BUILD_LOG = 0x1183;
constexpr cl_mem_flags CL_MEM_READ_WRITE = 1 << 0;
constexpr cl_mem_flags CL_MEM_WRITE_ONLY = 1 << 1;
constexpr cl_mem_flags CL_MEM_READ_ONLY = 1 << 2;

struct Device {
    cl_device_id id = nullptr;
    cl_platform_id platform = nullptr;
    std::string name, platform_name;
    cl_device_type type = 0;
    bool available = false, compiler = false, unified = false;
    uint64_t global_mem = 0, max_alloc = 0;
    bool is_gpu() const { return (type & CL_DEVICE_TYPE_GPU) != 0; }
};

// Throws if OpenCL.dll is missing. Platforms that fail to report devices add a notice.
std::vector<Device> devices(cl_device_type type, std::vector<std::string>* notices = nullptr);

class Buffer;

// One device with its own context, in-order queue and compiled program.
class Runtime {
public:
    Runtime(const Device& device, const char* source);
    ~Runtime();
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    cl_kernel kernel(const char* name);
    void set_arg(cl_kernel k, cl_uint index, const Buffer& buffer);
    void set_arg(cl_kernel k, cl_uint index, cl_uint value);
    void run(cl_kernel k, size_t global_size, size_t global_offset = 0, size_t local_size = 0);
    void write(const Buffer& buffer, const void* data, size_t size);
    void read(const Buffer& buffer, void* data, size_t size);
    cl_context context() const { return context_; }
    void release();
    const Device& device() const { return device_; }

private:
    Device device_;
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_program program_ = nullptr;
    std::vector<cl_kernel> kernels_;
};

class Buffer {
public:
    Buffer(Runtime& runtime, cl_mem_flags flags, size_t size);
    ~Buffer();
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    cl_mem handle() const { return mem_; }
    size_t size() const { return size_; }

private:
    cl_mem mem_ = nullptr;
    size_t size_ = 0;
};

}  // namespace ocl
