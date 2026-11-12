#include "ocl.h"

#include "common.h"

#include <cstring>

namespace ocl {
namespace {

using BuildNotify = void(__stdcall*)(cl_program, void*);
using ContextNotify = void(__stdcall*)(const char*, const void*, size_t, void*);

#define OCL_FUNCTIONS(X)                                                                                      \
    X(clGetPlatformIDs, cl_int, (cl_uint, cl_platform_id*, cl_uint*))                                         \
    X(clGetPlatformInfo, cl_int, (cl_platform_id, cl_uint, size_t, void*, size_t*))                           \
    X(clGetDeviceIDs, cl_int, (cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*))             \
    X(clGetDeviceInfo, cl_int, (cl_device_id, cl_uint, size_t, void*, size_t*))                               \
    X(clCreateContext, cl_context,                                                                            \
      (const cl_context_properties*, cl_uint, const cl_device_id*, ContextNotify, void*, cl_int*))            \
    X(clCreateCommandQueue, cl_command_queue, (cl_context, cl_device_id, cl_command_queue_properties, cl_int*)) \
    X(clCreateProgramWithSource, cl_program, (cl_context, cl_uint, const char**, const size_t*, cl_int*))     \
    X(clBuildProgram, cl_int, (cl_program, cl_uint, const cl_device_id*, const char*, BuildNotify, void*))    \
    X(clGetProgramBuildInfo, cl_int, (cl_program, cl_device_id, cl_uint, size_t, void*, size_t*))             \
    X(clCreateKernel, cl_kernel, (cl_program, const char*, cl_int*))                                          \
    X(clSetKernelArg, cl_int, (cl_kernel, cl_uint, size_t, const void*))                                      \
    X(clCreateBuffer, cl_mem, (cl_context, cl_mem_flags, size_t, void*, cl_int*))                             \
    X(clEnqueueWriteBuffer, cl_int,                                                                           \
      (cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*))  \
    X(clEnqueueReadBuffer, cl_int,                                                                            \
      (cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*))        \
    X(clEnqueueNDRangeKernel, cl_int,                                                                         \
      (cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint,            \
       const cl_event*, cl_event*))                                                                           \
    X(clFinish, cl_int, (cl_command_queue))                                                                   \
    X(clReleaseMemObject, cl_int, (cl_mem))                                                                   \
    X(clReleaseKernel, cl_int, (cl_kernel))                                                                   \
    X(clReleaseProgram, cl_int, (cl_program))                                                                 \
    X(clReleaseCommandQueue, cl_int, (cl_command_queue))                                                      \
    X(clReleaseContext, cl_int, (cl_context))

struct Api {
#define OCL_DECLARE(name, ret, args) ret(__stdcall* name) args = nullptr;
    OCL_FUNCTIONS(OCL_DECLARE)
#undef OCL_DECLARE
    std::string error;
};

const Api& api() {
    static const Api loaded = [] {
        Api a;
        HMODULE dll = LoadLibraryW(L"OpenCL.dll");
        if (!dll) {
            a.error = "OpenCL.dll was not found. Install a GPU driver with OpenCL support.";
            return a;
        }
#define OCL_LOAD(name, ret, args)                                                       \
    a.name = reinterpret_cast<decltype(a.name)>(GetProcAddress(dll, #name));            \
    if (!a.name && a.error.empty()) a.error = "OpenCL.dll does not export " #name;
        OCL_FUNCTIONS(OCL_LOAD)
#undef OCL_LOAD
        return a;
    }();
    if (!loaded.error.empty()) throw Failure(loaded.error);
    return loaded;
}

std::string code_name(cl_int code) {
    switch (code) {
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -11: return "CL_BUILD_PROGRAM_FAILURE";
        case -30: return "CL_INVALID_VALUE";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -48: return "CL_INVALID_KERNEL";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -61: return "CL_INVALID_BUFFER_SIZE";
        case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
        default: return "OpenCL error " + std::to_string(code);
    }
}

void check(cl_int code, const char* what) {
    if (code != CL_SUCCESS) throw Failure(std::string(what) + " failed: " + code_name(code));
}

std::string platform_string(cl_platform_id p, cl_uint param) {
    size_t n = 0;
    if (api().clGetPlatformInfo(p, param, 0, nullptr, &n) != CL_SUCCESS || !n) return {};
    std::string s(n, '\0');
    api().clGetPlatformInfo(p, param, n, s.data(), nullptr);
    while (!s.empty() && (s.back() == '\0' || s.back() == ' ')) s.pop_back();
    return s;
}

std::string device_string(cl_device_id d, cl_uint param) {
    size_t n = 0;
    if (api().clGetDeviceInfo(d, param, 0, nullptr, &n) != CL_SUCCESS || !n) return {};
    std::string s(n, '\0');
    api().clGetDeviceInfo(d, param, n, s.data(), nullptr);
    while (!s.empty() && (s.back() == '\0' || s.back() == ' ')) s.pop_back();
    size_t start = s.find_first_not_of(' ');
    return start == std::string::npos ? std::string() : s.substr(start);
}

template <class T>
T device_value(cl_device_id d, cl_uint param) {
    T value{};
    api().clGetDeviceInfo(d, param, sizeof value, &value, nullptr);
    return value;
}

}  // namespace

std::vector<Device> devices(cl_device_type type, std::vector<std::string>* notices) {
    const Api& cl = api();
    cl_uint count = 0;
    cl_int status = cl.clGetPlatformIDs(0, nullptr, &count);
    if (status == -1001 || count == 0) return {};
    check(status, "clGetPlatformIDs");
    std::vector<cl_platform_id> platforms(count);
    check(cl.clGetPlatformIDs(count, platforms.data(), nullptr), "clGetPlatformIDs");

    std::vector<Device> found;
    for (cl_platform_id p : platforms) {
        std::string platform_name = platform_string(p, CL_PLATFORM_NAME);
        cl_uint n = 0;
        status = cl.clGetDeviceIDs(p, type, 0, nullptr, &n);
        if (status == CL_DEVICE_NOT_FOUND || (status == CL_SUCCESS && n == 0)) continue;
        if (status != CL_SUCCESS) {
            if (notices) notices->push_back("Could not query OpenCL devices on " + platform_name + ".");
            continue;
        }
        std::vector<cl_device_id> ids(n);
        if (cl.clGetDeviceIDs(p, type, n, ids.data(), nullptr) != CL_SUCCESS) {
            if (notices) notices->push_back("Could not query OpenCL devices on " + platform_name + ".");
            continue;
        }
        for (cl_device_id id : ids) {
            Device d;
            d.id = id;
            d.platform = p;
            d.platform_name = platform_name;
            d.name = device_string(id, CL_DEVICE_NAME);
            d.type = device_value<cl_device_type>(id, CL_DEVICE_TYPE);
            d.available = device_value<cl_bool>(id, CL_DEVICE_AVAILABLE) != 0;
            d.compiler = device_value<cl_bool>(id, CL_DEVICE_COMPILER_AVAILABLE) != 0;
            d.unified = device_value<cl_bool>(id, CL_DEVICE_HOST_UNIFIED_MEMORY) != 0;
            d.global_mem = device_value<cl_ulong>(id, CL_DEVICE_GLOBAL_MEM_SIZE);
            d.max_alloc = device_value<cl_ulong>(id, CL_DEVICE_MAX_MEM_ALLOC_SIZE);
            found.push_back(std::move(d));
        }
    }
    return found;
}

Runtime::Runtime(const Device& device, const char* source) : device_(device) {
    const Api& cl = api();
    cl_int status = CL_SUCCESS;
    context_ = cl.clCreateContext(nullptr, 1, &device_.id, nullptr, nullptr, &status);
    check(status, "clCreateContext");
    queue_ = cl.clCreateCommandQueue(context_, device_.id, 0, &status);
    if (status != CL_SUCCESS) {
        cl.clReleaseContext(context_);
        check(status, "clCreateCommandQueue");
    }
    size_t length = std::strlen(source);
    program_ = cl.clCreateProgramWithSource(context_, 1, &source, &length, &status);
    if (status != CL_SUCCESS) {
        release();
        check(status, "clCreateProgramWithSource");
    }
    status = cl.clBuildProgram(program_, 1, &device_.id, "", nullptr, nullptr);
    if (status != CL_SUCCESS) {
        size_t n = 0;
        cl.clGetProgramBuildInfo(program_, device_.id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
        std::string log(n, '\0');
        if (n) cl.clGetProgramBuildInfo(program_, device_.id, CL_PROGRAM_BUILD_LOG, n, log.data(), nullptr);
        release();
        throw Failure("OpenCL kernel build failed on " + device_.name + ":\n" + log);
    }
}

Runtime::~Runtime() { release(); }

void Runtime::release() {
    const Api& cl = api();
    for (cl_kernel k : kernels_) cl.clReleaseKernel(k);
    kernels_.clear();
    if (program_) cl.clReleaseProgram(program_);
    if (queue_) {
        cl.clFinish(queue_);
        cl.clReleaseCommandQueue(queue_);
    }
    if (context_) cl.clReleaseContext(context_);
    program_ = nullptr;
    queue_ = nullptr;
    context_ = nullptr;
}

cl_kernel Runtime::kernel(const char* name) {
    cl_int status = CL_SUCCESS;
    cl_kernel k = api().clCreateKernel(program_, name, &status);
    check(status, "clCreateKernel");
    kernels_.push_back(k);
    return k;
}

void Runtime::set_arg(cl_kernel k, cl_uint index, const Buffer& buffer) {
    cl_mem mem = buffer.handle();
    check(api().clSetKernelArg(k, index, sizeof mem, &mem), "clSetKernelArg");
}

void Runtime::set_arg(cl_kernel k, cl_uint index, cl_uint value) {
    check(api().clSetKernelArg(k, index, sizeof value, &value), "clSetKernelArg");
}

void Runtime::run(cl_kernel k, size_t global_size, size_t global_offset, size_t local_size) {
    check(api().clEnqueueNDRangeKernel(queue_, k, 1, global_offset ? &global_offset : nullptr, &global_size,
                                       local_size ? &local_size : nullptr, 0, nullptr, nullptr),
          "clEnqueueNDRangeKernel");
}

void Runtime::write(const Buffer& buffer, const void* data, size_t size) {
    if (size) check(api().clEnqueueWriteBuffer(queue_, buffer.handle(), CL_TRUE, 0, size, data, 0, nullptr, nullptr),
                    "clEnqueueWriteBuffer");
}

void Runtime::read(const Buffer& buffer, void* data, size_t size) {
    if (size) check(api().clEnqueueReadBuffer(queue_, buffer.handle(), CL_TRUE, 0, size, data, 0, nullptr, nullptr),
                    "clEnqueueReadBuffer");
}

Buffer::Buffer(Runtime& runtime, cl_mem_flags flags, size_t size) : size_(size) {
    cl_int status = CL_SUCCESS;
    mem_ = api().clCreateBuffer(runtime.context(), flags, size, nullptr, &status);
    check(status, "clCreateBuffer");
}

Buffer::~Buffer() {
    if (mem_) api().clReleaseMemObject(mem_);
}

}  // namespace ocl
