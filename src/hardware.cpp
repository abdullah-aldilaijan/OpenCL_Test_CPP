// Port of hardware.py: CPU model names plus usable OpenCL GPUs for the device dropdown.
#include <algorithm>
#include <set>

#include "common.h"
#include "ocl.h"

static std::vector<std::string> cpu_names() {
    std::vector<std::string> names;
    std::set<std::string> seen;
    for (int index = 0; index < 1024; index++) {
        std::wstring key = L"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\" + std::to_wstring(index);
        wchar_t value[256];
        DWORD size = sizeof value;
        if (RegGetValueW(HKEY_LOCAL_MACHINE, key.c_str(), L"ProcessorNameString", RRF_RT_REG_SZ, nullptr, value,
                         &size) != ERROR_SUCCESS)
            break;
        std::string name = utf8(value);
        size_t a = name.find_first_not_of(' '), b = name.find_last_not_of(' ');
        if (a == std::string::npos) continue;
        name = name.substr(a, b - a + 1);
        if (seen.insert(name).second) names.push_back(name);
    }
    return names;
}

std::vector<DeviceOption> discover_devices(std::vector<std::string>& notices) {
    std::vector<std::string> cpus = cpu_names();
    if (cpus.empty()) {
        notices.push_back("CPU model lookup unavailable; using system CPU information.");
        cpus.push_back("System processor");
    }
    std::vector<ocl::Device> gpus;
    try {
        for (auto& d : ocl::devices(ocl::CL_DEVICE_TYPE_GPU, &notices))
            if (d.available && d.compiler) gpus.push_back(d);
        std::stable_sort(gpus.begin(), gpus.end(), [](const ocl::Device& a, const ocl::Device& b) {
            if (a.unified != b.unified) return !a.unified;  // discrete GPUs first
            return a.global_mem > b.global_mem;
        });
    } catch (const std::exception&) {
        notices.push_back("No usable OpenCL GPU runtime found. CPU mode remains available.");
    }
    std::vector<DeviceOption> options;
    auto has_label = [&](const std::string& label) {
        return std::any_of(options.begin(), options.end(), [&](const DeviceOption& o) { return o.label == label; });
    };
    for (auto& name : cpus) options.push_back({"CPU \xE2\x80\x94 " + name, "CPU"});
    for (auto& d : gpus) {
        std::string label = "GPU \xE2\x80\x94 " + d.name;
        if (has_label(label)) label += " (" + d.platform_name + ")";
        options.push_back({label, d.name});
    }
    // Extra option: the CPU and every GPU compress the same archive together.
    if (!gpus.empty())
        options.push_back({std::string("All \xE2\x80\x94 CPU + ") + (gpus.size() == 1 ? "GPU" : "all GPUs") +
                               " together",
                           "ALL"});
    if (gpus.empty() && notices.empty())
        notices.push_back("No OpenCL GPU with compiler support was found. CPU mode remains available.");
    return options;
}
