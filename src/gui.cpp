// Native Win32 desktop window (replaces desktop_launcher.py / tkinter).
#include "common.h"  // must come first: brings in <windows.h>

#include <commctrl.h>
#include <shobjidl.h>

#include <atomic>
#include <thread>

namespace {

enum : int { ID_DEVICE = 100, ID_REFRESH, ID_COMPRESS_FILE, ID_COMPRESS_FOLDER, ID_RESTORE, ID_LOG };
enum : UINT { WM_APP_LOG = WM_APP + 1, WM_APP_DONE };

HWND g_window, g_title, g_intro, g_device_label, g_device, g_refresh, g_buttons[3], g_log, g_status;
HFONT g_font, g_title_font;
std::vector<DeviceOption> g_options;
std::atomic<bool> g_busy{false};
std::wstring g_output;  // collected log text of the running operation
bool g_smoke_test = false;

int scale(int v) { return MulDiv(v, int(GetDpiForWindow(g_window)), 96); }

void set_status(const std::wstring& text) { SetWindowTextW(g_status, text.c_str()); }

std::wstring join_notices(const std::vector<std::string>& notices) {
    std::string all;
    for (auto& n : notices) all += (all.empty() ? "" : " ") + n;
    return wide(all);
}

void fill_devices(const std::string& keep_value_label) {
    std::vector<std::string> notices;
    g_options = discover_devices(notices);
    SendMessageW(g_device, CB_RESETCONTENT, 0, 0);
    int select = -1, first_gpu = -1;
    for (size_t i = 0; i < g_options.size(); i++) {
        SendMessageW(g_device, CB_ADDSTRING, 0, LPARAM(wide(g_options[i].label).c_str()));
        if (g_options[i].label == keep_value_label) select = int(i);
        if (first_gpu < 0 && g_options[i].value != "CPU") first_gpu = int(i);
    }
    if (select < 0) select = keep_value_label.empty() && first_gpu >= 0 ? first_gpu : 0;
    SendMessageW(g_device, CB_SETCURSEL, WPARAM(select), 0);
    if (!keep_value_label.empty()) set_status(notices.empty() ? L"Devices refreshed" : join_notices(notices));
    else set_status(notices.empty() ? L"Ready" : join_notices(notices));
}

std::string selected_label() {
    int i = int(SendMessageW(g_device, CB_GETCURSEL, 0, 0));
    return i >= 0 && size_t(i) < g_options.size() ? g_options[size_t(i)].label : std::string();
}

std::string selected_value() {
    int i = int(SendMessageW(g_device, CB_GETCURSEL, 0, 0));
    return i >= 0 && size_t(i) < g_options.size() ? g_options[size_t(i)].value : std::string("CPU");
}

void append_log(const std::wstring& text) {
    std::wstring crlf;
    for (wchar_t c : text) {
        if (c == L'\n') crlf += L'\r';
        crlf += c;
    }
    int end = GetWindowTextLengthW(g_log);
    SendMessageW(g_log, EM_SETSEL, WPARAM(end), LPARAM(end));
    SendMessageW(g_log, EM_REPLACESEL, FALSE, LPARAM(crlf.c_str()));
}

void enable_controls(bool on) {
    for (HWND b : g_buttons) EnableWindow(b, on);
    EnableWindow(g_device, on);
    EnableWindow(g_refresh, on);
}

// ---- Shell file dialogs ----

struct Filter {
    const wchar_t* name;
    const wchar_t* spec;
};

std::wstring item_path(IShellItem* item) {
    PWSTR p = nullptr;
    std::wstring out;
    if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &p))) {
        out = p;
        CoTaskMemFree(p);
    }
    return out;
}

void set_folder(IFileDialog* dialog, const std::wstring& folder) {
    IShellItem* item = nullptr;
    if (!folder.empty() && SUCCEEDED(SHCreateItemFromParsingName(folder.c_str(), nullptr, IID_PPV_ARGS(&item)))) {
        dialog->SetFolder(item);
        item->Release();
    }
}

std::wstring open_dialog(const wchar_t* title, bool folder, const std::vector<Filter>& filters) {
    IFileOpenDialog* dialog = nullptr;
    if (FAILED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&dialog)))) return {};
    DWORD options = 0;
    dialog->GetOptions(&options);
    dialog->SetOptions(options | FOS_FORCEFILESYSTEM | (folder ? FOS_PICKFOLDERS : FOS_FILEMUSTEXIST));
    dialog->SetTitle(title);
    std::vector<COMDLG_FILTERSPEC> specs;
    for (auto& f : filters) specs.push_back({f.name, f.spec});
    if (!specs.empty()) dialog->SetFileTypes(UINT(specs.size()), specs.data());
    std::wstring result;
    if (SUCCEEDED(dialog->Show(g_window))) {
        IShellItem* item = nullptr;
        if (SUCCEEDED(dialog->GetResult(&item))) {
            result = item_path(item);
            item->Release();
        }
    }
    dialog->Release();
    return result;
}

std::wstring save_dialog(const wchar_t* title, const std::wstring& name, const std::wstring& folder,
                         const wchar_t* extension, const std::vector<Filter>& filters) {
    IFileSaveDialog* dialog = nullptr;
    if (FAILED(CoCreateInstance(CLSID_FileSaveDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&dialog)))) return {};
    DWORD options = 0;
    dialog->GetOptions(&options);
    dialog->SetOptions((options | FOS_FORCEFILESYSTEM) & ~FOS_OVERWRITEPROMPT);
    dialog->SetTitle(title);
    dialog->SetFileName(name.c_str());
    set_folder(dialog, folder);
    std::vector<COMDLG_FILTERSPEC> specs;
    for (auto& f : filters) specs.push_back({f.name, f.spec});
    if (!specs.empty()) dialog->SetFileTypes(UINT(specs.size()), specs.data());
    if (extension) dialog->SetDefaultExtension(extension);
    std::wstring result;
    if (SUCCEEDED(dialog->Show(g_window))) {
        IShellItem* item = nullptr;
        if (SUCCEEDED(dialog->GetResult(&item))) {
            result = item_path(item);
            item->Release();
        }
    }
    dialog->Release();
    return result;
}

bool ends_with(const std::wstring& text, const std::wstring& suffix) {
    std::wstring a = fold_case(text), b = fold_case(suffix);
    return a.size() >= b.size() && a.compare(a.size() - b.size(), b.size(), b) == 0;
}

std::wstring stem(const std::wstring& name) {
    size_t dot = name.find_last_of(L'.');
    return dot == std::wstring::npos || dot == 0 ? name : name.substr(0, dot);
}

// ---- Actions ----

enum class Action { CompressFile, CompressFolder, Restore };

void start(Action action) {
    if (g_busy) return;
    std::wstring source;
    if (action == Action::CompressFolder) source = open_dialog(L"Select folder to compress", true, {});
    else if (action == Action::CompressFile) source = open_dialog(L"Select file", false, {{L"All files", L"*.*"}});
    else
        source = open_dialog(L"Select ZIP or OCLR archive", false,
                             {{L"Archives (*.zip; *.oclr)", L"*.zip;*.oclr"}, {L"All files", L"*.*"}});
    if (source.empty()) return;
    std::wstring name = name_of(source), parent = parent_of(source), target;
    bool to_folder = action == Action::Restore && (ends_with(source, L".zip") || ends_with(source, L".folder.oclr"));

    if (action == Action::Restore && to_folder) {
        target = save_dialog(L"Name for a NEW destination folder", stem(name) + L"-restored", parent, nullptr, {});
        if (target.empty()) return;
        std::wstring leaf = name_of(target);
        if (leaf.empty() || leaf == L"." || leaf == L"..") {
            MessageBoxW(g_window, L"Enter a single folder name.", L"Invalid folder name", MB_ICONERROR);
            return;
        }
    } else if (action == Action::Restore) {
        std::wstring suggested = ends_with(name, L".oclr") ? name.substr(0, name.size() - 5) : name + L".restored";
        target = save_dialog(L"Save output as", suggested, parent, nullptr, {{L"All files", L"*.*"}});
        if (target.empty()) return;
    } else {
        for (;;) {
            target = save_dialog(action == Action::CompressFolder ? L"Save archive outside the source folder"
                                                                  : L"Save output as",
                                 name + L".zip", parent, L"zip", {{L"ZIP archive", L"*.zip"}});
            if (target.empty()) return;
            if (action == Action::CompressFolder && is_within(target, source)) {
                MessageBoxW(g_window,
                            L"Save the archive outside the folder you are compressing.\n\nFor example, choose its "
                            L"parent folder or your Desktop. Your source folder has not been changed.",
                            L"Choose a different save location", MB_ICONINFORMATION);
                continue;
            }
            break;
        }
    }
    if (exists(target)) {
        MessageBoxW(g_window, L"Choose an output name that does not exist.", L"Choose a new name", MB_ICONINFORMATION);
        return;
    }

    std::string device = selected_value();
    g_busy = true;
    enable_controls(false);
    set_status(L"Working…");
    SetWindowTextW(g_log, L"");
    g_output.clear();
    std::thread([action, source, target, device] {
        int code = 0;
        double begin = seconds_now();
        try {
            if (action == Action::Restore) restore_any(source, target, device);
            else create_zip(source, target, device);
            char elapsed[64];
            std::snprintf(elapsed, sizeof elapsed, "%.3fs", seconds_now() - begin);
            say(std::string("Total elapsed: ") + elapsed);
        } catch (const std::exception& e) {
            say(std::string("Error: ") + e.what());
            code = 1;
        }
        PostMessageW(g_window, WM_APP_DONE, WPARAM(code), 0);
    }).detach();
}

// ---- Window ----

void apply_fonts() {
    if (g_font) DeleteObject(g_font);
    if (g_title_font) DeleteObject(g_title_font);
    UINT dpi = GetDpiForWindow(g_window);
    g_font = CreateFontW(-MulDiv(9, int(dpi), 72), 0, 0, 0, FW_NORMAL, 0, 0, 0, DEFAULT_CHARSET, 0, 0,
                         CLEARTYPE_QUALITY, 0, L"Segoe UI");
    g_title_font = CreateFontW(-MulDiv(16, int(dpi), 72), 0, 0, 0, FW_NORMAL, 0, 0, 0, DEFAULT_CHARSET, 0, 0,
                               CLEARTYPE_QUALITY, 0, L"Segoe UI");
    HWND all[] = {g_intro, g_device_label, g_device, g_refresh, g_buttons[0], g_buttons[1], g_buttons[2], g_log, g_status};
    for (HWND h : all) SendMessageW(h, WM_SETFONT, WPARAM(g_font), TRUE);
    SendMessageW(g_title, WM_SETFONT, WPARAM(g_title_font), TRUE);
    HFONT mono = CreateFontW(-MulDiv(9, int(dpi), 72), 0, 0, 0, FW_NORMAL, 0, 0, 0, DEFAULT_CHARSET, 0, 0,
                             CLEARTYPE_QUALITY, FIXED_PITCH, L"Consolas");
    static HFONT previous_mono = nullptr;
    SendMessageW(g_log, WM_SETFONT, WPARAM(mono), TRUE);
    if (previous_mono) DeleteObject(previous_mono);
    previous_mono = mono;
}

void layout() {
    RECT rc;
    GetClientRect(g_window, &rc);
    int pad = scale(16), w = rc.right - 2 * pad, y = pad, line = scale(20), button_h = scale(28);
    MoveWindow(g_title, pad, y, w, scale(32), TRUE);
    y += scale(40);
    MoveWindow(g_intro, pad, y, w, line * 2, TRUE);
    y += line * 2 + scale(10);
    MoveWindow(g_device_label, pad, y, w, line, TRUE);
    y += line;
    MoveWindow(g_device, pad, y, std::min(w, scale(560)), scale(300), TRUE);
    y += scale(34);
    MoveWindow(g_refresh, pad, y, scale(130), button_h, TRUE);
    y += button_h + scale(8);
    int x = pad;
    for (HWND b : g_buttons) {
        MoveWindow(b, x, y, scale(140), button_h, TRUE);
        x += scale(150);
    }
    y += button_h + scale(12);
    int status_h = line;
    int log_h = std::max(scale(60), int(rc.bottom) - y - pad - status_h - scale(8));
    MoveWindow(g_log, pad, y, w, log_h, TRUE);
    MoveWindow(g_status, pad, y + log_h + scale(8), w, status_h, TRUE);
}

HWND control(const wchar_t* cls, const wchar_t* text, DWORD style, int id, DWORD ex = 0) {
    return CreateWindowExW(ex, cls, text, WS_CHILD | WS_VISIBLE | style, 0, 0, 10, 10, g_window,
                           HMENU(INT_PTR(id)), GetModuleHandleW(nullptr), nullptr);
}

LRESULT CALLBACK window_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
        case WM_CREATE: {
            g_window = hwnd;
            g_title = control(L"STATIC", L"File and folder compression", 0, 0);
            g_intro = control(L"STATIC",
                              L"Choose CPU or a GPU for file and folder compression.\r\nCPU and GPU both create "
                              L"standard ZIP files. ZIP extraction currently uses CPU.",
                              0, 0);
            g_device_label = control(L"STATIC", L"Processing device:", 0, 0);
            g_device = control(WC_COMBOBOXW, L"", CBS_DROPDOWNLIST | WS_VSCROLL | WS_TABSTOP, ID_DEVICE);
            g_refresh = control(L"BUTTON", L"Refresh devices", BS_PUSHBUTTON | WS_TABSTOP, ID_REFRESH);
            g_buttons[0] = control(L"BUTTON", L"Compress file", BS_PUSHBUTTON | WS_TABSTOP, ID_COMPRESS_FILE);
            g_buttons[1] = control(L"BUTTON", L"Compress folder", BS_PUSHBUTTON | WS_TABSTOP, ID_COMPRESS_FOLDER);
            g_buttons[2] = control(L"BUTTON", L"Restore archive", BS_PUSHBUTTON | WS_TABSTOP, ID_RESTORE);
            g_log = control(L"EDIT", L"",
                            ES_MULTILINE | ES_READONLY | ES_AUTOVSCROLL | WS_VSCROLL | WS_TABSTOP, ID_LOG,
                            WS_EX_CLIENTEDGE);
            SendMessageW(g_log, EM_SETLIMITTEXT, 0, 0);
            g_status = control(L"STATIC", L"", SS_LEFTNOWORDWRAP | SS_ENDELLIPSIS, 0);
            apply_fonts();
            fill_devices("");
            layout();
            if (g_smoke_test) SetTimer(hwnd, 1, 300, nullptr);
            return 0;
        }
        case WM_TIMER:
            KillTimer(hwnd, 1);
            DestroyWindow(hwnd);
            return 0;
        case WM_SIZE:
            layout();
            return 0;
        case WM_DPICHANGED: {
            auto* r = reinterpret_cast<RECT*>(lp);
            SetWindowPos(hwnd, nullptr, r->left, r->top, r->right - r->left, r->bottom - r->top,
                         SWP_NOZORDER | SWP_NOACTIVATE);
            apply_fonts();
            layout();
            return 0;
        }
        case WM_GETMINMAXINFO: {
            if (!g_window) break;
            auto* info = reinterpret_cast<MINMAXINFO*>(lp);
            info->ptMinTrackSize = {scale(560), scale(380)};
            return 0;
        }
        case WM_CTLCOLORSTATIC:
            if (HWND(lp) == g_log) {  // keep the read-only log white instead of grey
                SetBkColor(HDC(wp), GetSysColor(COLOR_WINDOW));
                SetTextColor(HDC(wp), GetSysColor(COLOR_WINDOWTEXT));
                return LRESULT(GetSysColorBrush(COLOR_WINDOW));
            }
            SetBkMode(HDC(wp), TRANSPARENT);
            return LRESULT(GetSysColorBrush(COLOR_WINDOW));
        case WM_COMMAND:
            switch (LOWORD(wp)) {
                case ID_REFRESH: fill_devices(selected_label().empty() ? " " : selected_label()); break;
                case ID_COMPRESS_FILE: start(Action::CompressFile); break;
                case ID_COMPRESS_FOLDER: start(Action::CompressFolder); break;
                case ID_RESTORE: start(Action::Restore); break;
            }
            return 0;
        case WM_APP_LOG: {
            auto* text = reinterpret_cast<std::wstring*>(lp);
            g_output += *text;
            append_log(*text);
            delete text;
            return 0;
        }
        case WM_APP_DONE:
            g_busy = false;
            enable_controls(true);
            if (wp == 0)
                set_status(g_output.find(L"NO SPACE SAVED") != std::wstring::npos ? L"No space saved — output is larger"
                                                                                  : L"Finished");
            else
                set_status(L"Failed — see details above");
            return 0;
        case WM_CLOSE:
            if (g_busy) {
                MessageBoxW(hwnd, L"Please wait for the operation to finish before closing.", L"Still working",
                            MB_ICONINFORMATION);
                return 0;
            }
            DestroyWindow(hwnd);
            return 0;
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

}  // namespace

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR command_line, int show) {
    g_smoke_test = command_line && wcsstr(command_line, L"--smoke-test");
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
    INITCOMMONCONTROLSEX icc{sizeof icc, ICC_STANDARD_CLASSES};
    InitCommonControlsEx(&icc);
    set_sink([](const std::string& line) {
        // Called from the worker thread; the window thread appends it to the log.
        PostMessageW(g_window, WM_APP_LOG, 0, LPARAM(new std::wstring(wide(line) + L"\n")));
    });

    WNDCLASSEXW wc{sizeof wc};
    wc.lpfnWndProc = window_proc;
    wc.hInstance = instance;
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    wc.hIcon = LoadIconW(nullptr, IDI_APPLICATION);
    wc.hbrBackground = GetSysColorBrush(COLOR_WINDOW);
    wc.lpszClassName = L"OpenClZipWindow";
    RegisterClassExW(&wc);
    UINT dpi = GetDpiForSystem();
    HWND hwnd = CreateWindowExW(0, wc.lpszClassName, L"OpenCL compression prototype (C++)", WS_OVERLAPPEDWINDOW,
                                CW_USEDEFAULT, CW_USEDEFAULT, MulDiv(760, int(dpi), 96), MulDiv(520, int(dpi), 96),
                                nullptr, nullptr, instance, nullptr);
    ShowWindow(hwnd, show);
    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
        if (!IsDialogMessageW(hwnd, &msg)) {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }
    CoUninitialize();
    return 0;
}
