@echo off
rem Builds build\oclzip.exe (command line) and build\oclzip-gui.exe (window) with MSVC.
setlocal
cd /d "%~dp0"
where cl >nul 2>nul
if not errorlevel 1 goto :compile
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" goto :novs
for /f "usebackq delims=" %%i in (`call "%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if not defined VSDIR goto :novs
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul
if errorlevel 1 goto :novs

:compile
if not exist build\obj-cli mkdir build\obj-cli
if not exist build\obj-gui mkdir build\obj-gui
set "FLAGS=/nologo /std:c++20 /O2 /EHsc /utf-8 /W4 /MT /MP /DUNICODE /D_UNICODE"
set "CORE=src\common.cpp src\ocl.cpp src\hardware.cpp src\zip_create.cpp src\zip_restore.cpp src\legacy.cpp"
set "LIBS=bcrypt.lib advapi32.lib user32.lib"
cl %FLAGS% /Fobuild\obj-cli\ /Febuild\oclzip.exe src\cli.cpp %CORE% /link /MANIFEST:EMBED /MANIFESTINPUT:src\app.manifest %LIBS%
if errorlevel 1 exit /b 1
cl %FLAGS% /Fobuild\obj-gui\ /Febuild\oclzip-gui.exe src\gui.cpp %CORE% /link /SUBSYSTEM:WINDOWS /MANIFEST:EMBED /MANIFESTINPUT:src\app.manifest %LIBS% gdi32.lib comctl32.lib ole32.lib shell32.lib
if errorlevel 1 exit /b 1
echo Built build\oclzip.exe and build\oclzip-gui.exe
exit /b 0

:novs
echo Visual Studio with the "Desktop development with C++" workload was not found.
exit /b 1
