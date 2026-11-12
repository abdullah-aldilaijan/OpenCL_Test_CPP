@echo off
rem No arguments: open the window. "devices": list devices. Otherwise passed to oclzip.exe.
setlocal
set "EXE=%~dp0build\oclzip.exe"
if not exist "%EXE%" call "%~dp0build.cmd" || (pause & exit /b 1)
if "%~1"=="" (
  start "" "%~dp0build\oclzip-gui.exe"
  exit /b
)
"%EXE%" %*
if errorlevel 1 pause
