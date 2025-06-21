@echo off

call .\scripts\windows\compile.bat

echo Running the executables...

start /B .\bin\awesim_render_server.exe > render_server.log 2>&1
.\bin\awesim.exe
