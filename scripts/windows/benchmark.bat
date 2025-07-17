@echo off
REM Compile benchmark program and related source files for Windows.
REM Links SDL2 and related libraries. Enables useful warnings.

echo Compiling benchmark source...

if not exist bin mkdir bin

gcc ^
    -Iinclude ^
    -D_WIN32_WINNT=0x0A00 ^
    src/benchmark.c src/utils/*.c src/map/*.c src/sim/*.c src/awesim/*.c src/car/*.c src/ai/*.c src/procedures/*.c src/logging/*.c ^
    -o .\bin\benchmark.exe ^
    -lm -lws2_32 ^
    -Wall -Wunused-variable

REM Check if compilation succeeded
IF %ERRORLEVEL% NEQ 0 (
    echo ❌ Compilation failed. Please check the errors above.
    exit /b %ERRORLEVEL%
)

echo ✅ Benchmark executable successfully created at: .\bin\benchmark.exe
echo Running benchmark...

.\bin\benchmark.exe
