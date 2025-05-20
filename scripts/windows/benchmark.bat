@echo off
REM Compile benchmark program and related source files for Windows.
REM Links SDL2 and related libraries. Enables useful warnings.

echo Compiling benchmark source...

if not exist bin mkdir bin

gcc ^
    -Iinclude ^
    src/benchmark/*.c src/utils/*.c src/road/*.c src/render/*.c src/map/*.c src/sim/*.c src/awesim/*.c src/car/*.c src/ai/*.c ^
    -o .\bin\benchmark.exe ^
    -lSDL2main -lSDL2 -lSDL2_gfx -lSDL2_ttf -lm ^
    -Wall -Wunused-variable

REM Check if compilation succeeded
IF %ERRORLEVEL% NEQ 0 (
    echo ❌ Compilation failed. Please check the errors above.
    exit /b %ERRORLEVEL%
)

echo ✅ Benchmark executable successfully created at: .\bin\benchmark.exe
echo Running benchmark...

.\bin\benchmark.exe
