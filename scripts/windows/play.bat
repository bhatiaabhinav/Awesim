@echo off
REM Compile all source files into the play executable for Windows.
REM Links SDL2 and related libraries. Enables useful warnings.

echo Compiling source files...

if not exist bin mkdir bin

windres scripts/windows/resource.rc -O coff -o scripts/windows/resource.o
gcc ^
    -Iinclude ^
    src/*.c src/utils/*.c src/road/*.c src/render/*.c src/map/*.c src/sim/*.c src/awesim/*.c src/car/*.c src/ai/*.c src/logging/*.c ^
    scripts/windows/resource.o ^
    -o .\bin\awesim.exe ^
    -lSDL2main -lSDL2 -lSDL2_gfx -lSDL2_ttf -lSDL2_image -lm ^
    -Wall -Wunused-variable

REM Check if compilation succeeded
IF %ERRORLEVEL% NEQ 0 (
    echo ❌ Compilation failed. Please check the errors above.
    exit /b %ERRORLEVEL%
)

echo ✅ Compilation successful. Executable created at: .\bin\awesim.exe
echo Running the executable...

.\bin\awesim.exe
