@echo off
REM Compile all source files into a shared library for Windows.
REM Links SDL2 and related libraries. Enables useful warnings.

echo Compiling source files to shared library...

if not exist bin mkdir bin

gcc ^
    -fPIC -shared ^
    -Iinclude ^
    -D_WIN32_WINNT=0x0A00 ^
    src/*.c src/utils/*.c src/render/*.c src/map/*.c src/sim/*.c src/awesim/*.c src/car/*.c src/ai/*.c src/logging/*.c ^
    -o .\bin\libawesim.dll ^
    -lSDL2main -lSDL2 -lSDL2_gfx -lSDL2_ttf -lSDL2_image -lm ^
    -Wall -Wunused-variable

REM Check if compilation succeeded
IF %ERRORLEVEL% NEQ 0 (
    echo ❌ Compilation failed. Please check the errors above.
    exit /b %ERRORLEVEL%
)

echo ✅ Compilation successful. Shared library created at: .\bin\libawesim.dll

