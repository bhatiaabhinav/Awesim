#!/bin/sh

# Compile all source files into the play executable.
# Includes necessary SDL2 libraries and math/time support.
# -Wall enables general warnings, and -Wunused-variable helps catch potential dead code.
# You can add -Werror to treat warnings as errors during CI.

echo "Compiling source files..."

mkdir -p bin
# Check if the bin directory was created successfully
if [ $? -ne 0 ]; then
    echo "❌ Failed to create bin directory. Please check permissions."
    exit 1
fi

gcc \
    -Iinclude \
    src/*.c src/utils/*.c src/road/*.c src/render/*.c src/map/*.c src/sim/*.c src/awesim/*.c src/car/*.c \
    -o ./bin/play \
    -lSDL2 -lSDL2_gfx -lSDL2_ttf -lm -lrt \
    -Wall -Wunused-variable

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "✅ Compilation successful. Executable created at: ./bin/play"
    echo "Launching the simulation..."
    ./bin/play
else
    echo "❌ Compilation failed. Please check the errors above."
    exit 1
fi
