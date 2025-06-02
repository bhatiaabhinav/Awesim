#!/bin/sh

# Compile all source files into the play executable.
# Includes necessary SDL2 libraries and math/time support.
# -Wall enables general warnings, and -
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
    src/*.c src/utils/*.c src/render/*.c src/map/*.c src/sim/*.c src/awesim/*.c src/car/*.c src/ai/*.c src/logging/*.c \
    -o ./bin/play \
    `sdl2-config --cflags --libs` \
    `pkg-config --cflags --libs sdl2_ttf` \
    `pkg-config --cflags --libs sdl2_image` \
    -lSDL2 -lSDL2_gfx -lSDL2_ttf -lSDL2_image -lm \
    -Wall -Wunused-variable

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "✅ Compilation successful. Executable created at: ./bin/play"
else
    echo "❌ Compilation failed. Please check the errors above."
    exit 1
fi
