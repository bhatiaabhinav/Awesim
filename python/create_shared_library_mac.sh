#!/bin/bash

input_headers="include/utils.h include/map.h include/car.h include/ai.h include/sim.h include/awesim.h"
output_header="./bin/libawesim.h"

# Ensure bin directory exists
mkdir -p bin

# Process and clean headers
(
    for file in $input_headers; do
        echo "// Contents from $file"

        # Strip include guards and `#include` directives using awk and grep
        awk '
            /^#ifndef/ { skip=1 }
            /^#define/ && skip { next }
            /^#endif/ && skip { skip=0; next }
            skip { next }
            !/^#include/ && !/^#pragma once/ { print }
        ' "$file"

        echo
    done
) > "$output_header"

# Add dummy main for CFFI compatibility (if needed)
echo "int main(int argc, char* argv[]);" >> "$output_header"

echo "✅ Combined headers into $output_header"

# Compilation block (unchanged)
echo "Compiling source files to shared library..."

gcc \
    -fPIC -shared \
    -Iinclude \
    src/*.c src/utils/*.c src/render/*.c src/map/*.c src/sim/*.c src/awesim/*.c src/car/*.c src/ai/*.c src/logging/*.c \
    -o ./bin/libawesim.so \
    `sdl2-config --cflags --libs` \
    `pkg-config --cflags --libs sdl2_ttf` \
    `pkg-config --cflags --libs sdl2_image` \
    -lSDL2 -lSDL2_gfx -lSDL2_ttf -lSDL2_image -lm \
    -Wall -Wunused-variable

if [ $? -eq 0 ]; then
    echo "✅ Compilation successful. Shared library created at: ./bin/libawesim.so"
else
    echo "❌ Compilation failed. Please check the errors above."
    exit 1
fi
