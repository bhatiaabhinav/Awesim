#!/bin/sh

# Compile benchmark program and related source files.
# Links SDL2 and math libraries, and includes useful warning flags.

echo "Compiling benchmark source..."

mkdir -p bin
# Check if the bin directory was created successfully
if [ $? -ne 0 ]; then
    echo "❌ Failed to create bin directory. Please check permissions."
    exit 1
fi

gcc \
    -Iinclude \
    src/benchmark/*.c src/utils/*.c src/road/*.c src/render/*.c src/map/*.c src/sim/*.c src/awesim/*.c src/car/*.c \
    -o ./bin/benchmark \
    -lSDL2 -lSDL2_gfx -lSDL2_ttf -lm -lrt \
    -Wall -Wunused-variable

# Verify if compilation succeeded
if [ $? -eq 0 ]; then
    echo "✅ Benchmark executable successfully created at: ./bin/benchmark"
    echo "Running benchmark..."
    ./bin/benchmark
else
    echo "❌ Compilation failed. Please check the errors above."
    exit 1
fi
