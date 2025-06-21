#!/bin/sh

# Compile all source files into a shared library.
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
    -fPIC -shared \
    -Iinclude \
    src/utils/*.c src/map/*.c src/sim/*.c src/awesim/*.c src/car/*.c src/ai/*.c src/logging/*.c \
    -o ./bin/libawesim.so \
    -lm \
    -Wall -Wunused-variable

if [ $? -eq 0 ]; then
    echo "✅ Compilation successful. Shared library created at: ./bin/libawesim.so"
else
    echo "❌ Compilation failed. Please check the errors above."
    exit 1
fi