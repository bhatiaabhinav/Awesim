#!/bin/sh

# Save current directory
OLDPWD=$(pwd)

# Change to the directory containing this script
cd "$(dirname "$0")" || { echo "cd failed"; exit 1; }

# Run SWIG to generate wrappers
swig -python -doxygen bindings.i || { echo "SWIG failed"; cd "$OLDPWD"; exit 1; }

# Build the extension in-place
python compile_bindings.py build_ext --inplace || { echo "Build failed"; cd "$OLDPWD"; exit 1; }

# Clean up generated files
rm -rf build
rm -f bindings_wrap.c

# Return to the original directory
cd "$OLDPWD"
