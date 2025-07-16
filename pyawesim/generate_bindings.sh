#!/bin/bash

# Change to the directory containing this script
pushd "$(dirname "$0")" > /dev/null

# Run SWIG to generate wrappers
swig -python -doxygen bindings.i || { echo "SWIG failed"; exit 1; }

# Build the extension in-place
python compile_bindings.py build_ext --inplace || { echo "Build failed"; exit 1; }

# Clean up generated files
rm -rf build
rm -f bindings_wrap.c

# Return to the original directory
popd > /dev/null