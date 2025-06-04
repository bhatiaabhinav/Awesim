#!/bin/bash

input_headers="include/utils.h include/map.h include/car.h include/ai.h include/sim.h include/awesim.h"
output_header="./python/libawesim.h"

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