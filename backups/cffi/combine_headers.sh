#!/bin/bash

input_headers="include/utils.h include/map.h include/car.h include/ai.h include/sim.h include/actions.h include/awesim.h"
output_header="./python/libawesim.h"

# Ensure bin directory exists
mkdir -p bin

# Process and clean headers
(
    for file in $input_headers; do
        echo "// Contents from $file"

        # Strip include guards, `#include` directives, and extern variable declarations using awk and grep
        awk '
            /^#ifndef/ { skip=1 }
            /^#define/ && skip { next }
            /^#endif/ && skip { skip=0; next }
            skip { next }
            /^[ \t]*#include "/ { next }
            /^[ \t]*#pragma once/ { next }
            /^[ \t]*extern/ && !/\(/ && /;/ { next }  # Skip extern variable declarations (no parentheses, ends with semicolon)
            { print }
        ' "$file"

        echo
    done
) > "$output_header"

echo "✅ Combined headers into $output_header, excluding include guards, #include directives, and extern variable declarations"