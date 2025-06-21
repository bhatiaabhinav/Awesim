#!/bin/sh

chmod +x ./scripts/mac/compile.sh
./scripts/mac/compile.sh

echo "Running the executables..."

./bin/awesim_render_server > render_server.log 2>&1 &
sleep 1
./bin/awesim