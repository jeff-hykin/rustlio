#!/usr/bin/env bash
# Build the standalone C++ ICPLocalizer relocalization benchmark harness.
# Requires: cmake, Homebrew pcl + eigen. No GTSAM (relocalizer doesn't use it).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

cmake -S "$HERE" -B "$HERE/build" \
  -DEigen3_DIR=/opt/homebrew/opt/eigen/share/eigen3/cmake \
  -DCMAKE_PREFIX_PATH="/opt/homebrew/opt/pcl;/opt/homebrew"
cmake --build "$HERE/build" -j "$(sysctl -n hw.ncpu)"
echo "built: $HERE/build/reloc_bench"
echo "run with: DYLD_LIBRARY_PATH=/opt/homebrew/lib $HERE/build/reloc_bench ..."
