#!/usr/bin/env bash
# Build the standalone C++ PGO benchmark harness (pgo_bench) via nix.
#
# The harness links the in-tree C++ SimplePGO + the point-to-plane PlanePgo against
# GTSAM (jeff-hykin/gtsam-extended), PCL, and Eigen -- all from the SAME nixpkgs
# (flake.nix in this dir), so Eigen versions match across the PCL<->GTSAM ABI
# boundary. The binary's rpath is baked to the nix GTSAM lib so it runs
# standalone (no DYLD_LIBRARY_PATH).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

nix develop "$HERE" --command bash -c "
  cmake -S '$HERE' -B '$HERE/build' -DCMAKE_BUILD_TYPE=Release &&
  cmake --build '$HERE/build' -j \"\$(sysctl -n hw.ncpu)\"
"
echo "built: $HERE/build/pgo_bench"
