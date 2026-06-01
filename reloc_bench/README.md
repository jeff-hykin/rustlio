# reloc_bench — relocalization benchmark

Benchmarks the **current** relocalization algorithm — the C++ two-stage
point-to-point ICP in `localizer/src/localizers/icp_localizer.cpp` — and prints
a performance table. Built to be **backend-agnostic** so an enhanced Rust
relocalizer can be dropped in later and compared on identical trials.

## Run

```bash
./run/reloc_test                 # build harness, generate scenarios, benchmark
./run/reloc_test --regen         # force-regenerate scenarios
./run/reloc_test --quick         # synthetic only, fast
./run/reloc_test --json out.json # dump the table for diffing
```

Tests (characterization + spec for the current algorithm):

```bash
cd reloc_bench/scripts
uv run --with numpy --with scipy --with pytest python -m pytest test_reloc.py -v
```

## What it measures

For each scenario, trials are bucketed by **initial-guess error** (the operator
pose guess fed to `relocalize`). The headline is the *convergence basin*: how
much guess error the algorithm tolerates before it stops relocalizing correctly.

| column | meaning |
|---|---|
| `conv%` | fraction the algorithm reported converged (passed its score gate) |
| `correct%` | fraction recovered within tolerance of truth (0.30 m & 5°) |
| `te_med/p90` | translation error (m), median / p90, over converged trials |
| `re_med/p90` | rotation error (deg), median / p90 |
| `ms_med` | median align() wall time (ms) |

`conv%` vs `correct%` matters: ICP can "converge" to the wrong basin and still
pass its RMSE gate — those count as converged-but-incorrect.

## Scenarios

- **synthetic_room** — structured box-room map; each query is the room from a
  known pose. Clean observability → measures basin + best-case accuracy.
- **hk_village3** — real Go2 LiDAR (`data/loop_bench/hk_village3`). Prior map =
  stitched world-frame scans; each query is a real body-frame scan, truth =
  FAST-LIO odometry pose.

Scenarios are deterministic (seeded) and live in `scenarios/<name>/`:
`map.pcd`, `scans.bin`, and `manifest.json` (the driver's source of truth,
holding per-trial guess + truth + bucket).

## Backend CLI contract (for the Rust port)

A backend reads a map, query clouds, and a trials list, and writes one result
line per trial. Implement this exact interface for the Rust relocalizer:

```
BACKEND --map MAP.pcd --scans SCANS.bin --trials TRIALS.txt --out RESULTS.txt [cfg k=v ...]

MAP.pcd     prior map (binary PCD, fields: x y z intensity)
SCANS.bin   concatenated body-frame clouds; per cloud: [int32 n][n*(x,y,z,i) float32]
TRIALS.txt  one trial per line: "scan_idx tx ty tz qx qy qz qw"  (SE3 guess, body->map)
RESULTS.txt one line per trial:  "converged tx ty tz qx qy qz qw time_ms"
            pose = recovered body->map transform (== guess if not converged)
cfg k=v     ICP params: rough/refine _scan_resolution _map_resolution
            _max_iteration _score_thresh
```

Then: `./run/reloc_test --backend path/to/rust_backend` (or
`bench_reloc.py --backend ...`). Same scenarios, same scoring, comparable table.

## Layout

```
harness/reloc_bench.cpp   C++ backend wrapping the real ICPLocalizer (no ROS2)
harness/build.sh          cmake build (Homebrew PCL + Eigen, no GTSAM)
scripts/reloc_lib.py      neutral shared code: clouds, PCD/bin IO, metrics, perturbation
scripts/gen_scenarios.py  build scenarios (synthetic + hk_village3)
scripts/bench_reloc.py    run a backend over scenarios, print the table
scripts/test_reloc.py     pytest suite (drives the C++ harness)
```
