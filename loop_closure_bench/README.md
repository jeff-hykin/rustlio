# PGO loop-closure benchmark

Benchmarks the C++ `SimplePGO` (`pgo/src/pgos/simple_pgo.cpp`) on real Go2
recordings with AprilTag groundtruth and **artificial odometry drift**, so we
can quantify how much drift the pose-graph optimizer can actually correct before
porting it to Rust.

FAST-LIO odometry on these recordings is too accurate to exercise loop closure
on its own, so we inject accumulating drift into the clean trajectory, run the
C++ PGO on the drifted poses, and measure how well it recovers the original.

## Where the data comes from

Six recordings (`hk_village1..6`) from the dimos repo (autoresearch PGO recording set), pulled from Git LFS. Each is a Go2 looping a small
(~7×6 m) courtyard 3–4 times with a single 10 cm AprilTag (id 10) on a wall — an
ideal loop-closure scenario (repeated revisits of the same place).

Plus `outdoor_small_loop` — a Go2 + Mid-360 **outdoor** recording
(`recording_go2_mid360_outdoor_small_loop.db`), a 549 m loop over 12.6 min with a
10 cm AprilTag (id 7). Stresses the benchmark at much larger scale.

The dimos `.db` files are decoded once into neutral files (no dimos / OpenCV
dependency afterwards). See `scripts/export_dataset.py`.

### Stream layout differs between recordings
`hk_village*` store the FAST-LIO pose directly on each `lidar` cloud row. The
outdoor recording leaves cloud poses zero and carries the trajectory in a
separate higher-rate `fastlio_odometry` stream, with the deskewed body cloud in
`fastlio_lidar`. The exporter handles both via `--lidar-stream` / `--pose-stream`
(nearest-ts pose assignment). Camera-in-world poses are always read from the
`color_image` rows. Example for the outdoor recording:

```bash
uv run --no-sync python $S/export_dataset.py \
    --db /Volumes/USB/fastlio_recordings/recording_go2_mid360_outdoor_small_loop.db \
    --out .../outdoor_small_loop \
    --lidar-stream fastlio_lidar --pose-stream fastlio_odometry --image-stream color_image
```

## Pipeline

```
dimos .db  --export_dataset.py-->  data/loop_bench/<name>/
                                     meta.json        camera intrinsics + extrinsics
                                     lidar_poses.tum  clean FAST-LIO trajectory
                                     clouds.bin       per-frame body clouds
                                     markers.json     raw AprilTag detections (drift-free)
           --make_groundtruth.py-->  groundtruth.json loop events from marker revisits
           --run_bench.py--------->   inject drift -> C++ pgo_bench -> metrics + .rrd
           --run_all.py----------->   results.tsv across datasets/configs
```

### Metrics (`run_bench.py`)
- **Trajectory ATE (m)** vs the clean FAST-LIO trajectory, before (drifted) and
  after PGO. The primary "how much can it handle" number.
- **Marker spread (m)** — sum of pairwise distances between world positions of
  the same AprilTag across detections (the marker-spread metric). Drift smears it;
  good loop closure tightens it. Computed by applying the PGO world-correction
  to the camera poses, so it needs no AprilTag detection at run time.
- **Loop recall** — fraction of groundtruth marker-revisit loop events for which
  PGO detected a loop between keyframes near those times.

## Build

```bash
./run/pgo_bench            # builds the harness via nix if needed, then benchmarks
# or just build:
loop_closure_bench/harness/build_harness.sh
```

GTSAM comes from `github:jeff-hykin/gtsam-extended` (GTSAM 4.3a1); the harness
`flake.nix` follows its nixpkgs so PCL + Eigen match the version GTSAM was built
against — mixing Eigen majors across the PCL↔GTSAM ABI is what broke the earlier
brew build once brew Eigen reached 5.0.1. The binary's rpath is baked to the nix
GTSAM lib, so it runs standalone (no `DYLD_LIBRARY_PATH`).

`pgo_bench` is a standalone driver that feeds the neutral files through
`SimplePGO` (or `PlanePgo` with `impl=plane`) exactly as `pgo_node.cpp`'s `timerCB`
does, then dumps the pose graph
(raw + optimized keyframe poses) and detected loop edges (with ICP score +
offset) as JSON.

## Run

The Python tooling needs numpy/scipy/rerun/opencv. The dimos `uv` env has them:

```bash
cd ~/repos/dimos3
S=~/repos/FASTLIO2_ROS2/loop_closure_bench/scripts
# export (once per dataset)
uv run --no-sync python $S/export_dataset.py --db data/hk_village3.db \
    --out ~/repos/FASTLIO2_ROS2/data/loop_bench/hk_village3
uv run --no-sync python $S/make_groundtruth.py --in .../hk_village3
# benchmark one run + visualization
uv run --no-sync python $S/run_bench.py --in .../hk_village3 \
    --yaw-per-m 1.0 --rrd out.rrd  [pgo key=val ...]
# full table
uv run --no-sync python $S/run_all.py
```

Open a `.rrd` with `rerun out.rrd`: grey = clean trajectory + marker, red =
drifted, green = PGO-corrected, blue = loop edges.

## PGO implementations

`pgo_bench impl=stock` (default) drives the in-tree C++ `SimplePGO` (point-to-
point PCL ICP). `pgo_bench impl=plane` drives `plane_pgo.cpp` — a port of dimos
`pgo.py`'s loop closure (point-to-plane PCL ICP with normals, single-keyframe
source submap, decoupled rotation/translation noise) sharing the same iSAM2
backbone. `run_all.py` benchmarks stock / gated / plane together.

## Cloud frame (important)

The dimos cloud streams are stored in **world frame**. `SimplePGO` expects
**body-frame** clouds that it re-projects via each keyframe pose, so the exporter
unregisters world→body (`body = inv(pose) * world`) before writing `clouds.bin`.
Skipping this double-transforms every loop-closure submap and makes ICP slide
wildly — the harness should be sanity-checked (a body-frame cloud's centroid sits
near the origin, not out at the world position). See `CONCLUSIONS.md`.

## C++ changes

`simple_pgo.{h,cpp}` gained **backward-compatible** instrumentation/knobs (all
default to the original behavior):
- `cachePairs()` getter — read detected loops (score + offset) before they're
  consumed by `smoothAndUpdate`.
- `max_icp_correspondence_dist` (default 10.0 = original) — ICP correspondence
  gate.
- `max_loop_offset` (default 0 = disabled) — reject loops whose ICP correction
  exceeds this; catches false alignments.
- `loop_source_submap_half_range` (default 0 = original single keyframe).

See `CONCLUSIONS.md` for what the benchmark found.
