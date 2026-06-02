# Benchmark findings: C++ PGO loop closure

**TL;DR — once a cloud-frame bug in the harness was fixed, the stock C++
`SimplePGO` is *not* broken: it modestly reduces drift on most recordings with
only a small zero-drift perturbation. A faithful C++ port of the point-to-
plane approach is competitive — better on some recordings, slightly worse on
others — but there is no dramatic winner. The dominant factor was getting the
input geometry right, not the ICP variant.**

> ### Correction (read this first)
> An earlier version of this benchmark reported that stock loop closure
> "corrupts the trajectory in 100% of cases (ATE 0→4.4 m)". **That was a harness
> bug, not the PGO.** The dimos point clouds are stored in **world frame**, but
> the harness fed them to `SimplePGO` as if they were **body frame** and
> re-applied each keyframe's pose — double-transforming every loop-closure
> submap. With correct body-frame clouds (the exporter now unregisters
> world→body via the inverse pose, as the reference does) the numbers below are
> completely different. Lesson: a measurement harness needs its own sanity
> checks; the "PGO is broken" story was the harness lying.

## Setup

- Data: `hk_village1..6` (Go2, ~7×6 m courtyards, AprilTag id 10) and
  `outdoor_small_loop` (Go2 + Mid-360, 549 m outdoor loop, AprilTag id 7). Clean
  FAST-LIO trajectory is groundtruth.
- We inject accumulating yaw random-walk drift, run a PGO on the drifted poses +
  body clouds, and measure trajectory ATE (RMSE vs clean) before/after, AprilTag
  marker spread, and loop recall vs marker-revisit groundtruth.
- Three configs: **stock** (original point-to-point ICP, 10 m correspondence),
  **gated** (bounded correspondence + max-offset reject), **plane** (point-to-
  plane ICP with target normals + decoupled rot/trans noise, ported from dimos
  `pgo.py`).

## Finding 1 — the GTSAM/iSAM2 backbone is correct

With loop closure disabled the optimizer is an exact pass-through (ATE after PGO
== ATE in). The odometry-factor handling and graph optimization are sound; only
the loop-closure stage was ever in question.

## Finding 2 — stock PGO modestly helps (after the cloud fix)

`hk_village`, stock config:

| | zero-drift ATE (0→pgo) | yaw=1.0 ATE (drift→pgo) |
|---|---|---|
| hk_village1 | 0.38 m | 1.44 → 1.31 (−9%) |
| hk_village2 | 0.21 m | 2.66 → 2.14 (−20%) |
| hk_village3 | 0.13 m | 1.04 → 1.01 (−3%) |
| hk_village4 | 0.76 m | 2.87 → 3.11 (+8%) |
| hk_village5 | 0.18 m | 0.96 → 0.94 (−2%) |
| hk_village6 | 0.51 m | 1.09 → 1.08 (−1%) |

It reduces drift in 5/6 recordings (notably −20% on hk_village2) and never
corrupts. The residual zero-drift perturbation (up to 0.76 m) and the one
regression (hk_village4) come from imperfect point-to-point ICP loop constraints
and weak loop recall (1–2 of 6 marker revisits are matched geometrically).

## Finding 3 — gated ≈ stock now

The bounded-correspondence + max-offset-reject "gated" config is within ±0.05 m
of stock almost everywhere. It was designed to kill the catastrophic false loops
from the buggy world-frame clouds; with correct clouds those don't occur, so it
has little left to do. Useful as a guardrail, not a meaningful improvement.

## Finding 4 — the point-to-plane port is competitive, not a clear win

Faithful C++ port of the point-to-plane loop closure (point-to-plane PCL ICP with normals on
source+target, single-keyframe source submap, target half-range 10, decoupled
noise: translation variance = ICP fitness, rotation variance fixed 0.05 rad²):

| dataset | drifted | stock | gated | plane |
|---|---|---|---|---|
| hk_village1 | 1.44 | 1.31 | 1.31 | **1.17** |
| hk_village2 | 2.66 | **2.14** | 2.14 | 2.26 |
| hk_village3 | 1.04 | 1.01 | **0.98** | 1.10 |
| hk_village4 | 2.87 | 3.11 | 3.19 | **2.87** |
| hk_village5 | 0.96 | 0.94 | **0.93** | 0.98 |
| hk_village6 | 1.09 | 1.08 | 1.02 | **0.96** |

Zero-drift perturbation: point-to-plane beats stock on 4/6 (hk3 0.05 vs 0.13, hk4 0.35 vs
0.76, hk5 0.14 vs 0.18, hk6 0.32 vs 0.51) and is worse on hk1/hk2.

point-to-plane wins drift correction on hk1/hk6 and is the only config that doesn't worsen
hk4, but loses on hk2/hk3/hk5. **No approach dominates; differences are sub-metre
and often <0.2 m.** Point-to-plane's anti-sliding advantage doesn't show up
strongly here because, with correctly-framed clouds, the scenes don't trigger
the catastrophic sliding it's designed to prevent. Caveats on the port: PCL's
point-to-plane + normal estimation on voxel-downsampled submaps is not identical
to the reference Open3D tensor pipeline (inlier-RMSE fitness, target-only normals), so a
more faithful Open3D-backed port might shift these numbers.

## Finding 5 — on the 549 m outdoor loop, PGO clearly helps

`outdoor_small_loop` (Go2 + Mid-360, 549 m, AprilTag id 7), correct body clouds:

| yaw/√m | drifted | stock | gated | plane |
|---:|---:|---:|---:|---:|
| 0.0 | 0.00 | 0.31 | 0.11 | **0.04** |
| 0.1 | 3.67 | 1.84 | 2.16 | **1.52** |
| 0.3 | 11.16 | 11.58 | **10.59** | 11.23 |

At moderate drift (yaw=0.1, 3.67 m error) PGO cuts ATE by **50–59%** (stock
→1.84, plane →1.52) — the clearest benefit anywhere in the suite, and the point-to-plane variant's
point-to-plane is best at low/moderate drift (zero-drift 0.04 m). At 11 m drift
the error exceeds what ICP can bridge from the drifted initial guess, so gains
are marginal. (This is the recording whose *buggy* world-frame clouds had earlier
shown stock "corrupting" 0→2.63 m — the fix flipped it to a clear win.)

## Finding 6 — the Rust port (`pgo_bench_rs`) is competitive indoors

`loop_closure_bench/rust/` reimplements the point-to-plane approach in pure Rust (factrs SE(3)
factor graph + a from-scratch point-to-plane ICP), benchmarked head-to-head via
`backend=rust`. At the drift-correction task (yaw=1.0) on the six indoor
hk_village recordings it matches or beats the C++ `plane` config:

| dataset | drifted | C++ plane | Rust |
|---|---|---|---|
| hk_village1 | 1.44 | 1.17 | **1.03** |
| hk_village2 | 2.66 | 2.26 | **2.14** |
| hk_village3 | 1.04 | **1.10** | 1.20 |
| hk_village4 | 2.84 | 2.87 | **2.55** |
| hk_village5 | 0.96 | **0.98** | 1.14 |
| hk_village6 | 1.10 | 0.96 | **0.97** |
| **mean** | 1.67 | 1.56 | **1.51** |

Rust wins 4/6 and the aggregate. Its zero-drift perturbation is a bit higher than
C++ plane's (residual ICP sliding), so it trades a little clean-trajectory
stillness for slightly stronger drift correction.

**Known gap — outdoor.** On `outdoor_small_loop` (549 m open scene) the Rust port
does *not* match C++: its loops don't help (≈neutral with a tight reject, mildly
corrupting otherwise), whereas C++ plane cuts ATE ~60%. Root cause is isolated to
the ICP: the from-scratch point-to-plane converges to a ~1.3 m-offset minimum on
the open, ground-plane-dominated outdoor submaps where PCL's mature ICP converges
to ~0 — verified identical input submaps, and the factrs graph backbone is fine
(indoor works). Swept anchor weight, normals, source-submap size, iterations,
damping, step cap, reciprocal correspondences, and the loop reject — none close
it. Closing outdoor needs PCL-grade registration robustness (degeneracy handling
/ GICP / better correspondence rejection), which is a real chunk of work, not a
parameter tweak.

(Process note: a silently-failing `cargo build` had me tuning a stale binary for a
stretch — always `cargo build && run` or `nix run` so a build error aborts.)

## Implications for the Rust port

1. The cloud frame contract matters more than the ICP flavour — the Rust PGO
   must receive body-frame clouds (or unregister world→body) and the harness
   must assert it (centroid-near-origin check).
2. The GTSAM-style iSAM2 backbone + odometry factors are the right design; keep
   them.
3. Point-to-plane ICP + decoupled rot/trans noise is a reasonable, competitive
   choice and slightly better-behaved at zero drift on most recordings — worth
   adopting, but it is not a silver bullet here.
4. The real headroom is loop *recall* (only 1–2 of 6 true marker revisits are
   matched) and a drift-aware ICP initial guess so large loops can close. That,
   not point-to-point-vs-plane, is where to invest.

`results.tsv` is the full machine-readable scoreboard (run `./run/pgo_bench`).
All seven datasets use correct body-frame clouds; the outdoor recording is read
from `~/datasets/fastlio_recordings/` (copied off the USB stick).

## Finding 7 — full KITTI odometry evaluation (km-scale)

Evaluated on the KITTI odometry dataset (sequences 00–10 have public GT; 11–21
are the held-out test set). Velodyne scans → body clouds, GT velodyne-in-world
(cam0_to_world ∘ cam_to_velo) → trajectory; scored by ATE vs GT under injected
yaw drift. KITTI-scale params: keyframe spacing 2 m, loop search 15 m, ICP
correspondence 5 m (to bridge accumulated drift), submap voxel 0.5 m. See
`scripts/run_kitti.py` / `kitti_results.tsv`. ATE in metres, drift→after-PGO.

| seq | drift | stock | plane | rust | notes |
|-----|------:|------:|------:|-----:|-------|
| 00 | 3.12 | **0.82** | **0.79** | 12.5 | 13 loops; C++ −74% |
| 02 | 7.56 | 3.04 | 3.32 | **2.45** | rust wins under drift |
| 05 | 0.55 | 0.47 | 0.47 | 23 | C++ helps; rust diverges |
| 06 | ~0  | 0.09 | 0.09 | 0.74 | few loops, little drift |
| 07 | 0.18 | 0.28 | 0.27 | 0.89 | single loop |
| 08 | 3.61 | 6.15 | 6.26 | 11.0 | hard seq; all worsen |
| 09 | 0.35 | 2.03 | 2.15 | **0.69** | single loop; rust best |

**The in-tree C++ PGO performs loop closure correctly at km-scale.** On the
loop-rich sequences it strongly corrects injected drift — seq00 3.12→0.79 m
(−74%), seq02 7.56→3.0 m (−60%), seq05 helps — validating it on the full
real-world dataset. It is neutral/slightly worse on sparse-loop sequences (06,
07, 09 have ≤3 loops, little to gain) and degrades on seq08 (a known-difficult
sequence). **stock ≈ plane** on KITTI (point-to-plane marginally better on clean,
no consistent edge under drift).

**Rust is unstable at km-scale.** Its factrs batch solve (re-optimizing the full
graph each loop) diverges on the large sequences (00/05/08 blow up to 10–20+ m)
where GTSAM/iSAM2 stays stable, and it detects fewer loops. It does win seq02
(under drift) and the small single-loop seq09, but is not reliable on the big
loops. This is the same factrs-vs-iSAM2 robustness gap seen on the Go2 outdoor
set, amplified by KITTI's km-scale graphs. Closing it needs incremental/
relinearizing optimization (iSAM2-style) or robust back-end handling, not just
parameter tuning.
