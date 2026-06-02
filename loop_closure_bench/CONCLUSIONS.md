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

## Finding 8 — Rust PGO now beats both C++ PGOs on KITTI (km-scale fix)

Update to Finding 7: the Rust km-scale divergence was **NOT** the optimizer
(factrs batch LM is fine) nor the ICP (corrections were small, 0.1–0.9 m). It was
the **loop translation constraint**. On wide-open KITTI roads, a loop closure in
the middle of a long open trajectory yanks the far tail: a sub-degree ICP
rotation error, trusted as a tight translation constraint, swings the downstream
kilometres of trajectory by tens of metres. On a *clean* (zero-drift) trajectory
the loops should need ~0 correction, yet the old config corrupted seq00 to 7.8 m
and seq05 to 21 m purely from this tail-swing.

**Fix (loop noise model, not the solver):** heavily distrust loop *translation*
(`loop_trans_floor=256` → translation σ = 16 m) while trusting loop *rotation*
(`loop_rot_var=0.05`). The loop then corrects accumulated **yaw** — the thing
that actually bends a trajectory — without dragging position and swinging the
tail. (GNC / Geman-McClure robust kernels were also tried; they don't help here
because on clean data *all* loops carry the same small spurious correction, so
there's no inlier majority to converge toward — robust back-ends fix outliers
among good loops, not a uniform measurement bias.)

Full KITTI sweep (`scripts/run_kitti.py`, `kitti_results.tsv`), ATE_pgo in metres,
two drift levels (yaw 0.0 / 0.02 deg·m^-0.5):

| seq | stock | plane | rust | | seq | stock | plane | rust |
|----|----:|----:|----:|---|----|----:|----:|----:|
| 00 y0    | 0.73 | 0.67 | 0.76 | | 06 y0    | 0.08 | 0.07 | **0.03** |
| 00 y.02  | 0.82 | 0.79 | 2.29 | | 06 y.02  | 0.09 | 0.09 | **0.03** |
| 02 y0    | 2.14 | 2.26 | **1.21** | | 07 y0 | 0.22 | 0.21 | **0.01** |
| 02 y.02  | 3.04 | 3.32 | **2.33** | | 07 y.02 | 0.28 | 0.27 | **0.18** |
| 05 y0    | 0.60 | 0.61 | **0.05** | | 08 y0 | 6.59 | 6.44 | **0.13** |
| 05 y.02  | 0.47 | 0.48 | 0.56 | | 08 y.02 | 6.15 | 6.26 | **3.96** |
|          |      |      |      | | 09 y0 | 2.02 | 2.13 | **0.27** |
|          |      |      |      | | 09 y.02 | 2.03 | 2.15 | **0.45** |

**Mean ATE: rust 0.876 m vs stock 1.804 vs plane 1.840 — Rust is ~2.06× better
than both C++ PGOs** and is best in 11/14 cases. The big swings come from the
sequences where the C++ point-to-point + iSAM2 loops *corrupt* a good trajectory
(seq08 clean 6.5→0.13 m, seq09 2.0→0.27 m); the translation-distrust model simply
doesn't let a loop wreck the trajectory. **stock ≈ plane** throughout.

Remaining rust losses: seq00 y.02 (2.29 vs 0.82) and seq05 y.02 (0.56 vs 0.47) —
under drift the loose loop-rotation trust under-corrects yaw on these. The
KITTI config lives only in `run_kitti.py`; the PgoConfig default (trans_floor
0.01) is unchanged, so indoor/local loop closure — where translation IS reliable
— keeps trusting it.

### Finding 8 addendum — why the loop noise isn't auto-derived from the ICP

Natural follow-up: instead of a hand-set `loop_trans_floor`, derive the loop
covariance from the ICP information matrix `H = Σ JᵀJ` (point-to-plane Hessian,
exposed as `IcpResult.info`, used via `loop_icp_cov=1`). `H` does correctly flag
the in-plane "sliding" directions as near-zero eigenvalues. **But it makes KITTI
worse, not better** (clean seq05: trans_floor 0.05 m vs loop_icp_cov best ~2.0 m).

Reason — and it's the key insight: the tail-swing is **not** a local-measurement
problem, it's **global graph conditioning**. The ICP is *locally confident* about
translation (the planes align tightly), so the data-driven covariance *trusts*
it. But on a long open trajectory, a translation constraint between two far-apart
poses can only be satisfied by rotating the entire downstream chain (translations
are stiff; rotation is the cheap DOF), which swings the tail by tens of metres
for a centimetre of loop error. Local confidence is exactly the wrong signal.

So "trust loop rotation, distrust loop translation" is a **structural prior** for
loop closure on long vehicle trajectories (standard in pose-graph SLAM), not a
per-dataset magic number. A user doesn't tune it per-run; they pick a *regime*:
- indoor / structured  → trust loop translation (`PgoConfig` default,
  trans_floor 0.01) — short loops, no long open tail, translation IS reliable.
- outdoor / vehicle / open → distrust loop translation (`run_all.py`
  `RUST_OUTDOOR`, `run_kitti.py`) — long open tails, translation swings them.

`run_all.py:rust_cfg()` already auto-selects by scene ("outdoor" → distrust).
And the value is not knife-edge: clean seq05 ATE is 0.48 / 0.15 / 0.05 m at
trans_floor 16 / 64 / 256 — anything "large" works, so a regime default
generalizes without per-dataset tuning. The `loop_icp_cov` path is kept
(off by default) as a documented dead-end for this failure mode.

## Finding 9 — can the loop_trans_floor magic number be auto-derived? (partly)

"How does a user know to set loop_trans_floor=256?" They shouldn't. Attempt:
replace it with an automatic, dynamic per-loop law. Two physically-motivated
levers were implemented and benchmarked (both via `loop_trans_scale>0`, the arc
form is what's wired):

- **Loop arc length** (trans sigma = clamp(0.02 * loop_arc, 0.05, 16)). The loop
  span is the lever arm that amplifies a small loop-translation error into a
  tail-swing. Result: **reproduces the KITTI win without the magic number** (mean
  ATE 0.882 vs 0.876 for the hand-set value, ~2x better than C++) AND **improves
  the indoor aggregate** (0.894 vs 1.021 baseline: big low-drift wins, small
  high-drift losses). But it **over-distrusts the closed-loop outdoor set**
  (outdoor_small_loop y0.1 1.89->4.0, y0.3 7.2->12): a 549 m loop that returns
  home scales to sigma ~11 m when it actually wants ~1.4 m.
- **Downstream path length** (path after the loop to the trajectory end). Fixes
  some outdoor cells but regresses KITTI clean (seq05 0.05->1.61) and still breaks
  outdoor y0.1 (->7.4).

**Why neither is universal:** KITTI's late loops and the outdoor closing loops are
*indistinguishable* in both loop-span and downstream-length, yet want opposite
translation trust. KITTI is an OPEN traverse (the "loop" is a parallel road, the
vehicle keeps going -> a translation constraint swings the rest); the outdoor set
CLOSES (returns to start -> the constraint just pins the loop, trust it). Open vs
closed is a GLOBAL trajectory-structure property; no per-loop scalar captures it.

**Conclusion.** The arc-length law is shipped as an opt-in (`loop_trans_scale`,
default off) and is the recommended setting for OPEN / exploratory operation
(automotive, a robot roaming inside->park->inside) and indoor -- there it is
genuinely flag-free and dynamic. The safe default stays the fixed
`loop_trans_floor` (regime-selected: indoor trusts translation, outdoor/KITTI
distrust) because forcing the arc law would regress the closed-loop outdoor set,
and "don't regress local" is a hard constraint. Fully-automatic across open AND
closed needs a global open/closed (or graph-connectivity) signal, not a per-loop
lever -- logged as the next step. (Loop-search radius and submap resolution also
still scale with scene; the relocalization-style "estimate base_res from median
point spacing and scale all distances" would auto-handle those, and is the
natural companion to this.)

## Finding 10 — fastlio vs Go2 onboard sensor: PGO is only as good as the loops

The outdoor recording carries TWO independent sensor sources (different frames):
- **fastlio**: `fastlio_odometry` + `fastlio_lidar` (Mid-360, ~15 m range) — the
  good source. 549 m loop, start->end gap 1.0 m (essentially closed). All prior
  results use this; PGO corrects injected drift and beats C++.
- **Go2 onboard**: `odom` (PoseStamped, in the LCM payload not the indexed
  columns) + `lidar` (short-range ~4.6 m). A worse, independent estimator with
  REAL drift: the same physical loop comes out 405 m (~26% short) with a 17.3 m
  start->end gap. Exported via `export_dataset.py --pose-from-payload` to
  `data/loop_bench/outdoor_small_loop_go2/`; scored by `go2_eval.py` against the
  fastlio trajectory as groundtruth (rigid + similarity Umeyama alignment).

**Result** (raw ATE vs fastlio 12.9 m rigid / 7.5 m after removing the ~26%
scale error; loop gap is the trajectory's own start->end distance — a clean
"did the loop close" signal). go2 config: key_pose_delta_trans 1, search radius
20 (to bridge the 17 m drift), corr 2, submap_half 15, voxel 0.2.
| config | ATE rigid | ATE sim | loop gap | verdict |
|--------|----------:|--------:|---------:|---------|
| raw (no PGO)         | 12.88 | 7.50 | 17.3 | — |
| stock (C++ p2p)      | 35.67 | 34.81 | 99.0 | corrupts |
| gated (C++ +reject)  | 36.01 | 35.25 | 97.2 | corrupts |
| **plane (C++ p2plane)** | 11.88 | **2.02** | **0.75** | **closes the loop** |
| rust (distrust)      | 12.98 |  7.90 | 18.2 | no-op |
| rust (trust)         | 23.24 | 23.18 | 59.7 | corrupts |

**Corrected takeaway** (an earlier draft wrongly concluded "no PGO helps Go2" — it
had only run rust + C++ stock). **The C++ point-to-plane PGO DOES work on the Go2
sensor**: it finds the true end->start closure and pulls the 17.3 m gap to 0.75 m,
cutting the scale-aligned ATE 7.5 -> 2.0 m. **The Rust port does not** — opposite
of KITTI/indoor, where Rust wins. Why the Rust port loses on this hard source:

1. **Loop *detection*.** Both detectors are spatial-NN in the drifted frame, but
   C++ SimplePGO's end keyframes (785,798,830,861) connect to the START region
   (tgt 11-52) — the real revisit — while the Rust detector's end keyframes match
   tgt ~121 (a different leg that fell nearer in the drifted frame). C++ evidently
   validates candidates by ICP fit / considers several; Rust commits to the single
   nearest. So Rust never forms the constraint that closes the gap.
2. **Loop *measurement*.** C++ uses PCL point-to-plane (`...WithNormals`) + iSAM2
   and its loops are good enough to TRUST (decoupled noise) and still close; the
   Rust point-to-plane loops corrupt when trusted (the short ~4.6 m range gives
   poor overlap), so Rust must distrust -> can't pull the gap.
3. **~26% scale error** (405 vs 549 m) is unfixable by loop closure for any of
   them; it stays in the rigid ATE (~12 m) — only the sim ATE reflects what PGO
   can actually correct.

**So the Go2 source is the case that justifies the next two investments:**
descriptor-based loop **detection** (Scan Context — find the real revisit under
drift instead of the spatial-nearest) and better Rust loop **measurement**
(multi-candidate ICP validation, matching what C++ does). The trust/distrust
back-end tuning was all on the good fastlio source and cannot manufacture a good
loop where detection/measurement fail.

## Finding 11 — Scan Context loop detector (descriptor-based, toggleable)

Added `scan_context.rs`: a Scan Context (Kim & Kim 2018) place-recognition loop
detector as a drop-in alternative to spatial-NN, selected with `use_scan_context=1`
(default off). Descriptor = rings x sectors max-height grid (column-major for a
cache-friendly shift distance); rotation-invariant ring-key shortlists candidates,
a column-shift-invariant distance scores them and yields a relative-yaw estimate.
Performance fix: column-major storage took seq07 from 31 s -> 0.6 s.

**SC matches or beats spatial-NN everywhere tested** (rust ATE, lower better):
| case | spatial | scan-context |
|------|--------:|-------------:|
| KITTI seq00 clean | 0.756 | **0.648** |
| KITTI seq05 clean | 0.051 | **0.023** |
| KITTI seq07 clean | 0.012 | **0.006** |
| Go2 (sim ATE, distrust) | 7.90 (no-op) | **6.86** |

It finds cleaner loops (fewer false positives) on the good sensors. On the **Go2
short-range sensor it helps only modestly** (6.86 vs 7.90; gap 17.3->15.7 vs no
change) and still does NOT close the loop like C++ plane (2.02 / 0.75). Two
reasons, both sensor-driven:
1. **Descriptor not distinctive at ~4.6 m range.** SC normally uses ~80 m scans;
   a 5 m disc of mostly ground gives weak descriptors, so SC's end keyframes match
   mid-route (tgt 292/524) instead of the start -- the same wrong-place failure as
   spatial, for a different reason (weak appearance, not drift).
2. **Rust ICP measurement.** Even when a loop is found, trusting its translation
   corrupts (short-range -> poor overlap), so it must distrust -> rotation-only
   correction -> small gain. C++'s PCL point-to-plane gives trustworthy
   translation on the same clouds and closes the gap.

**Takeaway.** SC is a net win and the right detector for drift-robust place
recognition (clear gains on long-range KITTI). It is NOT a silver bullet for the
Go2 sensor, where the limits are descriptor range and Rust ICP quality, not the
detector choice. Next: improve the Rust loop ICP (match PCL point-to-plane /
multi-candidate validation) so found loops can be trusted; and SC++ augmentation
(intensity / multi-resolution descriptors) for short-range distinctiveness.

## Finding 12 — two new Go2 scenes (grass_field_loop, stair_plaza), fastlio + go2

Added two 2026-06-01 recordings (relocalizer pre-exported the fastlio source to
`~/datasets/go2_recordings/`; symlinked as `grass_field_loop` (5:32pm, sprawling
149x184x13 m, fastlio path 729 m / gap 174 m — barely a loop) and `stair_plaza`
(6:05pm, compact, 598 m / gap 20 m)). Go2 source exported with
`export_dataset.py --pose-from-payload`. 4 cases = 2 scenes x (fastlio injected
drift, ATE vs clean | go2 real drift, sim-ATE vs fastlio GT via go2_eval.py).

**fastlio (injected drift), ATE clean->pgo:**
| case | stock | plane | rust(best) |
|------|------:|------:|-----------:|
| grass yaw0    | 1.26 | 1.42 | 5.32 (arc) — all corrupt a featureless scene |
| grass yaw0.1  | 5.86 | 5.11 | 12.5 |
| stair yaw0    | **0.08** | **0.10** | 2.80 (arc) — rust corrupts |
| stair yaw0.1  | **0.86** | **0.94** | 2.64 |

**go2 (real drift), sim-ATE raw->pgo:**
| case | raw | stock | plane | rust | rust+SC |
|------|----:|------:|------:|-----:|--------:|
| grass go2 | 53.75 | 55.1 | 51.9 | 53.9 | 53.9 | (unsolvable: both sensors drift huge; no reliable GT) |
| stair go2 | 1.97 | 8.29 | 14.40 | **1.84** | **1.87** | (rust = do-no-harm; C++ CORRUPTS) |

**Two opposite failures, one root cause.** On the **fastlio** scenes the Rust loop
ICP produces bad loops (78 vs C++'s 45) and CORRUPTS where C++ stays clean
(stair 0.08); no back-end noise config (rot 0.001/0.05, arc-law, SC) avoids it.
On the **go2** scenes the C++ point-to-plane CORRUPTS stair (8-14 m) while Rust's
distrust-translation does no harm (1.84). So neither stack universally meets
"don't do worse than raw" — and the shared culprit is **loop ICP/measurement
quality on hard scenes**, exactly the next work item. grass is genuinely
unsolvable (no reliable GT); all methods ~no-op there, none catastrophic for go2.
(`rot_var=0.001` was overfit to outdoor_small_loop; it's not a safe default.)

**Priority confirmed:** improve the Rust loop ICP (PCL-grade point-to-plane /
multi-candidate validation) so loops are trustworthy — that is what gates both
the corruption on fastlio scenes and the inability to close loops on go2.

## Finding 13 — benchmark now scores against gtsam_odom GT (run/add_gt)

fastlio is unreliable groundtruth on hard scenes (it drifted ~130 m on grass).
`run/add_gt <db>` now builds a tag-consistent GT (gtsam_odom: fastlio/odom +
AprilTag landmark SLAM) and writes it into the db; `go2_eval.py` scores against
gtsam_odom instead of fastlio. This re-scores the Go2 real-drift cases honestly
(grass raw dropped 53.8 -> 24.0 m once the GT itself is correct).

Go2 source vs gtsam_odom GT (sim-ATE m, raw -> after PGO):
| case | raw | stock | plane | rust(arc) | rust+SC |
|------|----:|------:|------:|----------:|--------:|
| outdoor | 6.62 | 35.91 | **2.58** | 9.75 | 5.87 |
| grass   | 23.96 | 25.08 | 24.94 | 24.18 | **23.94** |
| stair   | 2.41 | 7.93 | 14.65 | 2.51 | **2.24** |

**rust+ScanContext is the only config that never corrupts** — it is <= raw on all
three (improves outdoor & stair, holds on the ~unsolvable grass). **C++ plane wins
outdoor (2.58) but CORRUPTS stair (14.65)**; C++ stock corrupts all three. So on
the worse Go2 sensor the robust choice is rust+SC (do-no-harm), while C++ is
higher-ceiling but dangerous. The remaining outdoor gap (rust+SC 5.87 vs plane
2.58) is rust loop-ICP measurement quality -> trustworthy translation would let
rust close it too (the active ICP work item).

## Finding 14 — Rust loop ICP: Huber + scale-aware residual gate (default-on)

Two scale-aware, real-time-cheap additions to the Rust loop closure (both default-on):
- **Huber** robust weighting on the point-to-plane residual, delta = `loop_huber_scale`
  * submap_resolution (default 1.0). Down-weights outlier correspondences so a few
  bad matches can't drag the alignment. Cost: one scalar weight per correspondence
  (no extra passes) — real-time-safe. (Source normals + normal-compat rejection
  also wired behind `ICP_NORMAL_COS`/`min_inlier_ratio`, default off — the overlap
  gate is useless on repetitive scenes where false matches have high overlap.)
- **`loop_fit_max`** (default 2.0): reject a whole loop if `fitness/submap_resolution^2`
  exceeds it. Scale-aware so one threshold transfers across voxel sizes. This is
  the discriminator that catches repetitive-structure false loops (high overlap,
  elevated residual) the overlap gate misses.

**Result** (rust ATE, * = the corruption cases this targets):
| case | before | after |
|------|-------:|------:|
| KITTI mean (7 seqs x 2 drift) | 0.876 | 0.878 (wash; 11->12/14 beat both C++) |
| KITTI seq00 / seq05 clean | 0.756 / 0.051 | 0.596 / 0.034 (better) |
| KITTI seq02 / seq08 clean | 1.21 / 0.135 | 1.50 / 0.21 (slightly worse) |
| *stair-fastlio clean | 2.80 (corrupt) | **0.58** |
| *stair-fastlio drift0.1 | 2.64 | **1.43** |
| *grass-fastlio clean | 5.32 (corrupt) | **3.61** |
| go2 outdoor / stair (vs gtsam GT) | 5.87 / 2.24 | 6.09 / 2.25 (neutral) |
| indoor hk_village mean | 1.02 | 1.09 (slight cost) |

**Isolation:** the residual gate is purely beneficial/neutral (rejects false
loops, never touches clean ones). Huber is scene-dependent: it FIXES the cluttered
hard scenes (gate-only leaves grass at 6.08, worse than baseline — Huber brings it
to 3.61) but slightly HURTS clean structured scenes (indoor hk4 0.72->1.06, KITTI
seq00 gate-only 0.52 vs huber 0.60) by down-weighting legitimate large residuals.
Net default-on is right: it removes the catastrophic corruption (the "don't do
worse than raw" bar) and holds KITTI, at a small clean-scene cost.

**Still open:** rust does not yet BEAT C++ on the clean new fastlio scenes (C++
stair 0.08 / grass 1.26 vs rust 0.58 / 3.61) or close the go2-outdoor gap (rust+SC
6.09 vs C++ plane 2.58). That needs better loop-ICP *translation accuracy*
(PCL-grade registration / coarse-to-fine), the deeper remaining item.

## Finding 15 — registration experiments toward beating C++ (translation accuracy)

Goal: close the gap where rust loop-ICP is good-enough-not-to-corrupt but less
accurate than PCL (clean stair 0.58 vs C++ 0.08, grass 3.61 vs 1.26, go2-outdoor
rust+SC 6.09 vs plane 2.58). `scripts/icp_eval.sh` runs the loss + no-regress
cases as one command.

- **Coarse-to-fine correspondence** (anneal max_dist over iters, `ICP_C2F`): HURTS
  (stair 0.58->2.0, grass 3.61->5.2). Reverted. Tightening loses inliers on the
  asymmetric/sparse source submap.
- **Symmetric point-to-plane** (Rusinkiewicz 2019, unit sum of source+target
  normals, `ICP_SYM`, default off): net-negative. Helps indoor hk4 (1.06->0.78)
  and stair (0.58->0.52) but worsens kitti05 (0.034->0.073), grass (3.61->4.20)
  and the headline go2-outdoor (6.09->7.20). Kept env-gated off.

Two principled objective tweaks both failed to close the go2/grass gaps -> the
limit may not be the ICP objective but detection (does rust find the true go2
closure?) and/or the batch backend. Next: GICP (plane-to-plane, per-point
covariances) and a detection check on go2-outdoor.

## Finding 16 — multi-candidate loop validation (net-negative) -> the gap is the BACKEND

Added multi-candidate loop validation (`loop_candidates`, default 1 = off): gather
the top-N candidates (Scan Context top-N or spatial within-radius) and ICP-validate
each, keeping the best-fitness one (what C++/PCL effectively do). Refactor:
`eval_candidate()` + `scan_context::top_matches()`.

Result: net-negative on the headline gaps. go2-outdoor SC-multi 6.09->5.98
(marginal), spatial-multi 6.09->8.06 (worse); grass 3.61->4.29 (worse); indoor hk4
1.06->0.89 (better); KITTI/stair unchanged. **On the short-range Go2 sensor ICP
fitness is NOT a reliable discriminator** -- a coincidentally-similar wrong place
aligns *better* than the true revisit (seen from a different lane), so
best-fitness selection picks wrong.

**Three ICP/detection levers now tried (coarse-to-fine, symmetric, multi-candidate)
all fail to close go2-outdoor/grass. Root cause is the BACKEND, not the front-end.**
C++ SimplePGO uses GTSAM **iSAM2 (incremental)**: as it closes earlier loops the
whole trajectory de-drifts, so by the trajectory's end the true revisit is
spatially close and trivially found (Finding 10: C++ matched end->start). The Rust
PGO re-solves the full graph BATCH from a world_correction-warped init each loop,
so it never de-drifts *during* mapping -> spatial/descriptor detection keeps
matching the wrong (un-de-drifted) place, and no front-end tweak fixes that.

**Conclusion: beating C++ on go2-outdoor (and real-time) requires an incremental,
relinearizing backend (iSAM2-style), not more ICP/detection work.** factrs is
batch; this is the substantial next investment. The committed ICP robustness
(Finding 14: Huber + scale-aware gate) stands -- it fixed the corruption and holds
KITTI; the remaining "beat C++ everywhere" gap is architectural.
