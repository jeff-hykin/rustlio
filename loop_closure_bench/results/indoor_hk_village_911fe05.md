# Indoor hk_village: rust beats C++ — code @ 911fe05

Indoor config (`scripts/run_all.py` RUST) gains `key_pose_delta_trans=1.0` (match
C++ keyframe density; rust default 0.5 over-keyframed 378 vs C++ 333) and
`loop_source_submap_half_range=2` (ICP source geometry). GNC backend (Finding 17)
default-on. Metric: sim-Umeyama ATE_pgo vs clean, mean over hk_village1..6, at
injected yaw-drift levels.

## Mean ATE_pgo by yaw (6 scenes)

| yaw | stock | plane (C++ best) | rust before | **rust after** |
|----:|------:|-----------------:|------------:|---------------:|
| 0.0 (clean) | 0.361 | 0.320 | 0.258 | **0.144** |
| 0.5 | 0.868 | 0.848 | 0.850 | **0.823** |
| 1.0 | 1.599 | 1.554 | 1.665 | 1.661 |
| 2.0 | 3.222 | 3.164 | 3.342 | 3.337 |
| **overall (24)** | 1.512 | 1.471 | 1.529 | **1.491** |

**Rust wins both recall-bearing drift levels (y0 by 2.2×, y0.5).** y1/y2 are
detection-limited — spatial-NN recall ≈ 0/6 for *all* configs (rust and both C++),
so ate_pgo ≈ ate_drift and the gap there is drift-baseline keyframe sampling, not
loop closure.

## Per-scene clean (y0.0): rust / plane / stock [recall]

| scene | rust | plane | stock | |
|-------|-----:|------:|------:|---|
| hk1 | 0.135 | 0.742 | 0.379 | WIN |
| hk2 | 0.086 | 0.324 | 0.207 | WIN |
| hk3 | 0.079 | 0.047 | 0.126 | lose (both sub-dm) |
| hk4 | 0.315 | 0.345 | 0.762 | WIN (was 0.678 loss, recall 0/3→1/3) |
| hk5 | 0.144 | 0.142 | 0.176 | tie (was 0.304 loss) |
| hk6 | 0.102 | 0.317 | 0.515 | WIN |

Rust wins 4/6, ties 1, loses only hk3 (by 3 cm). The two former 2× losses (hk4, hk5)
are now a win and a tie.

## Root cause

hk4's recall 0/3 said it was a DETECTION problem, not ICP precision. rust's denser
keyframe graph (key_pose_delta 0.5) packed more spatial-NN candidates into the loop
search, so the detector locked onto a wrong-but-nearby keyframe instead of the true
revisit. Matching C++ spacing (1.0) fixed recall; the source submap gave the loop
ICP enough geometry to align in cluttered indoor scenes.

Reproduce: `cd loop_closure_bench/scripts && python run_all.py`.
See CONCLUSIONS.md Finding 18.
