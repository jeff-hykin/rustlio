# Rust PGO: GNC robust batch backend — code @ 9e2e8ea

`loop_gnc` **default-on**. Graduated Non-Convexity (factrs
`GraduatedNonConvexity<GncGemanMcClure, LevenMarquardt>`, 30 outer mu-steps × 50
inner LM, `gnc_mu_step=1.4`, `gnc_percentile=0.95`) wraps the loop factors in a
Geman-McClure kernel graduated convex→non-convex while the odometry chain stays a
hard inlier. A slid/false loop that disagrees with trusted odometry is rejected by
the solve instead of smearing its error across the trajectory. Composes with the
Finding-14 Huber + scale-aware gate front-end.

Corruption scenes use the **GT-independent** clean→pgo perturbation metric (how far
PGO moves an already-clean fastlio trajectory). KITTI = sim-Umeyama ATE vs clean.

| case | rust pre-GNC (F14) | **rust + GNC** | C++ best |
|------|-------------------:|---------------:|---------:|
| KITTI mean (7 seqs × 2 drift) | 0.876 | **0.880** | 1.80 stock / 1.84 plane |
| KITTI seq00 clean / drift | 0.596 / 2.284 | 0.593 / 2.285 | 0.67 / 0.79 |
| KITTI seq02 clean / drift | 1.497 / 2.42 | 1.479 / 2.479 | 2.14 / 3.04 |
| KITTI seq05 clean / drift | 0.034 / 0.541 | 0.031 / 0.541 | 0.60 / 0.47 |
| KITTI seq08 clean / drift | 0.206 / 4.006 | 0.198 / 3.991 | 6.44 / 6.15 |
| **grass-fastlio clean** | 3.61 | **0.069** | 1.26 |
| **stair-fastlio clean** | 0.58 | **0.175** | 0.08 |
| indoor hk_village seq4 | 1.06 | **0.678** | ~0.6 |
| go2 outdoor (vs gtsam GT*) | 6.09 | 6.57 | 2.58 plane |
| go2 stair (vs gtsam GT*) | 2.25 | 2.37 | corrupts |

\* gtsam_odom GT is known-flawed on the stair scene (over-fits sparse AprilTags,
worse than fastlio); the go2 magnitudes are provisional. Everything else is
GT-independent and stands.

## Headline

- **grass-fastlio 3.61 → 0.069 now BEATS C++ 1.26** — first clean-scene cross-the-board win.
- stair-fastlio 0.58 → 0.175 (approaching C++ 0.08); indoor 1.06 → 0.678.
- KITTI: ties base on every seq (clean+drift) — no regression, the **2.05× lead**
  over both C++ PGOs holds (rust 0.880 vs 1.80 / 1.84, 12/14 beat both).
- **Real-time: zero measurable cost.** seq00 full run (1577 keyframes): 14.4s
  no-GNC vs 14.4s GNC. Graduation converges in a few outer steps; the batch solve
  is a tiny slice of per-loop work (dominated by submap build + ICP).

## Why this is the backend fix Finding 16 demanded

Finding 16 traced the clean-scene gap to the batch L2 solve faithfully applying a
slightly-slid ICP loop and smearing it. GNC makes the *solver* reject those bad
constraints — the cheap form of a robust backend, no iSAM2 rewrite. The residual
go2-outdoor gap is a genuinely mis-detected loop (wrong place under drift); GNC
correctly rejects it but therefore can't use it to de-drift — that one still wants
incremental relinearization.

Reproduce: `cd loop_closure_bench/scripts && ./icp_eval.sh` (baseline now = GNC),
full KITTI via `python run_kitti.py`.
