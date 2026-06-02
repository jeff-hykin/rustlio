# New Go2 recordings (2026-06-01) — relocalization across all methods

Two recordings from the USB drive (`go2_recordings`), copied to `~/datasets`,
processed like the hk_village set: dimos export (LiDAR + AprilTag detection) →
`make_groundtruth.py` (AprilTag loop closures). Processed neutral files +
`groundtruth.json` live under `~/datasets/go2_recordings/<name>/`; relocalization
result JSONs under `<name>/reloc_results/`.

| recording | frames | marker loops | scene extent | path |
|---|---|---|---|---|
| 2026-06-01_5-32pm-PST | 6888 | markers 18,21 → 2 loops | 149×184×13 m | 729 m |
| 2026-06-01_6-05pm-PST | 5635 | marker 21 (3 visits) → 3 loops | 82×60×4 m | 598 m |

## Relocalization — global (no guess), correct% within 0.30 m & 5°

| recording | stock ROS ICP | C++ global | **Rust** |
|---|---|---|---|
| go2_5_32pm | 0% | 3% | **9%** |
| go2_6_05pm | 0% | 69% | **84%** (te_med 2 cm) |

**Rust outperforms both on both recordings.** 6:05pm relocalizes well (84%).
5:32pm is low (9%) for two distinct reasons:

1. **Odometry drift corrupts the prior map (fixable).** The onboard FAST-LIO
   drifted on this recording: pose z climbs 0→13 m (ground robot, should be
   flat), and the SAME AprilTag lands far apart across visits — marker 18
   (trashcan) 9.6 m, marker 21 (ledge) 9.4 m. So the raw-odometry-stitched map
   is smeared: each landmark exists in ~two places ~10 m apart, the revisited
   passes don't align, and relocalization can't lock onto the structured
   trashcan/ledge regions. (NB: consecutive-frame overlap is 0.09 m but that
   only checks adjacent frames — it does NOT detect loop/revisit drift.)
   Fix: build the prior map from a loop-closed / PGO-corrected trajectory.
2. **The grass field is genuinely unrelocalizable** — featureless field
   crossings have no geometry to register against (expected to fail).

6:05pm drifts far less (marker 21 spread 3.7 m across 3 visits, flat z) → 84%.
stock ICP can't do no-guess global relocalization (0% — it needs a near-truth
guess).

Rust uses the density-adaptive pipeline (auto base_res ≈ map spacing); see
`three_dataset_comparison.md` for the technique.

## Reproduce

```bash
# processed data already at ~/datasets/go2_recordings/<name>/
cd reloc_bench/scripts
uv run --with numpy --with scipy --with typer python gen_scenarios.py --only global \
  --global-dataset ~/datasets/go2_recordings/2026-06-01_6-05pm-PST --global-name go2_6_05pm --global-query-stride 180
uv run --with numpy --with scipy --with typer python bench_reloc.py \
  --backend ../../rust/target/release/reloc_rust --scenario go2_6_05pm --cfg ransac_iters=400000
```
