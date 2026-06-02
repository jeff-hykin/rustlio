# Relocalization across three datasets — stock ROS vs. C++ global vs. Rust

Global relocalization (no initial guess) on three datasets spanning sensors and
environments:

- **hk_village3** — Go2 Livox, indoor-ish (`data/loop_bench/hk_village3`)
- **outdoor_small_loop** — Go2 Mid360, outdoor 154 m loop
- **kitti_06** — KITTI odometry seq 06, Velodyne HDL-64 automotive, ~457 m

Same global scenario per dataset (stitched prior map + per-query local submaps,
truth = odometry pose; correct = recovered within 0.30 m & 5°).

| dataset | stock ROS ICP | C++ global (FPFH+RANSAC) | **Rust** |
|---|---|---|---|
| hk_village3 | 0% | 17% | **29%** |
| outdoor_small_loop | 0% | 41% | **69%** |
| kitti_06 | 0% | 0% | **100%** (te_med 4 cm) |

**Rust outperforms both on all three.** stock ICP needs a near-truth guess so it
cannot do no-guess global relocalization (0% everywhere). The C++ global
relocalizer (port of dimos `relocalize.py`) works on Livox-density data but its
distance parameters (FPFH scales 0.2/0.3/0.8, normal radius, fine voxel 0.1,
rerank 0.15 m) are hardcoded for ~0.1 m indoor density and **collapse to 0% on
KITTI's 0.5 m automotive LiDAR** (normals degenerate — wall subset 1154/348034
points; a correct alignment has ~0 inliers within 0.15 m so fitness ≈ 0).

## The technique: density-adaptive `base_res`

The Rust relocalizer auto-estimates `base_res` = the prior map's median
nearest-neighbour spacing, then scales **all** distances proportionally:

- FPFH scale plan = `base_res · {2, 3, 8}` (0.1 → 0.2/0.3/0.8; 0.5 → 1.0/1.5/4.0)
- fine voxel = `base_res`, rerank distance = `base_res · 1.5`
- normal radius = `2·scale`, FPFH radius = `5·scale`

So the same binary adapts from indoor Livox (≈0.1 m) to outdoor Mid360 (≈0.25 m)
to KITTI Velodyne (≈0.3–0.5 m) with no manual tuning. On KITTI this turns
fitness 0.0004 → 0.84 and the wall subset 1154 → 276 358 points. It also lifted
outdoor from 44% → 69% by matching the 0.25 m map density.

Override with `--cfg base_res=<m>`; `<0` (default) auto-estimates.

## Reproduce

```bash
cd rust && cargo build --release --bin reloc_rust
cd ../reloc_bench/scripts
# scenarios (global): hk_village3_global, outdoor_small_loop, kitti_06
uv run --with numpy --with scipy --with typer python gen_scenarios.py --only global \
  --global-dataset ../../data/loop_bench/kitti_06 --global-name kitti_06 \
  --global-map-voxel 0.5 --global-sub-voxel 0.3 --global-map-stride 4
uv run --with numpy --with scipy --with typer python bench_reloc.py \
  --backend ../../rust/target/release/reloc_rust --scenario kitti_06 --cfg ransac_iters=400000
```

Result JSONs: `hk_rust.json`, `outdoor_rust_v2.json`, `kitti_rust.json` (Rust);
`outdoor_global.json`, `kitti_cppglobal.json` (C++); `outdoor_stock.json` (stock).
