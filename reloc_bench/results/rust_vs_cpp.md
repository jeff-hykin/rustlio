# Rust port of Ivan's relocalizer vs. the C++ original — outdoor_small_loop

`rust/src/bin/reloc_rust.rs` is a from-scratch Rust port of Ivan's global
FPFH+RANSAC relocalizer (same pipeline as the C++ `global_reloc_bench` /
dimos `relocalize.py`): multi-scale FPFH+RANSAC → 180° yaw-flip → gravity
filter → wall-only rerank → point-to-plane ICP. Pure Rust (nalgebra + rayon,
hand-rolled kd-tree); no PCL/Open3D.

Both backends implement the same reloc_bench CLI contract and run on the
identical scenario (`outdoor_small_loop`, 32 global queries, no initial guess,
400k RANSAC iters).

```
            n   conv%  correct%  te_med  re_med   <2m    <5m    ms_med
RUST port   32   50%     44%     0.010   0.06°   19/32  21/32   20.2s
C++ Ivan    32   44%     41%     0.010   0.09°   14/32  17/32   41.6s
```

**Result: the Rust port matches Ivan's original** — same centimetre accuracy
when it locks (te_med 1 cm), equal-or-slightly-better success rate (44% vs 41%,
within RNG noise), and **~2× faster** per query. Validated to ~1 mm on the
synthetic room as well.

Why the speedup: feature correspondences are computed once per scale (not per
RANSAC restart), and RANSAC scores models by *correspondence* inliers (O(1) per
correspondence) rather than a geometric nearest-neighbour search every
iteration. Same algorithm, cheaper inner loop.

## Run it

```bash
cd rust && cargo build --release --bin reloc_rust
cd ../reloc_bench/scripts
uv run --with numpy --with scipy --with typer python bench_reloc.py \
  --backend ../../rust/target/release/reloc_rust \
  --scenario outdoor_small_loop --cfg ransac_iters=400000
```

Results: `outdoor_rust.json` (this run), `outdoor_global.json` (C++), compared
against the stock ICP in `outdoor_comparison.md`.
