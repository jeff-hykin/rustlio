# pgo_bench_rs — Rust port of Ivan's PGO loop closure

A pure-Rust reimplementation of the dimos `pgo.py` loop-closure approach, drop-in
compatible with the C++ benchmark harness (same `--clouds --poses --out key=val`
CLI and the same keyframes/loops JSON), so `run/pgo_bench` benchmarks it head-to-
head against the C++ `stock` / `gated` / `ivan` configs (`backend=rust`).

## Pieces
- `io.rs` — read `clouds.bin` (body-frame) + `lidar_poses.tum`, write the JSON.
- `icp.rs` — point-to-plane ICP. kiddo KD-tree for correspondences, PCA normals
  on the target. Plus a small **point-to-point anchor term** and **Tikhonov
  damping**: plane-only ICP can slide freely within a wall plane (zero plane
  cost), and the Rust GN runs further into that slide than PCL's LLS does; the
  anchor + dt-based convergence curb it.
- `pgo.rs` — SE(3) pose graph via [`factrs`](https://crates.io/crates/factrs)
  (GTSAM-like factor graph): prior + odometry between-factors + loop factors with
  decoupled rotation/translation noise (translation variance = ICP fitness). KD-
  tree loop-candidate search (nearest, time-gated), batch re-optimize when a loop
  fires, plus a `max_loop_offset` reject for residual false slides.

## Build & run
```bash
cargo build --release
./target/release/pgo_bench_rs --clouds <clouds.bin> --poses <poses.tum> --out out.json \
    loop_search_radius=2.0 max_icp_correspondence_dist=1.0 loop_score_tresh=0.3 \
    loop_submap_half_range=10 submap_resolution=0.2 max_loop_offset=2.0
```
Or via the suite: `./run/pgo_bench` (builds and benchmarks all configs).

## Notes
- `factrs` pulls nalgebra 0.34, so this crate also uses 0.34 (the main
  `fastlio_rs` crate is on 0.33) — they don't share types.
- Debug: `ICP_LOG=1` prints per-iteration ICP convergence; `P2P_WEIGHT=<f>`
  overrides the point-to-point anchor weight (default 0.15).

See `../CONCLUSIONS.md` for the benchmark comparison vs the C++ implementations.
