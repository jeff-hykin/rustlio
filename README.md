# fastlio_rs

A pure-Rust reimplementation of the [FAST-LIO2](https://github.com/hku-mars/FAST_LIO)
LiDAR-Inertial Odometry algorithm — no ROS dependency and no required C++ FFI
(an optional PCL FFI sits behind a feature flag).

Published on crates.io as [`fastlio_rs`](https://crates.io/crates/fastlio_rs).

```toml
[dependencies]
fastlio_rs = "0.2"
```

## What it is

The core estimator is the same math as upstream FAST-LIO2 — an iterated
error-state Kalman filter (IESKF) over an incremental k-d tree (ikd-tree) map,
with IMU backward-propagation undistortion and point-to-plane residuals —
reimplemented in safe Rust. On top of the faithful port it adds:

- **Online gravity estimation** (24-dim error-state, 3-DOF additive gravity) so
  initial-tilt error is corrected instead of accumulating as vertical drift.
- **Corrected measurement Jacobian** (the rotation block uses the LiDAR→IMU
  extrinsic, not the world position) — the upstream-equivalent derivation.
- **Rayon-parallelized** per-point association and plane fitting.
- A **velocity-cap guardrail** in `MapBuilder` that rejects frames whose
  post-update speed exceeds `max_velocity` (rolls back, keeps bad poses out of
  the map). Disable with `max_velocity: 0`.
- A config loader that accepts **both** a flat schema and the upstream nested
  FAST-LIO schema (`common` / `preprocess` / `mapping`), so the stock
  `config_examples/*.yaml` work unmodified.

## Building

With [Nix](https://nixos.org/) (flakes):

```bash
nix build              # -> result/bin/{fastlio2, fastlio2-rerun, render, odom_rrd}
nix run .#default -- <args>   # always builds the current sources (no stale binary)
```

Or with a Rust toolchain:

```bash
cd rust
cargo build --release
```

## Binaries

### `fastlio2` — odometry runner
Processes an MCAP bag and prints odometry; optionally saves an `Nx7` `.npy`
(`[t, x, y, z, vx, vy, vz]`).
```bash
fastlio2 <config.yaml> <bag.mcap> [output.npy] [duration_s]
```

### `fastlio2-rerun` — Rerun visualizer
Same pipeline, logged to a [Rerun](https://rerun.io/) `.rrd` for 3D viewing.
```bash
fastlio2-rerun <config.yaml> <bag.mcap> [output.rrd]
```

### `render` — raw Livox `.pcap` → `.rrd`
Parses a raw Livox **mid360 SDK2** UDP capture directly (point cloud on
`:56301`, IMU on `:56401`), runs the LIO pipeline, and writes an `.rrd` with the
estimated trajectory and the **odom-adjusted world point cloud**.
```bash
render <input.pcap> <output.rrd> [config.yaml] [duration_s]
```

### `odom_rrd` — combine odom runs into one 3D recording
Reads several odometry `.npy` files and writes a single `.rrd` overlaying each
run's trajectory in 3D.
```bash
odom_rrd <out.rrd> <run0.npy> [run1.npy ...]
```

## Configuration

Configs may use the flat schema (see `fastlio2/config/lio.yaml`) or the upstream
nested FAST-LIO schema. The stock sensor configs in `config_examples/`
(`mid360.yaml`, `avia.yaml`, `velodyne.yaml`, …) are accepted as-is. Key fields:
topics, `lidar_type`, ranges, IMU noise covariances, the LiDAR→IMU extrinsic
(`extrinsic_R`/`extrinsic_T`), `fov_degree`, and `max_velocity`.

## Helper scripts

- `run/graph` — run the odometry N times (velocity cap 3.1 m/s), build a
  log-scale speed plot, and emit a 3D `.rrd` of all runs' trajectories. Uses
  `nix run`, so it always reflects the current sources.
- `run/rerun` — visualize a bag in Rerun.

## Repository layout

This repo also vendors the original C++ ROS2 stack that the Rust port grew out
of — `fastlio2/` (LIO node), `pgo/` (GTSAM loop-closure pose-graph
optimization), `localizer/` (coarse-to-fine ICP relocalization), `hba/`
(hierarchical bundle-adjustment map refinement), and `livox_ros_driver2/`. The
pure-Rust crate lives in `rust/`.

## Credits

- [FAST-LIO / FAST-LIO2](https://github.com/hku-mars/FAST_LIO) (HKU-MARS) — the original algorithm.
- [BALM/BLAM](https://github.com/hku-mars/BALM) and [HBA](https://github.com/hku-mars/HBA) — consistent map optimization (C++ stack).

## License

GPL-2.0 (matching upstream FAST-LIO).
