# rustlio2 — Rust FAST-LIO2

A pure-Rust reimplementation of the [FAST-LIO2](https://github.com/hku-mars/FAST_LIO) LiDAR-Inertial Odometry algorithm. No ROS dependency, no C++ FFI required (optional PCL FFI available behind a feature flag).

## Building

Requires [Nix](https://nixos.org/) with flakes enabled:

```bash
./run/build          # nix build → result/bin/fastlio2, result/bin/fastlio2-rerun
```

Or with a Rust toolchain (1.93+):

```bash
cd rust
cargo build --release
```

## Binaries

### `fastlio2` — Odometry runner

Processes an MCAP bag file and prints odometry. Optionally saves to `.npy`.

```bash
fastlio2 <config.yaml> <bag.mcap> [output.npy] [duration_seconds]
```

- `config.yaml` — FAST-LIO2 config (e.g. `fastlio2/config/lio.yaml`)
- `bag.mcap` — MCAP file with `/livox/imu` and `/livox/lidar` topics
- `output.npy` — (optional) save odometry as Nx7 float64 array `[time, x, y, z, vx, vy, vz]`
- `duration_seconds` — (optional, default 60) stop after this many seconds of data

### `fastlio2-rerun` — Rerun visualizer

Same pipeline, but logs to a [Rerun](https://rerun.io/) `.rrd` file for 3D visualization.

```bash
fastlio2-rerun <config.yaml> <bag.mcap> [output.rrd]
```

Logged entities:
| Entity path | Type | Description |
|---|---|---|
| `world/lidar/raw` | `Points3D` | Per-frame LiDAR point cloud, colored by intensity |
| `world/robot` | `Transform3D` | Robot pose (position + rotation quaternion) |
| `world/robot/origin` | `Points3D` | Red dot at robot origin |
| `world/trajectory` | `LineStrips3D` | Accumulated trajectory line |
| `metrics/speed` | `Scalars` | Speed magnitude (m/s) |
| `metrics/vx`, `vy`, `vz` | `Scalars` | Velocity components |

View with: `rerun output.rrd`

## Convenience scripts

```bash
./run/build                  # Build via nix
./run/graph [runs] [dur]     # Run N times (default 5, 60s), generate plotly comparison
./run/rerun [bag] [output]   # Run rerun visualizer
```

## Library API

The crate is also a library (`use rustlio2::*`). The core interface is:

### Quick start

```rust
use rustlio2::commons::*;
use rustlio2::map_builder::{MapBuilder, BuilderStatus};

// Load config from YAML or use defaults
let config = Config::default();
let mut builder = MapBuilder::new(config);

// Feed synchronized IMU + LiDAR data
let mut package = SyncPackage {
    imus: vec![IMUData { acc, gyro, time }],
    cloud: lidar_points,
    cloud_start_time: t0,
    cloud_end_time: t1,
};
builder.process(&mut package);

// Read state after processing
if builder.status() == BuilderStatus::Mapping {
    let state = &builder.kf.x;
    let position = state.imu_to_world_trans;  // Vector3<f64> — world position
    let rotation = state.imu_to_world_rot;    // Matrix3<f64> — world orientation
    let velocity = state.v;                   // Vector3<f64> — velocity
}
```

### Pipeline stages

`MapBuilder::process()` goes through three phases:

1. **`ImuInit`** — Collects IMU samples (default 20) to estimate gravity direction and gyro bias.
2. **`MapInit`** — First LiDAR scan initializes the ikd-tree map.
3. **`Mapping`** — Each scan: IMU forward-propagation → motion undistortion → IESKF update → map update.

### Modules

| Module | Description |
|---|---|
| `commons` | Core types: `Point`, `PointCloud`, `IMUData`, `SyncPackage`, `Config`, `Pose` |
| `map_builder` | Top-level pipeline orchestrator (`MapBuilder`) |
| `ieskf` | Iterated Error-State Kalman Filter — 21-DOF error state (rotation, position, LiDAR-IMU extrinsics, velocity, gyro/accel biases); gravity is fixed at init, not part of the error state |
| `imu_processor` | IMU initialization, forward propagation, motion undistortion |
| `lidar_processor` | Point-to-plane matching, IESKF update, local map management |
| `ikd_tree` | Incremental k-d tree for nearest-neighbor search with box deletion and downsampling |
| `voxel_grid` | Voxel grid downsampling (native Rust, closest-to-center strategy) |
| `so3` | SO(3) operations: `exp`, `log`, `hat`, left Jacobian and its inverse |
| `utils` | Timestamp conversion, Livox point filtering/conversion |
| `pcl_ffi` | Optional PCL VoxelGrid FFI (behind `pcl` feature flag, not used by default) |

### Key types

```rust
// 3D point with intensity and time offset
pub struct Point { pub x: f32, pub y: f32, pub z: f32, pub intensity: f32, pub curvature: f32 }

// IMU measurement
pub struct IMUData { pub acc: Vector3<f64>, pub gyro: Vector3<f64>, pub time: f64 }

// Synchronized sensor package — feed this to MapBuilder::process()
pub struct SyncPackage {
    pub imus: Vec<IMUData>,
    pub cloud: Vec<Point>,
    pub cloud_start_time: f64,
    pub cloud_end_time: f64,
}

// Kalman filter state (accessible via builder.kf.x)
pub struct State {
    pub imu_to_world_rot: Matrix3<f64>,    // IMU-to-world rotation
    pub imu_to_world_trans: Vector3<f64>,  // IMU-to-world translation
    pub lidar_to_imu_rot: Matrix3<f64>,    // LiDAR-to-IMU rotation (extrinsic)
    pub lidar_to_imu_trans: Vector3<f64>,  // LiDAR-to-IMU translation (extrinsic)
    pub v: Vector3<f64>,                   // Velocity
    pub bg: Vector3<f64>,                  // Gyroscope bias
    pub ba: Vector3<f64>,                  // Accelerometer bias
    pub g: Vector3<f64>,                   // Gravity vector (fixed at init, not in error state)
}

// Algorithm config — deserializable from YAML, has sensible defaults
pub struct Config { /* see commons.rs for all fields */ }
```

### Config fields

All fields have defaults and can be loaded from a YAML file via `serde_yaml`:

| Field | Default | Description |
|---|---|---|
| `lidar_filter_num` | 3 | Subsample every Nth LiDAR point |
| `lidar_min_range` | 0.5 | Minimum point range (m) |
| `lidar_max_range` | 20.0 | Maximum point range (m) |
| `scan_resolution` | 0.15 | Voxel size for scan downsampling |
| `map_resolution` | 0.3 | Voxel size for map downsampling |
| `cube_len` | 300.0 | Local map cube side length (m) |
| `det_range` | 60.0 | Detection range for map trimming |
| `move_thresh` | 1.5 | Movement threshold for map shift |
| `na`, `ng` | 0.01 | Accelerometer / gyroscope noise |
| `nba`, `nbg` | 0.0001 | Accelerometer / gyroscope bias noise |
| `imu_init_num` | 20 | IMU samples for initialization |
| `near_search_num` | 5 | Nearest neighbors for plane fitting |
| `ieskf_max_iter` | 5 | Max IESKF iterations per update |
| `gravity_align` | true | Align initial orientation to gravity |
| `esti_il` | false | Estimate IMU-LiDAR extrinsics online |
| `r_il`, `t_il` | identity, zeros | IMU-LiDAR extrinsic calibration |
| `lidar_cov_inv` | 1000.0 | Inverse LiDAR measurement covariance |
| `max_velocity` | 3.1 | Velocity-cap guardrail (m/s); 0 disables. Frames whose post-update speed exceeds this are rolled back and skipped |

## What's implemented

**Phase A (core FAST-LIO2) — complete:**
- IMU forward propagation with bias correction
- LiDAR motion undistortion (per-point compensation)
- Iterated Error-State Kalman Filter (IESKF) with 21-DOF error state (gravity fixed at init)
- Point-to-plane matching with incremental k-d tree
- Local map management (voxel downsampling, box trimming)
- Velocity-cap guardrail (rejects/rolls back frames that blow up the filter, per-instance state)
- MCAP bag reader for Livox LiDAR + IMU messages
- Rerun visualization output
- `.npy` odometry output

## What's NOT implemented

**Phase B (advanced features from the C++ repo):**
- `pgo` — Pose Graph Optimization (requires GTSAM)
- `hba` — Hierarchical Bundle Adjustment (requires GTSAM)
- `localizer` — Re-localization against a prior map (requires GTSAM)
- ROS2 node wrapper (publish/subscribe interface)
- Loop closure detection

## FFI

**No FFI is used by default.** Everything is implemented in pure Rust with `nalgebra` for linear algebra and `mcap` for bag reading.

An optional `pcl` feature flag exists (`cargo build --features pcl`) that provides FFI bindings to PCL's VoxelGrid filter via `cxx`. This requires PCL development headers and is not needed — the native Rust `voxel_grid::downsample` is used instead. The FFI module is defined in `src/pcl_ffi.rs` but the C++ shim is not included.

## Tests

```bash
cargo test    # 13 tests across so3, ikd_tree, ieskf, voxel_grid, imu_processor, map_builder, utils
```

## Dependencies

| Crate | Purpose |
|---|---|
| `nalgebra` | Linear algebra (matrices, vectors, SO(3)) |
| `serde` + `serde_yaml` | Config deserialization |
| `mcap` | MCAP bag file reading |
| `ndarray` + `ndarray-npy` | NumPy `.npy` output |
| `rerun` | Rerun `.rrd` visualization output |
