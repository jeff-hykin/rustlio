# AGENTS.md — FASTLIO2_ROS2

## What This Is

A ROS2 port of [FAST-LIO2](https://github.com/hku-mars/FAST_LIO) (Fast LiDAR-Inertial Odometry) plus loop closure (PGO), relocalization, and hierarchical bundle adjustment (HBA). Targets Livox LiDARs on Ubuntu 22.04 / ROS2 Humble.

## Packages

| Package | Purpose | Entry point |
|---------|---------|-------------|
| `fastlio2` | Core LiDAR-IMU odometry | `src/lio_node.cpp` |
| `pgo` | Pose-graph optimization (loop closure) | `src/pgo_node.cpp` |
| `localizer` | Coarse-to-fine ICP relocalization | `src/localizer_node.cpp` |
| `hba` | Hierarchical bundle adjustment | `src/hba_node.cpp` |
| `interface` | Shared ROS2 service definitions (.srv) | `srv/*.srv` |

## Dependencies

**System:** Ubuntu 22.04, ROS2 Humble, CMake 3.8+, C++17

**C++ libraries:**
- **PCL** — point cloud processing, ICP, voxel grid filtering
- **Eigen3** — all linear algebra (matrices, quaternions)
- **Sophus** (1.22.10) — SO3/SE3 Lie group operations. Built with `SOPHUS_USE_BASIC_LOGGING` to drop the fmt dependency
- **GTSAM** — factor graph optimization (PGO and HBA only)
- **yaml-cpp** — config file parsing
- **OpenMP** — parallel point processing

**ROS2 packages:** rclcpp, sensor_msgs, nav_msgs, geometry_msgs, tf2_ros, pcl_conversions, message_filters, visualization_msgs, livox_ros_driver2

**External SDKs:** Livox-SDK2, livox_ros_driver2 (see README.md for build steps)

## Build

```bash
# assuming ROS2 Humble sourced, Sophus/GTSAM/Livox installed
colcon build
```

All packages build with `-O3`, C++17, OpenMP. `fastlio2` uses `MP_PROC_NUM=2`; `hba` uses `MP_PROC_NUM=4`.

## Architecture & Data Flow

### Inputs
- **LiDAR:** `/livox/lidar` (Livox CustomMsg)
- **IMU:** `/livox/imu` (sensor_msgs/Imu)

### Core Pipeline (fastlio2)

```
Timer (20ms / 50Hz)
  │
  ├─ syncPackage()          Align IMU + LiDAR by timestamp
  │
  └─ MapBuilder::process()  State machine: IMU_INIT → MAP_INIT → MAPPING
      │
      ├─ IMUProcessor::undistort()      Propagate IMU, motion-compensate point cloud
      │
      └─ LidarProcessor::process()
          ├─ VoxelGrid downsample       scan_resolution=0.15m
          ├─ trimCloudMap()             Slide 300m³ local map cube
          ├─ IESKF::update()            Iterative EKF (calls updateLossFunc per iteration)
          │   └─ updateLossFunc()
          │       ├─ ikd-tree nearest search (5 neighbors per point, OpenMP parallel)
          │       ├─ Plane fitting (esti_plane, QR factorization)
          │       └─ Jacobian accumulation → H, b matrices
          └─ incrCloudMap()             Insert new points into ikd-tree
```

### Outputs
- `/fastlio2/body_cloud`, `/fastlio2/world_cloud` — point clouds
- `/fastlio2/lio_path` — trajectory
- `/fastlio2/lio_odom` — odometry with velocity
- TF: `lidar → body`

### Secondary Modules

**PGO** subscribes to body_cloud + odometry. Extracts key poses (>0.5m translation or >10° rotation), searches for loops (1m radius, >60s time gap), refines with ICP, optimizes with GTSAM ISAM2.

**Localizer** loads a reference PCD map, runs two-stage ICP (rough: 0.25m/5 iter, fine: 0.1m/10 iter) via ROS2 service call.

**HBA** builds a multi-level voxel pyramid, runs per-level bundle adjustment (OctoTree plane fitting + Gauss-Newton), refines poses coarse-to-fine.

## Key Data Structures

### State (21D Kalman filter state) — `ieskf.h`
```
r_wi (3)   — world-to-IMU rotation
t_wi (3)   — world-to-IMU translation
r_il (3)   — IMU-to-LiDAR rotation (extrinsic calibration)
t_il (3)   — IMU-to-LiDAR translation
v    (3)   — velocity in world frame
bg   (3)   — gyroscope bias
ba   (3)   — accelerometer bias
```

### Point type — `pcl::PointXYZINormal`
`curvature` field stores per-point timestamp offset (used for motion undistortion).

### ikd-Tree — `ikd_Tree.h`
Incremental KD-tree with dynamic insert/delete, box queries, built-in downsampling, AABB bounds per node, and multi-threaded rebuild (triggers at 1500+ points or 20% imbalance).

### Config — `commons.h` / `config/lio.yaml`
All tunable parameters live in YAML. Key ones:
- `scan_resolution` (0.15m) — pre-downsample
- `map_resolution` (0.3m) — ikd-tree voxel density
- `cube_len` (300m) — local map extent
- `near_search_num` (5) — neighbors for plane fitting
- `ieskf_max_iter` (5) — max EKF iterations per frame

## Expensive Operations (by cost)

1. **ikd-tree nearest search** — O(log N) × ~10k points per frame, with N up to ~1M. Called inside the IESKF iteration loop, so multiplied by iteration count. This is the dominant cost.

2. **IESKF update loop** — up to 5 iterations, each re-running the full loss function (nearest search + plane fit + Jacobian). The iteration count is the main throughput multiplier.

3. **Plane fitting** (`esti_plane`) — QR factorization on 5 neighbors, ~10k times per frame per iteration. Parallelized with OpenMP.

4. **ikd-tree insert/delete** — `Add_Points` and `Delete_Point_Boxes` after each frame. Amortized O(log N) but occasional full subtree rebuilds.

5. **Point cloud transforms** — per-point rotation+translation, ~10k points. Fast individually but adds up.

6. **IMU propagation / undistortion** — ~200 IMU samples per frame. Cheap per-sample (SO3 exponential map).

## Performance Note from README

All timer/subscriber/service callbacks run on a single thread in ROS2. On slower machines this causes blocking. The README recommends moving `timerCB` to a separate thread for better throughput.

## ROS2 Services

| Service | Package | Purpose |
|---------|---------|---------|
| `/pgo/save_maps` | pgo | Save point cloud patches + poses |
| `/localizer/relocalize` | localizer | Set initial pose from PCD + guess |
| `/localizer/relocalize_check` | localizer | Check relocalization validity |
| `/hba/refine_map` | hba | Run hierarchical bundle adjustment |

## File Map

```
fastlio2/
  src/lio_node.cpp                    — ROS2 node, subscribers, timer, publishers
  src/map_builder/map_builder.h/cpp   — State machine (IMU_INIT/MAP_INIT/MAPPING)
  src/map_builder/imu_processor.h/cpp — IMU integration, motion undistortion
  src/map_builder/lidar_processor.h/cpp — Scan matching, ikd-tree, loss function
  src/map_builder/ieskf.h/cpp         — 21D iterated extended Kalman filter
  src/map_builder/ikd_Tree.h/cpp      — Incremental KD-tree (~1000 lines, templated)
  src/map_builder/commons.h/cpp       — Types, Config, esti_plane()
  src/utils.h/cpp                     — Livox CustomMsg → PCL conversion
  config/lio.yaml                     — All tunable parameters
  launch/lio_launch.py                — Launch lio_node + rviz2

pgo/
  src/pgo_node.cpp                    — Key pose extraction, loop pub, map save service
  src/pgos/simple_pgo.h/cpp           — GTSAM ISAM2, ICP loop closure
  config/pgo.yaml

localizer/
  src/localizer_node.cpp              — Relocalize service handler
  src/localizers/icp_localizer.h/cpp  — Dual-resolution ICP
  config/localizer.yaml

hba/
  src/hba_node.cpp                    — Refine map service handler
  src/hba/hba.h/cpp                   — Multi-level pyramid optimization
  src/hba/blam.h/cpp                  — Per-level bundle adjustment (OctoTree + GN)
  config/hba.yaml

interface/
  srv/SaveMaps.srv, Relocalize.srv, RefineMap.srv, IsValid.srv, SavePoses.srv
```
