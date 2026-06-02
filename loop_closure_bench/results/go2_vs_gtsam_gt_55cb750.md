# Go2 source PGO vs gtsam_odom GT — code @ 55cb750

GT = gtsam_odom (AprilTag-corrected, from run/add_gt), not fastlio. Go2 onboard
sensor, REAL drift, sim-ATE (scale removed) vs gtsam_odom, raw -> after PGO (m).

| case | raw | stock | plane | rust(arc) | rust+SC | best |
|------|----:|------:|------:|----------:|--------:|------|
| outdoor_small_loop | 6.62 | 35.91 | 2.58 | 9.75 | 5.87 | plane (rust+SC safe) |
| grass_field_loop   | 23.96 | 25.08 | 24.94 | 24.18 | 23.94 | rust+SC (~unsolvable) |
| stair_plaza        | 2.41 | 7.93 | 14.65 | 2.51 | 2.24 | rust+SC |

- rust+SC never corrupts (<= raw everywhere). C++ plane wins outdoor but corrupts
  stair; C++ stock corrupts all three.
- Config: key_pose_delta_trans=1, search radius 20, corr 2, submap_half 15,
  voxel 0.2; rust uses loop_trans_scale=0.02 (arc-law); SC adds use_scan_context=1
  sc_max_range=8.
- grass raw is 24 m (not the old 53.8 vs fastlio) — gtsam GT removed fastlio's own
  ~130 m drift from the comparison.
