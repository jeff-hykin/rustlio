# Rust loop-ICP: Huber + scale-aware residual gate — code @ e6d6ab5

Default-on: loop_huber_scale=1.0 (Huber delta = 1.0*voxel), loop_fit_max=2.0
(reject loop if fitness/voxel^2 > 2). Both scale-aware, real-time-cheap.

| case | rust before | rust after | C++ best |
|------|------------:|-----------:|---------:|
| KITTI mean (7x2) | 0.876 | 0.878 | 1.80 (stock) |
| KITTI seq00 clean | 0.756 | 0.596 | 0.67 |
| KITTI seq05 clean | 0.051 | 0.034 | 0.60 |
| KITTI seq02 clean | 1.211 | 1.497 | 2.14 |
| stair-fastlio clean | 2.80✗ | 0.58 | 0.08 |
| stair-fastlio drift0.1 | 2.64 | 1.43 | 0.94 |
| grass-fastlio clean | 5.32✗ | 3.61 | 1.26 |
| go2 outdoor (vs gtsam GT) | 5.87 | 6.09 | 2.58 (plane) |
| go2 stair (vs gtsam GT) | 2.24 | 2.25 | corrupts |
| indoor hk_village mean | 1.02 | 1.09 | ~0.3-0.5 (plane) |

- Fixes the fastlio-scene corruption (stair/grass no longer blow up) -> meets the
  "PGO must not do worse than raw" bar.
- Holds KITTI (rust still ~2x better than C++, now 12/14 cases beat both).
- Gate alone is purely beneficial; Huber fixes cluttered scenes but slightly hurts
  clean structured ones (indoor, KITTI seq00/02) -- net positive.
- Still open: BEAT C++ on clean new fastlio scenes (stair 0.08 / grass 1.26) and
  close go2-outdoor (plane 2.58) -> needs better loop-ICP translation accuracy.
