# Scan Context vs spatial-NN loop detection — code @ d691edc

Rust PGO, `use_scan_context=1` vs default spatial-NN. ATE in metres (lower better).

| case | spatial-NN | scan-context | note |
|------|-----------:|-------------:|------|
| KITTI seq00 clean | 0.756 | **0.648** | SC cleaner loops (17 vs 19) |
| KITTI seq05 clean | 0.051 | **0.023** | SC cleaner (9 vs 11) |
| KITTI seq07 clean | 0.012 | **0.006** | both 1 loop |
| Go2 outdoor (sim ATE, distrust) | 7.90 (no-op) | **6.86** | modest; still not C++ plane (2.02) |

- SC matches/beats spatial-NN on every tested case (finds fewer false loops).
- On the Go2 short-range sensor (~4.6 m) SC helps only modestly: descriptors
  aren't distinctive at that range (its end keyframes match mid-route, not the
  start) and rust ICP measurements corrupt when trusted -> can only distrust ->
  rotation-only correction. C++ point-to-plane still wins Go2 (sim 2.02, gap 0.75).
- Perf: column-major descriptor storage took seq07 from 31 s to 0.6 s.
- Default off; spatial-NN remains the default. See CONCLUSIONS Finding 11.
