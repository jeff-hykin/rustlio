# PGO loop-closure benchmark — code @ 38bfa5c

All numbers are **trajectory ATE in metres (lower better)**. Each group has its
own drift setup and metric (noted), so compare *within* a group. Configs:
`stock` = C++ SimplePGO (point-to-point + iSAM2); `gated` = stock + bounded ICP /
max-offset reject; `plane` = C++ point-to-plane; `rust` = the Rust port.
Reproduce: `scripts/run_all.py` (HK + outdoor-fastlio), `scripts/run_kitti.py`
(KITTI), `scripts/go2_eval.py` (Go2). Rust default = arc-law OFF, regime-selected
`loop_trans_floor`.

## 1. HK_village indoor (avg of 6 recordings) — fastlio source, INJECTED yaw drift
ATE vs the clean fastlio trajectory.

| config | yaw=0 | yaw=1.0 |
|--------|------:|--------:|
| stock  | 0.361 | 1.599 |
| gated  | 0.351 | 1.594 |
| plane  | 0.320 | 1.554 |
| rust   | 0.539 | **1.505** |

## 2. outdoor_small_loop — fastlio source, INJECTED yaw drift
ATE vs the clean fastlio trajectory. (549 m loop, near-closed.)

| config | yaw=0 | yaw=0.1 | yaw=0.3 |
|--------|------:|--------:|--------:|
| stock  | 0.309 | 1.842 | 11.583 |
| gated  | 0.113 | 2.163 | 10.592 |
| plane  | **0.043** | **1.519** | 11.226 |
| rust   | 0.643 | 1.893 | **7.197** |

## 3. outdoor_small_loop — GO2 ONBOARD sensor, REAL drift (no injection)
Worse independent sensor (~4.6 m range, own frame): 405 m vs 549 m physical
(~26% scale error), 17.3 m start->end gap. Scored vs the fastlio trajectory as
groundtruth via Umeyama alignment. `sim` removes the unfixable global scale (what
PGO can actually correct); `gap` = trajectory's own start->end distance.

| config | ATE rigid | ATE sim | loop gap | verdict |
|--------|----------:|--------:|---------:|---------|
| raw (no PGO)      | 12.88 | 7.50 | 17.3 | — |
| stock             | 35.67 | 34.81 | 99.0 | corrupts |
| gated             | 36.01 | 35.25 | 97.2 | corrupts |
| **plane**         | 11.88 | **2.02** | **0.75** | **closes the loop** |
| rust (distrust)   | 12.98 | 7.90 | 18.2 | no-op |
| rust (trust)      | 23.24 | 23.18 | 59.7 | corrupts |

## 4. KITTI odometry (mean over seqs 00,02,05,06,07,08,09 × yaw 0 / 0.02)
ATE vs GT, injected yaw drift. (No `gated` run on KITTI.)

| config | mean ATE |
|--------|---------:|
| stock  | 1.804 |
| plane  | 1.840 |
| rust   | **0.876** |

## Reading it
- **Rust wins** KITTI (2× better) and is competitive indoor / best on the
  hardest fastlio-outdoor drift (yaw 0.3: 7.2 vs ~11).
- **Rust loses** on the Go2 onboard sensor: C++ point-to-plane closes the real
  loop (sim 7.5->2.0) while Rust no-ops — its spatial-NN detector matches the
  wrong leg and its short-range loop measurements corrupt when trusted.
- The Go2 case is what the Scan Context (descriptor loop detection) work targets.
- C++ `stock` ≈ `gated`; `plane` is the strongest C++ variant on the open/closed
  outdoor + Go2 cases.
