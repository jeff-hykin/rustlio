# faithful_check — is fastlio_rs a faithful port of upstream FAST-LIO?

`./run/faithful_check` overlays the **Rust** `fastlio_rs` trajectory against an
**upstream C++ FAST-LIO** run on the *same* raw sensor data, and reports how far
apart they are — truncated to the window *before* the C++ odometry catastrophically
diverges (all our Go2/Mid-360 recordings eventually blow up; see jhist
`dimos-fastlio-velocity-spike`).

## How it works

```
pcap (raw Livox Mid-360 UDP)                 DimOS recording .db
        │                                            │
        │ pcap_to_mcap.py                            │ extract_upstream.py
        ▼                                            ▼
  /livox/lidar + /livox/imu  (MCAP, CDR)      fastlio_odometry  (upstream C++ traj)
        │                                            │
        │ fastlio2 (Rust)                            │  find divergence: first
        ▼                                            ▼  inter-sample jump > 0.3 m
  rust odom  ───────────────  compare.py  ──────────┘
                                  │
                     yaw-aligned ATE (XY / Z split), scale, overlay PNG
```

Key choices and why:

- **Input is the pcap, not the DB clouds.** The DB's `lidar`/`fastlio_lidar`
  streams decode to *world-frame* open3d tensors with per-point `offset_time`/
  `tag`/`line` dropped — useless as raw LIO input. The pcap is the ground-truth
  raw sensor stream both stacks consumed. `pcap_to_mcap.py` decodes the
  Livox-SDK2 packets and re-encodes CDR byte-for-byte to match the Rust
  reader (`parse_livox_custom_msg` / `parse_imu_cdr`). Validated against the
  recording's decoded jsonl (sensor_ts, dot_num, xyz, refl/tag all match).

- **One clock.** pcap capture timestamps are epoch and match the DB odom epoch;
  lidar+IMU are stamped `sensor_ts + (epoch−sensor_ts)@packet0`, internally
  consistent and epoch-comparable. `compare.py` additionally refines a small
  clock offset by minimising the aligned ATE.

- **Divergence = teleport, not speed.** The 30 Hz odom has irregular dt, so a
  normal 0.18 m motion over a 3 ms gap reads as 50 m/s — instantaneous speed is
  unusable. Jump *magnitude* is dt-independent; the first step > `--jump-thresh`
  (0.3 m) marks the unrecoverable break (here ~71.8 s, matching the documented
  "~75 s first spike").

- **Yaw-only alignment.** Both stacks gravity-align (z = up) and start at the
  origin, so the only legitimate frame freedom is yaw + translation. A full
  SO(3) Umeyama would tilt to absorb XY drift into Z and hide real vertical
  drift, so we align yaw-only and report **XY and Z error separately**.

## Caveats (read before trusting the verdict)

- The "upstream" is the **live DimOS C++ FAST-LIO** recorded in the DB — a fork
  of FAST-LIO core (the divergence reproduces across hku-mars / Ericsii /
  liangheming forks). It is *not* a fresh hku-mars build re-run with a config
  matched to `config_examples/mid360.yaml`. Part of any residual difference can
  be config/extrinsic mismatch rather than Rust infidelity.
- `fastlio_rs` scales IMU accel by `*10.0` (see `main.rs`) vs the physical
  `*9.80665`; the reported similarity **scale ≈1.02** is consistent with this.
- Lidar frames are re-segmented into 100 ms windows; the C++ run's framing may
  differ slightly.

## Usage

```
./run/faithful_check                         # defaults: Go2 Mid-360 USB recording
./run/faithful_check --rebuild               # force re-extract / re-convert / rerun
./run/faithful_check --pcap A.pcap --db B.db --jump-thresh 0.3 --tol 0.5
```
Output: `logs.ignore/faithful/faithful_check.png` + a printed report.
Needs a python with `mcap`, `numpy`, `matplotlib` (auto-detects the dimos venv;
override with `FAITHFUL_PYTHON=/path/to/python`).
