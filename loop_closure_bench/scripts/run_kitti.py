#!/usr/bin/env python
"""Benchmark PGO loop closure on KITTI-odometry sequences (00-10 have GT).

KITTI is large urban driving (km-scale, sparse loops), so it needs different
params than the small Go2 sets: bigger keyframe spacing, larger loop search
radius + ICP correspondence distance (to bridge accumulated drift), coarser
submap voxel. Scored by trajectory ATE vs GT (no AprilTag markers).

    python run_kitti.py [--yaw 0.0 0.02 ...] [--seqs 00 05 06 ...]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent
DATA = BENCH.parents[1] / "data" / "loop_bench"

# KITTI-scale shared params (km-scale urban driving, sparse loops).
COMMON = [
    "key_pose_delta_trans=2.0",
    "loop_time_tresh=30",
    "loop_search_radius=15",
    "max_icp_correspondence_dist=5.0",
    "loop_score_tresh=2.0",
    "loop_submap_half_range=15",
    "submap_resolution=0.5",
]
STOCK = COMMON
PLANE = ["impl=plane"] + COMMON
# Rust: the km-scale fix is the LOOP NOISE MODEL, not the optimizer. On wide-open
# KITTI roads the loop's translation constraint yanks the long open trajectory
# tail (a small ICP error swings the far end metres), so we heavily distrust loop
# translation (loop_trans_floor=256 -> trans sigma 16 m) while trusting rotation
# (loop_rot_var=0.05) -- the loop then corrects accumulated yaw without dragging
# position. This took rust from diverging (00->12.5, 05->23, 08->11 m) to beating
# C++ on aggregate. max_loop_offset rejects gross false loops. See CONCLUSIONS.md.
RUST = ["backend=rust", "max_loop_offset=8.0", "loop_trans_floor=256.0"] + [
    c.replace("loop_time_tresh", "loop_time_thresh") for c in COMMON
]
CONFIGS = (("stock", STOCK), ("plane", PLANE), ("rust", RUST))


def run_one(ds: Path, yaw: float, cfg: list[str]) -> dict:
    cmd = [sys.executable, str(BENCH / "run_bench.py"), "--in", str(ds),
           "--yaw-per-m", str(yaw), "--trans-per-m", "0", *cfg]
    r = subprocess.run(cmd, capture_output=True, text=True)
    txt = r.stdout
    start = txt.find("{")
    return json.loads(txt[start:]) if start >= 0 else {"error": r.stderr[-300:]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaw", type=float, nargs="+", default=[0.0, 0.02])
    ap.add_argument("--seqs", nargs="+", default=["00", "02", "05", "06", "07", "08", "09"])
    args = ap.parse_args()

    header = "seq\tyaw_per_m\tconfig\tkeyframes\tloops\tate_drift\tate_pgo"
    print(header)
    rows = []
    for s in args.seqs:
        ds = DATA / f"kitti_{s}"
        if not (ds / "clouds.bin").exists():
            print(f"# skip kitti_{s}: not exported", file=sys.stderr)
            continue
        for yaw in args.yaw:
            for label, cfg in CONFIGS:
                d = run_one(ds, yaw, cfg)
                if "error" in d:
                    print(f"# kitti_{s} y={yaw} {label} ERROR {d['error'][:120]}", file=sys.stderr)
                    continue
                row = (f"{s}\t{yaw}\t{label}\t{d['keyframes']}\t{d['loops_detected']}\t"
                       f"{d['traj_ate_m']['drifted']}\t{d['traj_ate_m']['pgo']}")
                print(row, flush=True)
                rows.append(row)
    out = BENCH / "kitti_results.tsv"
    out.write_text(header + "\n" + "\n".join(rows) + "\n")
    print(f"\n# wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
