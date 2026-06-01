#!/usr/bin/env python
"""Run the PGO loop-closure benchmark across all datasets and configs.

Produces a single results table comparing the stock C++ SimplePGO loop-closure
config against a "gated" config (tighter ICP correspondence + max-offset reject),
at one or more drift levels. Writes results.tsv next to this script.

    python run_all.py [--yaw 1.0 ...] [--datasets hk_village1 ...]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent
DATA = BENCH.parents[1] / "data" / "loop_bench"

# pgo_bench config presets (key=val args forwarded by run_bench.py)
STOCK = ["loop_time_tresh=25", "loop_search_radius=2.0"]  # original ICP gate (corr=10m, no reject)
GATED = [
    "loop_time_tresh=25",
    "loop_search_radius=2.0",
    "max_icp_correspondence_dist=1.0",
    "max_loop_offset=1.5",
    "loop_source_submap_half_range=2",
]
# Ivan's approach: point-to-plane ICP, bounded correspondence, nearest candidate,
# decoupled rot/trans noise. loop_score_tresh is ICP fitness (m^2) here.
IVAN = [
    "impl=ivan",
    "loop_time_tresh=25",
    "loop_search_radius=2.0",
    "max_icp_correspondence_dist=1.0",
    "loop_score_tresh=0.3",
    "loop_submap_half_range=10",        # Ivan's target submap range
    "loop_source_submap_half_range=0",  # Ivan's source = single keyframe
    "submap_resolution=0.2",            # Ivan's default
]
# Rust port of Ivan's approach (backend=rust). Same params + a max-offset reject.
RUST = [
    "backend=rust",
    "loop_time_thresh=25",
    "loop_search_radius=2.0",
    "max_icp_correspondence_dist=1.0",
    "loop_score_tresh=0.3",
    "loop_submap_half_range=10",
    "loop_source_submap_half_range=0",
    "submap_resolution=0.2",
    "max_loop_offset=2.0",
]
# Open outdoor scenes produce translation-sliding loops (ground-plane dominant);
# distrust loop translation, trust rotation (the yaw drift is what matters).
RUST_OUTDOOR = RUST + [
    "key_pose_delta_trans=1.0",
    "min_loop_detect_duration=10.0",
    "loop_rot_var=0.001",
    "loop_trans_floor=2.0",
]


def rust_cfg(dataset: str) -> list[str]:
    return RUST_OUTDOOR if "outdoor" in dataset else RUST


CONFIGS = (("stock", STOCK), ("gated", GATED), ("ivan", IVAN), ("rust", RUST))


def run_one(ds: Path, yaw: float, cfg: list[str], rrd: str | None = None) -> dict:
    cmd = [
        sys.executable,
        str(BENCH / "run_bench.py"),
        "--in",
        str(ds),
        "--yaw-per-m",
        str(yaw),
        "--trans-per-m",
        "0",
    ]
    if rrd:
        cmd += ["--rrd", rrd]
    cmd += cfg
    r = subprocess.run(cmd, capture_output=True, text=True)
    # run_bench prints JSON summary; tolerate noisy stderr
    txt = r.stdout
    start = txt.find("{")
    return json.loads(txt[start:]) if start >= 0 else {"error": r.stderr[-500:]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaw", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--datasets", nargs="+", default=[f"hk_village{i}" for i in range(1, 7)])
    ap.add_argument("--append", action="store_true", help="append to results.tsv instead of overwriting")
    args = ap.parse_args()

    rows = []
    header = (
        "dataset\tyaw_per_m\tconfig\tkeyframes\tloops\trecall\tate_drift\tate_pgo\t"
        "spread_drift\tspread_pgo"
    )
    print(header)
    for name in args.datasets:
        ds = DATA / name
        if not (ds / "clouds.bin").exists():
            print(f"# skip {name}: not exported", file=sys.stderr)
            continue
        for yaw in args.yaw:
            for label, cfg in CONFIGS:
                if label == "rust":
                    cfg = rust_cfg(name)
                s = run_one(ds, yaw, cfg)
                if "error" in s:
                    print(f"# {name} yaw={yaw} {label} ERROR {s['error'][:120]}", file=sys.stderr)
                    continue
                r = s["loop_recall"]
                row = (
                    f"{name}\t{yaw}\t{label}\t{s['keyframes']}\t{s['loops_detected']}\t"
                    f"{r['recalled']}/{r['total']}\t{s['traj_ate_m']['drifted']}\t{s['traj_ate_m']['pgo']}\t"
                    f"{s['marker_spread_m']['drifted']:.0f}\t{s['marker_spread_m']['pgo']:.0f}"
                )
                print(row)
                rows.append(row)

    out = BENCH / "results.tsv"
    if args.append and out.exists():
        with out.open("a") as f:
            f.write("\n".join(rows) + "\n")
    else:
        out.write_text(header + "\n" + "\n".join(rows) + "\n")
    print(f"\n# wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
