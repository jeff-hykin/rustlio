#!/usr/bin/env python
"""Benchmark the C++ SimplePGO on a loop-closure dataset under artificial drift.

For a given drift level it: injects accumulating drift into the clean FAST-LIO
trajectory, runs the C++ pgo_bench harness on the drifted poses, then measures how
well PGO recovered the trajectory -- via (a) trajectory ATE vs the clean poses and
(b) AprilTag marker spread, plus loop-detection recall against the groundtruth.
Optionally writes a .rrd visualizing the pose graph before/after with loop edges.

    python run_bench.py --in data/loop_bench/hk_village3 \
        --yaw-per-m 1.5 --trans-per-m 0.01 --rrd out.rrd [pgo args: key=val]

Drift is swept by repeated invocation (see sweep_drift.py).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from bench_lib import (
    Correction,
    inject_drift,
    load_dataset,
    marker_world_positions,
    pose_mul,
    spread,
)

_HARNESS_CPP = Path(__file__).resolve().parents[1] / "harness" / "build" / "pgo_bench"
_HARNESS_RUST = Path(__file__).resolve().parents[1] / "rust" / "target" / "release" / "pgo_bench_rs"


def run_pgo(clouds: Path, poses_tum: Path, extra: list[str]) -> dict:
    # `backend=rust` (a pseudo-arg, stripped here) switches to the Rust harness;
    # otherwise the C++ harness. The C++ binary bakes the GTSAM lib dir into its
    # rpath (nix build) so both run standalone -- no DYLD_LIBRARY_PATH needed.
    backend = "cpp"
    passthru = []
    for a in extra:
        if a.startswith("backend="):
            backend = a.split("=", 1)[1]
        else:
            passthru.append(a)
    harness = _HARNESS_RUST if backend == "rust" else _HARNESS_CPP
    out = Path(tempfile.mktemp(suffix=".json"))
    cmd = [str(harness), "--clouds", str(clouds), "--poses", str(poses_tum), "--out", str(out), *passthru]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{harness.name} failed:\n{r.stderr}")
    data = json.loads(out.read_text())
    out.unlink(missing_ok=True)
    return data


def traj_ate(kf_ts: np.ndarray, kf_xyz: np.ndarray, clean_ts: np.ndarray, clean_xyz: np.ndarray) -> float:
    """RMSE of keyframe positions vs the clean trajectory interpolated at kf ts."""
    interp = np.stack([np.interp(kf_ts, clean_ts, clean_xyz[:, i]) for i in range(3)], axis=1)
    return float(np.sqrt(((kf_xyz - interp) ** 2).sum(axis=1).mean()))


def loop_recall(detected: list[dict], gt_events: list[dict], window: float = 4.0) -> dict:
    """A groundtruth event is recalled if some detected loop connects keyframes
    whose timestamps fall near both of the event's two times (order-agnostic)."""
    hit = 0
    for ev in gt_events:
        ea, eb = ev["ts_a"], ev["ts_b"]
        ok = False
        for d in detected:
            ds, dt = d["ts_source"], d["ts_target"]
            if (abs(ds - ea) < window and abs(dt - eb) < window) or (
                abs(ds - eb) < window and abs(dt - ea) < window
            ):
                ok = True
                break
        hit += ok
    return {"recalled": hit, "total": len(gt_events), "n_detected": len(detected)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--yaw-per-m", type=float, default=0.0, help="yaw drift std (deg) per sqrt(m)")
    ap.add_argument("--trans-per-m", type=float, default=0.0, help="translation drift std (m) per sqrt(m)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rrd", default=None, help="write rerun .rrd visualization here")
    ap.add_argument("pgo", nargs="*", help="extra key=val args forwarded to pgo_bench")
    args = ap.parse_args()

    ds = load_dataset(args.inp)
    gt_path = Path(args.inp) / "groundtruth.json"
    gt = json.loads(gt_path.read_text()) if gt_path.exists() else {"loop_events": [], "marker_world": {}}

    clean_ts = ds.ts
    clean_xyz = ds.poses[:, :3]

    # --- inject drift ---
    drifted, drift = inject_drift(
        ds.ts, ds.poses, trans_per_m=args.trans_per_m, yaw_deg_per_m=args.yaw_per_m, seed=args.seed
    )
    drift_at = drift.at  # ts -> Pose (world drift)

    # write drifted poses for the harness
    tmp = Path(tempfile.mkdtemp())
    drifted_tum = tmp / "drifted.tum"
    lines = [
        f"{ds.ts[i]:.9f} " + " ".join(f"{v:.9f}" for v in drifted[i]) for i in range(ds.n)
    ]
    drifted_tum.write_text("\n".join(lines) + "\n")

    # --- run C++ PGO on drifted poses ---
    pgo = run_pgo(Path(args.inp) / "clouds.bin", drifted_tum, args.pgo)
    kfs = pgo["keyframes"]
    kf_ts = np.array([k["ts"] for k in kfs])
    raw = np.array([k["raw"] for k in kfs])
    opt = np.array([k["opt"] for k in kfs])

    # correction field: world_corrected <- world_raw(drifted), from PGO output
    corr = Correction.from_keyframes(kf_ts, raw, opt)

    def corrected_cam(ts: float):
        # apply drift, then PGO correction (both world-frame)
        return pose_mul(corr.at(ts), drift_at(ts))

    # --- marker spread (lower = tighter loop closure) ---
    pos_clean = marker_world_positions(ds)
    pos_drift = marker_world_positions(ds, cam_transform=drift_at)
    pos_corr = marker_world_positions(ds, cam_transform=corrected_cam)
    sp_clean, sp_drift, sp_corr = spread(pos_clean), spread(pos_drift), spread(pos_corr)

    # --- trajectory ATE (vs clean groundtruth) ---
    ate_drift = traj_ate(kf_ts, raw[:, :3], clean_ts, clean_xyz)
    ate_pgo = traj_ate(kf_ts, opt[:, :3], clean_ts, clean_xyz)

    rec = loop_recall(pgo["loops"], gt["loop_events"])

    summary = {
        "dataset": Path(args.inp).name,
        "drift": {"yaw_per_m": args.yaw_per_m, "trans_per_m": args.trans_per_m, "seed": args.seed},
        "keyframes": len(kfs),
        "loops_detected": len(pgo["loops"]),
        "loop_recall": rec,
        "traj_ate_m": {"drifted": round(ate_drift, 4), "pgo": round(ate_pgo, 4)},
        "marker_spread_m": {
            "clean": round(sp_clean, 2),
            "drifted": round(sp_drift, 2),
            "pgo": round(sp_corr, 2),
        },
    }
    print(json.dumps(summary, indent=2))

    if args.rrd:
        _write_rrd(args.rrd, ds, clean_xyz, drifted, raw, opt, kf_ts, pgo["loops"], gt,
                   pos_clean, pos_drift, pos_corr)
        print(f"[rrd] wrote {args.rrd}")


def _write_rrd(path, ds, clean_xyz, drifted, raw, opt, kf_ts, loops, gt,
               pos_clean, pos_drift, pos_corr) -> None:
    import rerun as rr

    rr.init("pgo_loop_closure", spawn=False)
    rr.save(path)

    # trajectories
    rr.log("clean/trajectory", rr.LineStrips3D([clean_xyz], colors=[120, 120, 120]), static=True)
    rr.log("drifted/trajectory", rr.LineStrips3D([drifted[:, :3]], colors=[220, 80, 80]), static=True)
    rr.log("pgo/trajectory", rr.LineStrips3D([opt[:, :3]], colors=[80, 200, 120]), static=True)

    # keyframe nodes
    rr.log("drifted/keyframes", rr.Points3D(raw[:, :3], colors=[220, 80, 80], radii=0.05), static=True)
    rr.log("pgo/keyframes", rr.Points3D(opt[:, :3], colors=[80, 200, 120], radii=0.05), static=True)

    # loop edges (between optimized keyframe positions)
    def find(ts):
        i = int(np.argmin(np.abs(kf_ts - ts)))
        return i

    edges_pgo, edges_drift = [], []
    for lp in loops:
        s, t = find(lp["ts_source"]), find(lp["ts_target"])
        edges_pgo.append([opt[s, :3], opt[t, :3]])
        edges_drift.append([raw[s, :3], raw[t, :3]])
    if edges_pgo:
        rr.log("pgo/loop_edges", rr.LineStrips3D(edges_pgo, colors=[40, 120, 240]), static=True)
        rr.log("drifted/loop_edges", rr.LineStrips3D(edges_drift, colors=[40, 120, 240]), static=True)

    # marker detections as point clouds (shows the spread shrinking)
    for mid, p in pos_clean.items():
        rr.log(f"clean/marker_{mid}", rr.Points3D(p, colors=[150, 150, 150], radii=0.03), static=True)
    for mid, p in pos_drift.items():
        rr.log(f"drifted/marker_{mid}", rr.Points3D(p, colors=[220, 80, 80], radii=0.03), static=True)
    for mid, p in pos_corr.items():
        rr.log(f"pgo/marker_{mid}", rr.Points3D(p, colors=[80, 200, 120], radii=0.03), static=True)

    # groundtruth marker world positions
    if gt.get("marker_world"):
        gtp = np.array(list(gt["marker_world"].values()))
        rr.log("groundtruth/markers", rr.Points3D(gtp, colors=[255, 220, 0], radii=0.12), static=True)


if __name__ == "__main__":
    main()
