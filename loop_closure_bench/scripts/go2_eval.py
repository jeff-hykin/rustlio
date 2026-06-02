#!/usr/bin/env python3
"""Benchmark PGO on the Go2 onboard source (REAL drift), scored against the
fastlio trajectory as groundtruth.

The Go2 `odom` is a worse, independent sensor in its own frame with real
accumulated drift (its 549 m physical loop comes out ~405 m, start->end gap
~17 m). So unlike the fastlio benchmark we do NOT inject drift; we run PGO on the
raw Go2 trajectory + Go2 clouds and ask whether loop closure pulls it back toward
the (good) fastlio trajectory. Because the two sensors live in different frames,
we rigidly align (Umeyama, with and without scale) keyframe positions to the
fastlio groundtruth before computing ATE -- reported before (raw) and after PGO.

    python3 go2_eval.py [key=val ...]      # forwards pgo args to the rust harness
"""
import json, math, os, subprocess, sys, tempfile, bisect
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(ROOT)
HARNESS = f"{ROOT}/rust/target/release/pgo_bench_rs"
GO2 = f"{REPO}/data/loop_bench/outdoor_small_loop_go2"
GT = f"{REPO}/data/loop_bench/outdoor_small_loop/lidar_poses.tum"  # fastlio = groundtruth


def read_tum(path):
    ts, xyz = [], []
    for line in open(path):
        v = line.split()
        if len(v) >= 8:
            ts.append(float(v[0])); xyz.append([float(v[1]), float(v[2]), float(v[3])])
    return np.array(ts), np.array(xyz)


def umeyama(src, dst, with_scale):
    """Least-squares rigid (optionally similarity) transform src->dst (N x 3)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s, d = src - mu_s, dst - mu_d
    cov = d.T @ s / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    scale = (np.trace(np.diag(D) @ S) / (s ** 2).sum() * len(src)) if with_scale else 1.0
    t = mu_d - scale * R @ mu_s
    return scale, R, t


def ate(src, dst, with_scale):
    s, R, t = umeyama(src, dst, with_scale)
    aligned = (s * (R @ src.T).T) + t
    return float(np.sqrt(((aligned - dst) ** 2).sum(1).mean()))


def main():
    extra = [a for a in sys.argv[1:] if "=" in a]
    gts, gtx = read_tum(GT)

    out = tempfile.mktemp(suffix=".json")
    cmd = [HARNESS, "--clouds", f"{GO2}/clouds.bin", "--poses", f"{GO2}/lidar_poses.tum",
           "--out", out, *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("FAILED:", r.stderr[-400:]); sys.exit(1)
    d = json.load(open(out)); os.unlink(out)
    kf = d["keyframes"]

    def gt_at(t):
        j = bisect.bisect_left(gts, t)
        j = min(max(j, 0), len(gts) - 1)
        if j > 0 and abs(gts[j - 1] - t) < abs(gts[j] - t):
            j -= 1
        return gtx[j]

    g = np.array([gt_at(k["ts"]) for k in kf])
    raw = np.array([k["raw"][:3] for k in kf])
    opt = np.array([k["opt"][:3] for k in kf])

    # start->end gap (loop should close): in the trajectory's own frame
    gap_raw = np.linalg.norm(raw[0] - raw[-1])
    gap_opt = np.linalg.norm(opt[0] - opt[-1])

    print(f"keyframes={len(kf)} loops={len(d['loops'])}")
    for ws in (False, True):
        tag = "sim(scale)" if ws else "rigid"
        print(f"  [{tag:10}] ATE raw(go2 odom) = {ate(raw, g, ws):.3f} m  ->  "
              f"ATE pgo = {ate(opt, g, ws):.3f} m")
    print(f"  loop-closure gap: raw {gap_raw:.2f} m -> pgo {gap_opt:.2f} m")


if __name__ == "__main__":
    main()
