#!/usr/bin/env python3
"""Benchmark PGO on the Go2 onboard source (REAL drift), scored against the
GTSAM groundtruth (gtsam_odom) -- the AprilTag-corrected trajectory from
run/add_gt, NOT raw fastlio. fastlio is unreliable on hard scenes (it drifted
~130 m on the grass field); gtsam_odom is tag-consistent everywhere, so it is the
correct groundtruth. Pass --gt <gtsam_odom.tum>.

The Go2 `odom` is a worse, independent sensor in its own frame with real
accumulated drift. We do NOT inject drift; we run PGO on the raw Go2 trajectory +
Go2 clouds and ask whether loop closure pulls it toward the gtsam_odom truth.
Because the sensors live in different frames we rigidly align (Umeyama, with and
without scale) before computing ATE -- reported before (raw) and after PGO.

    python3 go2_eval.py --go2 <dir> --gt <gtsam_odom.tum> [key=val ...]
"""
import json, math, os, subprocess, sys, tempfile, bisect
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(ROOT)
HARNESS_RUST = f"{ROOT}/rust/target/release/pgo_bench_rs"
HARNESS_CPP = f"{ROOT}/harness/build/pgo_bench"
# Defaults; override with --go2 <dir> and --gt <gtsam_odom.tum>.
GO2 = f"{REPO}/data/loop_bench/outdoor_small_loop_go2"
GT = os.path.expanduser("~/datasets/fastlio_recordings/gtsam_odom.tum")  # gtsam_odom GT


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
    args = sys.argv[1:]
    go2_dir, gt_path = GO2, GT
    i = 0
    while i < len(args):
        if args[i] == "--go2":
            go2_dir = args[i + 1]; i += 2
        elif args[i] == "--gt":
            gt_path = args[i + 1]; i += 2
        else:
            i += 1
    extra = [a for a in args if "=" in a]
    gts, gtx = read_tum(gt_path)

    # backend=rust -> rust harness; otherwise the C++ harness (impl=plane selects
    # the point-to-plane C++ port). Mirrors run_bench.py's harness switch.
    backend = "cpp"
    passthru = []
    for a in extra:
        if a.startswith("backend="):
            backend = a.split("=", 1)[1]
        else:
            passthru.append(a)
    harness = HARNESS_RUST if backend == "rust" else HARNESS_CPP

    out = tempfile.mktemp(suffix=".json")
    cmd = [harness, "--clouds", f"{go2_dir}/clouds.bin", "--poses", f"{go2_dir}/lidar_poses.tum",
           "--out", out, *passthru]
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
