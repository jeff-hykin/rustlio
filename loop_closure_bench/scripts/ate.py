#!/usr/bin/env python3
"""Trusted ATE eval (matches run_bench.py / committed TSV). Runs the Rust harness
on a KITTI seq's CLEAN poses (yaw=0) and prints ATE of optimized vs clean GT.
For drift runs use run_bench.py. Usage: python3 ate.py <seq> [key=val ...]"""
import json, math, bisect, os, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(ROOT)
HARNESS = f"{ROOT}/rust/target/release/pgo_bench_rs"
seq = sys.argv[1]
extra = [a for a in sys.argv[2:] if "=" in a]
ds = f"{REPO}/data/loop_bench/kitti_{seq}"

ts, xyz = [], []
for l in open(f"{ds}/lidar_poses.tum"):
    v = l.split()
    if len(v) >= 8:
        ts.append(float(v[0])); xyz.append([float(v[1]), float(v[2]), float(v[3])])

def interp(t):
    j = bisect.bisect_left(ts, t)
    if j <= 0: return xyz[0]
    if j >= len(ts): return xyz[-1]
    t0, t1 = ts[j-1], ts[j]; a = 0 if t1 == t0 else (t-t0)/(t1-t0)
    return [xyz[j-1][k] + a*(xyz[j][k]-xyz[j-1][k]) for k in range(3)]

out = tempfile.mktemp(suffix=".json")
cmd = [HARNESS, "--clouds", f"{ds}/clouds.bin", "--poses", f"{ds}/lidar_poses.tum", "--out", out, *extra]
if os.environ.get("ATE_DEBUG"):
    print("argv:", sys.argv); print("extra:", extra)
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode != 0:
    print("FAILED:", r.stderr[-300:]); sys.exit(1)
stderr_kf = [l for l in r.stderr.splitlines() if "keyframes=" in l]
d = json.load(open(out)); os.unlink(out)
if os.environ.get("ATE_DEBUG"):
    print("stderr:", stderr_kf, "| json kf:", len(d["keyframes"]), "| out:", out)
se = [sum((k["opt"][i]-interp(k["ts"])[i])**2 for i in range(3)) for k in d["keyframes"]]
ate = math.sqrt(sum(se)/len(se))
print(f"seq{seq} kf={len(d['keyframes'])} loops={len(d['loops'])} ate_pgo={ate:.3f}")
