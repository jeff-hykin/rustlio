#!/usr/bin/env python3
"""
Faithfulness comparison: overlay the Rust rustlio trajectory against the
upstream C++ FAST-LIO trajectory, truncated where upstream diverges.

Inputs:
  upstream.npy : [t_epoch, x, y, z]             (from extract_upstream.py)
  rust.npy     : [t_epoch, x, y, z, vx, vy, vz] (fastlio2 binary output)

Divergence cutoff: the robot (Go2) is physically bounded (<=~3.1 m/s, flat
ground). When C++ FAST-LIO breaks it teleports — one odom step jumps far beyond
what the robot can travel in a frame. We cut at the FIRST inter-sample position
jump exceeding --jump-thresh metres and compare only the data before it.
(Instantaneous pose-delta *speed* is unusable: the 30 Hz odom has irregular dt,
so a normal 0.18 m motion over a 3 ms gap reads as 50 m/s. Jump magnitude is
dt-independent.)

Alignment: both stacks gravity-align (z = up) and start at the origin, so the
only legitimate frame freedom is YAW + translation. We align rust->upstream with
a yaw-only (rotation about z) + 3D-translation fit and report XY and Z error
SEPARATELY. A full-SO(3) fit would spuriously tilt to absorb XY drift into Z and
hide real vertical drift. We also fit a similarity scale (≈1 expected; a value
off 1 flags a metric error such as accel-unit scaling).
"""
import sys, os, argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def uniform_speed(t, xyz, hz, t_end):
    tu = np.arange(0.0, t_end, 1.0 / hz)
    xu = np.vstack([np.interp(tu, t, xyz[:, k]) for k in range(3)]).T
    return tu[1:], np.linalg.norm(np.diff(xu, axis=0), axis=1) * hz


def smooth(x, w):
    if w <= 1:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def yaw_align(src, dst):
    """Best yaw (about z) + 3D translation mapping src->dst. Returns (theta, aligned)."""
    a = src[:, :2] - src[:, :2].mean(0)
    b = dst[:, :2] - dst[:, :2].mean(0)
    H = a.T @ b
    theta = np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
    c, s = np.cos(theta), np.sin(theta)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    al = (Rz @ src.T).T
    al = al - al.mean(0) + dst.mean(0)
    return np.degrees(theta), al


def sim_scale(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(d0.T @ s0 / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    return float((D * np.diag(S)).sum() / ((s0 ** 2).sum() / len(src)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("upstream")
    ap.add_argument("rust")
    ap.add_argument("out_dir")
    ap.add_argument("--jump-thresh", type=float, default=0.3,
                    help="upstream single-step jump (m) marking divergence")
    ap.add_argument("--speed-limit", type=float, default=3.1)
    ap.add_argument("--tolerance", type=float, default=0.5,
                    help="3D aligned-ATE RMSE (m) PASS threshold")
    ap.add_argument("--hz", type=float, default=20.0)
    args = ap.parse_args()

    up = np.load(args.upstream)
    ru = np.load(args.rust)
    up_t, up_xyz = up[:, 0] - up[0, 0], up[:, 1:4]
    ru_t, ru_xyz = ru[:, 0] - ru[0, 0], ru[:, 1:4]
    ru_v = np.linalg.norm(ru[:, 4:7], axis=1) if ru.shape[1] >= 7 else None

    # --- divergence cutoff (first big inter-sample jump on upstream) ---
    jump = np.linalg.norm(np.diff(up_xyz, axis=0), axis=1)
    di = np.where(jump > args.jump_thresh)[0]
    diverged = len(di) > 0
    cutoff = up_t[di[0] + 1] if diverged else up_t[-1]
    print(f"[divergence] first upstream jump > {args.jump_thresh} m at t={cutoff:.2f}s "
          f"(jump={jump[di[0]]:.2f} m)" if diverged
          else f"[divergence] none > {args.jump_thresh} m; comparing full {cutoff:.1f}s")
    t_cmp = min(cutoff, ru_t[-1])

    # --- speed profiles (plot + coarse offset seed) ---
    up_st, up_spd = uniform_speed(up_t, up_xyz, args.hz, t_cmp)
    ru_st, ru_spd = (ru_t, ru_v) if ru_v is not None else uniform_speed(ru_t, ru_xyz, args.hz, t_cmp)

    win = up_t <= t_cmp
    ut, uxyz0 = up_t[win], up_xyz[win] - up_xyz[win][0]

    def at_offset(dt):
        r = np.vstack([np.interp(ut, ru_t + dt, ru_xyz[:, k]) for k in range(3)]).T
        r = r - r[0]
        _, al = yaw_align(r, uxyz0)
        return np.sqrt((np.linalg.norm(uxyz0 - al, axis=1) ** 2).mean()), r

    # joint time+space: clock offset minimising yaw-aligned RMSE
    cands = np.arange(-3.0, 3.0 + 1e-9, 1.0 / args.hz)
    dt_off = float(min(cands, key=lambda d: at_offset(d)[0]))
    _, rxyz0 = at_offset(dt_off)
    print(f"[time] rust clock offset {dt_off:+.2f}s (refined by min-ATE)")

    yaw, ral = yaw_align(rxyz0, uxyz0)
    scale = sim_scale(rxyz0, uxyz0)

    raw_err = np.linalg.norm(uxyz0 - rxyz0, axis=1)
    xy_err = np.linalg.norm(uxyz0[:, :2] - ral[:, :2], axis=1)
    z_err = np.abs(uxyz0[:, 2] - ral[:, 2])
    d3_err = np.linalg.norm(uxyz0 - ral, axis=1)

    def rms(e):
        return np.sqrt((e ** 2).mean())

    up_path = jump[up_t[1:] <= t_cmp].sum()
    ru_path = np.linalg.norm(np.diff(rxyz0, axis=0), axis=1).sum()
    within = float(np.mean(d3_err <= args.tolerance) * 100)
    passed = rms(d3_err) <= args.tolerance

    print("\n================ FAITHFULNESS REPORT ================")
    print(f"compared window     : 0 .. {t_cmp:.1f}s  ({win.sum()} upstream poses)")
    print(f"path length         : upstream {up_path:.1f} m | rust {ru_path:.1f} m "
          f"(ratio {ru_path/up_path:.3f})")
    print(f"similarity scale    : {scale:.4f}   yaw-align angle: {yaw:+.1f} deg")
    print(f"raw ATE (origin)    : mean {raw_err.mean():.3f}  RMSE {rms(raw_err):.3f}  "
          f"max {raw_err.max():.3f} m")
    print(f"yaw-aligned XY-RMSE : {rms(xy_err):.3f} m   (max {xy_err.max():.3f})")
    print(f"yaw-aligned  Z-RMSE : {rms(z_err):.3f} m   (rust raw z-range "
          f"[{rxyz0[:,2].min():.2f},{rxyz0[:,2].max():.2f}], upstream "
          f"[{uxyz0[:,2].min():.2f},{uxyz0[:,2].max():.2f}])")
    print(f"yaw-aligned 3D-RMSE : {rms(d3_err):.3f} m   (max {d3_err.max():.3f}, "
          f"final {d3_err[-1]:.3f})")
    print(f"within {args.tolerance:.2f} m (3D): {within:.1f}% of samples")
    print(f"RESULT              : {'PASS' if passed else 'FAIL'} "
          f"(3D-RMSE {rms(d3_err):.3f} {'<=' if passed else '>'} tol {args.tolerance:.2f} m)")
    print("=====================================================\n")

    # --- plots ---
    os.makedirs(args.out_dir, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(14, 11))
    m = 2.0
    lim_x = (min(uxyz0[:, 0].min(), ral[:, 0].min()) - m, max(uxyz0[:, 0].max(), ral[:, 0].max()) + m)
    lim_y = (min(uxyz0[:, 1].min(), ral[:, 1].min()) - m, max(uxyz0[:, 1].max(), ral[:, 1].max()) + m)
    ax[0, 0].plot(uxyz0[:, 0], uxyz0[:, 1], "b-", lw=1.8, label="upstream C++")
    ax[0, 0].plot(ral[:, 0], ral[:, 1], "r-", lw=1.8, label="rust (yaw-aligned)")
    ax[0, 0].plot(0, 0, "ko", ms=6)
    ax[0, 0].set(title="XY top-down (pre-divergence window)", xlabel="x [m]", ylabel="y [m]",
                 xlim=lim_x, ylim=lim_y)
    ax[0, 0].set_aspect("equal", "box"); ax[0, 0].legend(fontsize=9); ax[0, 0].grid(alpha=0.3)

    ax[0, 1].plot(ut, uxyz0[:, 2], "b-", label="upstream z")
    ax[0, 1].plot(ut, ral[:, 2], "r-", label="rust z (aligned)")
    ax[0, 1].set(title=f"Z vs time  (Z-RMSE {rms(z_err):.2f} m — vertical drift)",
                 xlabel="t [s]", ylabel="z [m]")
    ax[0, 1].legend(fontsize=9); ax[0, 1].grid(alpha=0.3)

    ax[1, 0].plot(up_st, smooth(up_spd, 11), "b-", lw=1.2, label="upstream (pose-Δ, smoothed)")
    ax[1, 0].plot(ru_st + dt_off, ru_spd, "r-", lw=1.0, alpha=0.8, label="rust (EKF v)")
    ax[1, 0].axhline(args.speed_limit, color="k", ls="--", lw=0.8, label=f"{args.speed_limit} m/s")
    ax[1, 0].set(title="speed vs time", xlabel="t [s]", ylabel="m/s",
                 xlim=(0, t_cmp), ylim=(0, args.speed_limit * 1.6))
    ax[1, 0].legend(fontsize=9); ax[1, 0].grid(alpha=0.3)

    ax[1, 1].plot(ut, xy_err, "g-", lw=1.3, label="XY error")
    ax[1, 1].plot(ut, z_err, color="purple", lw=1.3, label="Z error")
    ax[1, 1].plot(ut, d3_err, "k-", lw=1.0, alpha=0.6, label="3D error")
    ax[1, 1].axhline(args.tolerance, color="k", ls="--", lw=0.8, label=f"tol {args.tolerance} m")
    ax[1, 1].set(title="aligned position error vs time", xlabel="t [s]", ylabel="error [m]")
    ax[1, 1].legend(fontsize=9); ax[1, 1].grid(alpha=0.3)

    fig.suptitle(f"rustlio vs upstream C++ FAST-LIO  —  {'PASS' if passed else 'FAIL'}  "
                 f"(3D-RMSE {rms(d3_err):.2f} m, XY {rms(xy_err):.2f}, Z {rms(z_err):.2f}, "
                 f"scale {scale:.3f})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    png = os.path.join(args.out_dir, "faithful_check.png")
    fig.savefig(png, dpi=110)
    print(f"saved overlay plot -> {png}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
