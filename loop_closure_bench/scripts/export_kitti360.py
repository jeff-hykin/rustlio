#!/usr/bin/env python
"""Export a KITTI-360 sequence to the loop-closure-bench neutral format.

KITTI-360 velodyne scans are in the velodyne (body) frame; the GT trajectory is
the velodyne pose in world, derived from cam0_to_world and the cam0->velo
calibration. This lets the PGO harness run on real urban driving sequences with
large loops, scored by ATE vs the GT trajectory (no AprilTag markers here).

    python export_kitti360.py --root ~/datasets/kitti360 --seq 0008 \
        --out ~/repos/FASTLIO2_ROS2/data/loop_bench/kitti_0008 --voxel 0.5

Outputs: clouds.bin (body, voxel-downsampled), lidar_poses.tum (velo-in-world),
markers.json ([] -- no markers), meta.json.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def load_mat34(path: Path) -> dict[int, np.ndarray]:
    """Parse 'frame m00 m01 ... m23' lines -> {frame: 4x4}."""
    out = {}
    for line in path.read_text().splitlines():
        t = line.split()
        if not t:
            continue
        frame = int(t[0])
        vals = np.array([float(x) for x in t[1:]], dtype=np.float64)
        m = np.eye(4)
        m[:3, :4] = vals[:12].reshape(3, 4)
        out[frame] = m
    return out


def load_calib_4x4(path: Path) -> np.ndarray:
    vals = np.array([float(x) for x in path.read_text().split()], dtype=np.float64)
    m = np.eye(4)
    m[:3, :4] = vals[:12].reshape(3, 4)
    return m


def voxel_downsample(pts: np.ndarray, res: float) -> np.ndarray:
    if res <= 0 or len(pts) == 0:
        return pts
    keys = np.floor(pts[:, :3] / res).astype(np.int64)
    # unique voxel -> first point (fast; centroid not needed for ICP submaps)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(idx)]


def mat_to_tum(m: np.ndarray) -> tuple[float, ...]:
    from numpy.linalg import norm

    t = m[:3, 3]
    r = m[:3, :3]
    # rotation matrix -> quaternion (x,y,z,w)
    tr = np.trace(r)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    q /= norm(q)
    return (t[0], t[1], t[2], q[0], q[1], q[2], q[3])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--seq", required=True, help="e.g. 0008")
    ap.add_argument("--out", required=True)
    ap.add_argument("--voxel", type=float, default=0.5)
    ap.add_argument("--rate-hz", type=float, default=10.0, help="velodyne rate for synthetic ts")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    seq = f"2013_05_28_drive_{args.seq}_sync"
    velo_dir = root / "data_3d_raw" / seq / "velodyne_points" / "data"
    cam0_to_world = load_mat34(root / "data_poses" / seq / "cam0_to_world.txt")
    T_velo_cam = load_calib_4x4(root / "calibration" / "calib_cam_to_velo.txt")
    cam0_from_velo = np.linalg.inv(T_velo_cam)  # velo point -> cam0 point

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    bins = sorted(velo_dir.glob("*.bin"))
    tum_lines = []
    n_written = 0
    with open(out / "clouds.bin", "wb") as cf:
        for i, b in enumerate(bins):
            frame = int(b.stem)
            if frame not in cam0_to_world:
                continue
            velo_to_world = cam0_to_world[frame] @ cam0_from_velo
            pts = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
            pts = voxel_downsample(pts, args.voxel)
            n = pts.shape[0]
            ts = frame / args.rate_hz
            x, y, z, qx, qy, qz, qw = mat_to_tum(velo_to_world)
            tum_lines.append(
                f"{ts:.9f} {x:.9f} {y:.9f} {z:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}"
            )
            cf.write(struct.pack("<i", n))
            xyzi = np.zeros((n, 4), dtype=np.float32)
            xyzi[:, :3] = pts[:, :3]
            xyzi[:, 3] = pts[:, 3]
            cf.write(xyzi.tobytes())
            n_written += 1
    (out / "lidar_poses.tum").write_text("\n".join(tum_lines) + "\n")
    (out / "markers.json").write_text("[]")
    (out / "meta.json").write_text(
        json.dumps(
            {
                "source": f"kitti360/{seq}",
                "num_lidar": n_written,
                "num_marker_obs": 0,
                "voxel": args.voxel,
                "note": "KITTI-360 velodyne; trajectory = velo-in-world (cam0_to_world @ inv(cam_to_velo))",
            },
            indent=2,
        )
    )
    print(f"[kitti] seq {args.seq}: wrote {n_written} scans (of {len(bins)} bins) -> {out}")


if __name__ == "__main__":
    main()
