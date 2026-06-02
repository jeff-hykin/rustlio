#!/usr/bin/env python
"""Export a KITTI odometry sequence (00-10 have GT) to the neutral format.

velodyne/*.bin are in the velodyne (body) frame; GT trajectory = velodyne pose in
world = (cam0 pose in world, poses/XX.txt) @ (velo->cam, calib Tr). Scored by ATE
vs GT (no markers). Sequences 11-21 have no public GT.

    python export_kitti_odometry.py --root ~/datasets/kitti_odometry/dataset \
        --seq 00 --out ~/repos/FASTLIO2_ROS2/data/loop_bench/kitti_00 --voxel 0.5
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

from export_kitti360 import mat_to_tum, voxel_downsample  # reuse helpers


def load_poses(path: Path) -> list[np.ndarray]:
    out = []
    for line in path.read_text().splitlines():
        v = np.array([float(x) for x in line.split()], dtype=np.float64)
        m = np.eye(4)
        m[:3, :4] = v.reshape(3, 4)
        out.append(m)
    return out


def load_tr(calib_path: Path) -> np.ndarray:
    for line in calib_path.read_text().splitlines():
        if line.startswith("Tr:"):
            v = np.array([float(x) for x in line.split()[1:]], dtype=np.float64)
            m = np.eye(4)
            m[:3, :4] = v.reshape(3, 4)
            return m
    raise ValueError("no Tr in calib")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help=".../kitti_odometry/dataset")
    ap.add_argument("--seq", required=True, help="e.g. 00")
    ap.add_argument("--out", required=True)
    ap.add_argument("--voxel", type=float, default=0.5)
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    seq_dir = root / "sequences" / args.seq
    poses = load_poses(root / "poses" / f"{args.seq}.txt")
    tr = load_tr(seq_dir / "calib.txt")  # velo -> cam
    times_path = seq_dir / "times.txt"
    times = (
        [float(x) for x in times_path.read_text().split()]
        if times_path.exists()
        else None
    )

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    bins = sorted((seq_dir / "velodyne").glob("*.bin"))

    tum_lines = []
    n = 0
    with open(out / "clouds.bin", "wb") as cf:
        for b in bins:
            frame = int(b.stem)
            if frame >= len(poses):
                continue
            velo_to_world = poses[frame] @ tr
            pts = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
            pts = voxel_downsample(pts, args.voxel)
            ts = times[frame] if times and frame < len(times) else frame / 10.0
            x, y, z, qx, qy, qz, qw = mat_to_tum(velo_to_world)
            tum_lines.append(
                f"{ts:.9f} {x:.9f} {y:.9f} {z:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}"
            )
            cf.write(struct.pack("<i", pts.shape[0]))
            xyzi = np.zeros((pts.shape[0], 4), dtype=np.float32)
            xyzi[:, :3] = pts[:, :3]
            xyzi[:, 3] = pts[:, 3]
            cf.write(xyzi.tobytes())
            n += 1
    (out / "lidar_poses.tum").write_text("\n".join(tum_lines) + "\n")
    (out / "markers.json").write_text("[]")
    (out / "meta.json").write_text(
        json.dumps(
            {"source": f"kitti_odometry/{args.seq}", "num_lidar": n, "num_marker_obs": 0,
             "voxel": args.voxel}, indent=2
        )
    )
    print(f"[kitti-odo] seq {args.seq}: wrote {n} scans (of {len(bins)} bins) -> {out}")


if __name__ == "__main__":
    main()
