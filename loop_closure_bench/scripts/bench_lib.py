"""Shared utilities for the PGO loop-closure benchmark.

Reads only the neutral files produced by export_dataset.py. No dimos / AprilTag
dependency. Pose convention everywhere: a pose is (t, q) with t = (3,) translation
and q = (4,) quaternion in (x, y, z, w) order, representing child-in-parent
(i.e. point_parent = R(q) @ point_child + t).
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

Pose = tuple[np.ndarray, np.ndarray]  # (t[3], q[4] xyzw)


# ----------------------------- pose algebra -----------------------------
def pose_mul(a: Pose, b: Pose) -> Pose:
    """Compose: result = a ∘ b (apply b, then a)."""
    ta, qa = a
    tb, qb = b
    Ra = Rotation.from_quat(qa)
    t = Ra.apply(tb) + ta
    q = (Ra * Rotation.from_quat(qb)).as_quat()
    return t, q


def pose_inv(a: Pose) -> Pose:
    ta, qa = a
    Rinv = Rotation.from_quat(qa).inv()
    return -Rinv.apply(ta), Rinv.as_quat()


def pose_apply(a: Pose, p: np.ndarray) -> np.ndarray:
    t, q = a
    return Rotation.from_quat(q).apply(p) + t


def pose_from_row(row: np.ndarray) -> Pose:
    """row = [tx ty tz qx qy qz qw]."""
    return np.asarray(row[:3], float), np.asarray(row[3:7], float)


def pose_to_row(p: Pose) -> list[float]:
    t, q = p
    return [*map(float, t), *map(float, q)]


# ----------------------------- loaders -----------------------------
@dataclass
class Dataset:
    root: Path
    meta: dict
    ts: np.ndarray            # (N,)
    poses: np.ndarray         # (N,7) tx ty tz qx qy qz qw  (clean lidar-in-world)
    markers: list[dict]       # raw marker obs

    @property
    def n(self) -> int:
        return len(self.ts)


def load_dataset(root: str | Path) -> Dataset:
    root = Path(root)
    meta = json.loads((root / "meta.json").read_text())
    tum = np.loadtxt(root / "lidar_poses.tum")
    if tum.ndim == 1:
        tum = tum[None, :]
    ts = tum[:, 0].copy()
    poses = tum[:, 1:8].copy()
    markers = json.loads((root / "markers.json").read_text())
    return Dataset(root=root, meta=meta, ts=ts, poses=poses, markers=markers)


def iter_clouds(root: str | Path):
    """Yield (idx, np.ndarray[N,4]) body-frame clouds in frame order."""
    path = Path(root) / "clouds.bin"
    with open(path, "rb") as f:
        idx = 0
        while True:
            head = f.read(4)
            if len(head) < 4:
                break
            (n,) = struct.unpack("<i", head)
            buf = f.read(n * 4 * 4)
            arr = np.frombuffer(buf, dtype=np.float32).reshape(n, 4)
            yield idx, arr
            idx += 1


# ----------------------------- marker metric -----------------------------
def marker_world_positions(ds: Dataset, cam_transform=None) -> dict[int, np.ndarray]:
    """Map marker_id -> (K,3) world positions of the marker across observations.

    Each obs gives T_world_marker = cam_in_world ∘ t_cam_marker. ``cam_transform``
    is an optional callable ts -> Pose applied (world-frame) to the camera pose
    first: cam' = cam_transform(ts) ∘ cam. Chain drift then PGO-correction here.
    """
    out: dict[int, list[np.ndarray]] = {}
    for m in ds.markers:
        cam = pose_from_row(np.asarray(m["cam_in_world"]))
        if cam_transform is not None:
            cam = pose_mul(cam_transform(m["ts"]), cam)
        t_cm = pose_from_row(np.asarray(m["t_cam_marker"]))
        world = pose_mul(cam, t_cm)
        out.setdefault(int(m["marker_id"]), []).append(world[0])
    return {k: np.array(v) for k, v in out.items()}


def _pairwise_sum(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    d = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(d, axis=2)
    return float(np.triu(dist, k=1).sum())


def spread(positions: dict[int, np.ndarray]) -> float:
    """Total marker spread (m): sum over marker_ids of pairwise-distance sum."""
    return float(sum(_pairwise_sum(p) for p in positions.values()))


# ----------------------------- corrections -----------------------------
class Correction:
    """ts -> world correction Transform (world_corrected <- world_raw), interpolated.

    Built from PGO output: per keyframe, correction = optimized ∘ raw^-1.
    SLERP rotation, lerp translation, clip to endpoints.
    """

    def __init__(self, ts: np.ndarray, corr_poses: np.ndarray):
        order = np.argsort(ts)
        ts = ts[order]
        corr_poses = corr_poses[order]
        # Slerp requires strictly increasing times; drop duplicate timestamps.
        keep = np.concatenate([[True], np.diff(ts) > 0])
        self.ts = ts[keep]
        self.t = corr_poses[keep, :3]
        self.rots = Rotation.from_quat(corr_poses[keep, 3:7])
        if len(self.ts) == 1:  # pad so Slerp works
            self.ts = np.array([self.ts[0], self.ts[0] + 1.0])
            self.t = np.vstack([self.t, self.t])
            self.rots = Rotation.from_quat(np.vstack([self.rots.as_quat(), self.rots.as_quat()]))
        self.slerp = Slerp(self.ts, self.rots)

    def at(self, ts: float) -> Pose:
        ts = float(np.clip(ts, self.ts[0], self.ts[-1]))
        t = np.array([np.interp(ts, self.ts, self.t[:, i]) for i in range(3)])
        q = self.slerp([ts])[0].as_quat()
        return t, q

    @staticmethod
    def from_keyframes(ts: np.ndarray, raw: np.ndarray, optimized: np.ndarray) -> "Correction":
        corr = np.zeros((len(ts), 7))
        for i in range(len(ts)):
            c = pose_mul(pose_from_row(optimized[i]), pose_inv(pose_from_row(raw[i])))
            corr[i] = pose_to_row(c)
        return Correction(ts, corr)


# ----------------------------- drift injection -----------------------------
def inject_drift(
    ts: np.ndarray,
    poses: np.ndarray,
    *,
    trans_per_m: float = 0.0,
    yaw_deg_per_m: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Add accumulating SE(3) drift to a clean trajectory.

    Models odometry error as a small per-meter-travelled body-frame perturbation
    that integrates along the path (so error grows with distance, like real
    SLAM drift). trans_per_m: random-walk translation std per metre travelled.
    yaw_deg_per_m: random-walk yaw std (deg) per metre travelled -- yaw error is
    what actually bends a trajectory into needing loop closure.

    Returns (drifted_poses (N,7), drift_field) where drift_field is a Correction
    giving D(ts) = drifted ∘ clean^-1, so callers can apply the same drift to the
    camera poses by ts. Drift is a world-frame left-multiply accumulated over the
    path, keeping the first pose fixed.
    """
    rng = np.random.default_rng(seed)
    n = len(ts)
    out = np.zeros_like(poses)
    drift_poses = np.zeros((n, 7))  # the D transforms themselves
    drift_t = np.zeros(3)
    drift_R = Rotation.identity()
    prev_p = poses[0, :3]
    for i in range(n):
        p = poses[i]
        step = float(np.linalg.norm(p[:3] - prev_p)) if i > 0 else 0.0
        prev_p = p[:3]
        if step > 0:
            dyaw = np.deg2rad(yaw_deg_per_m) * np.sqrt(step) * rng.standard_normal()
            dtr = trans_per_m * np.sqrt(step) * rng.standard_normal(3)
            drift_R = Rotation.from_euler("z", dyaw) * drift_R
            drift_t = drift_t + drift_R.apply(dtr)
        # apply accumulated drift in world frame: pose' = D ∘ pose
        out[i, :3] = drift_R.apply(p[:3]) + drift_t
        out[i, 3:7] = (drift_R * Rotation.from_quat(p[3:7])).as_quat()
        drift_poses[i, :3] = drift_t
        drift_poses[i, 3:7] = drift_R.as_quat()
    return out, Correction(ts.copy(), drift_poses)
