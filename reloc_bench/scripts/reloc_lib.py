"""Shared utilities for the relocalization benchmark.

The relocalizer under test (C++ ICPLocalizer) is treated as a black-box backend
with a fixed CLI contract (see harness/reloc_bench.cpp). This module owns the
*neutral* pieces a future Rust backend will share unchanged: synthetic-cloud
generation, PCD / scans.bin IO, deterministic guess perturbation, and the
error metrics. Keeping these here (not in the backend) is what makes the
C++-vs-Rust comparison apples-to-apples.

Pose convention everywhere: a pose is (t, q) with t=(3,) translation and
q=(4,) quaternion in (x,y,z,w) order, representing child-in-parent
(point_parent = R(q) @ point_child + t). For relocalization, the pose we care
about is body->map (a.k.a. lidar-in-world): the transform the algorithm must
recover that maps a live body-frame scan into the prior-map frame.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

Pose = tuple[np.ndarray, np.ndarray]  # (t[3], q[4] xyzw)


# ----------------------------- pose algebra -----------------------------
def pose_mul(a: Pose, b: Pose) -> Pose:
    """Compose: result = a ∘ b (apply b, then a)."""
    ta, qa = a
    tb, qb = b
    Ra = Rotation.from_quat(qa)
    return Ra.apply(tb) + ta, (Ra * Rotation.from_quat(qb)).as_quat()


def pose_inv(a: Pose) -> Pose:
    ta, qa = a
    Rinv = Rotation.from_quat(qa).inv()
    return -Rinv.apply(ta), Rinv.as_quat()


def pose_apply(a: Pose, pts: np.ndarray) -> np.ndarray:
    """Apply pose to an (N,3) array of points."""
    t, q = a
    return Rotation.from_quat(q).apply(pts) + t


def pose_to_row(p: Pose) -> list[float]:
    t, q = p
    return [*map(float, t), *map(float, q)]


def pose_from_row(row) -> Pose:
    row = np.asarray(row, float)
    return row[:3].copy(), row[3:7].copy()


def trans_rot_err(recovered: Pose, truth: Pose) -> tuple[float, float]:
    """(translation error m, rotation error deg) between two poses."""
    tr = float(np.linalg.norm(recovered[0] - truth[0]))
    dR = Rotation.from_quat(recovered[1]) * Rotation.from_quat(truth[1]).inv()
    return tr, float(np.degrees(dR.magnitude()))


# ----------------------------- guess perturbation -----------------------------
def perturb(truth: Pose, trans_m: float, yaw_deg: float, rng: np.random.Generator) -> Pose:
    """Offset a truth pose by ~trans_m metres (random direction) and yaw_deg
    degrees about world-z, plus a little roll/pitch (10% of yaw). Deterministic
    given ``rng``. Models the operator's initial-pose guess fed to relocalize().
    """
    t, q = truth
    direction = rng.standard_normal(3)
    norm = np.linalg.norm(direction)
    direction = direction / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])
    dt = direction * trans_m
    sign = 1.0 if rng.random() < 0.5 else -1.0
    dyaw = np.deg2rad(yaw_deg) * sign
    small = np.deg2rad(yaw_deg * 0.1)
    dR = Rotation.from_euler(
        "ZYX", [dyaw, rng.uniform(-small, small), rng.uniform(-small, small)]
    )
    # world-frame left-multiply, matching how a global pose guess is expressed
    return t + dt, (dR * Rotation.from_quat(q)).as_quat()


# ----------------------------- synthetic clouds -----------------------------
def synthetic_room(n_points: int = 12000, size=(10.0, 10.0, 3.0), seed: int = 42) -> np.ndarray:
    """Structured (N,3) cloud: floor + ceiling + 4 walls of a box room.

    Structured walls give ICP translation+yaw observability; a bare floor or a
    single wall is degenerate. Mirrors the synthetic room in dimos'
    test_relocalize.py so synthetic results are comparable across stacks.
    """
    rng = np.random.default_rng(seed)
    hx, hy, hz = size[0] / 2, size[1] / 2, size[2]
    per = n_points // 6
    faces = [
        np.column_stack([rng.uniform(-hx, hx, per), rng.uniform(-hy, hy, per), np.zeros(per)]),
        np.column_stack([rng.uniform(-hx, hx, per), rng.uniform(-hy, hy, per), np.full(per, hz)]),
        np.column_stack([rng.uniform(-hx, hx, per), np.full(per, hy), rng.uniform(0, hz, per)]),
        np.column_stack([rng.uniform(-hx, hx, per), np.full(per, -hy), rng.uniform(0, hz, per)]),
        np.column_stack([np.full(per, hx), rng.uniform(-hy, hy, per), rng.uniform(0, hz, per)]),
        np.column_stack([np.full(per, -hx), rng.uniform(-hy, hy, per), rng.uniform(0, hz, per)]),
    ]
    return np.concatenate(faces).astype(np.float64)


def add_noise(pts: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    if sigma <= 0:
        return pts
    rng = np.random.default_rng(seed)
    return pts + rng.normal(0.0, sigma, pts.shape)


# ----------------------------- voxel downsample -----------------------------
def voxel_downsample(pts: np.ndarray, voxel: float) -> np.ndarray:
    """Centroid voxel-grid downsample of an (N,3) cloud (numpy, no PCL)."""
    if voxel <= 0 or len(pts) == 0:
        return pts
    keys = np.floor(pts / voxel).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.ravel()
    m = inv.max() + 1
    counts = np.bincount(inv, minlength=m).astype(np.float64)
    out = np.empty((m, 3))
    for d in range(3):
        out[:, d] = np.bincount(inv, weights=pts[:, d], minlength=m) / counts
    return out


# ----------------------------- IO -----------------------------
def write_pcd(path: str | Path, pts: np.ndarray, intensity: np.ndarray | None = None) -> None:
    """Write an (N,3) cloud as a binary PCD with x y z intensity (float32 packed),
    the layout pcl::PCDReader expects for pcl::PointXYZI on-disk data."""
    pts = np.asarray(pts, np.float32)
    n = len(pts)
    if intensity is None:
        intensity = np.zeros(n, np.float32)
    intensity = np.asarray(intensity, np.float32).reshape(n, 1)
    packed = np.hstack([pts, intensity]).astype(np.float32)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA binary\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(packed.tobytes())


def write_scans_bin(path: str | Path, clouds: list[np.ndarray]) -> None:
    """Write clouds in the dataset clouds.bin layout: per cloud [int32 n][n*(x,y,z,i) f32]."""
    with open(path, "wb") as f:
        for c in clouds:
            c = np.asarray(c, np.float32)
            n = len(c)
            if c.shape[1] == 3:
                c = np.hstack([c, np.zeros((n, 1), np.float32)])
            f.write(struct.pack("<i", n))
            f.write(c.astype(np.float32).tobytes())


def iter_scans_bin(path: str | Path):
    """Yield (idx, (N,4) float32) clouds from a clouds.bin / scans.bin file."""
    with open(path, "rb") as f:
        idx = 0
        while True:
            head = f.read(4)
            if len(head) < 4:
                break
            (n,) = struct.unpack("<i", head)
            buf = f.read(n * 4 * 4)
            yield idx, np.frombuffer(buf, dtype=np.float32).reshape(n, 4)
            idx += 1


def load_tum(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a TUM file -> (ts (N,), poses (N,7) tx ty tz qx qy qz qw)."""
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr[:, 0].copy(), arr[:, 1:8].copy()


def write_trials(path: str | Path, trials: list[tuple[int, Pose]]) -> None:
    """trials: (scan_idx, guess_pose). One 'idx tx ty tz qx qy qz qw' line each."""
    with open(path, "w") as f:
        for idx, guess in trials:
            f.write(f"{idx} " + " ".join(f"{v:.9g}" for v in pose_to_row(guess)) + "\n")


def read_results(path: str | Path) -> list[tuple[bool, Pose, float]]:
    """Read the harness output -> [(converged, recovered_pose, time_ms), ...]."""
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        v = line.split()
        conv = v[0] == "1"
        pose = (np.array(list(map(float, v[1:4]))), np.array(list(map(float, v[4:8]))))
        out.append((conv, pose, float(v[8])))
    return out
