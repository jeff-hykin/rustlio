"""Generate relocalization benchmark scenarios (deterministic, seeded).

Each scenario is a directory under reloc_bench/scenarios/<name>/ containing:
  map.pcd        prior map (loaded by the backend via its real loadMap path)
  scans.bin      query clouds, dataset clouds.bin layout (body frame)
  manifest.json  {scenario, map, scans, buckets, trials:[{scan_idx, bucket,
                  guess[7], truth[7]}]} — the driver's source of truth.

The backend (C++ ICPLocalizer, or a future Rust one) only ever sees map.pcd,
scans.bin, and a trials.txt the driver derives from the manifest. Truth lives
only here, so every backend is scored identically.

Two scenario families:
  synthetic_room  full box-room map; each query is the room seen from a known
                  pose. Clean observability — measures the convergence basin and
                  best-case accuracy without real-sensor confounds.
  hk_village3     real Go2 LiDAR. Prior map = stitched world-frame scans; each
                  query is a real body-frame scan whose truth pose is the
                  FAST-LIO odometry. The realistic case.

Initial-guess error is swept in BUCKETS (trans_m, yaw_deg): the operator's pose
guess is rarely exact, and how much error the algorithm tolerates before it
stops converging is the headline number for comparing against an enhanced Rust
relocalizer.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer
from scipy.spatial.transform import Rotation

import reloc_lib as R

# (label, translation offset m, yaw offset deg)
BUCKETS = [
    ("exact", 0.0, 0.0),
    ("near", 0.3, 5.0),
    ("mid", 1.0, 15.0),
    ("far", 2.5, 30.0),
    ("extreme", 5.0, 60.0),
]

HERE = Path(__file__).resolve().parent
SCEN_ROOT = HERE.parent / "scenarios"


def _emit(scen_dir: Path, scenario: str, map_path: Path, scans_path: Path, trials: list[dict]) -> None:
    manifest = {
        "scenario": scenario,
        "map": str(map_path),
        "scans": str(scans_path),
        "buckets": [b[0] for b in BUCKETS],
        "trials": trials,
    }
    (scen_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  {scenario}: {len(trials)} trials -> {scen_dir}")


def gen_synthetic(reps: int, seed: int) -> None:
    scen_dir = SCEN_ROOT / "synthetic_room"
    scen_dir.mkdir(parents=True, exist_ok=True)
    room = R.synthetic_room(n_points=15000, seed=seed)
    R.write_pcd(scen_dir / "map.pcd", room)

    # A handful of viewpoints inside the room; each becomes one query scan
    # (the room as seen from that pose, body frame = inv(truth) @ map).
    rng = np.random.default_rng(seed)
    truths: list[R.Pose] = []
    for _ in range(6):
        t = np.array([rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(0.3, 1.5)])
        yaw = rng.uniform(-np.pi, np.pi)
        truths.append((t, Rotation.from_euler("ZYX", [yaw, 0, 0]).as_quat()))

    scans = [R.pose_apply(R.pose_inv(tr), room).astype(np.float32) for tr in truths]
    R.write_scans_bin(scen_dir / "scans.bin", scans)

    prng = np.random.default_rng(seed + 1)
    trials: list[dict] = []
    for idx, truth in enumerate(truths):
        for label, dt, dy in BUCKETS:
            n = 1 if dt == 0.0 and dy == 0.0 else reps
            for _ in range(n):
                guess = R.perturb(truth, dt, dy, prng)
                trials.append({
                    "scan_idx": idx, "bucket": label,
                    "guess": R.pose_to_row(guess), "truth": R.pose_to_row(truth),
                })
    _emit(scen_dir, "synthetic_room", scen_dir / "map.pcd", scen_dir / "scans.bin", trials)


def gen_real(dataset: Path, reps: int, seed: int, query_stride: int,
             map_stride: int, map_voxel: float, name: str = "hk_village3") -> None:
    if not dataset.exists():
        print(f"  (skipping {name}: {dataset} not found)")
        return
    scen_dir = SCEN_ROOT / name
    scen_dir.mkdir(parents=True, exist_ok=True)

    _ts, poses = R.load_tum(dataset / "lidar_poses.tum")
    clouds = [c for _, c in R.iter_scans_bin(dataset / "clouds.bin")]
    n = min(len(poses), len(clouds))

    # Frame guard: the stock relocalizer aligns a BODY-frame scan to the map via
    # a body->world guess, so queries must be body-frame. Recover body if the
    # export is world-frame (see gen_global detection).
    samp = range(0, n, max(1, n // 40))
    cmag = np.median([np.linalg.norm(clouds[i][:, :3].mean(0)) for i in samp])
    pmag = np.median([np.linalg.norm(poses[i][:3]) for i in samp])
    world_frame = cmag > max(15.0, 0.5 * pmag)
    print(f"  {name}: {len(clouds)} clouds, {len(poses)} poses (using {n}); "
          f"cloud frame: {'WORLD->recovering body' if world_frame else 'body'}")

    def body(i: int) -> np.ndarray:
        pts = clouds[i][:, :3].astype(np.float64)
        return R.pose_apply(R.pose_inv(R.pose_from_row(poses[i])), pts) if world_frame else pts

    # Build the prior map: stitch strided body scans into the world frame.
    blocks = [R.pose_apply(R.pose_from_row(poses[i]), body(i))
              for i in range(0, n, map_stride) if clouds[i].size]
    stitched = np.concatenate(blocks)
    world_map = R.voxel_downsample(stitched, map_voxel)
    print(f"  stitched {len(stitched)} -> {len(world_map)} map points (voxel {map_voxel}m)")
    R.write_pcd(scen_dir / "map.pcd", world_map)

    # Query scans are body clouds; truth is the odometry pose; guess = truth + offset.
    R.write_scans_bin(scen_dir / "scans.bin", [body(i).astype(np.float32) for i in range(n)])

    prng = np.random.default_rng(seed + 2)
    trials: list[dict] = []
    for idx in range(0, n, query_stride):
        truth = R.pose_from_row(poses[idx])
        for label, dt, dy in BUCKETS:
            k = 1 if dt == 0.0 and dy == 0.0 else reps
            for _ in range(k):
                guess = R.perturb(truth, dt, dy, prng)
                trials.append({
                    "scan_idx": idx, "bucket": label,
                    "guess": R.pose_to_row(guess), "truth": R.pose_to_row(truth),
                })
    _emit(scen_dir, name, scen_dir / "map.pcd", scen_dir / "scans.bin", trials)


def gen_global(dataset: Path, name: str, query_stride: int, window: int,
               map_stride: int, map_voxel: float, sub_voxel: float, seed: int) -> None:
    """Scenario for Ivan's GLOBAL relocalizer: a stitched prior map + per-query
    local submaps (accumulated windows of frames, in the center frame's body
    frame). No initial guess is used; truth = center frame's world pose. A
    single sparse Livox frame can't be globally registered, so we accumulate a
    +/-window neighbourhood via relative odometry (locally accurate).
    """
    if not dataset.exists():
        print(f"  (skipping {name}: {dataset} not found)")
        return
    scen_dir = SCEN_ROOT / name
    scen_dir.mkdir(parents=True, exist_ok=True)

    _ts, poses = R.load_tum(dataset / "lidar_poses.tum")
    clouds = [c for _, c in R.iter_scans_bin(dataset / "clouds.bin")]
    n = min(len(poses), len(clouds))

    # Auto-detect cloud frame: if per-frame centroids track the pose translation,
    # the export already put clouds in WORLD frame (a known export bug for some
    # datasets); otherwise they're body-frame. Either way we derive world points.
    samp = range(0, n, max(1, n // 40))
    cmag = np.median([np.linalg.norm(clouds[i][:, :3].mean(0)) for i in samp])
    pmag = np.median([np.linalg.norm(poses[i][:3]) for i in samp])
    # Body-frame clouds are sensor-centred (centroid a few m from origin); a
    # world-frame export's centroid sits at the robot's world position (~pmag).
    world_frame = cmag > max(15.0, 0.5 * pmag)
    frame = "WORLD (clouds pre-transformed)" if world_frame else "body"
    print(f"  {name}: {len(clouds)} clouds, {len(poses)} poses (using {n}); detected cloud frame: {frame}")

    def world_pts(i: int) -> np.ndarray:
        pts = clouds[i][:, :3].astype(np.float64)
        return pts if world_frame else R.pose_apply(R.pose_from_row(poses[i]), pts)

    blocks = [world_pts(i) for i in range(0, n, map_stride) if clouds[i].size]
    world_map = R.voxel_downsample(np.concatenate(blocks), map_voxel)
    print(f"  stitched -> {len(world_map)} map points (voxel {map_voxel}m)")
    R.write_pcd(scen_dir / "map.pcd", world_map)

    # Build a local submap per query center, in that center's body frame.
    submaps: list[np.ndarray] = []
    trials: list[dict] = []
    centers = list(range(window, n - window, query_stride))
    for si, c in enumerate(centers):
        Tc_inv = R.pose_inv(R.pose_from_row(poses[c]))
        acc = []
        for i in range(c - window, c + window + 1):
            if clouds[i].size == 0:
                continue
            acc.append(R.pose_apply(Tc_inv, world_pts(i)))  # world -> center body frame
        submap = R.voxel_downsample(np.concatenate(acc), sub_voxel)
        submaps.append(submap.astype(np.float32))
        truth = R.pose_from_row(poses[c])  # body_c -> world == what reloc must recover
        trials.append({"scan_idx": si, "bucket": "global",
                       "guess": R.pose_to_row((np.zeros(3), np.array([0, 0, 0, 1.0]))),
                       "truth": R.pose_to_row(truth)})
    R.write_scans_bin(scen_dir / "scans.bin", submaps)
    manifest = {"scenario": name, "map": str(scen_dir / "map.pcd"),
                "scans": str(scen_dir / "scans.bin"), "buckets": ["global"], "trials": trials}
    (scen_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  {name}: {len(trials)} query submaps -> {scen_dir}")


def main(
    dataset: Path = typer.Option(
        HERE.parents[1] / "data" / "loop_bench" / "hk_village3", "--dataset"),
    reps: int = typer.Option(3, "--reps", help="perturbations per (query, bucket)"),
    seed: int = typer.Option(7, "--seed"),
    query_stride: int = typer.Option(18, "--query-stride", help="sample every Nth real scan"),
    map_stride: int = typer.Option(2, "--map-stride", help="stitch every Nth scan into map"),
    map_voxel: float = typer.Option(0.1, "--map-voxel"),
    real_name: str = typer.Option("hk_village3", "--real-name", help="scenario dir name for --only real"),
    only: str = typer.Option("", "--only", help="synthetic|real|global (default: synthetic+real)"),
    # global-relocalizer scenario (Ivan's FPFH+RANSAC)
    global_dataset: Path = typer.Option(
        HERE.parents[1] / "data" / "loop_bench" / "outdoor_small_loop", "--global-dataset"),
    global_name: str = typer.Option("outdoor_small_loop", "--global-name"),
    global_query_stride: int = typer.Option(220, "--global-query-stride"),
    global_window: int = typer.Option(15, "--global-window", help="+/- frames per submap"),
    global_map_stride: int = typer.Option(8, "--global-map-stride"),
    global_map_voxel: float = typer.Option(0.25, "--global-map-voxel"),
    global_sub_voxel: float = typer.Option(0.1, "--global-sub-voxel"),
) -> None:
    SCEN_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"generating scenarios -> {SCEN_ROOT}")
    if only in ("", "synthetic"):
        gen_synthetic(reps, seed)
    if only in ("", "real"):
        gen_real(dataset, reps, seed, query_stride, map_stride, map_voxel, name=real_name)
    if only == "global":
        gen_global(global_dataset, global_name, global_query_stride, global_window,
                   global_map_stride, global_map_voxel, global_sub_voxel, seed)


if __name__ == "__main__":
    typer.run(main)
