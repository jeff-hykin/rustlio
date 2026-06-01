"""Relocalization test suite — drives the real C++ ICPLocalizer harness.

These are characterization + spec tests for the *current* relocalizer
(localizer/src/localizers/icp_localizer.cpp), warts and all. Tests that assert
behavior the current point-to-point ICP cannot deliver (wide convergence basin,
symmetry disambiguation) are marked xfail with a reason — they document the gap
the enhanced Rust relocalizer is expected to close, and will flip to green when
it does (run with --runxfail to see).

Run:
    cd reloc_bench/scripts
    uv run --with numpy --with scipy --with pytest pytest test_reloc.py -v
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import reloc_lib as R

HARNESS = Path(__file__).resolve().parent.parent / "harness" / "build" / "reloc_bench"
ENV = dict(os.environ, DYLD_LIBRARY_PATH="/opt/homebrew/lib")

pytestmark = pytest.mark.skipif(not HARNESS.exists(), reason="build harness first (run/reloc_test)")


def yaw_pose(x=0.0, y=0.0, z=0.0, yaw_deg=0.0) -> R.Pose:
    return np.array([x, y, z], float), Rotation.from_euler("Z", yaw_deg, degrees=True).as_quat()


def run_trials(map_pts, scans, trials, cfg=None, expect_ok=True):
    """Write a one-off scenario, run the harness, return [(conv, pose, ms), ...].

    map_pts: (N,3) prior map. scans: list of (N,3) body clouds. trials:
    list of (scan_idx, guess_pose). cfg: dict of ICP overrides.
    """
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        R.write_pcd(d / "map.pcd", np.asarray(map_pts))
        R.write_scans_bin(d / "scans.bin", scans)
        R.write_trials(d / "trials.txt", trials)
        cmd = [str(HARNESS), "--map", str(d / "map.pcd"), "--scans", str(d / "scans.bin"),
               "--trials", str(d / "trials.txt"), "--out", str(d / "res.txt")]
        for k, v in (cfg or {}).items():
            cmd.append(f"{k}={v}")
        proc = subprocess.run(cmd, env=ENV, capture_output=True, text=True)
        if expect_ok and proc.returncode != 0:
            raise AssertionError(f"harness failed ({proc.returncode}):\n{proc.stderr}")
        if proc.returncode != 0:
            return None
        return R.read_results(d / "res.txt")


def reloc_once(map_pts, scan, truth, guess, cfg=None):
    res = run_trials(map_pts, [scan], [(0, guess)], cfg)
    conv, pose, ms = res[0]
    te, re_ = R.trans_rot_err(pose, truth)
    return conv, te, re_, ms


# ----------------------------- fixtures -----------------------------
@pytest.fixture(scope="module")
def room():
    return R.synthetic_room(n_points=15000, seed=1)


def scan_from(map_pts, truth):
    """Body-frame scan that, transformed by truth, lands back on the map."""
    return R.pose_apply(R.pose_inv(truth), map_pts).astype(np.float32)


# ----------------------------- accuracy / basin -----------------------------
def test_exact_guess_recovers_pose(room):
    truth = yaw_pose(1.0, -0.5, 0.1, 8.0)
    conv, te, re_, _ = reloc_once(room, scan_from(room, truth), truth, truth)
    assert conv and te < 0.05 and re_ < 1.0, f"te={te:.3f} re={re_:.3f}"


def test_near_guess_recovers_pose(room):
    truth = yaw_pose(1.2, -0.7, 0.05, 8.0)
    guess = R.perturb(truth, 0.3, 5.0, np.random.default_rng(0))
    conv, te, re_, _ = reloc_once(room, scan_from(room, truth), truth, guess)
    assert conv and te < 0.10 and re_ < 2.0, f"te={te:.3f} re={re_:.3f}"


def test_mid_guess_recovers_pose(room):
    truth = yaw_pose(0.5, 0.5, 0.0, -10.0)
    guess = R.perturb(truth, 1.0, 12.0, np.random.default_rng(3))
    conv, te, re_, _ = reloc_once(room, scan_from(room, truth), truth, guess)
    assert conv and te < 0.30 and re_ < 5.0, f"te={te:.3f} re={re_:.3f}"


@pytest.mark.xfail(reason="point-to-point ICP basin is too narrow for >2m / >25deg "
                          "guess error; enhanced Rust relocalizer should fix this")
def test_far_guess_recovers_pose(room):
    truth = yaw_pose(0.0, 0.0, 0.0, 0.0)
    guess = R.perturb(truth, 2.5, 30.0, np.random.default_rng(5))
    conv, te, re_, _ = reloc_once(room, scan_from(room, truth), truth, guess)
    assert conv and te < 0.30 and re_ < 5.0, f"te={te:.3f} re={re_:.3f}"


@pytest.mark.parametrize("trans_m,yaw_deg", [(0.0, 0.0), (0.3, 5.0), (0.8, 10.0), (1.2, 18.0)])
def test_basin_sweep_converges(room, trans_m, yaw_deg):
    truth = yaw_pose(0.3, -0.2, 0.0, 5.0)
    guess = R.perturb(truth, trans_m, yaw_deg, np.random.default_rng(11))
    conv, te, re_, _ = reloc_once(room, scan_from(room, truth), truth, guess)
    assert conv and te < 0.30, f"trans={trans_m} yaw={yaw_deg}: conv={conv} te={te:.3f}"


# ----------------------------- rejection -----------------------------
def test_rejects_junk_cloud(room):
    rng = np.random.default_rng(0)
    junk = rng.normal((20, 20, 20), 0.5, size=(3000, 3)).astype(np.float32)
    conv, *_ = reloc_once(room, junk, yaw_pose(), yaw_pose())
    assert not conv, "junk cloud should not relocalize"


def test_rejects_far_displaced_scan(room):
    # A real scan but the guess places it 50m away — no overlap, must reject.
    truth = yaw_pose(0.0, 0.0, 0.0, 0.0)
    conv, *_ = reloc_once(room, scan_from(room, truth), truth, yaw_pose(50.0, 50.0, 0.0, 0.0))
    assert not conv


# ----------------------------- robustness -----------------------------
def test_noisy_scan_still_recovers(room):
    truth = yaw_pose(0.8, -0.3, 0.05, 6.0)
    scan = R.add_noise(scan_from(room, truth), sigma=0.03, seed=2).astype(np.float32)
    guess = R.perturb(truth, 0.3, 5.0, np.random.default_rng(0))
    # noise inflates inlier_rmse; loosen the refine gate so it can still accept.
    conv, te, re_, _ = reloc_once(room, scan, truth, guess, cfg={"refine_score_thresh": 0.3})
    assert conv and te < 0.15, f"te={te:.3f} re={re_:.3f}"


def test_partial_overlap_recovers():
    # Map is the full room; the scan sees only 3 walls + floor (no ceiling, no
    # west wall) — a realistic partial view. Should still localize.
    full = R.synthetic_room(n_points=18000, seed=4)
    # keep floor (z~0) + east/north/south walls (drop ceiling z~3 and west x~-5)
    keep = (full[:, 2] < 2.5) & (full[:, 0] > -4.5)
    truth = yaw_pose(0.5, 0.4, 0.0, 7.0)
    scan = R.pose_apply(R.pose_inv(truth), full[keep]).astype(np.float32)
    guess = R.perturb(truth, 0.3, 5.0, np.random.default_rng(0))
    conv, te, re_, _ = reloc_once(full, scan, truth, guess)
    assert conv and te < 0.20, f"te={te:.3f} re={re_:.3f}"


def test_determinism_same_trial_same_result(room):
    truth = yaw_pose(0.7, 0.1, 0.0, 4.0)
    guess = R.perturb(truth, 0.5, 6.0, np.random.default_rng(9))
    scan = scan_from(room, truth)
    a = reloc_once(room, scan, truth, guess)
    b = reloc_once(room, scan, truth, guess)
    assert a[0] == b[0] and abs(a[1] - b[1]) < 1e-6, "ICP must be deterministic for fair comparison"


# ----------------------------- symmetry / ambiguity -----------------------------
@pytest.mark.xfail(reason="a square room is 90deg-symmetric; point-to-point ICP with a "
                          "near-90deg guess locks to the wrong symmetric basin and the "
                          "RMSE gate can't tell — needs a disambiguating global step")
def test_square_room_90deg_guess_disambiguates():
    room = R.synthetic_room(n_points=16000, size=(10.0, 10.0, 3.0), seed=8)
    truth = yaw_pose(0.0, 0.0, 0.0, 0.0)
    guess = yaw_pose(0.0, 0.0, 0.0, 88.0)  # near a symmetry multiple
    conv, te, re_, _ = reloc_once(room, scan_from(room, truth), truth, guess)
    assert conv and re_ < 5.0, f"locked to symmetric basin: re={re_:.2f}deg"


# ----------------------------- degenerate maps / IO -----------------------------
def test_missing_map_file_fails_cleanly():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        R.write_scans_bin(d / "scans.bin", [R.synthetic_room(3000)])
        R.write_trials(d / "trials.txt", [(0, yaw_pose())])
        cmd = [str(HARNESS), "--map", str(d / "nope.pcd"), "--scans", str(d / "scans.bin"),
               "--trials", str(d / "trials.txt"), "--out", str(d / "res.txt")]
        proc = subprocess.run(cmd, env=ENV, capture_output=True, text=True)
        assert proc.returncode != 0, "missing map should fail, not silently succeed"


def test_empty_map_does_not_converge():
    # A near-empty map can't support alignment; align() must return false
    # (target size 0 guard), never a bogus pose.
    empty = np.zeros((1, 3), np.float32)
    res = run_trials(empty, [R.synthetic_room(3000)], [(0, yaw_pose())])
    assert res is None or not res[0][0], "empty map must not yield a converged pose"


def test_time_is_reported(room):
    truth = yaw_pose(0.5, 0.0, 0.0, 3.0)
    _, _, _, ms = reloc_once(room, scan_from(room, truth), truth, truth)
    assert ms > 0.0
