"""Run a relocalization backend over the generated scenarios and print a
performance table.

A "backend" is any binary implementing the reloc_bench CLI contract:
    BACKEND --map M.pcd --scans S.bin --trials T.txt --out R.txt [cfg k=v ...]
Today that's the C++ ICPLocalizer harness; later the enhanced Rust relocalizer
implements the same contract and is compared on the identical scenarios/manifests.

Headline metric: per initial-guess-error bucket, what fraction of trials the
algorithm relocalizes CORRECTLY (recovered pose within tolerance of truth) —
the convergence basin — plus the accuracy and latency among those.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import typer

import reloc_lib as R

HERE = Path(__file__).resolve().parent
SCEN_ROOT = HERE.parent / "scenarios"
DEFAULT_BACKEND = HERE.parent / "harness" / "build" / "reloc_bench"

# A trial counts as a correct relocalization when the recovered pose is within
# these of truth. (Loose enough that a real downstream nav stack would be happy;
# tight enough to exclude ICP that "converged" to the wrong basin.)
SUCCESS_TRANS_M = 0.30
SUCCESS_ROT_DEG = 5.0


def run_backend(backend: Path, manifest: dict, cfg: list[str]) -> list[tuple[bool, R.Pose, float]]:
    trials = manifest["trials"]
    with tempfile.TemporaryDirectory() as d:
        trials_txt = Path(d) / "trials.txt"
        results_txt = Path(d) / "results.txt"
        R.write_trials(trials_txt, [(t["scan_idx"], R.pose_from_row(t["guess"])) for t in trials])
        env = dict(os.environ, DYLD_LIBRARY_PATH="/opt/homebrew/lib")
        cmd = [str(backend), "--map", manifest["map"], "--scans", manifest["scans"],
               "--trials", str(trials_txt), "--out", str(results_txt), *cfg]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"backend failed:\n{proc.stderr}")
        for ln in proc.stderr.splitlines():
            print(f"    [{Path(backend).name}] {ln}")
        return R.read_results(results_txt)


def _pct(x: list, q: float) -> float:
    return float(np.percentile(x, q)) if x else float("nan")


def score(manifest: dict, results: list[tuple[bool, R.Pose, float]]) -> list[dict]:
    """Per-bucket stats, in manifest bucket order."""
    trials = manifest["trials"]
    by_bucket: dict[str, dict] = {b: {"te": [], "re": [], "ms": [], "conv": 0, "ok": 0, "n": 0}
                                  for b in manifest["buckets"]}
    for t, (conv, pose, ms) in zip(trials, results):
        b = by_bucket[t["bucket"]]
        b["n"] += 1
        b["ms"].append(ms)
        te, re_ = R.trans_rot_err(pose, R.pose_from_row(t["truth"]))
        if conv:
            b["conv"] += 1
            b["te"].append(te)
            b["re"].append(re_)
            if te <= SUCCESS_TRANS_M and re_ <= SUCCESS_ROT_DEG:
                b["ok"] += 1
    rows = []
    for label in manifest["buckets"]:
        b = by_bucket[label]
        if b["n"] == 0:
            continue
        rows.append({
            "bucket": label, "n": b["n"],
            "conv_rate": b["conv"] / b["n"],
            "correct_rate": b["ok"] / b["n"],
            "te_med": _pct(b["te"], 50), "te_p90": _pct(b["te"], 90),
            "re_med": _pct(b["re"], 50), "re_p90": _pct(b["re"], 90),
            "ms_med": _pct(b["ms"], 50),
        })
    return rows


def print_table(scenario: str, rows: list[dict]) -> None:
    print(f"\n=== {scenario} ===")
    hdr = f"{'guess err':<9} {'n':>4} {'conv%':>6} {'correct%':>9} " \
          f"{'te_med':>7} {'te_p90':>7} {'re_med':>7} {'re_p90':>7} {'ms_med':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['bucket']:<9} {r['n']:>4} {r['conv_rate']*100:>5.0f}% "
              f"{r['correct_rate']*100:>8.0f}% "
              f"{r['te_med']:>7.3f} {r['te_p90']:>7.3f} "
              f"{r['re_med']:>7.2f} {r['re_p90']:>7.2f} {r['ms_med']:>7.1f}")
    print("(te=translation err m, re=rotation err deg; med/p90 over converged trials; "
          f"correct = within {SUCCESS_TRANS_M}m & {SUCCESS_ROT_DEG}deg)")


def main(
    backend: Path = typer.Option(DEFAULT_BACKEND, "--backend"),
    scenario: str = typer.Option("", "--scenario", help="one scenario name (default: all)"),
    cfg: list[str] = typer.Option([], "--cfg", help="backend cfg override key=val (repeatable)"),
    out_json: Path = typer.Option(None, "--json", help="also dump results table as JSON"),
) -> None:
    if not backend.exists():
        raise SystemExit(f"backend not found: {backend} (build it first)")
    scen_dirs = ([SCEN_ROOT / scenario] if scenario
                 else sorted(p.parent for p in SCEN_ROOT.glob("*/manifest.json")))
    if not scen_dirs:
        raise SystemExit(f"no scenarios under {SCEN_ROOT} (run gen_scenarios.py)")

    print(f"backend: {backend}")
    if cfg:
        print(f"cfg overrides: {' '.join(cfg)}")
    all_out = {}
    for sd in scen_dirs:
        manifest = json.loads((sd / "manifest.json").read_text())
        results = run_backend(backend, manifest, cfg)
        rows = score(manifest, results)
        print_table(manifest["scenario"], rows)
        all_out[manifest["scenario"]] = rows

    if out_json:
        out_json.write_text(json.dumps(all_out, indent=2))
        print(f"\nwrote {out_json}")


if __name__ == "__main__":
    typer.run(main)
