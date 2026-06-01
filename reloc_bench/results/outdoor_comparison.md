# Relocalization on outdoor_small_loop — stock ICP vs. Ivan's global FPFH+RANSAC

Both methods run on the same dataset (`data/loop_bench/outdoor_small_loop`,
6961 frames, ~154 m outdoor loop, Go2 Mid360), corrected to body-frame clouds.
Prior map = odometry-stitched, 0.25 m voxel → 323k points.

- stock: `reloc_bench` backend (localizer two-stage point-to-point ICP, with the
  bounded-correspondence fix). Scenario `outdoor_guess`: query = one body-frame
  scan, **needs an initial guess** = truth ± offset bucket. `outdoor_stock.json`.
- ivan: `global_reloc_bench` (C++ port of dimos
  `dimos/mapping/relocalization/relocalize.py`). Scenario `outdoor_small_loop`:
  query = ±15-frame submap, **no guess** (global). 400k RANSAC iters.
  `outdoor_global.json`.
- regenerate: `gen_scenarios.py --only global` / `--only real --real-name outdoor_guess`,
  then `bench_reloc.py --backend <…> --scenario <…>`.

## Stock two-stage ICP (needs a near-truth guess)

```
guess err    n  conv%  correct%  te_med  te_p90  re_med  re_p90  ms_med
exact       31   100%     100%   0.021   0.035    0.14    0.25   160.1
near(0.3m)  93    99%      95%   0.062   0.171    0.63    1.31   162.6
mid (1.0m)  93    44%      15%   0.254   0.746    4.97    9.13    81.0
far (2.5m)  93     0%       0%     —       —        —       —      79.8
extreme     93     0%       0%     —       —        —       —      83.8
```

Tracker/refiner: 95–100% correct and fast (~160 ms) when the guess is within
~0.5 m; collapses past ~1 m; useless with no guess. (Much better on
outdoor-near than on hk_village3, where it was 0% — denser/more-structured
outdoor scans + the correspondence-distance bound.)

## Ivan global FPFH+RANSAC (no guess)

```
            n  conv%  correct%  te_med  te_p90  re_med  re_p90  ms_med
global     32    44%      41%   0.010   0.035    0.09    0.17  41585
all-trial translation err: median 3.91 m, 14/32 <2 m, 17/32 <5 m
```

Cold-start recoverer: 41% correct **from no guess at all**, centimetre-accurate
when it locks (te_med 1 cm), and the fitness gate reliably knows when it
succeeded (conv 44% ≈ correct 41% → fails safe). Misses ~59% to the loop's
along-track self-similarity, and is slow (~42 s/query).

## Takeaway

They solve different halves of the problem and compose into the standard
two-tier localization stack:

| | stock ICP | Ivan global |
|---|---|---|
| needs initial guess | yes (<~0.5 m) | **no** |
| success | 95–100% if guess good | 41% from scratch |
| accuracy when it works | 2–6 cm | **1 cm** |
| latency | ~160 ms | ~42 s |
| failure mode | diverges silently past basin | fails *safe* (low fitness) |

Production pattern: **Ivan global for cold-start / kidnapped recovery → hand its
cm-accurate pose to stock ICP as the high-rate tracker.** Neither alone is
enough on this outdoor loop; together they cover it. Open levers for the global
tier (the enhanced Rust target): the ~59% along-track misses want either richer
descriptors or a place-recognition prior (Scan Context) to disambiguate the
self-similar path, and the ~42 s/query needs the SAC/rerank made faster.
