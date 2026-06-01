# Relocalization benchmark — baseline vs. bounded-correspondence fix

Compares the committed [baseline](baseline.md) against the first improvement
attempt: bounding the ICP correspondence search distance in
`ICPLocalizer::align` (rough ≈ 2.0 m, refine ≈ 0.8 m). Previously PCL's default
was effectively infinite.

- baseline: `baseline.json` (commit a89f331)
- fixed: `fixed.json` (this change)
- regenerate either with `./run/reloc_test --json <file>`

## Verdict (honest)

**Hygiene, not a rescue.** Bounding correspondences is a legitimate correctness
fix — an infinite correspondence distance is a latent bug the Rust port must not
copy — and it causes **no regression** (synthetic is unchanged). But it does
**not** make real-data relocalization usable: hk_village3 stays at **0% correct**
for every non-exact guess.

The decisive signal is the `exact` bucket: with the guess set *equal to truth*,
real-data relocalization is still only **87% correct** — i.e. point-to-point ICP
slides a perfectly-placed scan >0.3 m away 13% of the time. Correspondence
distance can't fix that. Root cause: point-to-point ICP on **sparse single-frame
Livox scans** over a ground-dominated map has almost no in-plane constraint, so
it drifts along the ground regardless of how the search is bounded.

**The real fix is a stronger registration** — point-to-plane / GICP (constrains
the normal direction, kills ground-sliding), or the tightly-coupled
IESKF-against-prior-map approach (reuses the LIO engine, IMU-constrained). That
is the planned Rust enhancement; this benchmark is set up to validate it.

## hk_village3 (real Go2 LiDAR) — the metric that matters

```
            baseline                         fixed
guess err  conv%  correct%  te_med   |  conv%  correct%  te_med
exact       100%      87%   0.056    |   100%      87%   0.056
near         95%       0%   2.056    |    94%       0%   2.040
mid          85%       0%   2.716    |    83%       0%   2.549
far          83%       0%   3.625    |    81%       0%   3.564
extreme      68%       0%   6.748    |    56%       0%   6.475
```

No headline change. (te_med dips a hair but correct% — within 0.30 m & 5° — is 0%
across the board either way.)

## synthetic_room (clean box room) — regression check

```
            baseline                         fixed
guess err  conv%  correct%  re_med   |  conv%  correct%  re_med
exact       100%     100%   0.01     |   100%     100%   0.01
near        100%     100%   0.01     |   100%     100%   0.01
mid         100%     100%   2.15     |   100%     100%   2.58
far           0%       0%   nan      |     0%       0%   nan
extreme       0%       0%   nan      |     0%       0%   nan
```

No regression. (An intermediate tuning with a tighter 0.4 m refine distance *did*
regress `mid` to 0% via worse rotation accuracy — hence refine ≈ 0.8 m.)
