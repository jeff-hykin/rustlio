# grass_field_loop + stair_plaza benchmark — code @ 5fddc4f

Two 2026-06-01 Go2 recordings. grass_field_loop (5:32pm): sprawling 149x184x13 m,
fastlio 729 m / gap 174 m (barely a loop, ~unsolvable). stair_plaza (6:05pm):
compact, 598 m / gap 20 m. Each x (fastlio source: inject yaw drift, ATE vs clean
fastlio | go2 source: REAL drift, sim-ATE vs fastlio GT via Umeyama). ATE metres.

## fastlio source (ATE clean -> after PGO)
| case | stock | plane | rust (best cfg) |
|------|------:|------:|----------------:|
| grass yaw0   | 1.26 | 1.42 | 5.32 (arc-law) |
| grass yaw0.1 | 5.86 | 5.11 | 12.53 |
| stair yaw0   | **0.08** | **0.10** | 2.80 (arc-law) |
| stair yaw0.1 | **0.86** | **0.94** | 2.64 |
rust corrupts these scenes regardless of noise cfg (rot 0.001/0.05, arc, SC) —
loop ICP quality, not back-end tuning.

## go2 source (sim-ATE raw -> after PGO; raw = no PGO)
| case | raw | stock | plane | rust | rust+SC |
|------|----:|------:|------:|-----:|--------:|
| grass go2 | 53.75 | 55.08 | 51.94 | 53.86 | 53.88 |
| stair go2 | 1.97 | 8.29 | 14.40 | **1.84** | **1.87** |
grass go2: unsolvable (both sensors drift huge, no reliable GT); all ~no-op.
stair go2: C++ CORRUPTS (8-14 m); rust does no harm (1.84, slight gain).

## Verdict vs "PGO must not do worse than raw"
- fastlio scenes: C++ holds (stair 0.08); **rust violates** (corrupts) — rust ICP.
- go2 scenes: rust holds (stair 1.84); **C++ violates** (corrupts) — C++ over-trusts.
- Shared root cause: loop ICP/measurement quality on hard scenes.
