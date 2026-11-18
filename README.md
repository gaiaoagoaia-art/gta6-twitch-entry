[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/gaiaoagoaia-art/gta6-twitch-entry/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/gaiaoagoaia-art/gta6-twitch-entry/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

# GTA VI Twitch CCU forecast — make.fun entry

A program that forecasts, from everything recorded up to the end of hour 1 of
Grand Theft Auto VI's release day (2026-11-19, UTC), how many people will be
watching the game live on Twitch in each of hours 2–24 — five quantiles
(5%, 25%, 50%, 75%, 95%) per hour, scored by mean pinball loss on the log
scale. `main.py` (Python 3.13, standard library only, no network) reads every
`*.in` case file in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes one
`<id>.out` per case to `$MAKEFUN_OUTPUT_DIR` (else `/output`). It is
deterministic — two runs on the same case produce byte-identical output — and
runs all four practice cases in about 0.3 s total.

## The model

The forecast anchors on the case's own hour-1 print `p1` and multiplies it by
a ratio distribution fitted **at run time from the case's own `comparables`** —
the past big game launches whose full release-day hourly curves (SullyGnome's
hourly Twitch category averages) ship inside every case file. Nothing is
hard-coded from outside data; the same code fits the practice cases (38
comparables) and the final (42).

1. **Ratio curves.** For each comparable whose release day has a positive
   hour-1 value, form `L(h) = ln(viewers at hour h) − ln(viewers at hour 1)`
   for hours 2–24 of its release day (37 of the 38 comparables qualify; the
   GTA VI category's own pre-release history has no release-day hour 1 and
   drops out on its own).
2. **Shape family.** GTA VI and all four practice launches are console games
   that unlock at rolling local midnight (New Zealand first, then Europe at
   23:00–00:00 UTC, the US coasts at 05:00/08:00 UTC), so their UTC day has a
   characteristic shape: a soft first hour, a shallow dip, then a climb to a
   peak of roughly 2–2.3× hour 1 around hours 20–22 as the US comes online.
   Comparables whose `unlock` note describes rolling local midnight form the
   primary family (12 of 37). Each hour's predictive distribution is a
   **mixture: 0.65 × the rolling family's empirical distribution of L(h) +
   0.35 × all launches'** — the subset supplies the shape, the global pool
   fattens the tails (a few rolling launches, e.g. Resident Evil Requiem,
   ramp all day to ~12× hour 1 instead).
3. **Blend, smooth and shrink.** The mixture's quantiles are averaged in log
   space with a normal fit (mean, sd) of the rolling family's L(h). Adjacent
   hours are fitted independently from nearly the same pool, so the fitted
   log-quantile curves are lightly smoothed across neighbouring hours with a
   [0.25, 0.5, 0.25] kernel, then shrunk 20% in log space toward the
   persistence baseline's quantiles
   (`p1 · exp(z_τ · 0.15 · √(h−1))`). Hours or cases with no usable
   comparables fall back to that baseline outright. Quantiles are rounded to
   2 decimals, clamped to (0, 1e9], and forced non-decreasing.

## Backtest (leave-one-out on the comparables)

The kit's practice cases carry no local ground truth (the server grades them
against the historical source), so the model was fitted and evaluated by
leave-one-out over the 37 usable comparable launches: each launch in turn is
treated as the target — its hour-1 value as the anchor, the other 36 as the
fit set — and scored with the contest's exact pinball loss. The slice that
matters is the 12 rolling-midnight launches, the family GTA VI and all four
practice games belong to:

| forecaster | mean pinball loss, rolling-midnight targets | all 37 targets |
|---|---|---|
| house persistence baseline | 0.3568 | 0.9296 |
| **this model** | **0.3045** | 0.7911 |

On the rolling-midnight targets the model's 90% interval covers 93.8% of
realised hours and its median's typical miss is 61.7%. The cross-hour
smoothing is a small, measured win rather than a leap: at full precision it
moves the rolling-family loss from 0.30450 to 0.30448 and the all-37 loss
from 0.79156 to 0.79114 — it never hurts either slice, which is why it
stays. (For reference, the
house baseline's official practice loss, graded by the server on the four
practice games, is ≈ 0.387 with a typical miss of ≈ 82.8%; the LOO figures
above are measured on the comparable launches instead, so they compare models
to each other, not to that number.) The shipped `main.py` was re-scored
through the same LOO harness and reproduces the table exactly. The mixture
weight (0.65), blend and shrinkage (0.20) sit on a flat region of the LOO
surface — weights 0.6–0.7 and shrinkage 0.15–0.25 all score within 0.303–0.307
— so they are not knife-edge fits to twelve launches.

## Running it locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
# writes out/p00.out … out/p03.out for the kit's practice cases
```

Output format, one object per case: `{"hours": [{"hour": 2, "q": [q05, q25,
q50, q75, q95]}, …]}` with exactly 23 rows, hours 2–24.

## License

MIT — see `LICENSE`. The practice and comparable figures quoted by the case
files are SullyGnome.com's hourly Twitch category averages, used as shipped
inside the competition kit.
