[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/pyx378952739/makefun-gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/pyx378952739/makefun-gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Built for the [make.fun gta6-trophy-rarity-forecast-01 bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

# makefun-gta6-trophy-rarity-forecast-01

Forecast how many GTA VI players earn each trophy in week one: a program
that reads one case (`/input/<id>.in`) and writes five quantiles of every
trophy's Exophase earn rate at the target (`/output/<id>.out`), for the
lowest pinball loss on the logit scale.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Node 22, standard library only. No network, no shell-outs, deterministic:
two runs on the same case write the same bytes.

## Model

A counts model on the trophy page's own refresh dynamics. Exophase does not
update the page continuously: the house saw it frozen for hours, then
refreshed in one jump (638 → 3,868 players tracked on launch night, every
rate but the 100% one falling). The program therefore works in holder counts
and players tracked, which is what a refresh actually moves:

- **Plateaus.** Consecutive reads with identical players-tracked and rates
  are one plateau; a change starts a new one (a refresh). From the case's
  own history it reads the refresh cadence (median gap between plateau
  starts) and, per trophy, the trend of its log rate over refreshes
  (least-squares slope — a median of per-refresh ratios would read zero
  once small rates sit on the page's 2-decimal rounding floor).
- **Forward simulation.** To the target it applies the expected number of
  refreshes, each step's log change decaying toward zero (growth saturates
  as the launch surge passes).
- **Spread.** Nine deterministic scenarios (refresh count × growth) give
  the scenario dispersion of each trophy's logit rate, combined with the
  residual dispersion around the fitted trends and a horizon-scaled floor,
  clamped to [0.45, 1.5] logit units; the upper spread is 1.4× the lower
  one, since upside risk (slower dilution, faster progression) is larger
  than downside risk.
- **Flat history.** Launch night's practice histories show no refresh, yet
  the answers sit after the page's first refresh. The program then uses a
  prior from the competition's public description of that refresh: one
  refresh before a ≥ 60-minute target, players tracked ×6, new players
  holding each trophy at a small fraction of its current rate (tilted so
  the 100% trophy stays 100%), i.e. a median of roughly 0.22× the last
  rate, plus a small upward drift with horizon and a wide asymmetric
  spread whose upper tail reaches the undiluted rate.
- **Trophies at 100%** stay at 99.99 across all quantiles (what the scorer
  clamps 100% to). Any failure falls back, trophy by trophy, to the house
  baseline's persistence forecast, so the output is always valid.

## Local validation

The kit's `grade-local` does not grade this competition locally ("its
scorer and reference data are not published in the kit, so it is graded on
the server only"). The practice answers were therefore reconstructed from
the hard facts in `competition.yaml` (04:30Z read: 3,868 players, 4,276
trophies earned; every rate but the 100% one fell; the 103 non-platinum
trophy-earns by the 3,230 new players distributed four ways:
proportional to old holders, all on the highest-rate trophies, uniform,
and the impossible zero-inflow bound). Against all four reconstructions:

| case | this entry | house baseline (reconstructions; stated ≈ 1.51) |
|------|-----------|--------------------------------------------------|
| p00  | 0.19 – 0.25 | 1.38 – 1.69 |
| p01  | 0.19 – 0.25 | 1.38 – 1.69 |

Runtime ≈ 0.1 s per case (cap 8 s). Output shape verified: 55 rows, five
quantiles each, 0–100, non-decreasing. A synthetic final-like case (12
simulated refreshes over 2 days, 6,510-minute horizon) and edge cases
(single-read history, null rates, 0% rates) all produce valid,
deterministic output.
