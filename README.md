# GTA VI Trophy Rarity Forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Goatleaf12/makefun-gta6-trophy-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Goatleaf12/makefun-gta6-trophy-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

An entry for make.fun's **gta6-trophy-rarity-forecast-01** bounty: forecast,
for every PlayStation 5 trophy of Grand Theft Auto VI, the percent of
Exophase-tracked players who will have earned it at 2026-11-26 00:00 UTC, as
five quantiles per trophy, scored by pinball loss on the logit scale.

## How it works

`main.py` (Python 3.13, standard library only) reads one case file and:

- collapses the read history into *refresh states* — Exophase updates its
  counts in steps, so only reads where the page actually changed carry
  information;
- projects the tracked-player pool and each trophy's *earners* separately
  on a log-time axis with recency-weighted least squares (recent states
  weigh more, old states fade), then takes the median rate as
  earners / players at the target;
- shrinks each trophy's fit spread and two-point slopes toward the
  cross-trophy pool, damps earners growth for saturation over the 4.5-day
  final horizon, and blends the decomposition with a direct rate-trend
  projection;
- builds the quantile band on the logit scale from the fit spread, a
  refresh walk paced by the observed refresh cadence, a target-staleness
  allowance, and a counting term for trophies held by only a few players;
- when the history holds a single state, falls back to explicit priors:
  saturating pool growth, earners trailing the pool with a dilution
  exponent, and an unseen-refresh band that grows with the page's staleness.

Every trophy or case that breaks the model falls back to the house
persistence row, so the output is always a valid forecast file.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

where `input/` holds the case files (`<id>.in`) and each forecast is written
to `out/<id>.out`.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
