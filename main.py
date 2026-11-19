"""Trophy rarity forecaster for GTA VI (make.fun gta6-trophy-rarity-forecast-01).

What the program predicts: for every PlayStation 5 trophy of the game, the
share of Exophase-tracked players who will hold it at the case target time,
given as five quantiles (5/25/50/75/95 percent) of the printed earn rate.

How it thinks about the data
----------------------------
Exophase does not stream its numbers. The page sits frozen for hours, then a
refresh lands and the tracked-player count jumps while every earn rate moves
in the same instant. Two consequences drive the whole design:

1.  Repeated reads of the same page carry no new information, so the history
    is first collapsed into *states* — the distinct pages, in order. Every
    fit below runs on states, never on raw reads.

2.  A rate is a ratio, earners / players, and the two sides move for
    different reasons during launch week: the tracked pool grows by sign-ups
    (fast, then saturating), earners grow by people actually playing (slower,
    trophy by trophy). Forecasting the ratio directly mixes the two motions,
    so players and earners are projected separately on a log-time axis and
    the median rate is their quotient at the target.

Uncertainty is assembled per trophy on the logit scale (the scoring scale)
from four parts: the misfit of the trophy's own earners curve shrunk toward
the cross-trophy pool, a refresh walk whose length follows the observed
refresh cadence, an allowance for the unknown age of the page at the target
instant, and a counting term that widens the band for trophies held by only
a handful of tracked players. When the history holds a single state (the
launch-night practice cases) no curve can be fitted, so explicit priors take
over: the pool keeps growing on a saturating power curve, earners trail it
with a dilution exponent, and the band carries a priced "a refresh we have
not seen yet" term that grows with the page's staleness.

Any trophy — or a whole case — that breaks the model falls back to the house
persistence row, so the output file is always well formed. Standard library
only, fully deterministic, and far inside the 8 second budget.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SRC_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
DST_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

TAUS = (0.05, 0.25, 0.5, 0.75, 0.95)
ZNORM = (-1.645, -0.674, 0.0, 0.674, 1.645)

# Behaviour of the tracked pool when only one page state is known: the pool
# grows roughly like (age + AGE_PAD) ** POOL_EXP and earners trail the pool
# with exponent DILUTE, because late sign-ups have had less time to play.
POOL_EXP = 1.9
AGE_PAD_H = 4.0
DILUTE = 0.5

# Spread priors, all in logit units.
UNSEEN_BASE = 0.55        # sd of the log pool-multiplier of an unseen refresh
UNSEEN_STALE = 0.10       # extra sd per sqrt(hour) the page stayed frozen
UNSEEN_MAX = 0.85
WALK_FALLBACK = 0.30      # random-walk rate per sqrt(day) when unobserved
BAND_MIN = 0.10
BAND_MAX = 2.20
SHIFT_LIMIT = 2.2         # a projected median may not move further than this
COUNT_COEF = 0.5          # counting-noise weight for thin trophies


# --------------------------------------------------------------------------
# small numeric helpers
# --------------------------------------------------------------------------

def pct_clamp(v):
    return 99.99 if v > 99.99 else (0.01 if v < 0.01 else v)


def to_logit(pct):
    p = pct_clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def from_logit(x):
    if x >= 0.0:
        return 100.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return 100.0 * ex / (1.0 + ex)


def q4(v):
    """Round to 4 decimals, halves away from zero, ints stay ints."""
    scaled = v * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    out = whole / 10000.0
    return int(out) if float(out).is_integer() else out


def read_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def middle(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def wls(points):
    """Weighted least squares y ~ a + b*x over (x, y, w) triples.

    Returns (a, b, resid_sd) where resid_sd is the weighted residual spread
    with a degrees-of-freedom correction, or None for a degenerate input.
    """
    n = len(points)
    if n == 0:
        return None
    wsum = sum(w for _, _, w in points)
    if wsum <= 0.0:
        return None
    mx = sum(w * x for x, _, w in points) / wsum
    my = sum(w * y for _, y, w in points) / wsum
    sxx = sum(w * (x - mx) ** 2 for x, _, w in points)
    if sxx <= 1e-12:
        return (my, 0.0, 0.0)
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in points)
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        ss = sum(w * (y - (a + b * x)) ** 2 for x, y, w in points)
        resid = math.sqrt(ss / wsum * n / (n - 2))
    else:
        resid = 0.0
    return (a, b, resid)


def persistence_row(rate, horizon_minutes):
    """The house baseline row: last rate, random-walk spread. Fallback only."""
    width = 0.5 * math.sqrt(max(horizon_minutes, 0.0) / 1440.0)
    centre = to_logit(rate if rate is not None else 0.0)
    return [q4(from_logit(centre + z * width)) for z in ZNORM]


def monotone(qs):
    out = []
    for v in qs:
        v = float(v)
        if v < 0.0:
            v = 0.0
        elif v > 100.0:
            v = 100.0
        if out and v < out[-1]:
            v = out[-1]
        out.append(v)
    return out


# --------------------------------------------------------------------------
# history handling
# --------------------------------------------------------------------------

def collapse_states(history):
    """Distinct pages in order: (moment, players, rates) per refresh state."""
    states = []
    seen_key = None
    for read in history:
        players = read.get("players_tracked")
        if players is None:
            continue
        key = (players, tuple(read.get("rates") or ()))
        if key != seen_key:
            states.append((read_ts(read["at"]), float(players), read["rates"]))
            seen_key = key
    return states


def recency_weights(count, ages_h):
    """Weights for the curve fits: newer states speak louder, and states left
    far behind the newest one fade toward a floor instead of vanishing."""
    if count == 1:
        return [1.0]
    out = []
    for idx in range(count):
        ramp = 1.0 + 2.0 * (idx / (count - 1.0))
        fade = 0.35 + 0.65 * math.exp(-ages_h[idx] / 96.0)
        out.append(ramp * fade)
    return out


# --------------------------------------------------------------------------
# the forecast itself
# --------------------------------------------------------------------------

def forecast_case(case, label):
    if list(case["levels"]) != list(TAUS):
        raise ValueError(label + ": unexpected levels " + str(case["levels"]))
    history = case["history"]
    if not history:
        raise ValueError(label + ": empty history")
    trophies = case["trophies"]
    n = len(trophies)
    horizon_min = float(case["horizon_minutes"])
    horizon_h = horizon_min / 60.0
    horizon_d = horizon_min / 1440.0

    states = collapse_states(history)
    if not states:
        tail = history[-1]
        return {"trophies": [
            {"id": t["id"], "q": persistence_row(tail["rates"][i], horizon_min)}
            for i, t in enumerate(trophies)]}

    origin = states[0][0]
    target = read_ts(case["target"])
    checkpoint = read_ts(case["at"])

    def hours_from_origin(moment):
        return max(0.0, (moment - origin).total_seconds() / 3600.0)

    def logtime(moment):
        return math.log1p(hours_from_origin(moment))

    u_target = logtime(target)
    last_moment, last_players, _ = states[-1]
    stale_h = max(0.0, (checkpoint - last_moment).total_seconds() / 3600.0)

    # The freshest known rate of each trophy (a read may lack a trophy).
    freshest = [None] * n
    for read in reversed(history):
        rates = read["rates"]
        for i in range(n):
            if freshest[i] is None and rates[i] is not None:
                freshest[i] = rates[i]
        if all(v is not None for v in freshest):
            break

    state_u = [logtime(m) for m, _, _ in states]
    state_age = [(last_moment - m).total_seconds() / 3600.0 for m, _, _ in states]
    weights = recency_weights(len(states), state_age)

    # ---- refresh cadence and the size of one refresh step ----------------
    gaps_h = []
    step_moves = []
    for j in range(1, len(states)):
        gaps_h.append((states[j][0] - states[j - 1][0]).total_seconds() / 3600.0)
        before, after = states[j - 1][2], states[j][2]
        for i in range(n):
            if before[i] is not None and after[i] is not None:
                step_moves.append(abs(to_logit(after[i]) - to_logit(before[i])))
    if step_moves:
        rms_step = math.sqrt(sum(v * v for v in step_moves) / len(step_moves))
        med_step = middle(step_moves)
    else:
        rms_step = 0.0
        med_step = 0.0
    med_gap_h = middle(gaps_h) if gaps_h else 0.0

    # ---- tracked-pool projection ------------------------------------------
    if len(states) >= 2:
        fit_p = wls([(state_u[j], math.log(states[j][1]), weights[j])
                     for j in range(len(states))])
        a_p, b_p, resid_p = fit_p
        players_star = math.exp(a_p + b_p * u_target)
        if players_star < last_players * 0.99:
            players_star = last_players * 0.99
        pool_log_growth = math.log(players_star / last_players)
    else:
        resid_p = 0.0
        b_p = 0.0
        span_last = hours_from_origin(last_moment)
        span_target = hours_from_origin(target)
        growth = ((span_target + AGE_PAD_H) / (span_last + AGE_PAD_H)) ** POOL_EXP
        players_star = last_players * growth
        pool_log_growth = math.log(growth)

    # ---- first pass over trophies: earners curves --------------------------
    # For each trophy keep its earners points, its fitted slope and residual,
    # so the second pass can shrink noisy trophies toward the pooled values.
    per_trophy = []
    slopes = []
    resids = []
    for i in range(n):
        pts = []
        for j, (moment, players, rates) in enumerate(states):
            r = rates[i]
            if r is not None:
                earners = pct_clamp(r) / 100.0 * players
                pts.append((state_u[j], math.log(max(earners, 0.05)), weights[j]))
        entry = {"points": pts, "fit": None, "trend": None}
        if len(pts) >= 2:
            entry["fit"] = wls(pts)
            slopes.append(entry["fit"][1])
            if len(pts) >= 3:
                resids.append(entry["fit"][2])
        if len(pts) >= 3:
            tpts = []
            for j, (moment, players, rates) in enumerate(states):
                r = rates[i]
                if r is not None:
                    tpts.append((state_u[j], to_logit(r), weights[j]))
            entry["trend"] = wls(tpts)
        per_trophy.append(entry)
    pooled_slope = middle(slopes) if slopes else DILUTE * max(b_p, 0.0)
    pooled_resid = middle(resids) if resids else 0.15

    # ---- spread pieces that do not depend on the trophy -------------------
    if len(states) >= 2 and med_gap_h > 0.0:
        expected_refreshes = horizon_h / max(med_gap_h, 0.25)
        walk_sd = rms_step * math.sqrt(min(expected_refreshes, 60.0))
        stale_sd = 0.35 * rms_step
        unseen_sd = 0.10
    elif len(states) >= 2:
        walk_sd = (med_step if med_step > 0 else WALK_FALLBACK) * math.sqrt(horizon_d)
        stale_sd = 0.0
        unseen_sd = 0.10
    else:
        sd_pool = min(UNSEEN_MAX, UNSEEN_BASE + UNSEEN_STALE * math.sqrt(stale_h))
        walk_sd = WALK_FALLBACK * math.sqrt(horizon_d)
        stale_sd = 0.0
        unseen_sd = (1.0 - DILUTE) * sd_pool

    # ---- second pass: median and band per trophy ---------------------------
    rows = []
    for i, trophy in enumerate(trophies):
        try:
            rows.append(single_trophy(
                trophy, freshest[i], last_players, players_star,
                pool_log_growth, per_trophy[i], pooled_slope, pooled_resid,
                resid_p, u_target, walk_sd, stale_sd, unseen_sd, horizon_min))
        except Exception:
            rows.append({"id": trophy["id"],
                         "q": persistence_row(freshest[i], horizon_min)})
    return {"trophies": rows}


def single_trophy(trophy, last_rate, last_players, players_star,
                  pool_log_growth, entry, pooled_slope, pooled_resid,
                  resid_p, u_target, walk_sd, stale_sd, unseen_sd,
                  horizon_min):
    if last_rate is None:
        return {"id": trophy["id"], "q": persistence_row(0.05, horizon_min)}
    base_rate = pct_clamp(last_rate)
    earners_last = base_rate / 100.0 * last_players

    # Held by literally every tracked player: it can only stay pinned.
    if earners_last >= 0.999 * last_players:
        centre = to_logit(99.99)
        return {"id": trophy["id"], "q": monotone(
            [q4(from_logit(centre + z * 0.05)) for z in ZNORM])}

    pts = entry["points"]
    fit = entry["fit"]
    if fit is not None:
        a_e, b_e, resid_e = fit
        if len(pts) == 2:
            # A two-point line is exact and therefore overconfident in its
            # slope; pull the slope halfway toward the pooled earners slope.
            b_e = 0.5 * b_e + 0.5 * pooled_slope
            a_e = pts[-1][1] - b_e * pts[-1][0]
        earners_star = math.exp(a_e + b_e * u_target)
        # Hierarchical shrinkage of the fit spread toward the pool.
        k = len(pts)
        resid_shrunk = (k * resid_e + 4.0 * pooled_resid) / (k + 4.0)
        fit_sd = math.sqrt(resid_shrunk ** 2 + resid_p ** 2)
    elif len(pts) == 1:
        earners_star = earners_last * math.exp(DILUTE * pool_log_growth)
        fit_sd = 0.15
    else:
        earners_star = earners_last * math.exp(DILUTE * pool_log_growth)
        fit_sd = 0.0

    # Saturation damping for long horizons: earners growth read off a
    # log-time line keeps accelerating in levels, while real earners bend
    # toward the share of players who will ever earn the trophy. Shave the
    # extrapolated log-growth once the horizon passes two days.
    horizon_days = horizon_min / 1440.0
    if fit is not None and horizon_days > 2.0 and earners_star > earners_last:
        log_growth = math.log(earners_star / earners_last)
        log_growth /= (1.0 + 0.18 * (horizon_days - 2.0))
        earners_star = earners_last * math.exp(log_growth)

    earners_star = min(max(earners_star, earners_last * 0.97), players_star)
    median_rate = pct_clamp(100.0 * earners_star / players_star)
    centre = to_logit(median_rate)

    # Blend with the direct rate-trend projection when one exists: the trend
    # sees saturation the decomposition misses, the decomposition sees pool
    # dilution the trend misses. The decomposition keeps the larger voice.
    trend = entry["trend"]
    if trend is not None:
        a_t, b_t, _ = trend
        trend_centre = a_t + b_t * u_target
        centre = 0.65 * centre + 0.35 * trend_centre

    anchor = to_logit(base_rate)
    if centre - anchor > SHIFT_LIMIT:
        centre = anchor + SHIFT_LIMIT
    elif anchor - centre > SHIFT_LIMIT:
        centre = anchor - SHIFT_LIMIT

    count_sd = COUNT_COEF / math.sqrt(max(earners_last, 1.0))
    if count_sd > 0.6:
        count_sd = 0.6
    band = math.sqrt(fit_sd ** 2 + walk_sd ** 2 + stale_sd ** 2
                     + unseen_sd ** 2 + count_sd ** 2 + BAND_MIN ** 2)
    if band > BAND_MAX:
        band = BAND_MAX
    elif band < BAND_MIN:
        band = BAND_MIN

    return {"id": trophy["id"], "q": monotone(
        [q4(from_logit(centre + z * band)) for z in ZNORM])}


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def run():
    DST_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(SRC_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            result = forecast_case(case, path.name)
        except Exception:
            tail = case["history"][-1]
            result = {"trophies": [
                {"id": t["id"],
                 "q": persistence_row(tail["rates"][i], case["horizon_minutes"])}
                for i, t in enumerate(case["trophies"])]}
        (DST_DIR / (path.stem + ".out")).write_text(
            json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
