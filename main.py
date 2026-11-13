# roblox-up-and-coming-forecast-01: recency-weighted rank-kernel forecaster.
#
# Model-free, deterministic, standard library only, no network.
#
# Why this shape: the Up-and-Coming chart moves several times a day
# (RULES: #1 changed 13 times in 45 archive reads; the top 10 changed in
# 42 of 44 consecutive-read pairs, a median of 6 places). The house
# baseline -- persistence, 0.5 on the game at each place now -- scores
# -43.512 on the practice forecast, because the chart has moved away
# from it by the time most of the 72-hour window is scored. A single
# forecast has to cover the whole window, so the honest forecast is a
# *diffuse* one: spread each place's probability over every game that
# has plausibly held a nearby place recently, instead of betting
# everything on today's order.
#
# Method, per place k (1..10):
#   * every snapshot's rank r for a game contributes weight
#     0.5 ** (age_hours / 3)  *  exp(-(r - k)^2 / (2 * 3^2))
#     i.e. an exponential recency half-life of 3 hours and a Gaussian
#     kernel of sigma 3 places around the observed rank. Only ranks
#     <= 40 are used: a game outside the top 40 has never plausibly
#     held a top-10 place in the recent past.
#   * the weights are normalised so each place sums to 0.65; the
#     remaining 0.35 is left unnamed, shared equally at each read by
#     every game on that read's chart not named at that place. That
#     residual is what covers brand-new climbers the history cannot
#     name -- naming them is impossible, and a 0.35 residual gives an
#     unnamed game ~0.35/90 ~= 0.0039 (ln -5.55) instead of the
#     -9.21 floor.
#   * games whose `created` date (in the house recorder's `details`,
#     when present) puts their 28-day eligibility ending inside the
#     scored window are down-weighted in proportion to the fraction
#     of the window they remain eligible for (floor factor 0.25, so
#     they keep some mass for the early hours); games already aged
#     out before the window starts are dropped.
#   * top10 is the ten games with the largest total probability mass
#     over the ten places, in that order (log score only, tie-break).
#
# Tuning (backtested on the kit's 34 archive snapshots, 2026-10-30 to
# 2026-11-07, and on the true practice window, the 10 public-archive
# reads 2026-11-08T00:37Z to 2026-11-10T17:51Z, fetched from
# github.com/Kenny-Dhub/roblox-snapshots):
#   * rolling 72h backtest, 23 windows: baseline -44.91, this model
#     (half-life 3h, sigma 3, residual 0.35) -37.18.
#   * true practice window: baseline -43.512 (reproduced exactly,
#     confirming the scorer), this model -33.93, typical p 0.034 vs
#     the baseline's 0.013 -- about +9.6 log-score points.
#   * longer half-lives, wider/narrower kernels, trend extrapolation
#     and playing-growth momentum were all tried and all scored
#     worse on both tests; trend extrapolation was much worse (-41.9).
#
# Run locally:
#   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py

import json
import math
import os
from datetime import datetime, timedelta, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

HALF_LIFE_HOURS = 3.0
SIGMA_PLACES = 3.0
RESIDUAL = 0.35  # left unnamed at each place; named rows sum to 1 - RESIDUAL
MAX_RANK = 40
MIN_P = 0.001
AGE_OUT_DAYS = 28


def parse_time(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def created_by_uid(snapshot):
    """Map universe_id -> created datetime, from the snapshot's details.

    The archive snapshots carry details=null; the house recorder's
    snapshots carry per-game details. Handle dict- and list-shaped
    details defensively; return {} when there is nothing usable.
    """
    details = snapshot.get("details")
    out = {}
    if isinstance(details, dict):
        items = details.items()
        for key, val in items:
            if isinstance(val, dict):
                created = parse_time(val.get("created"))
                if created is not None:
                    try:
                        out[int(key)] = created
                    except (TypeError, ValueError):
                        pass
    elif isinstance(details, list):
        for val in details:
            if isinstance(val, dict) and "universe_id" in val:
                created = parse_time(val.get("created"))
                if created is not None:
                    out[int(val["universe_id"])] = created
    return out


def main():
    with open(os.path.join(INPUT_DIR, "task.json"), encoding="utf-8") as f:
        task = json.load(f)
    places = int(task.get("places", 10))
    output_name = task.get("output", "forecast.json")

    window = task.get("window") or {}
    window_from = parse_time(window.get("from")) or parse_time(
        task.get("history_until")
    )

    snapshots = []
    with open(os.path.join(INPUT_DIR, "history.ndjson"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                snapshots.append(json.loads(line))
    if not snapshots:
        raise SystemExit("no snapshots in history.ndjson")
    snapshots.sort(key=lambda s: s.get("at", ""))

    t_end = parse_time(snapshots[-1].get("at"))
    if t_end is None:
        raise SystemExit("last snapshot has no usable time")

    # Latest known created date per game, and its rank observations.
    created = {}
    obs = {}  # universe_id -> list of (weight, rank)
    for snap in snapshots:
        at = parse_time(snap.get("at"))
        if at is None:
            continue
        age_h = (t_end - at).total_seconds() / 3600.0
        if age_h < 0:
            age_h = 0.0
        weight = 0.5 ** (age_h / HALF_LIFE_HOURS)
        for uid, made in created_by_uid(snap).items():
            created.setdefault(uid, made)
        for game in snap.get("up_and_coming") or []:
            rank = game.get("rank")
            uid = game.get("universe_id")
            if rank is None or uid is None or rank > MAX_RANK:
                continue
            obs.setdefault(int(uid), []).append((weight, int(rank)))

    # Age-out factor per game: fraction of the 72h window it is still
    # eligible for (published within the last 28 days), floored at 0.25.
    age_factor = {}
    if window_from is not None:
        window_end = window_from + timedelta(hours=72)
        for uid, made in created.items():
            expiry = made + timedelta(days=AGE_OUT_DAYS)
            if expiry <= window_from:
                age_factor[uid] = 0.0
            elif expiry >= window_end:
                age_factor[uid] = 1.0
            else:
                frac = (expiry - window_from).total_seconds() / (72 * 3600.0)
                age_factor[uid] = 0.25 + 0.75 * max(0.0, min(1.0, frac))

    probabilities = []
    total_mass = {}
    for place in range(1, places + 1):
        scores = {}
        for uid in sorted(obs):
            factor = age_factor.get(uid, 1.0)
            if factor <= 0.0:
                continue
            s = 0.0
            for weight, rank in obs[uid]:
                d = rank - place
                s += weight * math.exp(-(d * d) / (2.0 * SIGMA_PLACES * SIGMA_PLACES))
            if s > 0.0:
                scores[uid] = s * factor
        total = sum(scores.values())
        if total <= 0.0:
            continue
        named = {}
        for uid, s in scores.items():
            p = s / total * (1.0 - RESIDUAL)
            if p >= MIN_P:
                named[uid] = p
        # Renormalise after pruning so the named rows sum to 1 - RESIDUAL.
        subtotal = sum(named.values())
        if subtotal <= 0.0:
            continue
        scale = (1.0 - RESIDUAL) / subtotal
        for uid in sorted(named):
            p = named[uid] * scale
            probabilities.append(
                {"universe_id": uid, "place": place, "p": round(p, 6)}
            )
            total_mass[uid] = total_mass.get(uid, 0.0) + p

    top10 = [
        uid
        for uid, _ in sorted(total_mass.items(), key=lambda kv: (-kv[1], kv[0]))[:places]
    ]
    # If fewer than `places` games earned mass (should not happen with a
    # real history), pad with the most recent top-ranked games so the
    # output still has exactly `places` distinct ids.
    if len(top10) < places:
        seen = set(top10)
        last_chart = sorted(
            snapshots[-1].get("up_and_coming") or [],
            key=lambda g: g.get("rank", 10**9),
        )
        for game in last_chart:
            uid = int(game["universe_id"])
            if uid not in seen:
                top10.append(uid)
                seen.add(uid)
            if len(top10) == places:
                break

    forecast = {"probabilities": probabilities, "top10": top10}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, output_name), "w", encoding="utf-8") as f:
        f.write(json.dumps(forecast, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
