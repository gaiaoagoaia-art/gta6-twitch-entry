"""gta6-twitch-ccu-forecast-01 entry: a launch-day Twitch audience forecaster.

Reads one case file (schema makefun/twitch-ccu-case/v1) per run from the input
directory and writes, for each of hours 2..24 of the release day, five quantiles
of that hour's concurrent-viewer count in the game's Twitch category.

Model
-----
The forecast anchors on the case's own hour-1 print p1 and multiplies it by a
ratio distribution learned at run time from the case's `comparables` — past big
game launches whose full release-day hourly curves ship inside the case file:

  * For every comparable with an hour-1 value, form the log-ratios
    L(h) = ln(viewers at hour h) - ln(viewers at hour 1) of its release day.
  * GTA VI and the practice launches are console games that unlock at rolling
    local midnight, so launches whose `unlock` note says so are the closest
    shape family. Each hour's predictive distribution is a mixture: 0.65 of the
    rolling-midnight launches' empirical distribution of L(h) and 0.35 of all
    usable launches', which keeps the subset's shape but fattens the tails
    toward the global one (some rolling launches ramp all day instead).
  * The mixture's quantiles are blended (geometric mean) with a normal fit
    (mean, sd) of the rolling subset's L(h). Adjacent hours are fitted
    independently from nearly the same data, so the fitted log-quantile
    curves are then lightly smoothed across neighbouring hours with a
    [0.25, 0.5, 0.25] kernel, trimming hour-to-hour fitting noise, and the
    result is shrunk 20% in log space toward the persistence baseline's
    quantiles, which hedges the small sample. Hours or cases with no usable
    comparables fall back to that baseline outright.

Everything is a deterministic function of the case file: no randomness, no
network, standard library only, and a run takes well under a second.
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

DEFAULT_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
NORMAL = statistics.NormalDist()

# The persistence baseline's per-root-hour log spread (the house starter's).
BASE_SIGMA = 0.15
# Mixture weight on the rolling-midnight subset's empirical distribution.
MIX_ROLLING = 0.65
# Log-space shrinkage of the fitted quantiles toward the baseline's.
SHRINK = 0.20


def round2(x: float) -> float:
    """Round half up to 2 decimals, as the forecast format allows fractions."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    return whole / 100


def parse_ts(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def is_rolling(unlock: str) -> bool:
    """True when a comparable's unlock note describes rolling local midnight."""
    text = (unlock or "").lower()
    return "midnight local" in text or ("midnight" in text and "rolling" in text)


def day_ratios(comp: dict) -> dict[int, float] | None:
    """A comparable's release-day log-ratios L(hour) for hours 2..24, or None.

    The comparable is usable only when its release day's first hour is present
    and positive: every ratio is measured against that hour, as the forecast
    is measured against the case's hour-1 print.
    """
    try:
        start = parse_ts(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    by_hour: dict[int, float] = {}
    for point in comp.get("hourly") or []:
        try:
            offset = (parse_ts(point["from"]) - start).total_seconds() / 3600
            viewers = point["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        hour = round(offset)
        if abs(offset - hour) < 1e-6 and 0 <= hour < 24 and viewers is not None:
            by_hour[int(hour)] = float(viewers)
    first = by_hour.get(0)
    if first is None or first <= 0:
        return None
    log_first = math.log(first)
    return {h + 1: math.log(v) - log_first for h, v in by_hour.items() if h >= 1 and v > 0}


def baseline_quantiles(p1: float, hour: int, levels: list[float]) -> list[float]:
    spread = BASE_SIGMA * math.sqrt(hour - 1)
    return [p1 * math.exp(NORMAL.inv_cdf(tau) * spread) for tau in levels]


def mixture_quantile(rolling: list[float], overall: list[float], tau: float) -> float:
    """The tau quantile of MIX_ROLLING * F_rolling + (1 - MIX_ROLLING) * F_all.

    Both empirical CDFs are evaluated on their pooled support; the quantile is
    the smallest support point whose mixture CDF reaches tau.
    """
    support = sorted(set(rolling) | set(overall))

    def cdf(x: float) -> float:
        total = 0.0
        if rolling:
            total += MIX_ROLLING * (sum(1 for v in rolling if v <= x) / len(rolling))
        if overall:
            total += (1 - MIX_ROLLING) * (sum(1 for v in overall if v <= x) / len(overall))
        return total

    for point in support:
        if cdf(point) >= tau:
            return point
    return support[-1]


def fitted_log_quantiles(
    levels: list[float],
    rolling_ratios: list[float],
    all_ratios: list[float],
) -> list[float] | None:
    """One hour's fitted log-quantile offsets, or None when nothing is usable."""
    if not all_ratios:
        return None
    fitted = []
    for tau in levels:
        log_q = mixture_quantile(rolling_ratios, all_ratios, tau)
        if len(rolling_ratios) >= 3:
            mean = sum(rolling_ratios) / len(rolling_ratios)
            var = sum((v - mean) ** 2 for v in rolling_ratios) / (len(rolling_ratios) - 1)
            normal_q = mean + NORMAL.inv_cdf(tau) * math.sqrt(var)
            log_q = (log_q + normal_q) / 2
        fitted.append(log_q)
    return fitted


def smooth_hours(by_hour: dict[int, list[float]]) -> dict[int, list[float]]:
    """Lightly smooth fitted log-quantiles across adjacent hours.

    Each hour's fit is estimated independently from almost the same pool of
    launches, so the hour-to-hour wiggle in the fitted curves is mostly
    fitting noise. A [0.25, 0.5, 0.25] kernel over neighbouring hours trades
    a little hour-specific shape for lower variance; edge hours renormalise
    over the neighbours they have.
    """
    hours = sorted(by_hour)
    out = {}
    for i, hour in enumerate(hours):
        acc = None
        wsum = 0.0
        for j, w in ((i - 1, 0.25), (i, 0.5), (i + 1, 0.25)):
            if 0 <= j < len(hours):
                vals = by_hour[hours[j]]
                if acc is None:
                    acc = [w * v for v in vals]
                else:
                    acc = [a + w * v for a, v in zip(acc, vals)]
                wsum += w
        out[hour] = [a / wsum for a in acc]
    return out


def tidy(values: list[float]) -> list[float]:
    """Round, clamp into (0, 1e9] and enforce the non-decreasing order."""
    out = []
    prev = 0.0
    for v in values:
        v = round2(min(max(v, 0.01), 1_000_000_000.0))
        v = max(v, prev)
        out.append(int(v) if float(v).is_integer() else v)
        prev = float(v)
    return out


def forecast_case(case: dict) -> dict:
    levels = [float(t) for t in case.get("levels") or DEFAULT_LEVELS]
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    p1 = 1.0
    if first is not None and first.get("viewers"):
        p1 = max(1.0, float(first["viewers"]))

    rolling: dict[int, list[float]] = {}
    overall: dict[int, list[float]] = {}
    for comp in case.get("comparables") or []:
        ratios = day_ratios(comp)
        if not ratios:
            continue
        roll = is_rolling(comp.get("unlock", ""))
        for hour, value in ratios.items():
            overall.setdefault(hour, []).append(value)
            if roll:
                rolling.setdefault(hour, []).append(value)

    hours = [int(row["hour"]) for row in case.get("forecast") or []]
    fitted = {}
    for hour in hours:
        fq = fitted_log_quantiles(
            levels,
            sorted(rolling.get(hour, [])),
            sorted(overall.get(hour, [])),
        )
        if fq is not None:
            fitted[hour] = fq
    fitted = smooth_hours(fitted)
    log_p1 = math.log(p1)
    rows = []
    for hour in hours:
        base = baseline_quantiles(p1, hour, levels)
        if hour in fitted:
            qs = [
                math.exp((1 - SHRINK) * (log_p1 + lq) + SHRINK * math.log(b))
                for lq, b in zip(fitted[hour], base)
            ]
        else:
            qs = base
        rows.append({"hour": hour, "q": tidy(qs)})
    return {"hours": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
