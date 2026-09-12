"""Per-stint lap-time analysis shared by the dashboard and the CLI report.

Raw lap times inside a stint fall steadily as the fuel burns (the car gets
lighter every lap), which masks any genuine tyre-wear trend.  To surface
wear we remove each stint's own linear trend and express every lap as a
delta vs that trend line:

    delta = lap_time - (intercept + slope * tyre_age)

Positive delta = slower than the stint's own pace line (cold tyres,
traffic, wear); negative = faster.  Within one stint there is exactly one
fuel trajectory, so the fitted line is the fuel curve (plus any linear
wear); subtracting it cancels the fuel effect and leaves the non-linear
part of wear -- e.g. a tyre cliff at high age -- visible as positive
deltas.

Pit in/out laps are excluded (they are pit-lane / cold-tyre laps, not
representative pace), and warm-up laps (age <= WARMUP_AGE_MAX) are
excluded from the trend fit so the cold-tyre spike at the start of a
stint shows up as a genuine positive delta instead of skewing the slope.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

# A pit event shorter than this is a spurious "stop" (the known 2.3 s
# artifact), not a real wheel change -- it must not split a stint.
PIT_EVENT_MIN_DURATION_S = 15.0

# First laps of a stint are cold-tyre / race-start laps.  They are shown
# on the chart (the cold-tyre spike is real) but must not shape the fit.
WARMUP_AGE_MAX = 2


def _num(value: Any) -> Optional[float]:
    """Best-effort float conversion; None for missing/garbage values."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def segment_stints(laps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Split laps into stints.

    A new stint starts on:
      1. the first lap,
      2. the lap immediately after a lap flagged as a pit stop,
      3. the lap where the tyre compound changed,
      4. the lap where tyre age dropped (age reset by a wheel change).

    Returns stints as ``{"stint_number": n, "compound": str, "laps": [...]}``
    in lap order.  The input dicts are not modified.
    """
    ordered = sorted(
        (l for l in laps if l.get("lap_number") is not None),
        key=lambda l: int(l["lap_number"]),
    )
    stints: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for lap in ordered:
        prev = current["laps"][-1] if current and current["laps"] else None
        is_new = False
        if current is None:
            is_new = True
        elif prev is not None:
            if prev.get("has_pit_stop"):
                is_new = True
            elif (
                lap.get("tyre_compound") and prev.get("tyre_compound")
                and lap["tyre_compound"] != prev["tyre_compound"]
            ):
                is_new = True
            elif (
                _num(lap.get("tyre_age")) is not None
                and _num(prev.get("tyre_age")) is not None
                and _num(lap["tyre_age"]) < _num(prev["tyre_age"])
            ):
                is_new = True

        if is_new:
            current = {
                "stint_number": (current["stint_number"] + 1) if current else 1,
                "compound": lap.get("tyre_compound") or "Unknown",
                # Only a stint that directly follows a pit stop has an
                # out-lap as its first lap.  A stint that starts from a
                # compound change or an age reset mid-race (no pit) begins
                # with a normal lap that must be kept.
                "started_after_pit": bool(
                    prev is not None and prev.get("has_pit_stop")
                ),
                "laps": [],
            }
            stints.append(current)
        current["laps"].append(lap)

    return stints


def _fit_trend(fit_laps: List[Dict[str, Any]]) -> Optional[tuple]:
    """Linear least-squares trend of lap_time vs tyre_age.

    Returns ``(slope, intercept)`` or None when a fit is not possible
    (fewer than 3 points, or a single tyre age in the stint).
    """
    if len(fit_laps) < 3:
        return None
    ages = np.array([_num(l["tyre_age"]) for l in fit_laps], dtype=float)
    times = np.array([_num(l["lap_time"]) for l in fit_laps], dtype=float)
    if np.std(ages) < 1e-9:
        return None
    slope, intercept = np.polyfit(ages, times, 1)
    return float(slope), float(intercept)


def _detrend_stint(stint: Dict[str, Any]) -> None:
    """Fill ``stint_number`` / ``stint_delta`` on every lap of one stint."""
    laps = stint["laps"]

    # The in-lap (pit lane) is excluded from the chart -- a transition lap,
    # not representative pace.  The out-lap is the first lap of the stint
    # that follows, but only when that stint really follows a pit stop.
    excluded = {i for i, l in enumerate(laps) if l.get("has_pit_stop")}
    if stint.get("started_after_pit") and laps:
        excluded.add(0)

    def usable_for_delta(l: Dict[str, Any]) -> bool:
        return (
            _num(l.get("lap_time")) is not None
            and _num(l.get("tyre_age")) is not None
        )

    delta_laps = [l for i, l in enumerate(laps)
                  if i not in excluded and usable_for_delta(l)]

    # A stint with fewer than 2 usable laps cannot fit a trend.
    # When pit laps stole the usable count (short stint after pit),
    # non-excluded laps still deserve a pace delta via the median.
    if len(delta_laps) < 2:
        if excluded and delta_laps:
            fallback = float(np.median([_num(l["lap_time"]) for l in delta_laps]))
            for i, l in enumerate(laps):
                l["stint_number"] = stint["stint_number"]
                if i in excluded:
                    l["is_pit_lap"] = 1
                    l["stint_delta"] = None
                    pit_time = _num(l.get("lap_time"))
                    if pit_time is not None:
                        l["pit_delta"] = float(pit_time - fallback)
                elif _num(l.get("lap_time")) is not None and _num(l.get("tyre_age")) is not None:
                    l["stint_delta"] = float(_num(l["lap_time"]) - fallback)
                else:
                    l["stint_delta"] = None
        else:
            for l in laps:
                l["stint_number"] = stint["stint_number"]
                l["stint_delta"] = None
        return

    # Fit on representative laps only: no in/out laps, no warm-up laps,
    # no invalid laps.
    fit_laps = [
        l for i, l in enumerate(laps)
        if (i not in excluded
            and usable_for_delta(l)
            and l.get("is_valid") in (1, True)
            and _num(l["tyre_age"]) > WARMUP_AGE_MAX)
    ]
    trend = _fit_trend(fit_laps)
    fallback = float(np.median([_num(l["lap_time"]) for l in delta_laps]))

    for i, l in enumerate(laps):
        l["stint_number"] = stint["stint_number"]
        if i in excluded:
            # Pit in/out laps are transition laps, not pace -- so they stay
            # out of the fit and carry no pace delta (stint_delta).  But
            # give them a pit_delta: their offset from the stint's pace
            # line, so the chart can plot the pit stop as a visible spike
            # instead of a hole in the line.
            l["is_pit_lap"] = 1
            l["stint_delta"] = None
            if _num(l.get("lap_time")) is None or _num(l.get("tyre_age")) is None:
                l["pit_delta"] = None
            elif trend is not None:
                slope, intercept = trend
                l["pit_delta"] = float(
                    _num(l["lap_time"]) - (intercept + slope * _num(l["tyre_age"]))
                )
            else:
                l["pit_delta"] = float(_num(l["lap_time"]) - fallback)
        elif _num(l.get("lap_time")) is None or _num(l.get("tyre_age")) is None:
            l["stint_delta"] = None
        elif trend is not None:
            slope, intercept = trend
            l["stint_delta"] = float(
                _num(l["lap_time"]) - (intercept + slope * _num(l["tyre_age"]))
            )
        else:
            # Too short to fit a trend: fall back to the stint median.
            l["stint_delta"] = float(_num(l["lap_time"]) - fallback)


def detrend_laps(laps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return copies of ``laps`` with ``stint_number`` and ``stint_delta``.

    ``stint_delta`` is seconds relative to the stint's own pace line
    (fuel-corrected); None marks laps that carry no meaningful pace delta
    (laps without a tyre age, or single-lap stints).  Pit in/out laps
    carry ``is_pit_lap=1`` and a separate ``pit_delta`` (their offset from
    the pace line, for visualization) instead of a pace delta.
    """
    out = [dict(l) for l in laps]
    for stint in segment_stints(out):
        _detrend_stint(stint)
    return out
