"""Clean up spurious / inconsistent PitStop events and re-validate validity.

The database should carry exactly one PitStop event per real pit stop,
attached to the in-lap.  Three classes of inconsistency exist:

  * spurious stops   -- events with duration_sec < 15 (the old 2.3 s
                        artifact attached to a normal lap).  Purged.
  * phantom stops    -- events whose following lap shows no tyre change at
                        all (no compound change, no age drop): a "pit"
                        that never happened.  Deleted.
  * missing stops    -- mid-race tyre changes (compound change or age drop)
                        with no event on the in-lap, so the real in/out-lap
                        pair is unmarked.  A PitStop event is inserted with
                        duration_sec = NULL (the stop happened; the box time
                        is unknown).

Race-start / red-flag tyre data quirks on lap 1 (compound jumps between the
first and second existing laps) are deliberately left alone: they are not
box stops and usually have no in-lap row to attach an event to.

Run without arguments for a dry-run report; pass --apply to repair the DB.

Usage:
    python scripts/cleanup_pit_events.py            # dry run
    python scripts/cleanup_pit_events.py --apply    # repair
"""

import argparse
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from config import get_db_connection

# Anything shorter than this is the known 2.3 s glitch, not a wheel change.
PIT_MIN_DURATION_S = 15.0


# ---------------------------------------------------------------------------
# Pure reconciliation logic (no DB) -- unit-testable.
# ---------------------------------------------------------------------------

def _stint_changed(prev: Dict[str, Any], nxt: Dict[str, Any]) -> bool:
    """True when the tyre changed between two consecutive existing laps."""
    if nxt is None:
        return False
    # Compound change (only when both compounds are known).
    if (prev.get("tyre_compound") and nxt.get("tyre_compound")
            and prev["tyre_compound"] != nxt["tyre_compound"]):
        return True
    # Age drop (only when both ages are known) -- e.g. fresh intermediates.
    a, b = prev.get("tyre_age"), nxt.get("tyre_age")
    if a is not None and b is not None and b < a:
        return True
    return False


def reconcile_pit_events(
    laps: List[Dict[str, Any]]
) -> Tuple[List[int], List[Dict[str, Any]]]:
    """Classify laps into phantom and missing pit events.

    ``laps``: dicts with keys session_id, lap_number, lap_id,
    tyre_compound, tyre_age, has_pit_event (bool), sorted by
    (session_id, lap_number).

    Returns ``(phantoms, missing)``:

    * phantoms -- lap_ids carrying a PitStop event with no tyre change on
      the next existing lap (fully-known tyre data required, so a data gap
      never deletes a real event).
    * missing  -- dicts {session_id, lap_id, lap_number} for mid-race tyre
      changes whose in-lap has no PitStop event.  Lap-1 compound jumps are
      race-start quirks, not pits, and are excluded.
    """
    by_sess: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for l in laps:
        by_sess[l["session_id"]].append(l)

    phantoms: List[int] = []
    missing: List[Dict[str, Any]] = []

    for sid in sorted(by_sess):
        ls = by_sess[sid]
        for i, l in enumerate(ls):
            nxt = ls[i + 1] if i + 1 < len(ls) else None
            changed = _stint_changed(l, nxt)

            if l["has_pit_event"]:
                # Phantom: a pit event followed by a lap with fully-known
                # tyre data that shows no change at all.
                if (nxt is not None and not changed
                        and nxt.get("tyre_compound") is not None
                        and nxt.get("tyre_age") is not None):
                    phantoms.append(l["lap_id"])
            elif changed and i > 0:
                # Missing: mid-race stint change (not lap 1) with no event.
                missing.append({
                    "session_id": sid,
                    "lap_id": l["lap_id"],
                    "lap_number": l["lap_number"],
                })

    return phantoms, missing


# ---------------------------------------------------------------------------
# DB plumbing
# ---------------------------------------------------------------------------

def load_laps(conn) -> List[Dict[str, Any]]:
    """All laps with a real time plus whether they carry a PitStop event."""
    cur = conn.cursor(dictionary=True)
    cur.execute(
        """
        SELECT l.session_id, l.lap_number, l.lap_id,
               l.tyre_compound, l.tyre_age,
               EXISTS(SELECT 1 FROM strategy_events se
                      WHERE se.lap_id = l.lap_id
                        AND se.event_type = 'PitStop') AS has_pit_event
        FROM laps l
        WHERE l.lap_time_ms > 0
        ORDER BY l.session_id, l.lap_number
        """
    )
    laps = cur.fetchall()
    for l in laps:
        l["has_pit_event"] = bool(l["has_pit_event"])
    return laps


def find_inconsistencies(conn) -> Dict[str, Any]:
    """Report what is wrong without changing anything."""
    cur = conn.cursor(dictionary=True)
    cur.execute(
        """
        SELECT event_id, lap_id, duration_sec
        FROM strategy_events
        WHERE event_type = 'PitStop' AND duration_sec < %s
        """,
        (PIT_MIN_DURATION_S,),
    )
    spurious = cur.fetchall()

    laps = load_laps(conn)
    phantoms, missing = reconcile_pit_events(laps)

    # Map lap_id -> (session, lap_number) for readable output.
    by_id = {l["lap_id"]: l for l in laps}

    return {
        "spurious": [{"event_id": e["event_id"], "lap_id": e["lap_id"],
                      "duration_sec": e["duration_sec"]} for e in spurious],
        "phantoms": [{"lap_id": lid, "session_id": by_id[lid]["session_id"],
                      "lap_number": by_id[lid]["lap_number"]}
                     for lid in phantoms],
        "missing": missing,
    }


def apply_fixes(conn) -> Dict[str, Any]:
    """Purge spurious, delete phantoms, insert missing events.  Commits."""
    inc = find_inconsistencies(conn)
    cur = conn.cursor()

    if inc["spurious"]:
        cur.execute(
            "DELETE FROM strategy_events "
            "WHERE event_type = 'PitStop' AND duration_sec < %s",
            (PIT_MIN_DURATION_S,),
        )
    if inc["phantoms"]:
        lap_ids = [p["lap_id"] for p in inc["phantoms"]]
        fmt = ",".join(["%s"] * len(lap_ids))
        cur.execute(
            "DELETE FROM strategy_events "
            f"WHERE event_type = 'PitStop' AND lap_id IN ({fmt})",
            lap_ids,
        )
    for m in inc["missing"]:
        # The stop happened; only the box duration is unknown.
        cur.execute(
            "INSERT INTO strategy_events (lap_id, event_type, duration_sec) "
            "VALUES (%s, 'PitStop', NULL)",
            (m["lap_id"],),
        )
    conn.commit()
    return inc


def verify(conn) -> Dict[str, Any]:
    """Re-run the checks; everything should be zero after a repair."""
    inc = find_inconsistencies(conn)
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT COUNT(*) AS n FROM laps WHERE lap_time_ms = 0 AND is_valid = 1")
    zero_ms_valid = cur.fetchone()["n"]
    cur.execute("SELECT COUNT(*) AS n FROM laps WHERE lap_time_ms = 0")
    zero_ms_total = cur.fetchone()["n"]
    return {
        "spurious": len(inc["spurious"]),
        "phantoms": len(inc["phantoms"]),
        "missing": len(inc["missing"]),
        "zero_ms_valid_laps": zero_ms_valid,
        "zero_ms_total_laps": zero_ms_total,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_lap(d: Dict[str, Any]) -> str:
    return f"sess {d['session_id']} lap {d['lap_number']}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Purge spurious pit events and reconcile in/out-laps.")
    ap.add_argument("--apply", action="store_true",
                    help="repair the database (default: dry run)")
    args = ap.parse_args()

    conn = get_db_connection()
    try:
        inc = find_inconsistencies(conn)

        print("=" * 60)
        print("PIT-EVENT CONSISTENCY CHECK")
        print("=" * 60)
        print(f"  Spurious events (duration < {PIT_MIN_DURATION_S:.0f}s): "
              f"{len(inc['spurious'])}")
        for e in inc["spurious"]:
            print(f"    event {e['event_id']} on lap_id {e['lap_id']} "
                  f"({e['duration_sec']}s)")
        print(f"  Phantom events (no tyre change follows): {len(inc['phantoms'])}")
        for p in inc["phantoms"]:
            print(f"    {_fmt_lap(p)}")
        print(f"  Missing events (mid-race tyre change, no event): "
              f"{len(inc['missing'])}")
        for m in inc["missing"]:
            print(f"    {_fmt_lap(m)} (in-lap)")

        if not args.apply:
            print("\nDry run -- nothing changed. Re-run with --apply to repair.")
        else:
            applied = apply_fixes(conn)
            print(f"\n[APPLIED] purged {len(applied['spurious'])} spurious, "
                  f"deleted {len(applied['phantoms'])} phantom, "
                  f"inserted {len(applied['missing'])} missing.")

        v = verify(conn)
        print("\nPOST-CHECK")
        print(f"  spurious: {v['spurious']}   phantom: {v['phantoms']}   "
              f"missing: {v['missing']}")
        print(f"  0ms laps marked valid: {v['zero_ms_valid_laps']} / "
              f"{v['zero_ms_total_laps']} total 0ms laps")
        ok = (v["spurious"] == 0 and v["phantoms"] == 0 and v["missing"] == 0
              and v["zero_ms_valid_laps"] == 0)
        print("  VALIDITY: " + ("OK - pit events and laps are consistent"
                                if ok else "ISSUES REMAIN"))
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
