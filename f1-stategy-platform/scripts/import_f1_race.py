import argparse
import os
import re
import sys
import logging
from datetime import date, datetime
from pathlib import Path

# Ensure scripts directory is in sys.path for sibling imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

# fastf1 is only needed to import historical race data, not by sibling
# modules (e.g. capture_telemetry) that import helpers from this file, so it
# is imported lazily and guarded.
try:
    import fastf1
except ImportError:
    fastf1 = None

import pandas as pd
from fuel_estimation import estimate_fuel_load
from config import get_db_connection

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(message)s")

# ---------------------------------------------------------------------------
# FastF1 cache
# ---------------------------------------------------------------------------
_CACHE_DIR = str(Path(__file__).resolve().parent.parent / "f1_cache")
if fastf1 is not None:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(_CACHE_DIR)

# ---------------------------------------------------------------------------
# Tyre compound normalisation (FastF1 raw → DB ENUM)
# ---------------------------------------------------------------------------
TYRE_COMPOUND_MAP = {
    "HYPERSOFT":    "Hypersoft",
    "ULTRASOFT":    "Ultrasoft",
    "SUPERSOFT":    "Supersoft",
    "SOFT":         "Soft",
    "MEDIUM":       "Medium",
    "HARD":         "Hard",
    "SUPERHARD":    "Superhard",
    "INTERMEDIATE": "Intermediate",
    "WET":          "Wet",
}


# Race calendar used by import_f1_dataset for the track menu
RACE_CALENDAR = {
    2018: ["Australia", "Bahrain", "China", "Azerbaijan", "Spain", "Monaco",
           "Canada", "France", "Austria", "Silverstone", "Germany", "Hungary",
           "Belgium", "Italy", "Singapore", "Russia", "Japan", "United States",
           "Mexico", "Brazil", "Abu Dhabi"],
    2019: ["Australia", "Bahrain", "China", "Azerbaijan", "Spain", "Monaco",
           "Canada", "France", "Austria", "Silverstone", "Germany", "Hungary",
           "Belgium", "Italy", "Singapore", "Russia", "Japan", "Mexico",
           "United States", "Brazil", "Abu Dhabi"],
    2020: ["Austria", "Styria", "Hungary", "Silverstone", "70th Anniversary",
           "Spain", "Belgium", "Italy", "Mugello", "Russia", "Eifel", "Portugal",
           "Imola", "Turkey", "Bahrain", "Sakhir", "Abu Dhabi"],
    2021: ["Bahrain", "Emilia Romagna", "Portugal", "Spain", "Monaco",
           "Azerbaijan", "France", "Styria", "Austria", "Silverstone", "Hungary",
           "Belgium", "Netherlands", "Italy", "Russia", "Turkey", "United States",
           "Mexico", "Brazil", "Qatar", "Saudi Arabia", "Abu Dhabi"],
    2022: ["Bahrain", "Saudi Arabia", "Australia", "Emilia Romagna", "Miami",
           "Spain", "Monaco", "Azerbaijan", "Canada", "Silverstone", "Austria",
           "France", "Hungary", "Belgium", "Netherlands", "Italy", "Singapore",
           "Japan", "United States", "Mexico", "Brazil", "Abu Dhabi"],
    2023: ["Bahrain", "Saudi Arabia", "Australia", "Azerbaijan", "Miami",
           "Monaco", "Spain", "Canada", "Austria", "Silverstone", "Hungary",
           "Belgium", "Netherlands", "Italy", "Singapore", "Japan",
           "Qatar", "United States", "Mexico", "Brazil", "Las Vegas", "Abu Dhabi"],
    2024: ["Bahrain", "Saudi Arabia", "Australia", "Japan", "China", "Miami",
           "Emilia Romagna", "Monaco", "Canada", "Spain", "Austria", "Silverstone",
           "Hungary", "Belgium", "Netherlands", "Italy", "Azerbaijan", "Singapore",
           "United States", "Mexico", "Brazil", "Las Vegas", "Qatar", "Abu Dhabi"],
    2025: ["Australia", "China", "Japan", "Bahrain", "Saudi Arabia", "Miami",
           "Emilia Romagna", "Monaco", "Spain", "Canada", "Austria", "Silverstone",
           "Belgium", "Hungary", "Netherlands", "Italy", "Azerbaijan", "Singapore",
           "United States", "Mexico", "Brazil", "Las Vegas", "Qatar", "Abu Dhabi"],
}


# ---------------------------------------------------------------------------
# Helpers shared with import_f1_dataset
# ---------------------------------------------------------------------------

def normalize_compound(compound_raw) -> str | None:
    """Normalize a raw FastF1 compound string to a DB ENUM value.

    Returns None (never a fabricated guess) when the raw value is missing
    or not in the map; the caller decides how to record the lap.
    """
    if not pd.notna(compound_raw):
        return None
    return TYRE_COMPOUND_MAP.get(str(compound_raw).strip().upper())


def fetch_drivers_from_db() -> list[dict]:
    """Return all rows from drivers table as a list of dicts, sorted by driver_id."""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT driver_id, driver_code, driver_name FROM drivers ORDER BY driver_id")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows


def print_driver_menu(drivers: list[dict]) -> None:
    """Pretty-print the driver list."""
    print("\n  DRIVERS (enter the F1 number shown on the left):")
    print("  " + "-" * 40)
    for d in drivers:
        print(f"  {d['driver_id']:>3}  -  {d['driver_name']:<22} ({d['driver_code']})")
    print()


def print_track_menu(year: int) -> list[str]:
    """Pretty-print the race calendar for a year and return the list."""
    calendar = RACE_CALENDAR.get(year)
    if not calendar:
        print(f"  (No default calendar for {year}; type a race name manually.)")
        return []
    print(f"\n  RACES for {year} (enter the number shown):")
    print("  " + "-" * 40)
    for i, name in enumerate(calendar, 1):
        print(f"  {i:>3}  -  {name}")
    print()
    return calendar


def resolve_driver_input(raw_input: str, db_drivers: list[dict]) -> dict:
    """
    Resolve a user-supplied string (F1 number or 3-letter code) to a drivers row.

    Raises ValueError with a clear message if the driver cannot be found.
    Never falls back silently.
    """
    token = raw_input.strip()

    # Try numeric — match driver_id
    if token.isdigit():
        number = int(token)
        for d in db_drivers:
            if d["driver_id"] == number:
                return d
        raise ValueError(
            f"Driver number {number} not found in the drivers table. "
            f"Run the importer for a race that includes this driver first, "
            f"or check your input."
        )

    # Try 3-letter code (case-insensitive)
    code_upper = token.upper()
    for d in db_drivers:
        if d["driver_code"] == code_upper:
            return d
    raise ValueError(
        f"Driver code '{token}' not found in the drivers table. "
        f"Valid codes: {', '.join(d['driver_code'] for d in db_drivers)}"
    )


def resolve_race_input(raw_input: str, calendar: list[str]) -> str:
    """
    Resolve a user-supplied track input (number from menu or free-text name).
    Returns the race name string to pass to FastF1.
    Raises ValueError on bad numeric input.
    """
    token = raw_input.strip()
    if token.isdigit() and calendar:
        idx = int(token) - 1
        if 0 <= idx < len(calendar):
            return calendar[idx]
        raise ValueError(
            f"Track number {token} is out of range (1–{len(calendar)})."
        )
    # Free-text: pass directly to FastF1 as-is
    return token


# ---------------------------------------------------------------------------
# Reference-table helpers
# ---------------------------------------------------------------------------

def get_or_create_source(cursor, source_name="FastF1", source_type="Real World") -> int:
    version = getattr(fastf1, "__version__", "unknown") if fastf1 is not None else "unknown"
    cursor.execute("SELECT source_id FROM data_sources WHERE source_name = %s", (source_name,))
    row = cursor.fetchone()
    if row:
        return row[0]
    cursor.execute(
        "INSERT INTO data_sources (source_name, source_type, version) VALUES (%s, %s, %s)",
        (source_name, source_type, version),
    )
    return cursor.lastrowid


def get_or_create_regulation(cursor, year: int) -> int:
    cursor.execute(
        "SELECT regulation_id FROM regulations WHERE year_start <= %s AND year_end >= %s LIMIT 1",
        (year, year),
    )
    row = cursor.fetchone()
    if row:
        return row[0]
    if 2014 <= year <= 2021:
        name, start, end = "Turbo Hybrid Era", 2014, 2021
    elif 2022 <= year <= 2025:
        name, start, end = "Ground Effect Era", 2022, 2025
    elif year >= 2026:
        name, start, end = "2026 Power Unit Era", 2026, 2030
    else:
        name, start, end = f"{year} Regulations", year, year
    cursor.execute(
        "INSERT INTO regulations (name, year_start, year_end) VALUES (%s, %s, %s)",
        (name, start, end),
    )
    return cursor.lastrowid


def get_or_create_season(cursor, year: int, regulation_id: int) -> int:
    cursor.execute("SELECT season_id FROM seasons WHERE year = %s", (year,))
    row = cursor.fetchone()
    if row:
        return row[0]
    cursor.execute(
        "INSERT INTO seasons (year, regulation_id) VALUES (%s, %s)",
        (year, regulation_id),
    )
    return cursor.lastrowid


def get_or_create_track(cursor, location_name: str, country: str = None) -> tuple[int, str]:
    canonical = location_name.strip().title()
    cursor.execute(
        """
        SELECT t.track_id, t.canonical_name
        FROM tracks t
        LEFT JOIN track_aliases ta ON t.track_id = ta.track_id
        WHERE LOWER(t.canonical_name) = %s OR LOWER(ta.alias) = %s
        LIMIT 1
        """,
        (canonical.lower(), canonical.lower()),
    )
    row = cursor.fetchone()
    if row:
        return row[0], row[1]
    cursor.execute(
        "INSERT INTO tracks (canonical_name, country, short_code) VALUES (%s, %s, %s)",
        (canonical, country or canonical, canonical[:3].upper()),
    )
    track_id = cursor.lastrowid
    cursor.execute(
        "INSERT INTO track_aliases (track_id, alias) VALUES (%s, %s)",
        (track_id, canonical),
    )
    return track_id, canonical


def upsert_driver_from_fastf1(cursor, driver_id: int, driver_code: str, driver_name: str) -> None:
    """
    Ensure the driver row exists in the DB.
    driver_id  = FastF1 DriverNumber (the real F1 race number).
    driver_code = 3-letter abbreviation.
    driver_name = best available name string.
    Does NOT overwrite an existing row's name (user may have corrected it).
    """
    cursor.execute("SELECT driver_id FROM drivers WHERE driver_id = %s", (driver_id,))
    if not cursor.fetchone():
        cursor.execute(
            "INSERT INTO drivers (driver_id, driver_code, driver_name) VALUES (%s, %s, %s)",
            (driver_id, driver_code, driver_name),
        )


def check_existing_session(cursor, track_id: int, season_id: int, driver_id: int,
                            session_type: str, event_date) -> int | None:
    """
    A session is unique per (track, season, driver, session_type, date).
    Including driver_id prevents collisions when two drivers at the same
    race are imported separately.
    """
    cursor.execute(
        """
        SELECT session_id FROM sessions
        WHERE track_id = %s AND season_id = %s AND driver_id = %s
          AND session_type = %s AND date = %s
        """,
        (track_id, season_id, driver_id, session_type, event_date),
    )
    row = cursor.fetchone()
    return row[0] if row else None


def is_live_import(event_date, today=None) -> bool:
    """
    True when a FastF1 import is "live" data: the race ran the same day it
    is being imported.  Same-day races are stamped captured_at=NOW() so the
    dashboard's top cards treat them like game/live-feed telemetry; anything
    older stays NULL and never qualifies as live.

    ``today`` is injectable for deterministic tests.
    """
    if today is None:
        today = date.today()
    return bool(event_date) and event_date == today


# ---------------------------------------------------------------------------
# Weather classification
# ---------------------------------------------------------------------------

def classify_weather(session) -> str:
    """
    Derive a meaningful weather label from FastF1 session.weather_data.

    FastF1 provides a weather_data DataFrame with (at minimum) columns:
      Rainfall (bool/int)  - 1/True when it is raining
      TrackTemp (float)    - track surface temperature in °C
      AirTemp   (float)    - ambient air temperature in °C

    Classification rules
    --------------------
    Wet   : Rainfall is True/1 for >= 50% of weather samples.
    Mixed : Rainfall present in 10–49% of samples (e.g. shower + dry spells).
    Dry   : Rainfall absent in > 90% of samples.

    Falls back to "Dry" if weather_data is unavailable or empty.
    """
    try:
        wd = getattr(session, "weather_data", None)
        if wd is None or wd.empty:
            return "Dry"
        if "Rainfall" not in wd.columns:
            return "Dry"

        # Rainfall column is bool or 0/1 int
        rain_fraction = wd["Rainfall"].astype(bool).mean()  # 0.0 – 1.0

        if rain_fraction >= 0.50:
            return "Wet"
        if rain_fraction >= 0.10:
            return "Mixed"
        return "Dry"
    except Exception as exc:
        logging.warning(f"Weather classification failed ({exc}); defaulting to 'Dry'")
        return "Dry"


# ---------------------------------------------------------------------------
# Race-control event extraction (Safety Car / VSC / Red Flag)
# ---------------------------------------------------------------------------

# FastF1 >= 3.2 track_status Status codes come from the live-timing
# TrackStatus feed:
#   1 = AllClear, 2 = Yellow, 4 = SCDeployed, 5 = Red,
#   6 = VSCDeployed, 7 = VSCEnding
# (Older FastF1 releases used 3 = SC, 4 = Red, 5 = VSC, 6 = VSC-ending.)
# The human-readable Message text is stable across versions, so it is the
# preferred signal; numeric codes are used only as a fallback.

# Generic end keywords close whatever race-control period is currently open.
_GENERIC_RC_END_KEYWORDS = ("TRACK CLEAR", "GREEN FLAG", "RACE RESUMED", "ALL CLEAR")

# Text-feed fallback patterns: deploy and end keywords per event type.
# Note: "SAFETY CAR IN THIS LAP" is deliberately a deploy/keep-alive keyword
# ONLY — a repeated deployment message must NOT close a still-running period.
_RC_MESSAGE_PATTERNS = {
    "SafetyCar": {
        "deploy": ("SAFETY CAR DEPLOYED", "SAFETY CAR IN THIS LAP",
                    "ALL CARS MAY OVERTAKE THE SAFETY CAR"),
        "end":    ("SAFETY CAR WITHDRAWN",),
    },
    "VSC": {
        # "VIRTUAL SAFETY CAR ..." is normalised to "VSC ..." before
        # matching (see _events_from_messages) so it can never collide
        # with the SafetyCar keywords as a substring.
        "deploy": ("VSC DEPLOYED",),
        "end":    ("VSC ENDING",),
    },
    "RedFlag": {
        "deploy": ("RED FLAG", "SESSION SUSPENDED"),
        "end":    ("SESSION RESTARTED", "SESSION RESUMED"),
    },
}


def _status_to_event_type(status_value, message_text) -> str | None:
    """Map one track_status row to a race-control event type (or None).

    None means the status is an end/neutral marker (AllClear, Yellow,
    VSCEnding) or is unknown.
    """
    msg = str(message_text).strip().lower() if message_text is not None else ""
    if msg:
        if msg == "scdeployed":
            return "SafetyCar"
        if msg == "red":
            return "RedFlag"
        if msg == "vscdeployed":
            return "VSC"
        return None
    # No message text — fall back to numeric codes (current FastF1 scheme,
    # plus the legacy SafetyCar code 3 for older releases).
    try:
        code = int(status_value)
    except (TypeError, ValueError):
        return None
    if code in (3, 4):
        return "SafetyCar"
    if code == 5:
        return "RedFlag"
    if code == 6:
        return "VSC"
    return None


def _delta_seconds(end, start) -> float | None:
    """Seconds between two timestamps; None when unavailable or non-positive."""
    if end is None or start is None:
        return None
    try:
        delta = (end - start).total_seconds()
        return round(delta, 1) if delta > 0 else None
    except Exception:
        return None


def _contains_any(text: str, keywords) -> bool:
    """Word-boundary substring check.

    Plain substring matching is unsafe on free text: e.g. "RED FLAG" is a
    substring of "CHEQUERED FLAG".  Word boundaries prevent such false
    positives.
    """
    return any(re.search(r"\b" + re.escape(kw) + r"\b", text) for kw in keywords)


def _events_from_track_status(track_status) -> list[tuple[str, float | None]]:
    """Convert the structured track_status DataFrame into (event_type, duration_sec).

    Each row is a status change.  A deploy row opens a period; an end row
    (AllClear / Yellow / VSCEnding) closes any open period; a different-type
    deploy while one is open closes the previous period first.  Repeated
    deploys of the same type are keep-alives and are ignored.
    """
    events: list[tuple[str, float | None]] = []
    open_type: str | None = None
    open_time = None

    for _, row in track_status.iterrows():
        ts = row.get("Time")
        ev_type = _status_to_event_type(row.get("Status"), row.get("Message"))

        if ev_type is None:
            # End / neutral marker — close any open period.
            if open_type is not None:
                events.append((open_type, _delta_seconds(ts, open_time)))
                open_type, open_time = None, None
        elif ev_type == open_type:
            # Repeated deploy of the same type — keep-alive, ignore.
            continue
        else:
            # New period — close a differently-typed open period first.
            if open_type is not None:
                events.append((open_type, _delta_seconds(ts, open_time)))
            open_type, open_time = ev_type, ts

    if open_type is not None:
        # Period still open at the end of the session (no closing status).
        events.append((open_type, None))
    return events


def _events_from_messages(rcm) -> list[tuple[str, float | None]]:
    """Fallback: derive (event_type, duration_sec) from the text race-control feed.

    A single pending period is tracked at a time.  Generic end keywords close
    whatever is pending; type-specific end keywords close only their type;
    repeated deploy keywords of the open type are keep-alives.
    """
    events: list[tuple[str, float | None]] = []
    time_col = "Time" if "Time" in rcm.columns else ("Date" if "Date" in rcm.columns else None)
    open_type: str | None = None
    open_time = None

    msgs = rcm.copy()
    msgs["_upper"] = (
        msgs["Message"].astype(str).str.upper().str.strip()
        .str.replace("VIRTUAL SAFETY CAR", "VSC")
    )

    for _, row in msgs.iterrows():
        text = row["_upper"]
        ts = row.get(time_col) if time_col else None

        if open_type is None:
            for ev_type, pat in _RC_MESSAGE_PATTERNS.items():
                if _contains_any(text, pat["deploy"]):
                    open_type, open_time = ev_type, ts
                    break
            continue

        if _contains_any(text, _GENERIC_RC_END_KEYWORDS):
            events.append((open_type, _delta_seconds(ts, open_time)))
            open_type, open_time = None, None
        elif _contains_any(text, _RC_MESSAGE_PATTERNS[open_type]["end"]):
            events.append((open_type, _delta_seconds(ts, open_time)))
            open_type, open_time = None, None
        elif _contains_any(text, _RC_MESSAGE_PATTERNS[open_type]["deploy"]):
            # Repeated deploy of the open type — keep-alive, ignore.
            continue
        else:
            # A deploy of a different type — close the current period and
            # open the new one.
            for ev_type, pat in _RC_MESSAGE_PATTERNS.items():
                if ev_type != open_type and _contains_any(text, pat["deploy"]):
                    events.append((open_type, _delta_seconds(ts, open_time)))
                    open_type, open_time = ev_type, ts
                    break

    if open_type is not None:
        events.append((open_type, None))
    return events


def extract_race_control_events(session) -> list[tuple[str, float | None]]:
    """Extract (event_type, duration_sec) periods for SafetyCar / VSC / RedFlag.

    Primary source: the structured ``session.track_status`` feed.  Fallback:
    the text ``session.race_control_messages`` feed.  Returns an empty list
    when neither source is available.  Duration is None when a period has no
    closing status/message.
    """
    try:
        ts = getattr(session, "track_status", None)
        if (ts is not None and hasattr(ts, "size") and ts.size > 0
                and "Status" in getattr(ts, "columns", ())):
            # The structured feed is authoritative — even an eventless
            # session (green/yellow only) is a valid "no events" answer, so
            # do NOT fall through to the noisier text feed.
            return _events_from_track_status(ts)
    except Exception as exc:
        logging.warning(f"track_status extraction failed ({exc}); falling back to messages")

    try:
        rcm = getattr(session, "race_control_messages", None)
        if (rcm is not None and hasattr(rcm, "empty") and not rcm.empty
                and "Message" in getattr(rcm, "columns", ())):
            return _events_from_messages(rcm)
    except Exception as exc:
        logging.warning(f"race_control_messages extraction failed (non-fatal): {exc}")
    return []


# ---------------------------------------------------------------------------
# Core importer
# ---------------------------------------------------------------------------

def import_race(year: int, race_name: str, driver_id: int,
                session_type: str = "R", allow_duplicate: bool = False) -> int | None:
    """
    Import laps + telemetry for ONE driver from a FastF1 session into MySQL.

    Parameters
    ----------
    year          : Championship year (e.g. 2024).
    race_name     : FastF1 race name or partial match (e.g. "Bahrain", "Monaco").
    driver_id     : Real F1 driver number (= drivers.driver_id in the DB).
    session_type  : "R" (Race), "Q" (Qualifying), "FP1", "FP2", "FP3".
    allow_duplicate : If False (default) skip if session already in DB.

    Returns the new (or existing) session_id, or None on failure.
    """
    mapped_type = {"R": "Race", "Q": "Qualifying"}.get(session_type.upper(), "Practice")

    # ------------------------------------------------------------------
    # 1. Load FastF1 session
    # ------------------------------------------------------------------
    logging.info("=" * 60)
    logging.info(f"F1 IMPORT  |  {year} {race_name}  |  {session_type}  |  driver #{driver_id}")
    logging.info("=" * 60)

    if fastf1 is None:
        logging.error(
            "fastf1 is not installed — run `pip install fastf1` to import "
            "historical race data."
        )
        return None

    try:
        session = fastf1.get_session(year, race_name, session_type)
        session.load(telemetry=True, laps=True, weather=True, messages=True)
    except Exception as exc:
        logging.error(f"FastF1 could not load session: {exc}")
        return None

    # ------------------------------------------------------------------
    # 2. Identify the driver inside this FastF1 session
    # ------------------------------------------------------------------
    # FastF1 DriverNumber column is the real F1 number — same as driver_id.
    session_laps = session.laps
    driver_laps = session_laps[session_laps["DriverNumber"].astype(str) == str(driver_id)]

    if driver_laps.empty:
        logging.error(
            f"Driver #{driver_id} not found in the FastF1 session "
            f"(drivers present: {sorted(session_laps['DriverNumber'].dropna().unique().tolist())}). "
            f"Aborting."
        )
        return None

    # Resolve 3-letter code and best available name from FastF1
    fastf1_code = str(driver_laps.iloc[0]["Driver"]).strip().upper()[:3]
    fastf1_number = int(driver_laps.iloc[0]["DriverNumber"])
    # FastF1 lap rows don't carry full names; use code as placeholder unless
    # the drivers table already has a better name.
    fastf1_name = fastf1_code

    logging.info(f"Resolved driver: #{fastf1_number} {fastf1_name} ({fastf1_code})")

    # ------------------------------------------------------------------
    # 3. Database operations (single transaction)
    # ------------------------------------------------------------------
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        source_id     = get_or_create_source(cursor)
        regulation_id = get_or_create_regulation(cursor, year)
        season_id     = get_or_create_season(cursor, year, regulation_id)

        location = session.event.get("Location", race_name)
        country  = session.event.get("Country", location)
        track_id, track_name = get_or_create_track(cursor, location, country)

        event_date = session.event["EventDate"].date()
        # A race imported on the day it actually ran is live data: stamp
        # captured_at so the dashboard's top cards treat it as live, just
        # like game UDP capture.  Historical imports keep captured_at NULL.
        live_import = is_live_import(event_date)
        if live_import:
            logging.info(
                f"Same-day race ({event_date}) — laps will be stamped "
                f"captured_at=NOW() and count as live data"
            )

        # Ensure driver row exists (won't overwrite existing name)
        upsert_driver_from_fastf1(cursor, fastf1_number, fastf1_code, fastf1_name)

        # Fetch the final driver_name from DB (may be richer than FastF1 code)
        cursor.execute(
            "SELECT driver_name FROM drivers WHERE driver_id = %s", (driver_id,)
        )
        db_driver_name = cursor.fetchone()[0] or fastf1_code

        # Classify weather from FastF1 weather_data (Dry / Wet / Mixed)
        weather_label = classify_weather(session)
        logging.info(f"Weather classification: {weather_label}")

        # Duplicate check (per driver per race)
        existing_id = check_existing_session(
            cursor, track_id, season_id, driver_id, mapped_type, event_date
        )
        if existing_id and not allow_duplicate:
            logging.info(
                f"Session already exists (ID={existing_id}) for driver #{driver_id} "
                f"at {track_name}. Use --allow-duplicate to re-import."
            )
            return existing_id

        # ------------------------------------------------------------------
        # 4. Create session row
        # ------------------------------------------------------------------
        cursor.execute(
            """
            INSERT INTO sessions
              (track_name, session_type, weather, date,
               season_id, source_id, track_id, regulation_id, driver_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (track_name, mapped_type, weather_label, event_date,
             season_id, source_id, track_id, regulation_id, driver_id),
        )
        session_id = cursor.lastrowid
        logging.info(
            f"Session created: ID={session_id}  "
            f"Driver={driver_id} - {db_driver_name} ({fastf1_code})  "
            f"Track={track_name}  Date={event_date}  Weather={weather_label}"
        )

        # ------------------------------------------------------------------
        # 5. Import laps + pit stops
        # ------------------------------------------------------------------
        lap_count       = 0
        telem_count     = 0
        telem_failures  = 0

        # Build a map of lap_number -> lap_id once inserted, so the pit-stop
        # block on the in-lap can look up the *next* lap's PitOutTime.
        # We also keep a simple ordered list of (lap_number, PitInTime, PitOutTime_this_row)
        # rows so we can do a forward-look after the loop.
        lap_id_by_number: dict[int, int] = {}
        pit_in_rows: list[tuple] = []  # (lap_number, lap_id, PitInTime, PitOutTime_same_row)

        for _, lap in driver_laps.iterrows():
            try:
                lap_num = int(lap["LapNumber"])
                if not pd.notna(lap["LapTime"]):
                    continue
                # The official F1 timing feed reports lap times at 1 ms
                # granularity, and lap times cluster tightly, so two distinct
                # laps can legitimately share an identical ms value.  That is
                # source data, not a duplication bug -- keep the faithful copy.
                lap_time_ms = int(lap["LapTime"].total_seconds() * 1000)

                compound  = normalize_compound(lap["Compound"])
                if compound is None:
                    logging.warning(
                        f"Lap {lap_num}: unknown tyre compound "
                        f"{lap.get('Compound')!r} - importing with NULL compound"
                    )
                tyre_age  = int(lap["TyreLife"]) if pd.notna(lap["TyreLife"]) and int(lap["TyreLife"]) > 0 else 1
                is_valid  = 0 if ("Deleted" in lap and pd.notna(lap["Deleted"]) and bool(lap["Deleted"])) else 1
                fuel_load = estimate_fuel_load(lap_num)

                cursor.execute(
                    """
                    INSERT INTO laps
                      (session_id, driver_id, lap_number, lap_time_ms,
                       tyre_compound, tyre_age, fuel_load, is_valid, captured_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (session_id, driver_id, lap_num, lap_time_ms,
                     compound, tyre_age, fuel_load, is_valid,
                     datetime.now() if live_import else None),
                )
                lap_id = cursor.lastrowid
                lap_count += 1
                lap_id_by_number[lap_num] = lap_id

                # Collect pit-in rows for post-loop processing
                pit_in_time = lap.get("PitInTime")
                if pd.notna(pit_in_time):
                    pit_out_same = lap.get("PitOutTime")  # may or may not exist on same row
                    pit_in_rows.append((lap_num, lap_id, pit_in_time, pit_out_same))

                # Telemetry — log failures, do not silently swallow them
                try:
                    telem = lap.get_telemetry()
                    if telem is not None and not telem.empty:
                        step    = max(1, len(telem) // 5)
                        sampled = telem.iloc[::step]
                        for _, row in sampled.iterrows():
                            speed    = int(row["Speed"])    if pd.notna(row["Speed"])    else 0
                            throttle = round(float(row["Throttle"]) / 100.0, 2) if pd.notna(row["Throttle"]) else 0.0
                            brake    = round(float(row["Brake"])    / 100.0, 2) if pd.notna(row["Brake"])    else 0.0
                            gear     = int(row["nGear"])    if pd.notna(row["nGear"])    else 0
                            rpm      = int(row["RPM"])      if pd.notna(row["RPM"])      else 0
                            drs      = 1 if (pd.notna(row["DRS"]) and int(row["DRS"]) in (10, 12, 14)) else 0
                            cursor.execute(
                                "INSERT INTO telemetry (lap_id, speed, throttle, brake, gear, rpm, drs) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                                (lap_id, speed, throttle, brake, gear, rpm, drs),
                            )
                            telem_count += 1
                except Exception as telem_err:
                    telem_failures += 1
                    logging.warning(f"Telemetry failed on lap {lap_num}: {telem_err}")

                if lap_count % 20 == 0:
                    logging.info(f"  ... {lap_count} laps, {telem_count} telem samples so far")

            except Exception as lap_err:
                logging.error(f"Error on lap row: {lap_err}")

        # ------------------------------------------------------------------
        # 5b. Pit-stop strategy events
        #
        # Duration = PitOutTime(out-lap) - PitInTime(in-lap).
        # The out-lap is the lap AFTER the in-lap; FastF1 sets PitOutTime on
        # the out-lap row, not the in-lap row.  We look forward one lap.
        # If the next-lap PitOutTime is unavailable we use PitOutTime on the
        # same row as a fallback.  If neither is valid (implausible or absent)
        # the event is still recorded -- the stop happened -- but with a NULL
        # duration instead of a meaningless estimate.
        # ------------------------------------------------------------------
        pit_stop_count = 0

        # Build a lookup of lap_number -> PitOutTime from the driver_laps frame
        pit_out_by_lapnum: dict[int, object] = {}
        for _, lap in driver_laps.iterrows():
            if pd.notna(lap.get("PitOutTime")):
                pit_out_by_lapnum[int(lap["LapNumber"])] = lap["PitOutTime"]

        for lap_num, lap_id, pit_in_time, pit_out_same in pit_in_rows:
            duration_sec: float | None = None

            # Primary: next-lap PitOutTime (the real box stop window)
            next_lap_out = pit_out_by_lapnum.get(lap_num + 1)
            if next_lap_out is not None:
                try:
                    delta = (next_lap_out - pit_in_time).total_seconds()
                    if 2.0 <= delta <= 120.0:
                        duration_sec = round(delta, 2)
                except Exception:
                    pass

            # Fallback: PitOutTime on the same lap row (rare but possible)
            if duration_sec is None and pd.notna(pit_out_same):
                try:
                    delta = (pit_out_same - pit_in_time).total_seconds()
                    if 2.0 <= delta <= 120.0:
                        duration_sec = round(delta, 2)
                except Exception:
                    pass

            if duration_sec is None:
                logging.info(
                    f"Lap {lap_num}: PitInTime set but no reliable PitOutTime found "
                    f"- recording PitStop event without a duration "
                    f"(DNF/final-lap/data gap)."
                )
            else:
                logging.info(f"  PitStop on lap {lap_num}: {duration_sec:.2f}s (lap_id={lap_id})")

            cursor.execute(
                "INSERT INTO strategy_events (lap_id, event_type, duration_sec) VALUES (%s, 'PitStop', %s)",
                (lap_id, duration_sec),
            )
            pit_stop_count += 1

        # ------------------------------------------------------------------
        # 5c. Race-control events: SafetyCar, VSC, RedFlag
        #
        # Session-wide events are stored with lap_id = NULL because they
        # affect every driver — pinning them to one driver's lap would be
        # misleading.
        #
        # Derived primarily from the structured session.track_status feed
        # (FastF1 Status codes: 4=SCDeployed, 5=Red, 6=VSCDeployed,
        # 7=VSCEnding, 1=AllClear, 2=Yellow), with the text
        # race_control_messages feed as a fallback.
        # ------------------------------------------------------------------
        rc_event_count = 0

        try:
            rc_events = extract_race_control_events(session)
            for event_type, duration_sec in rc_events:
                cursor.execute(
                    "INSERT INTO strategy_events (lap_id, event_type, duration_sec) VALUES (NULL, %s, %s)",
                    (event_type, duration_sec),
                )
                rc_event_count += 1
                dur_str = f"{duration_sec:.1f}s" if duration_sec is not None else "unknown"
                logging.info(f"  {event_type} event: duration={dur_str}")
            if not rc_events:
                logging.info("  No race-control events found for this session.")
        except Exception as rc_err:
            logging.warning(f"Race-control event processing failed (non-fatal): {rc_err}")

        conn.commit()

        # ------------------------------------------------------------------
        # 6. Final report
        # ------------------------------------------------------------------
        print("\n" + "=" * 60)
        print("IMPORT COMPLETE")
        print("=" * 60)
        print(f"  Driver          : #{driver_id} - {db_driver_name} ({fastf1_code})")
        print(f"  Session ID      : {session_id}")
        print(f"  Track           : {track_name}  ({year})")
        print(f"  Laps            : {lap_count}")
        print(f"  Pit stops       : {pit_stop_count}")
        print(f"  SC/VSC/RF events: {rc_event_count}")
        print(f"  Telemetry       : {telem_count} samples")
        print(f"  Telem fails     : {telem_failures}")
        print("=" * 60 + "\n")

        return session_id

    except Exception as fatal:
        conn.rollback()
        logging.error(f"Fatal error — transaction rolled back: {fatal}")
        return None
    finally:
        cursor.close()
        conn.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _interactive_prompt() -> dict:
    """
    Gather all import parameters interactively, showing the DB driver list
    and a numbered race calendar.  Never silently falls back.
    """
    print("=" * 60)
    print("F1 RACE DATA IMPORTER — INTERACTIVE MODE")
    print("=" * 60)

    # Year
    year_raw = input("\nEnter championship year [default: 2024]: ").strip()
    year = int(year_raw) if year_raw.isdigit() else 2024

    # Track from calendar
    calendar = print_track_menu(year)
    race_raw = input("Enter race number from the list above, or type a race name: ").strip()
    if not race_raw:
        print("ERROR: A race name or number is required.")
        sys.exit(1)
    try:
        race = resolve_race_input(race_raw, calendar)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    # Session type
    session_raw = input("\nSession type  R=Race  Q=Qualifying  FP1/FP2/FP3  [default: R]: ").strip()
    session_type = session_raw if session_raw else "R"

    # Driver from DB
    db_drivers = fetch_drivers_from_db()
    if not db_drivers:
        print(
            "ERROR: The drivers table is empty. Import at least one session first "
            "(the importer auto-populates drivers from FastF1 DriverNumber data)."
        )
        sys.exit(1)
    print_driver_menu(db_drivers)
    driver_raw = input("Enter the F1 number (left column) or 3-letter code: ").strip()
    if not driver_raw:
        print("ERROR: A driver selection is required.")
        sys.exit(1)
    try:
        db_driver = resolve_driver_input(driver_raw, db_drivers)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print(
        f"\n  Selected: #{db_driver['driver_id']} - "
        f"{db_driver['driver_name']} ({db_driver['driver_code']})"
    )

    dup_raw = input("Allow duplicate import if session already exists? (y/N): ").strip().lower()
    allow_duplicate = dup_raw in ("y", "yes")

    return dict(
        year=year,
        race_name=race,
        session_type=session_type,
        driver_id=db_driver["driver_id"],
        allow_duplicate=allow_duplicate,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 Race Data Importer (single driver)")
    parser.add_argument("--year",       type=int,  default=None, help="Championship year (e.g. 2024)")
    parser.add_argument("--race",       type=str,  default=None, help="Race name (e.g. Bahrain, Monaco)")
    parser.add_argument("--session",    type=str,  default=None, help="Session type: R, Q, FP1, FP2, FP3")
    parser.add_argument("--driver",     type=str,  default=None, help="Driver F1 number (e.g. 16) or code (e.g. LEC)")
    parser.add_argument("--allow-duplicate", action="store_true", help="Re-import even if session exists")
    parser.add_argument("--interactive", "-i", action="store_true", help="Force interactive prompt")
    args = parser.parse_args()

    use_interactive = len(sys.argv) == 1 or args.interactive or args.year is None or args.driver is None

    if use_interactive:
        params = _interactive_prompt()
    else:
        # CLI mode — resolve driver through the DB (no silent fallback)
        db_drivers = fetch_drivers_from_db()
        if not db_drivers:
            print("ERROR: drivers table is empty. Run interactively on a session first.")
            sys.exit(1)
        try:
            db_driver = resolve_driver_input(args.driver, db_drivers)
        except ValueError as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)
        params = dict(
            year=args.year,
            race_name=args.race,
            session_type=args.session or "R",
            driver_id=db_driver["driver_id"],
            allow_duplicate=args.allow_duplicate,
        )

    import_race(**params)
