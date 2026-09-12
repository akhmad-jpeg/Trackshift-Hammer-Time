"""
Batch ingestion utility.

Imports one driver across multiple races/seasons by calling import_race()
from import_f1_race.  All driver and track selection goes through the same
DB-backed helpers used by the single-race importer, ensuring identical
resolution behaviour in both scripts.
"""

import argparse
import sys
import logging
from pathlib import Path

# Ensure scripts directory is on sys.path for sibling imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from import_f1_race import (
    import_race,
    fetch_drivers_from_db,
    resolve_driver_input,
    resolve_race_input,
    print_driver_menu,
    print_track_menu,
    RACE_CALENDAR,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(message)s")


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def import_dataset(
    seasons: list[int],
    races: list[str] | None,
    driver_id: int,
    session_type: str = "R",
    allow_duplicate: bool = False,
) -> None:
    """
    Import one driver across multiple seasons and races.

    Parameters
    ----------
    seasons        : List of championship years, e.g. [2023, 2024].
    races          : Explicit race-name list, or None to use RACE_CALENDAR.
    driver_id      : Real F1 driver number (= drivers.driver_id in the DB).
    session_type   : "R", "Q", "FP1", "FP2", or "FP3".
    allow_duplicate: If False, skip sessions already in the DB.
    """
    logging.info("=" * 60)
    logging.info("BATCH F1 DATASET INGESTION")
    logging.info("=" * 60)

    successful = 0
    failed     = 0
    skipped    = 0

    for year in seasons:
        race_list = races if races else RACE_CALENDAR.get(year, ["Bahrain", "Monaco"])
        logging.info(f"\n--- Season {year}: {len(race_list)} race(s), driver #{driver_id} ---")

        for race_name in race_list:
            try:
                session_id = import_race(
                    year=year,
                    race_name=race_name,
                    driver_id=driver_id,
                    session_type=session_type,
                    allow_duplicate=allow_duplicate,
                )
                if session_id is not None:
                    successful += 1
                else:
                    failed += 1
            except Exception as exc:
                logging.error(f"  Batch error for {year} {race_name}: {exc}")
                failed += 1

    print("\n" + "=" * 60)
    print("BATCH COMPLETE")
    print("=" * 60)
    print(f"  Successful / already-existed : {successful}")
    print(f"  Failed                       : {failed}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Interactive prompt (identical resolution logic to import_f1_race)
# ---------------------------------------------------------------------------

def _interactive_prompt() -> dict:
    print("=" * 60)
    print("BATCH F1 DATASET INGESTION — INTERACTIVE MODE")
    print("=" * 60)

    # Driver (same DB lookup as single-race importer)
    db_drivers = fetch_drivers_from_db()
    if not db_drivers:
        print(
            "ERROR: The drivers table is empty. Run import_f1_race.py for at least one "
            "session first so FastF1 can populate the drivers table."
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

    # Years
    years_raw = input("\nEnter year(s) separated by spaces [default: 2024]: ").strip()
    if years_raw:
        seasons = [int(y) for y in years_raw.split() if y.isdigit()]
        if not seasons:
            print("ERROR: No valid years entered.")
            sys.exit(1)
    else:
        seasons = [2024]

    # Races (optional — blank → full calendar)
    races: list[str] | None = None
    use_calendar = input("\nImport the full season calendar? (Y/n) [default: Y]: ").strip().lower()
    if use_calendar in ("n", "no"):
        races = []
        for year in seasons:
            calendar = print_track_menu(year)
            while True:
                raw = input(
                    "Enter race numbers separated by spaces (or race names), "
                    "blank line to finish: "
                ).strip()
                if not raw:
                    break
                for token in raw.split():
                    try:
                        races.append(resolve_race_input(token, calendar))
                    except ValueError as exc:
                        print(f"  Warning: {exc}")
        if not races:
            print("ERROR: No valid races selected.")
            sys.exit(1)

    # Session type
    session_raw = input("\nSession type  R=Race  Q=Qualifying  FP1/FP2/FP3  [default: R]: ").strip()
    session_type = session_raw if session_raw else "R"

    # Duplicates
    dup_raw = input("Allow duplicate import if session already exists? (y/N): ").strip().lower()
    allow_duplicate = dup_raw in ("y", "yes")

    return dict(
        seasons=seasons,
        races=races,
        driver_id=db_driver["driver_id"],
        session_type=session_type,
        allow_duplicate=allow_duplicate,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch F1 Dataset Ingestion (single driver, multiple races)")
    parser.add_argument("--years",    type=int, nargs="+", default=None, help="Season years (e.g. 2023 2024)")
    parser.add_argument("--races",    type=str, nargs="+", default=None, help="Race names (e.g. Bahrain Monaco)")
    parser.add_argument("--driver",   type=str, default=None,            help="Driver F1 number or 3-letter code")
    parser.add_argument("--session-type", type=str, default=None,        help="Session type: R, Q, FP1, FP2, FP3")
    parser.add_argument("--allow-duplicate", action="store_true",        help="Re-import even if session exists")
    parser.add_argument("--interactive", "-i", action="store_true",      help="Force interactive prompt")
    args = parser.parse_args()

    use_interactive = len(sys.argv) == 1 or args.interactive or args.years is None or args.driver is None

    if use_interactive:
        params = _interactive_prompt()
    else:
        db_drivers = fetch_drivers_from_db()
        if not db_drivers:
            print("ERROR: drivers table is empty. Run import_f1_race.py interactively first.")
            sys.exit(1)
        try:
            db_driver = resolve_driver_input(args.driver, db_drivers)
        except ValueError as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)
        params = dict(
            seasons=args.years,
            races=args.races,
            driver_id=db_driver["driver_id"],
            session_type=args.session_type or "R",
            allow_duplicate=args.allow_duplicate,
        )

    import_dataset(**params)
