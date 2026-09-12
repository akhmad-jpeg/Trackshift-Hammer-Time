"""Head-to-head driver comparison using the per-driver lap-time models
trained by ml_lap_predictions.py.

How it works
------------
Every driver with enough clean race laps gets their OWN model (same
pipeline: tyre_age + one-hot tyre/track features, fitted on their own
laps).  Comparing two drivers therefore means predicting BOTH models on the
SAME (track, tyre compound, tyre age) inputs and taking the time
difference.  Because the models are additive in the one-hot features, the
intercept gap on a shared track is the driver pace gap — the whole point of
a per-driver model.

Fairness caveats (also returned in the payload)
-----------------------------------------------
* Only tracks AND tyres that BOTH drivers' models cover can be compared —
  a driver model has no feature for a track they never raced.
* Each driver model is fitted on that driver's own races, so different
  sessions, weather and track evolution are baked into each driver's
  intercept.  Same-team (same car) comparisons are the most meaningful;
  cross-team gaps mix driver AND car performance.
* Drivers with few laps produce noisy models; the trainer skips drivers
  below MIN_DRIVER_LAPS clean laps.
"""

import sys
import json
import joblib
import argparse
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from feature_pipeline import covered_tracks, covered_tyres

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DRIVER_MODELS_DIR = PROJECT_ROOT / 'ml_models' / 'drivers'
COMPARISON_DIR = PROJECT_ROOT / 'ml_models' / 'comparisons'

# Tyre ages evaluated for every shared (track, tyre) pair.
DEFAULT_AGES = (1, 5, 10, 15, 20)


def list_driver_models(models_dir=None):
    """Scan ml_models/drivers/* for trained per-driver models.

    Returns a list of dicts (code, name, laps, tracks, tyres, years, mae,
    model, tyre_age_coefficient, fuel_burn_rate) sorted by driver code.
    `years` lists the seasons with their own per-driver-per-year model
    (ml_models/drivers/<code>/<year>/), so a UI can offer same-year
    comparisons; an empty list means only the aggregate model exists.
    """
    base = Path(models_dir) if models_dir else DRIVER_MODELS_DIR
    if not base.exists():
        return []
    drivers = []
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        if not (d / 'best_model.pkl').exists() or not (d / 'feature_names.pkl').exists():
            continue
        info = {}
        info_path = d / 'model_info.json'
        if info_path.exists():
            try:
                info = json.loads(info_path.read_text(encoding='utf-8'))
            except Exception:
                info = {}
        code = str(info.get('driver', {}).get('code', d.name))
        drivers.append({
            "code": code,
            "name": str(info.get('driver', {}).get('name', d.name)),
            "laps": int(info.get('clean_laps') or info.get('training_samples') or 0),
            "tracks": list(info.get('coverage', {}).get('tracks', [])),
            "tyres": list(info.get('coverage', {}).get('tyres', [])),
            "years": list_driver_years(code, base=base),
            "mae": info.get('metrics', {}).get('within_track', {}).get('mae'),
            "model": info.get('best_model'),
            "tyre_age_coefficient": info.get('tyre_age_coefficient'),
            "fuel_burn_rate": info.get('fuel_burn_rate'),
        })
    return drivers


def list_driver_years(code, base=None):
    """Seasons with their own per-driver-per-year model for a driver code.

    Scans ml_models/drivers/<code>/<year>/ directories that contain a
    trained model.  Returns a sorted list of ints (empty = aggregate only).
    """
    base = Path(base) if base else DRIVER_MODELS_DIR
    ddir = base / code
    if not ddir.is_dir():
        return []
    years = []
    for sub in ddir.iterdir():
        if sub.is_dir() and (sub / 'best_model.pkl').exists():
            try:
                years.append(int(sub.name))
            except ValueError:
                continue
    return sorted(years)


def load_driver_model(code, year=None, models_dir=None):
    """Return (model, feature_names, info_dict, used_year) for a driver code.

    If year is provided, attempts to load ml_models/drivers/<code>/<year>/
    first (per-driver-per-year model).  Falls back to the aggregate
    per-driver model in ml_models/drivers/<code>/ when the year-specific
    model is not found — `used_year` is then None so callers can report
    the fallback.

    Raises FileNotFoundError when neither model exists.
    """
    base = Path(models_dir) if models_dir else DRIVER_MODELS_DIR

    # Try year-specific model first
    if year is not None:
        year_dir = base / code / str(year)
        if year_dir.exists() and (year_dir / 'best_model.pkl').exists():
            model = joblib.load(year_dir / 'best_model.pkl')
            feature_names = joblib.load(year_dir / 'feature_names.pkl')
            info = {}
            info_path = year_dir / 'model_info.json'
            if info_path.exists():
                try:
                    info = json.loads(info_path.read_text(encoding='utf-8'))
                except Exception:
                    info = {}
            return model, feature_names, info, int(year)

    # Fallback to aggregate per-driver model
    ddir = base / code
    if not ddir.exists():
        raise FileNotFoundError(
            f"No per-driver model for '{code}' at {ddir} — run "
            f"scripts/ml_lap_predictions.py first (it trains one model "
            f"per driver with enough laps)."
        )
    model = joblib.load(ddir / 'best_model.pkl')
    feature_names = joblib.load(ddir / 'feature_names.pkl')
    info = {}
    info_path = ddir / 'model_info.json'
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding='utf-8'))
        except Exception:
            info = {}
    return model, feature_names, info, None


def _significance(deltas):
    """Paired t-test on the per-track deltas: is the observed gap
    distinguishable from zero given the spread across tracks?

    The per-driver models each carry ~1 s of noise, so a gap that is small
    relative to the between-track spread should not be read as a definitive
    pace advantage.  Returns a dict with the test statistics and a plain
    verdict label for the UI/CLI:

      significant  - p < 0.05   (gap larger than track-to-track spread)
      suggestive   - 0.05 <= p < 0.20
      inconclusive - p >= 0.20  (within model noise - do not over-read)
      insufficient - fewer than 2 shared tracks
    """
    n = len(deltas)
    if n < 2:
        return {"n": n, "delta_mean": None, "delta_std": None,
                "t_statistic": None, "p_value": None, "label": "insufficient",
                "note": "Fewer than 2 shared tracks - no significance test possible."}

    mean = sum(deltas) / n
    var = sum((d - mean) ** 2 for d in deltas) / (n - 1)
    std = var ** 0.5

    if std == 0.0:
        # Every track agrees on the exact same gap: fully consistent.
        consistent = mean != 0.0
        return {"n": n, "delta_mean": round(mean, 4), "delta_std": 0.0,
                "t_statistic": None, "p_value": 0.0 if consistent else 1.0,
                "label": "significant" if consistent else "inconclusive",
                "note": ("Every shared track agrees on the same gap - the "
                          "result is fully consistent." if consistent else
                          "All shared tracks show an identical zero gap.")}

    t = mean / (std / n ** 0.5)
    df = n - 1
    try:
        from scipy.stats import t as t_dist
        p = 2.0 * t_dist.sf(abs(t), df)
    except Exception:
        # scipy unavailable: two-tailed normal approximation.
        from math import erf, sqrt
        p = 1.0 - erf(abs(t) / sqrt(2.0))

    if p < 0.05:
        label = "significant"
        note = ("The gap is consistent enough across tracks to be "
                "distinguished from model noise.")
    elif p < 0.20:
        label = "suggestive"
        note = ("The gap is larger than typical track-to-track spread, "
                "but not conclusively.")
    else:
        label = "inconclusive"
        note = ("The gap is smaller than the spread between tracks - treat "
                "it as within model noise.")

    return {"n": n, "delta_mean": round(mean, 4), "delta_std": round(std, 4),
            "t_statistic": round(t, 3), "p_value": round(p, 4),
            "label": label, "note": note}


def _predict(model, feature_names, track, tyre, age):
    """Predict one lap time with a driver's model on an aligned feature row."""
    row = pd.DataFrame(0, index=[0], columns=feature_names)
    if 'tyre_age' in row.columns:
        row['tyre_age'] = age
    tyre_feature = f'tyre_{tyre}'
    track_feature = f'track_{track}'
    if tyre_feature in row.columns:
        row[tyre_feature] = 1
    if track_feature in row.columns:
        row[track_feature] = 1
    return float(model.predict(row)[0])


def compare_drivers(code_a, code_b, ages=DEFAULT_AGES, models_dir=None, year=None):
    """Predict both drivers on every shared (track, tyre) at several ages.

    If year is provided, attempts to use per-driver-per-year models for
    that season (same-year = apples-to-apples comparison).  Falls back
    to aggregate per-driver models when year-specific models are not available.

    delta is defined as driver_a_time - driver_b_time, so a POSITIVE delta
    means driver B is faster (lower time) and a NEGATIVE delta means
    driver A is faster.
    """
    if code_a == code_b:
        raise ValueError("Pick two different drivers to compare.")
    model_a, feats_a, info_a, used_year_a = load_driver_model(
        code_a, year=year, models_dir=models_dir)
    model_b, feats_b, info_b, used_year_b = load_driver_model(
        code_b, year=year, models_dir=models_dir)

    tracks_a, tracks_b = covered_tracks(feats_a), covered_tracks(feats_b)
    tyres_a, tyres_b = covered_tyres(feats_a), covered_tyres(feats_b)
    shared_tracks = sorted(set(tracks_a) & set(tracks_b))
    shared_tyres = sorted(set(tyres_a) & set(tyres_b))

    per_track = []
    for track in shared_tracks:
        per_tyre = []
        for tyre in shared_tyres:
            rows = []
            for age in ages:
                ta = _predict(model_a, feats_a, track, tyre, age)
                tb = _predict(model_b, feats_b, track, tyre, age)
                rows.append({
                    "age": int(age),
                    "driver_a": round(ta, 3),
                    "driver_b": round(tb, 3),
                    "delta": round(ta - tb, 3),
                })
            avg = sum(r['delta'] for r in rows) / len(rows)
            per_tyre.append({"tyre": tyre, "rows": rows, "avg_delta": round(avg, 3)})
        track_avg = sum(t['avg_delta'] for t in per_tyre) / len(per_tyre)
        per_track.append({
            "track": track,
            "tyres": per_tyre,
            "avg_delta": round(track_avg, 3),
        })

    if per_track:
        overall = sum(t['avg_delta'] for t in per_track) / len(per_track)
        faster_code = code_b if overall > 0 else code_a
        faster_name = (info_b.get('driver', {}).get('name', code_b)
                       if overall > 0 else info_a.get('driver', {}).get('name', code_a))
    else:
        overall = None
        faster_code, faster_name = None, None

    def _driver_summary(code, info, feats, used_year):
        return {
            "code": code,
            "name": str(info.get('driver', {}).get('name', code)),
            "tracks": covered_tracks(feats),
            "tyres": covered_tyres(feats),
            "laps": int(info.get('clean_laps') or info.get('training_samples') or 0),
            "mae": info.get('metrics', {}).get('within_track', {}).get('mae'),
            "tyre_age_coefficient": info.get('tyre_age_coefficient'),
            # Season whose model was actually used; None = aggregate
            # (all-seasons) fallback.
            "year": used_year,
        }

    # Both drivers resolved the same year model -> a true same-season
    # comparison.  Otherwise at least one driver fell back to the
    # aggregate model.
    same_year = used_year_a is not None and used_year_a == used_year_b
    return {
        "driver_a": _driver_summary(code_a, info_a, feats_a, used_year_a),
        "driver_b": _driver_summary(code_b, info_b, feats_b, used_year_b),
        "shared": {"tracks": shared_tracks, "tyres": shared_tyres},
        "per_track": per_track,
        "summary": {
            "shared_tracks": len(shared_tracks),
            "shared_tyres": len(shared_tyres),
            "avg_delta": round(overall, 3) if overall is not None else None,
            "faster_code": faster_code,
            "faster_name": faster_name,
            # Which models produced the gap: the requested season when both
            # drivers have it, or None when at least one fell back.
            "year_used": used_year_a if same_year else None,
            "note": "delta = driver_a_time - driver_b_time; positive means "
                    "driver B is faster",
            # Paired t-test over per-track deltas: labels the headline gap
            # significant / suggestive / inconclusive so it is not over-read
            # when the models' noise (~1 s) exceeds the gap itself.
            "significance": _significance(
                [t['avg_delta'] for t in per_track]
            ),
        },
        "caveats": [
            "Only tracks and tyres BOTH drivers' models cover are compared.",
            "Each model is fitted on its driver's own races, so weather, "
            "track evolution and car performance are baked into the gap — "
            "same-team (same car) comparisons are the most meaningful.",
            "Small-sample driver models are noisy; the trainer skips drivers "
            "under 60 clean laps.",
        ],
    }


def _pick_driver(drivers, prompt, exclude=None):
    print(f"\n{prompt}")
    for i, d in enumerate(drivers, 1):
        if d['code'] == exclude:
            continue
        print(f"  {i:>2}. {d['code']:<4} {d['name']:<26} "
              f"laps={d['laps']:>5} tracks={len(d['tracks']):>2} "
              f"MAE={d['mae'] if d['mae'] is not None else '?':>6}")
    while True:
        try:
            choice = int(input("\nSelect driver: "))
            shown = [d for d in drivers if d['code'] != exclude]
            if 1 <= choice <= len(shown):
                return shown[choice - 1]['code']
            print(f"   Enter a number between 1 and {len(shown)}")
        except ValueError:
            print("   Please enter a number, not text")


def _save_chart(code_a, code_b, per_track, out_dir=None):
    """Bar chart of per-track average delta (A - B).  Positive = B faster."""
    import matplotlib.pyplot as plt
    out = Path(out_dir) if out_dir else COMPARISON_DIR
    out.mkdir(parents=True, exist_ok=True)
    tracks = [t['track'] for t in per_track]
    deltas = [t['avg_delta'] for t in per_track]
    colors = ['#2ecc71' if d <= 0 else '#ff6b6b' for d in deltas]  # green = A faster
    plt.figure(figsize=(10, max(4, 0.6 * len(tracks))))
    plt.barh(tracks[::-1], deltas[::-1], color=colors[::-1])
    plt.axvline(0, color='white', lw=1)
    plt.xlabel('Average delta (s) — negative means ' + code_a + ' faster, positive means ' + code_b + ' faster')
    plt.title(f'{code_a} vs {code_b} — per-track pace gap')
    plt.tight_layout()
    path = out / f'{code_a}_vs_{code_b}.png'
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    parser = argparse.ArgumentParser(description='Compare two drivers head-to-head.')
    parser.add_argument('--driver-a', help='Driver code of the first driver')
    parser.add_argument('--driver-b', help='Driver code of the second driver')
    parser.add_argument('--year', type=int, help='Compare using same-year models (e.g. 2021)')
    args = parser.parse_args()

    print("=" * 60)
    print("F1 DRIVER COMPARISON — HEAD TO HEAD")
    print("=" * 60)

    drivers = list_driver_models()
    if not drivers:
        print("\n[ERROR] No per-driver models found in "
              f"{DRIVER_MODELS_DIR}.")
        print("Run scripts/ml_lap_predictions.py first — it now trains one "
              "model per driver with >= 60 clean laps.")
        sys.exit(1)

    codes = [d['code'] for d in drivers]
    code_a = args.driver_a
    if code_a and code_a not in codes:
        print(f"\n[ERROR] '{code_a}' has no model. Available: {', '.join(codes)}")
        sys.exit(1)
    code_b = args.driver_b
    if code_b and code_b not in codes:
        print(f"\n[ERROR] '{code_b}' has no model. Available: {', '.join(codes)}")
        sys.exit(1)

    if not code_a:
        code_a = _pick_driver(drivers, "DRIVER A")
    if not code_b:
        code_b = _pick_driver(drivers, "DRIVER B", exclude=code_a)
    if code_a == code_b:
        print("\n[ERROR] Pick two different drivers.")
        sys.exit(1)

    try:
        result = compare_drivers(code_a, code_b, year=args.year)
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)

    a, b = result['driver_a'], result['driver_b']
    year_label = f" ({args.year})" if args.year else ""
    print(f"\n  {a['name']} ({a['code']}){year_label}  vs  {b['name']} ({b['code']}){year_label}")
    used = result['summary']['year_used']
    if used:
        print(f"  Models: same-season {used} per-driver models")
    elif args.year:
        print(f"  Models: {args.year} model missing for at least one driver — "
              f"fell back to aggregate (all-seasons) models")
    else:
        print(f"  Models: aggregate (all-seasons) per-driver models")
    print(f"  Shared coverage: {len(result['shared']['tracks'])} tracks "
          f"{result['shared']['tracks']}, {len(result['shared']['tyres'])} tyres "
          f"{result['shared']['tyres']}\n")

    if not result['per_track']:
        print("  No shared tracks — nothing to compare.")
        sys.exit(0)

    header = (f"  {'TRACK':<42}{'AVG Δ (s)':>12}  FASTER")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for t in result['per_track']:
        faster = a['code'] if t['avg_delta'] < 0 else b['code']
        print(f"  {t['track']:<42}{t['avg_delta']:>10.3f}s  {faster}")
    print("  " + "-" * (len(header) - 2))

    s = result['summary']
    print(f"\n  OVERALL: {s['avg_delta']:+.3f}s per lap "
          f"({s['faster_name']} faster)")

    sig = s.get('significance', {})
    if sig.get('label'):
        t_str = (f", t={sig['t_statistic']:.2f}"
                 if sig.get('t_statistic') is not None else "")
        p_str = (f"p={sig['p_value']:.3f}"
                 if sig.get('p_value') is not None else "n/a")
        print(f"  SIGNIFICANCE: {sig['label'].upper()} "
              f"({p_str}{t_str}, n={sig.get('n', '?')} tracks)")
        print(f"    {sig.get('note', '')}")

    try:
        chart = _save_chart(code_a, code_b, result['per_track'])
        print(f"\n  Chart saved: {chart}")
    except Exception as exc:
        print(f"\n  (Chart not saved: {exc})")

    print("\n" + "=" * 60)
    print("HOW TO READ THIS")
    print("=" * 60)
    for c in result['caveats']:
        print(f"  • {c}")
    print("=" * 60)


if __name__ == '__main__':
    main()
