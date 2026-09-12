import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import json
from datetime import datetime
from pathlib import Path

from config import get_db_connection

# ---------------------------------------------------------------------------
# Model artifacts always live in <project root>/ml_models — never relative to
# the current working directory.  (Running this script from scripts/ vs the
# root used to produce two divergent model directories.)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "ml_models"

# ---------------------------------------------------------------------------
# Training data
#
# Only representative steady-state racing laps are used.  Excluded:
#   * invalid laps (track-limit deletions etc.)
#   * laps outside the 60–180 s window (red-flag laps, formation laps)
#   * pit in-laps  — the lap that carries a real PitStop event
#   * pit out-laps — the lap immediately after a real pit in-lap
#   * the first two laps of every stint (cold-tyre / traffic / race-start
#     laps that are not representative of steady-state pace)
#   * SC / VSC / red-flag laps — laps > 130% of their session's median lap
#     time (red-flag periods are stored session-wide with lap_id = NULL, so
#     they cannot be excluded by lap id; the session-relative time ratio is
#     the reliable signal).
#
# A PitStop event only excludes the laps around it when the stop duration is
# plausible (>= 15 s) or unknown (NULL -- recorded by the importer when the
# box time could not be resolved; the stop still happened).  Implausibly
# short "stops" (data glitches that attach a 2.3 s pit event to a normal
# lap) are ignored so they do not drop valid racing laps.
# ---------------------------------------------------------------------------
TRAINING_QUERY = """
SELECT
    l.lap_time_ms / 1000.0 AS lap_time,
    l.lap_number,
    l.tyre_age,
    l.tyre_compound,
    l.session_id,
    s.track_name,
    s.date AS session_date,
    l.driver_id,
    d.driver_code,
    COALESCE(d.driver_name, d.driver_code) AS driver_name
FROM laps l
JOIN sessions s ON l.session_id = s.session_id
LEFT JOIN drivers d ON l.driver_id = d.driver_id
WHERE l.is_valid = 1
  AND l.lap_time_ms BETWEEN 60000 AND 180000
  -- Pit in-lap: the lap carrying a real PitStop event
  AND l.lap_id NOT IN (
      SELECT lap_id FROM strategy_events
      WHERE event_type = 'PitStop'
        AND (duration_sec IS NULL OR duration_sec >= 15.0)
  )
  -- Pit out-lap: the lap immediately after a real pit in-lap
  AND l.lap_id NOT IN (
      SELECT l2.lap_id
      FROM laps l2
      JOIN strategy_events se ON se.lap_id = l2.lap_id
      WHERE se.event_type = 'PitStop'
        AND (se.duration_sec IS NULL OR se.duration_sec >= 15.0)
        AND l2.session_id = l.session_id
        AND l.lap_number = l2.lap_number + 1
  )
"""

# Laps slower than this multiple of their session's median lap time are
# treated as SC / VSC / red-flag / formation laps and excluded.
SC_VSC_REDFLAG_RATIO = 1.30

# Cold-tyre / traffic / race-start laps dropped from the start of each stint.
STINT_WARMUP_LAPS = 2

# A driver needs at least this many CLEAN training laps to get their own
# model.  Below that the model is too noisy to be a meaningful signature.
MIN_DRIVER_LAPS = 60


def clean_training_data(df, verbose=True, label="Training data"):
    """Return the cleaned training DataFrame for a driver subset (or all).

    Applies the shared cleaning used by the global model and every
    per-driver model:

      * normalises track/tyre casing
      * drops the first STINT_WARMUP_LAPS laps of each stint
      * drops SC/VSC/red-flag/formation laps (> SC_VSC_REDFLAG_RATIO x the
        session's median lap time)

    Stints are detected per (session, driver): one driver's pit stop must
    never split another driver's stint, which is why the global model also
    groups by driver_id here.
    """
    df = df.copy()
    df['track_name']    = df['track_name'].str.strip().str.title()
    df['tyre_compound'] = df['tyre_compound'].str.strip()

    # -------------------------------------------------------------------
    # Drop the first STINT_WARMUP_LAPS laps of every stint.  A stint is a
    # run of consecutive laps on the same compound within a session (and
    # driver).  These opening laps are distorted by cold tyres, race-start
    # traffic and tyre warm-up, and were the main driver of the old model's
    # bogus "laps get faster as tyres age" coefficient.
    # -------------------------------------------------------------------
    df = df.sort_values(['session_id', 'driver_id', 'lap_number'])
    df['_stint'] = (
        df['tyre_compound'] != df.groupby(['session_id', 'driver_id'])['tyre_compound'].shift()
    ).groupby([df['session_id'], df['driver_id']]).cumsum()
    df['_lap_in_stint'] = df.groupby(['session_id', 'driver_id', '_stint']).cumcount()
    warmup_mask = df['_lap_in_stint'] < STINT_WARMUP_LAPS
    df = df[~warmup_mask].drop(columns=['_stint', '_lap_in_stint', 'lap_number'])
    if verbose:
        print(f"[INFO] Excluded {int(warmup_mask.sum())} stint warm-up laps "
              f"(first {STINT_WARMUP_LAPS} laps of each stint)")

    # -------------------------------------------------------------------
    # SC / VSC / red-flag lap exclusion (session-relative outlier filter)
    # -------------------------------------------------------------------
    session_medians = df.groupby('session_id')['lap_time'].transform('median')
    outlier_mask = df['lap_time'] <= session_medians * SC_VSC_REDFLAG_RATIO
    dropped_outliers = int((~outlier_mask).sum())
    df = df[outlier_mask]
    if verbose:
        print(f"[INFO] Excluded {dropped_outliers} SC/VSC/red-flag/formation laps "
              f"(> {SC_VSC_REDFLAG_RATIO:.2f}x session median)")
    return df


print("=" * 60)
print("F1 LAP TIME PREDICTION - MODEL TRAINING")
print("=" * 60)

conn = get_db_connection()

print("\n[DATABASE] Loading training data...")
df = pd.read_sql(TRAINING_QUERY, conn)
conn.close()

if df.empty:
    print("[ERROR] No valid laps found in database. Train data is empty!")
    sys.exit(1)

print(f"[INFO] Raw candidate laps: {len(df)}")

# Keep a pre-cleaning copy so each driver's model re-runs the stint warm-up
# and outlier filters on their OWN laps (grouped by session + driver).
raw_df = df.copy()

df = clean_training_data(df)
print(f"[INFO] Training laps after cleaning: {len(df)}")
print(f"[INFO] Tracks covered: {df['track_name'].nunique()} — {sorted(df['track_name'].unique())}")
print(f"[INFO] Tyres:  {df['tyre_compound'].nunique()}")

# ---------------------------------------------------------------------------
# Feature engineering
#
# NOTE: the synthetic fuel-load feature (max(0, 110 - 2 * lap_number)) is
# deliberately NOT used.  It is a deterministic function of lap_number and is
# therefore perfectly collinear with tyre_age inside every stint, which makes
# the linear coefficients unstable.  tyre_age carries the within-stint pace
# progression (its cleaned-data slope is the net of tyre wear minus fuel
# burn, which is physically meaningful and near zero once warm-up laps are
# excluded).
# ---------------------------------------------------------------------------
df_encoded = pd.get_dummies(df, columns=['tyre_compound', 'track_name'], prefix=['tyre', 'track'])

y = df_encoded['lap_time']
groups = df_encoded['session_id']
X = df_encoded.drop(columns=['lap_time', 'session_id', 'driver_id', 'driver_code',
                              'driver_name', 'session_date'])
feature_names = list(X.columns)

print(f"[INFO] Feature Count: {len(feature_names)}")

# ---------------------------------------------------------------------------
# Evaluation
#
# Two metrics are reported:
#   * Within-track accuracy (random 80/20 split) — this is the deployment
#     scenario: the API/CLI only predict on tracks the model has seen
#     (unseen tracks are rejected with a clear error), so the useful
#     question is "how well does the model predict remaining laps of a
#     known track?".
#   * Unseen-track generalization (GroupShuffleSplit on session_id) — held
#     out sessions are whole tracks the model never saw, which a one-hot
#     track model structurally cannot predict.  Its R2 is expected to be
#     poor; it is reported for transparency and is NOT the deployment
#     metric.
# Model selection uses the within-track MAE.
# ---------------------------------------------------------------------------
def train_and_eval(X_tr, y_tr, X_te, y_te):
    results = {}
    lr = LinearRegression()
    lr.fit(X_tr, y_tr)
    yp = lr.predict(X_te)
    results['LinearRegression'] = (lr, {
        'mae': mean_absolute_error(y_te, yp),
        'rmse': np.sqrt(mean_squared_error(y_te, yp)),
        'r2': r2_score(y_te, yp),
        'predictions': yp,
        'y_test': y_te,
    })

    rf = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_split=4, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    yp = rf.predict(X_te)
    results['RandomForest'] = (rf, {
        'mae': mean_absolute_error(y_te, yp),
        'rmse': np.sqrt(mean_squared_error(y_te, yp)),
        'r2': r2_score(y_te, yp),
        'predictions': yp,
        'y_test': y_te,
    })
    return results

print("=" * 60)
print("EVALUATION 1/2 — WITHIN-TRACK (random 80/20 split)")
print("=" * 60)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
within = train_and_eval(X_train, y_train, X_test, y_test)
for name, (_, m) in within.items():
    print(f"  {name:<16} MAE {m['mae']:.3f}s  RMSE {m['rmse']:.3f}s  R2 {m['r2']:.3f}")

lr_within = within['LinearRegression'][0]
lr_coefs = dict(zip(feature_names, lr_within.coef_))
print(f"\n  tyre_age coefficient (within-track): "
      f"{lr_coefs.get('tyre_age', 0.0):+.4f} s/lap  "
      f"[positive = tyres slow you down as they age]")

# Fuel-burn rate: the pace effect of fuel burn (and other non-wear age
# effects like track evolution) that the tyre_age coefficient conflates with
# tyre wear.  The strategy advisor detrends predictions by this rate so that
# stay-out vs pit comparisons are made on equal fuel footing — fuel is burned
# identically by both scenarios, so crediting the stay-out scenario's higher
# ages with it would bias the comparison by |rate| * laps_rem * cur_age in
# its favour.  The cleaned dataset shows no net within-stint wear (per-stint
# slope ~0.00 after warm-up laps are excluded), so the whole negative age
# slope is treated as fuel; a positive slope (wear dominating) would be
# clamped to zero, leaving the full wear effect in the comparison.
fuel_burn_rate = min(0.0, lr_coefs.get('tyre_age', 0.0))

print("=" * 60)
print("EVALUATION 2/2 — UNSEEN-TRACK (GroupShuffleSplit by session)")
print("=" * 60)
print("  Held-out sessions are whole tracks the model has never seen.")
print("  One-hot track models cannot predict them; the API rejects such")
print("  requests by design.  Reported for transparency only.")
if groups.nunique() > 1:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups))
    unseen = train_and_eval(X.iloc[tr_idx], y.iloc[tr_idx], X.iloc[te_idx], y.iloc[te_idx])
    for name, (_, m) in unseen.items():
        print(f"  {name:<16} MAE {m['mae']:.3f}s  RMSE {m['rmse']:.3f}s  R2 {m['r2']:.3f}")
else:
    unseen = None
    print("  Only one session — skipping.")

# Select best model on the within-track (deployment) metric
best_name  = min(within, key=lambda k: within[k][1]['mae'])
best_model = within[best_name][0]
best_metrics = within[best_name][1]

print("\n" + "=" * 60)
print(f"[BEST MODEL] {best_name} (selected on within-track MAE)")
print(f"   Within-track MAE:  {best_metrics['mae']:.3f}s")
print(f"   Within-track RMSE: {best_metrics['rmse']:.3f}s")
print(f"   Within-track R2:   {best_metrics['r2']:.3f}")
print("=" * 60)

# Save artifacts
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(best_model,   MODEL_DIR / 'best_model.pkl')
joblib.dump(feature_names, MODEL_DIR / 'feature_names.pkl')
print(f"\n[INFO] Saved: {MODEL_DIR / 'best_model.pkl'}")
print(f"[INFO] Saved: {MODEL_DIR / 'feature_names.pkl'}")

# Track / tyre coverage (used by the predictors to reject unseen inputs)
covered_tracks = sorted(f.replace('track_', '') for f in feature_names if f.startswith('track_'))
covered_tyres  = sorted(f.replace('tyre_', '') for f in feature_names
                        if f.startswith('tyre_') and f not in ('tyre_age', 'tyre_load'))
per_track_laps = df.groupby('track_name').size().sort_values(ascending=False)

def fmt_metrics(m):
    return (f"MAE: {m['mae']:.3f}s\n"
            f"RMSE: {m['rmse']:.3f}s\n"
            f"R2: {m['r2']:.3f}")

# Text metadata
with open(MODEL_DIR / 'model_info.txt', 'w', encoding='utf-8') as f:
    f.write("F1 LAP TIME PREDICTION MODEL\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Best Model:       {best_name}\n")
    f.write(f"Within-track:     {fmt_metrics(best_metrics)}\n")
    if unseen:
        f.write(f"Unseen-track:     {fmt_metrics(unseen[best_name][1])} "
                f"(informational — unseen tracks are rejected)\n")
    f.write(f"Training samples: {len(X_train)}\n")
    f.write(f"Test samples:     {len(X_test)}\n")
    f.write(f"Features:         {len(feature_names)}\n")
    f.write(f"Trained on:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Tracks covered:   {len(covered_tracks)}\n")
    f.write(f"Tyres covered:    {len(covered_tyres)}\n")
    f.write(f"Fuel burn rate:   {fuel_burn_rate:+.4f} s/lap "
            f"(detrended out of advisor comparisons)\n\n")
    f.write("Laps per track:\n")
    for track, n in per_track_laps.items():
        f.write(f"  {track:<50} {n}\n")
    f.write("\nFeatures used:\n")
    for fn in feature_names:
        f.write(f"  {fn}\n")

print(f"[INFO] Saved: {MODEL_DIR / 'model_info.txt'}")

# JSON metadata
metadata = {
    "best_model": best_name,
    "metrics": {
        "within_track": {k: round(v, 4) for k, v in best_metrics.items() if k not in ('predictions', 'y_test')},
    },
    "unseen_track": (
        {k: round(v, 4) for k, v in unseen[best_name][1].items() if k not in ('predictions', 'y_test')}
        if unseen else None
    ),
    "training_samples": len(X_train),
    "test_samples": len(X_test),
    "features": feature_names,
    "coverage": {
        "tracks": covered_tracks,
        "tyres": covered_tyres,
        "laps_per_track": {str(k): int(v) for k, v in per_track_laps.items()},
    },
    "tyre_age_coefficient": round(lr_coefs.get('tyre_age', 0.0), 4),
    "fuel_burn_rate": round(fuel_burn_rate, 4),
    "trained_at": datetime.now().isoformat()
}
with open(MODEL_DIR / 'model_info.json', 'w', encoding='utf-8') as f_json:
    json.dump(metadata, f_json, indent=2, ensure_ascii=False)
print(f"[INFO] Saved: {MODEL_DIR / 'model_info.json'}")

# ---------------------------------------------------------------------------
# Per-driver models
#
# One model per driver, trained on the driver's OWN race laps with the same
# cleaning + feature pipeline as the global model.  Each driver model covers
# only the tracks/tyres that driver has raced, which is exactly what makes
# a head-to-head comparison meaningful: on a shared (track, tyre) the
# difference between two drivers' predicted times is their pace gap.
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PER-DRIVER MODELS")
print("=" * 60)
print(f"  Training one model per driver with >= {MIN_DRIVER_LAPS} clean laps.")
print("  Artifacts: ml_models\\drivers\\<driver_code>\\")

drivers_dir = MODEL_DIR / 'drivers'
drivers_dir.mkdir(parents=True, exist_ok=True)

driver_summary = []
# NULL driver ids (telemetry-only laps) are dropped by groupby's dropna.
for (driver_code, driver_name), grp in raw_df.groupby(['driver_code', 'driver_name']):
    grp_clean = clean_training_data(grp, verbose=False)
    if len(grp_clean) < MIN_DRIVER_LAPS:
        print(f"  [SKIP ] {str(driver_code):<4} {str(driver_name):<24} "
              f"only {len(grp_clean)} clean laps (< {MIN_DRIVER_LAPS})")
        continue

    df_enc = pd.get_dummies(grp_clean, columns=['tyre_compound', 'track_name'],
                            prefix=['tyre', 'track'])
    y_drv = df_enc['lap_time']
    drop_cols = [c for c in ('lap_time', 'session_id', 'driver_id', 'driver_code',
                             'driver_name', 'session_date')
                 if c in df_enc.columns]
    X_drv = df_enc.drop(columns=drop_cols)
    drv_features = list(X_drv.columns)

    X_tr, X_te, y_tr, y_te = train_test_split(X_drv, y_drv, test_size=0.2, random_state=42)
    within_drv = train_and_eval(X_tr, y_tr, X_te, y_te)
    drv_best = min(within_drv, key=lambda k: within_drv[k][1]['mae'])
    drv_model = within_drv[drv_best][0]
    drv_metrics = within_drv[drv_best][1]

    drv_lr = within_drv['LinearRegression'][0]
    drv_coefs = dict(zip(drv_features, drv_lr.coef_))
    drv_fuel_burn = min(0.0, drv_coefs.get('tyre_age', 0.0))

    drv_tracks = sorted(f.replace('track_', '') for f in drv_features if f.startswith('track_'))
    drv_tyres = sorted(f.replace('tyre_', '') for f in drv_features
                       if f.startswith('tyre_') and f not in ('tyre_age', 'tyre_load'))

    ddir = drivers_dir / str(driver_code)
    ddir.mkdir(parents=True, exist_ok=True)
    joblib.dump(drv_model, ddir / 'best_model.pkl')
    joblib.dump(drv_features, ddir / 'feature_names.pkl')

    drv_info = {
        "driver": {"code": str(driver_code), "name": str(driver_name)},
        "best_model": drv_best,
        "metrics": {"within_track": {k: round(v, 4) for k, v in drv_metrics.items()
                                     if k not in ('predictions', 'y_test')}},
        "training_samples": int(len(X_tr)),
        "test_samples": int(len(X_te)),
        "clean_laps": int(len(grp_clean)),
        "features": drv_features,
        "coverage": {"tracks": drv_tracks, "tyres": drv_tyres},
        "tyre_age_coefficient": round(drv_coefs.get('tyre_age', 0.0), 4),
        "fuel_burn_rate": round(drv_fuel_burn, 4),
        "trained_at": datetime.now().isoformat()
    }
    with open(ddir / 'model_info.json', 'w', encoding='utf-8') as f:
        json.dump(drv_info, f, indent=2, ensure_ascii=False)

    driver_summary.append((str(driver_code), str(driver_name), len(grp_clean),
                           len(drv_tracks), drv_best, drv_metrics['mae']))
    print(f"  [TRAIN] {str(driver_code):<4} {str(driver_name):<24} "
          f"laps={len(grp_clean):>5} tracks={len(drv_tracks):>2} "
          f"model={drv_best:<16} MAE={drv_metrics['mae']:.3f}s")

# -----------------------------------------------------------------------
# Per-driver-per-year models
#
# One model per (driver, season) pair, trained on that driver's own race
# laps from that specific year.  Same pipeline as the global model, but
# the intercept now absorbs the driver + car + regulations of that exact
# season, so cross-year comparisons mix car changes with driver pace.
# These models are stored in:
#   ml_models/drivers/<code>/<year>/
# and are the primary comparison unit in the dashboard (same-year =
# apples-to-apples).  The per-driver aggregate model in
# ml_models/drivers/<code>/ is kept as a fallback for drivers with only
# one year of data, and for multi-year 'career shape' queries.
# -----------------------------------------------------------------------
MIN_YEAR_LAPS = 60

print("\n" + "=" * 60)
print("PER-DRIVER-PER-YEAR MODELS")
print("=" * 60)
print(f"  Training one model per (driver, year) with >= {MIN_YEAR_LAPS} clean laps.")
print("  Artifacts: ml_models\\drivers\\<driver_code>\\<year>\\")

year_summary = []
for (driver_code, driver_name), grp in raw_df.groupby(['driver_code', 'driver_name']):
    if pd.isna(driver_code):
        continue
    # session_date is in the query result; extract year for grouping
    grp = grp.copy()
    grp['_year'] = pd.to_datetime(grp['session_date'], errors='coerce').dt.year
    for year, year_grp in grp.groupby('_year'):
        if pd.isna(year):
            continue
        year_int = int(year)
        year_clean = clean_training_data(year_grp, verbose=False)
        if len(year_clean) < MIN_YEAR_LAPS:
            continue

        df_enc = pd.get_dummies(year_clean, columns=['tyre_compound', 'track_name'],
                                prefix=['tyre', 'track'])
        y_drv = df_enc['lap_time']
        drop_cols = [c for c in ('lap_time', 'session_id', 'driver_id', 'driver_code',
                                 'driver_name', 'session_date', '_year')
                     if c in df_enc.columns]
        X_drv = df_enc.drop(columns=drop_cols)
        drv_features = list(X_drv.columns)

        X_tr, X_te, y_tr, y_te = train_test_split(X_drv, y_drv, test_size=0.2, random_state=42)
        within_drv = train_and_eval(X_tr, y_tr, X_te, y_te)
        drv_best = min(within_drv, key=lambda k: within_drv[k][1]['mae'])
        drv_model = within_drv[drv_best][0]
        drv_metrics = within_drv[drv_best][1]

        drv_lr = within_drv['LinearRegression'][0]
        drv_coefs = dict(zip(drv_features, drv_lr.coef_))
        drv_fuel_burn = min(0.0, drv_coefs.get('tyre_age', 0.0))

        drv_tracks = sorted(f.replace('track_', '') for f in drv_features if f.startswith('track_'))
        drv_tyres = sorted(f.replace('tyre_', '') for f in drv_features
                           if f.startswith('tyre_') and f not in ('tyre_age', 'tyre_load'))

        ydir = drivers_dir / str(driver_code) / str(year_int)
        ydir.mkdir(parents=True, exist_ok=True)
        joblib.dump(drv_model, ydir / 'best_model.pkl')
        joblib.dump(drv_features, ydir / 'feature_names.pkl')

        yr_info = {
            "driver": {"code": str(driver_code), "name": str(driver_name)},
            "year": year_int,
            "best_model": drv_best,
            "metrics": {"within_track": {k: round(v, 4) for k, v in drv_metrics.items()
                                         if k not in ('predictions', 'y_test')}},
            "training_samples": int(len(X_tr)),
            "test_samples": int(len(X_te)),
            "clean_laps": int(len(year_clean)),
            "features": drv_features,
            "coverage": {"tracks": drv_tracks, "tyres": drv_tyres},
            "tyre_age_coefficient": round(drv_coefs.get('tyre_age', 0.0), 4),
            "fuel_burn_rate": round(drv_fuel_burn, 4),
            "trained_at": datetime.now().isoformat()
        }
        with open(ydir / 'model_info.json', 'w', encoding='utf-8') as f:
            json.dump(yr_info, f, indent=2, ensure_ascii=False)

        year_summary.append((str(driver_code), str(driver_name), year_int, len(year_clean),
                               len(drv_tracks), drv_best, drv_metrics['mae']))
        print(f"  [TRAIN] {str(driver_code):<4} {year_int} {str(driver_name):<20} "
              f"laps={len(year_clean):>5} tracks={len(drv_tracks):>2} "
              f"model={drv_best:<16} MAE={drv_metrics['mae']:.3f}s")

if not year_summary:
    print("  No (driver, year) pair met the minimum-lap threshold.")
else:
    print(f"\n  Trained {len(year_summary)} per-driver-per-year models.")

if not driver_summary:
    print("  No driver met the minimum-lap threshold — only the global model was saved.")
else:
    print(f"\n  Trained {len(driver_summary)} per-driver models in {drivers_dir}.")
    print("  Compare two drivers head-to-head with: python scripts/driver_comparison.py")

# Visualizations
if best_name == 'RandomForest':
    importances = best_model.feature_importances_
    feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feat_imp = feat_imp.sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    top15 = feat_imp.head(15)
    plt.barh(range(len(top15)), top15['importance'])
    plt.yticks(range(len(top15)), top15['feature'])
    plt.xlabel('Importance')
    plt.title('Top 15 Feature Importances — Random Forest')
    plt.tight_layout()
    plt.savefig(MODEL_DIR / 'feature_importance.png', dpi=300)
    plt.close()
    print(f"[INFO] Saved: {MODEL_DIR / 'feature_importance.png'}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, best_metrics['predictions'], alpha=0.5, label='Predictions')
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--', lw=2, label='Perfect Prediction'
)
plt.xlabel('Actual Lap Time (seconds)')
plt.ylabel('Predicted Lap Time (seconds)')
plt.title(f'Actual vs Predicted Lap Times ({best_name}, within-track)')
plt.legend()
plt.tight_layout()
plt.savefig(MODEL_DIR / 'predictions_vs_actual.png', dpi=300)
plt.close()
print(f"[INFO] Saved: {MODEL_DIR / 'predictions_vs_actual.png'}")
