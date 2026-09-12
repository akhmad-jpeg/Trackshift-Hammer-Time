"""Model selection benchmark for the lap-time / strategy model.

Compares the deployed LinearRegression against a track-random-effects
mixed model, a stint-random-effects mixed model, and tree baselines, on
two evaluation contracts:

  * known-track (random lap split) -- the deployment case, since the API
    rejects unseen tracks;
  * unseen-track (session-grouped split) -- informational; every model is
    expected to fail here, which is why coverage rejection exists.

Also probes each model's tyre-age response curve (smoothness matters more
than raw MAE for the strategy advisor, whose decision reads that curve).

Run:  python scripts/benchmark_models.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import get_db_connection

# Mirrors ml_lap_predictions.TRAINING_QUERY (NULL-safe pit-event filter).
TRAINING_QUERY = """
SELECT
    l.lap_time_ms / 1000.0 AS lap_time,
    l.lap_number,
    l.tyre_age,
    l.tyre_compound,
    l.session_id,
    s.track_name
FROM laps l
JOIN sessions s ON l.session_id = s.session_id
WHERE l.is_valid = 1
  AND l.lap_time_ms BETWEEN 60000 AND 180000
  AND l.lap_id NOT IN (
      SELECT lap_id FROM strategy_events
      WHERE event_type = 'PitStop'
        AND (duration_sec IS NULL OR duration_sec >= 15.0)
  )
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

TYRE_COLS = ["tyre_Medium", "tyre_Soft", "tyre_Intermediate"]  # Hard = base


def load_cleaned():
    """Same cleaning as the trainer: warm-up laps, SC/VSC/red-flag laps."""
    conn = get_db_connection()
    df = pd.read_sql(TRAINING_QUERY, conn)
    conn.close()

    df["tyre_compound"] = df["tyre_compound"].str.strip()
    df = df.sort_values(["session_id", "lap_number"])
    df["_stint"] = (
        df["tyre_compound"] != df.groupby("session_id")["tyre_compound"].shift()
    ).groupby(df["session_id"]).cumsum()
    df["_lap_in_stint"] = df.groupby(["session_id", "_stint"]).cumcount()
    df = df[df["_lap_in_stint"] >= 2]
    med = df.groupby("session_id")["lap_time"].transform("median")
    df = df[df["lap_time"] <= med * 1.30]
    df["stint_id"] = (df["session_id"].astype(str) + "_"
                      + df["_stint"].astype(str))
    return df.drop(columns=["_stint", "_lap_in_stint"])


def onehot_features(df):
    """One-hot track x compound + tyre_age (the deployed LR design)."""
    return pd.get_dummies(
        df[["tyre_compound", "track_name"]],
        columns=["tyre_compound", "track_name"], prefix=["tyre", "track"],
    ).join(df[["tyre_age"]])


def evaluate(name, y_true, y_pred):
    y_pred = np.asarray(y_pred).ravel()
    print(f"  {name:<24} MAE {mean_absolute_error(y_true, y_pred):6.3f}s   "
          f"R2 {r2_score(y_true, y_pred):+6.2f}")


def fit_lr(X_tr, y_tr):
    return LinearRegression().fit(X_tr, y_tr)


def fit_mixed_track(df_tr):
    """lap_time ~ compound + age, random track intercepts (shrinkage)."""
    from statsmodels.regression.mixed_linear_model import MixedLM
    m = MixedLM.from_formula(
        "lap_time ~ C(tyre_compound) + tyre_age",
        groups="track_name", data=df_tr)
    return m.fit(reml=True)


def predict_mixed_track(model, df):
    """Fixed-effects prediction + track BLUP (0 for unseen tracks).

    statsmodels' predict() returns only the fixed part, so the random
    track intercept has to be added from model.random_effects.
    """
    fe = np.asarray(model.predict(df[["tyre_compound", "tyre_age",
                                      "track_name"]])).ravel()
    blups = {g: float(v[0]) for g, v in model.random_effects.items()}
    re = np.array([blups.get(t, 0.0) for t in df["track_name"]],
                  dtype=float)
    return fe + re


def fit_mixed_stint(df_tr, track_cols):
    """LR design + random stint intercept and stint age slope."""
    from statsmodels.regression.mixed_linear_model import MixedLM

    def design(df, cols=None):
        comp = pd.get_dummies(df["tyre_compound"], prefix="tyre",
                              drop_first=True).reindex(columns=TYRE_COLS,
                                                       fill_value=0).astype(int)
        # Track dummies drop the first track so the const is identifiable;
        # an absent track then predicts at the population level instead of
        # collapsing (no intercept) as a full one-hot would.
        tracks = pd.get_dummies(df["track_name"], prefix="track",
                                drop_first=True).reindex(
            columns=track_cols[1:], fill_value=0).astype(int)
        exog = pd.concat([comp, df[["tyre_age"]], tracks], axis=1)
        exog["const"] = 1.0
        exog = exog.astype(float)
        if cols is not None:
            exog = exog.reindex(columns=cols, fill_value=0.0)
        return exog

    # Trim all-zero columns (a track/compound absent from this split would
    # otherwise make X'X singular).
    full = design(df_tr)
    cols = [c for c in full.columns if full[c].abs().sum() > 0]
    exog_tr = full[cols]
    exog_re = pd.DataFrame({"const": 1.0,
                            "tyre_age": df_tr["tyre_age"]}).astype(float)
    m = MixedLM(df_tr["lap_time"], exog_tr, groups=df_tr["stint_id"],
                exog_re=exog_re)
    return m.fit(reml=True), lambda df: design(df, cols)


def predict_mixed_stint(model, design, df, track_cols):
    """Fixed effects + stint BLUPs for known stints (0 for unseen).

    MixedLM results.predict() returns only the fixed mean structure, so
    the random stint intercept/slope BLUPs are added per stint.
    """
    exog = design(df)
    pred = np.asarray(model.predict(exog)).ravel().copy()
    stint_ids = df.get("stint_id")
    if stint_ids is not None:
        for i, stint in enumerate(stint_ids):
            blup = model.random_effects.get(stint)
            if blup is not None:
                pred[i] += blup[0] + blup[1] * df["tyre_age"].iloc[i]
    return pred


def main():
    df = load_cleaned()
    print(f"Data: {len(df)} laps, {df['track_name'].nunique()} tracks, "
          f"{df['stint_id'].nunique()} stints")

    track_cols = [f"track_{t}" for t in sorted(df["track_name"].unique())]
    X = onehot_features(df)
    y = df["lap_time"]
    groups = df["session_id"]

    print("\n" + "=" * 64)
    print("KNOWN-TRACK EVALUATION (random 80/20 lap split)")
    print("=" * 64)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42)
    df_tr, df_te = df.loc[X_tr.index], df.loc[X_te.index]

    lr = fit_lr(X_tr, y_tr)
    evaluate("LinearRegression", y_te, lr.predict(X_te))

    rf = RandomForestRegressor(n_estimators=400, random_state=42)
    rf.fit(X_tr, y_tr)
    evaluate("RandomForest", y_te, rf.predict(X_te))

    mt = fit_mixed_track(df_tr)
    evaluate("MixedTrack (RE: track)", y_te, predict_mixed_track(mt, df_te))

    ms, design = fit_mixed_stint(df_tr, track_cols)
    evaluate("MixedStint (RE: stint)",
             y_te, predict_mixed_stint(ms, design, df_te, track_cols))

    print("\nFixed tyre-age slope (fuel + wear conflated, as expected):")
    print(f"  LinearRegression : {lr.coef_[X.columns.get_loc('tyre_age')]:+.4f} s/lap")
    print(f"  MixedTrack       : {mt.params['tyre_age']:+.4f} s/lap")
    print(f"  MixedStint       : {ms.params['tyre_age']:+.4f} s/lap")

    print("\n" + "=" * 64)
    print("NEW-STINT EVALUATION (stint-grouped 80/20, RE=0)")
    print("=" * 64)
    # The advisor predicts hypothetical future stints: no stint BLUP is
    # available, so random effects are always 0.  A stint-grouped split is
    # the honest version of the deployment case.
    gss_stint = GroupShuffleSplit(n_splits=1, test_size=0.2,
                                  random_state=42)
    (tr_i, te_i) = next(gss_stint.split(X, y, groups=df["stint_id"]))
    df_tr3, df_te3 = df.iloc[tr_i], df.iloc[te_i]
    X_tr3, X_te3 = X.iloc[tr_i], X.iloc[te_i]
    y_tr3, y_te3 = y.iloc[tr_i], y.iloc[te_i]

    evaluate("LinearRegression", y_te3, fit_lr(X_tr3, y_tr3).predict(X_te3))
    rf3 = RandomForestRegressor(n_estimators=400, random_state=42)
    rf3.fit(X_tr3, y_tr3)
    evaluate("RandomForest", y_te3, rf3.predict(X_te3))
    mt3 = fit_mixed_track(df_tr3)
    evaluate("MixedTrack (RE: track)",
             y_te3, predict_mixed_track(mt3, df_te3))
    ms3, design3 = fit_mixed_stint(df_tr3, track_cols)
    evaluate("MixedStint (RE=0, new stints)",
             y_te3, predict_mixed_stint(ms3, design3, df_te3, track_cols))

    print("\n" + "=" * 64)
    print("UNSEEN-TRACK EVALUATION (session-grouped 75/25)")
    print("=" * 64)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    (tr_i, te_i) = next(gss.split(X, y, groups=groups))
    df_tr2, df_te2 = df.iloc[tr_i], df.iloc[te_i]
    X_tr2, X_te2 = X.iloc[tr_i], X.iloc[te_i]
    y_tr2, y_te2 = y.iloc[tr_i], y.iloc[te_i]

    evaluate("LinearRegression", y_te2, fit_lr(X_tr2, y_tr2).predict(X_te2))
    rf2 = RandomForestRegressor(n_estimators=400, random_state=42)
    rf2.fit(X_tr2, y_tr2)
    evaluate("RandomForest", y_te2, rf2.predict(X_te2))

    mt2 = fit_mixed_track(df_tr2)
    evaluate("MixedTrack (RE=0 on unseen)",
             y_te2, predict_mixed_track(mt2, df_te2))
    ms2, design2 = fit_mixed_stint(df_tr2, track_cols)
    evaluate("MixedStint (RE=0 on unseen)",
             y_te2, predict_mixed_stint(ms2, design2, df_te2, track_cols))

    print("\n" + "=" * 64)
    print("TYRE-AGE RESPONSE (Medium, ages 10-50)")
    print("=" * 64)
    probe_lap = df.sample(1, random_state=3).iloc[0]
    probe_track = probe_lap["track_name"]
    print(f"  track: {probe_track}")
    lr_full = fit_lr(X, y)
    mt_full = fit_mixed_track(df)
    ms_full, design_full = fit_mixed_stint(df, track_cols)
    print("  age :   Linear   MixedTrack   MixedStint")
    for age in (10, 20, 30, 40, 50):
        r = X.loc[probe_lap.name].copy()
        r["tyre_age"] = age
        lr_v = lr_full.predict([r])[0]
        mt_v = predict_mixed_track(
            mt_full, pd.DataFrame([{"tyre_compound": "Medium",
                                    "tyre_age": age,
                                    "track_name": probe_track}]))[0]
        ms_v = predict_mixed_stint(
            ms_full, design_full,
            pd.DataFrame([{"tyre_compound": "Medium", "tyre_age": age,
                           "track_name": probe_track}]), track_cols)[0]
        print(f"  age {age:>2} : {lr_v:7.2f}s  {mt_v:9.2f}s  {ms_v:9.2f}s")


if __name__ == "__main__":
    main()
