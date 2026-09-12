"""Shared feature engineering and vector alignment pipeline for F1 Telemetry Platform."""

import pandas as pd


def preprocess_laps_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess raw database laps dataframe for ML training/prediction.

    1. Normalizes string casing for track_name and tyre_compound.
    2. Performs one-hot encoding for tyre_compound and track_name.

    NOTE: no fuel-load feature is derived here.  The legacy synthetic fuel
    estimate (max(0, 110 - 2 * lap_number)) is a deterministic function of
    lap_number and perfectly collinear with tyre_age inside every stint,
    which destabilised the linear coefficients (the old model "predicted"
    laps get faster as tyres age).  tyre_age carries the within-stint pace
    progression instead.
    """
    df = df.copy()
    if 'track_name' in df.columns:
        df['track_name'] = df['track_name'].str.strip().str.title()
    if 'tyre_compound' in df.columns:
        df['tyre_compound'] = df['tyre_compound'].str.strip()
    if 'lap_number' in df.columns:
        # Not a model feature — within a stint it is collinear with tyre_age.
        df = df.drop(columns=['lap_number'])

    df_encoded = pd.get_dummies(df, columns=['tyre_compound', 'track_name'], prefix=['tyre', 'track'])
    return df_encoded


def covered_tracks(feature_names: list) -> list:
    """Sorted list of track names the trained model can predict for."""
    return sorted(f.replace('track_', '') for f in feature_names if f.startswith('track_'))


def covered_tyres(feature_names: list) -> list:
    """Sorted list of tyre compounds the trained model can predict for."""
    return sorted(f.replace('tyre_', '') for f in feature_names
                  if f.startswith('tyre_') and f not in ('tyre_age', 'tyre_load'))


def _normalise_track_name(track_name: str) -> str:
    """Apply the same normalisation used during training (strip + title-case).

    Track names from the sessions API keep the raw FastF1 casing (e.g.
    "Circuit de Barcelona-Catalunya") while the model features are trained
    on the title-cased form ("Circuit De Barcelona-Catalunya") — both must
    resolve to the same feature.
    """
    return str(track_name).strip().title()


def validate_model_inputs(tyre_compound: str, track_name: str, feature_names: list) -> None:
    """Raise ValueError with a clear message when a tyre/track is not covered.

    The one-hot features only exist for values seen during training, so any
    unseen track or compound would otherwise produce an all-zero row and a
    silently meaningless prediction.
    """
    tyre_compound = str(tyre_compound).strip()
    track_name    = _normalise_track_name(track_name)

    tyre_feature  = f'tyre_{tyre_compound}'
    track_feature = f'track_{track_name}'

    if tyre_feature not in feature_names:
        available = covered_tyres(feature_names)
        raise ValueError(
            f"Unknown tyre compound '{tyre_compound}'. "
            f"Model was trained on: {', '.join(available) or '(none)'}."
        )
    if track_feature not in feature_names:
        available = covered_tracks(feature_names)
        raise ValueError(
            f"Unknown track '{track_name}'. "
            f"Model was trained on: {', '.join(available) or '(none)'}."
        )


def construct_prediction_input(tyre_age: float, lap_number: int, tyre_compound: str, track_name: str, feature_names: list) -> pd.DataFrame:
    """Construct a 1-row feature DataFrame aligned with expected feature_names order.

    lap_number is accepted for call compatibility but no fuel-load feature is
    derived from it — see preprocess_laps_dataframe().
    """
    tyre_compound = str(tyre_compound).strip()
    track_name    = _normalise_track_name(track_name)

    tyre_feature = f'tyre_{tyre_compound}'
    track_feature = f'track_{track_name}'

    input_data = pd.DataFrame(0, index=[0], columns=feature_names)
    if 'tyre_age' in input_data.columns:
        input_data['tyre_age'] = tyre_age
    if tyre_feature in input_data.columns:
        input_data[tyre_feature] = 1
    if track_feature in input_data.columns:
        input_data[track_feature] = 1

    return input_data
