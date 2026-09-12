"""Shared synthetic fuel-load feature for F1 Telemetry & Race Strategy Platform.

NOTE: This is a synthetic, model-derived race-progress proxy feature and NOT
measured real-world fuel mass from telemetry sensors. Both FastF1 and Legacy
telemetry pipelines use this exact synthetic feature to maintain consistency
with the trained ML model.

Formula: max(0.0, 110.0 - lap_number * 2.0)
"""

STARTING_FUEL_KG = 110.0
FUEL_BURN_PER_LAP_KG = 2.0


def estimate_fuel_load(lap_number: int) -> float:
    """Return synthetic estimated fuel load feature for a non-negative lap number.

    Lap 0 represents the pre-race/start baseline estimate (110.0 kg). Lap 1 is 108.0 kg.
    Values are clamped at zero because the feature must never become negative.
    """
    if isinstance(lap_number, bool) or int(lap_number) != lap_number:
        raise ValueError("lap_number must be a whole number")

    lap_number = int(lap_number)
    if lap_number < 0:
        raise ValueError("lap_number must be non-negative")

    return max(0.0, STARTING_FUEL_KG - (lap_number * FUEL_BURN_PER_LAP_KG))

