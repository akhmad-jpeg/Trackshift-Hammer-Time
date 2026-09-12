from flask import Flask, render_template, jsonify, request
import os
import sys
import joblib
import json
import traceback
import datetime
from pathlib import Path
from fuel_estimation import estimate_fuel_load
from config import get_db_connection
from stint_analysis import detrend_laps
from feature_pipeline import (
    construct_prediction_input,
    covered_tracks,
    covered_tyres,
    validate_model_inputs,
)
import driver_comparison

app = Flask(__name__)

# ── ML model (loaded once at startup) ───────────────────────
# Model artifacts always live in <project root>/ml_models so the dashboard
# finds them regardless of the working directory it is launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / 'ml_models' / 'best_model.pkl'
FEATURES_PATH = PROJECT_ROOT / 'ml_models' / 'feature_names.pkl'

model = None
feature_names = []

# Fuel-burn rate: the pace effect the model bakes into tyre_age that is fuel
# (and track evolution), not tyre wear.  The strategy advisor detrends every
# scenario's predictions by this rate so stay-out vs pit comparisons are made
# on equal fuel footing — both scenarios burn the same fuel over the
# remaining laps, so crediting the stay-out scenario's higher ages with it
# would bias the comparison by |rate| * laps_rem * cur_age in its favour
# (the old advisor therefore ALWAYS said "Stay Out", regardless of tyre
# age).  Written at training time as min(0, tyre_age coefficient): the
# cleaned dataset shows no net wear after warm-up laps are excluded, so the
# whole negative age slope is treated as fuel; a positive slope (wear
# dominating) would be clamped to 0 and left fully in the comparison.
MODEL_INFO_PATH = PROJECT_ROOT / 'ml_models' / 'model_info.json'
fuel_burn_rate = 0.0

if MODEL_PATH.exists() and FEATURES_PATH.exists():
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    print(f"[INFO] Model loaded: {type(model).__name__} | {len(feature_names)} features")
    print(f"[INFO] Covers {len(covered_tracks(feature_names))} tracks, "
          f"{len(covered_tyres(feature_names))} tyres")
    if MODEL_INFO_PATH.exists():
        try:
            info = json.loads(MODEL_INFO_PATH.read_text(encoding='utf-8'))
            fuel_burn_rate = min(0.0, float(info.get('fuel_burn_rate', 0.0)))
            print(f"[INFO] Fuel burn rate (detrended from advisor comparisons): "
                  f"{fuel_burn_rate:+.4f} s/lap")
        except Exception:
            traceback.print_exc()
else:
    print("[WARNING] Model not found — run scripts/ml_lap_predictions.py first")


# PAGE ROUTES
@app.route('/')
def index():
    return render_template('dashboard.html')


# SESSION / TELEMETRY API
@app.route('/api/sessions')
def get_sessions():
    conn = None
    cursor = None
    try:
        # Pagination: ?limit=&offset= (defaults keep the historical 50-row cap;
        # limit is clamped to [1, 500]).  Optional ?driver=<CODE> filters to
        # one driver's sessions (used by the dashboard driver selector).
        limit = max(1, min(request.args.get('limit', default=50, type=int), 500))
        offset = max(0, request.args.get('offset', default=0, type=int))
        driver = request.args.get('driver', '').strip().upper()
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        base_sql = """
            SELECT
                s.session_id,
                s.track_name,
                s.session_type,
                s.weather,
                s.date,
                d.driver_code,
                COUNT(l.lap_id) AS total_laps,
                MIN(CASE WHEN l.is_valid = 1 AND l.lap_time_ms > 0
                         THEN l.lap_time_ms END) / 1000 AS fastest_lap
            FROM sessions s
            LEFT JOIN drivers d ON s.driver_id = d.driver_id
            LEFT JOIN laps l ON s.session_id = l.session_id
        """
        if driver:
            cursor.execute(base_sql + """
                WHERE d.driver_code = %s
                GROUP BY s.session_id, s.track_name, s.session_type, s.weather,
                         s.date, d.driver_code
                ORDER BY s.date DESC, s.session_id DESC
                LIMIT %s OFFSET %s
            """, (driver, limit, offset))
        else:
            cursor.execute(base_sql + """
                GROUP BY s.session_id, s.track_name, s.session_type, s.weather,
                         s.date, d.driver_code
                ORDER BY s.date DESC, s.session_id DESC
                LIMIT %s OFFSET %s
            """, (limit, offset))
        sessions = cursor.fetchall()

        for s in sessions:
            if s['date']:
                s['date'] = s['date'].strftime('%Y-%m-%d')
            s['fastest_lap'] = float(s['fastest_lap']) if s['fastest_lap'] else None

        return jsonify(sessions)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if cursor:
            try: cursor.close()
            except: pass
        if conn:
            try: conn.close()
            except: pass


@app.route('/api/session/<int:session_id>/laps')
def get_session_laps(session_id):
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT
                l.lap_number,
                l.lap_time_ms / 1000.0 AS lap_time,
                l.tyre_compound,
                l.tyre_age,
                l.fuel_load,
                l.is_valid,
                MAX(CASE WHEN se.event_type = 'PitStop' THEN 1 ELSE 0 END) AS has_pit_stop
            FROM laps l
            LEFT JOIN strategy_events se ON l.lap_id = se.lap_id
            WHERE l.session_id = %s AND l.lap_time_ms > 0
            GROUP BY l.lap_id, l.lap_number, l.lap_time_ms, l.tyre_compound, l.tyre_age, l.fuel_load, l.is_valid
            ORDER BY l.lap_number
        """, (session_id,))
        laps = cursor.fetchall()

        for lap in laps:
            lap['lap_time'] = float(lap['lap_time']) if lap['lap_time'] is not None else None
            lap['fuel_load'] = float(lap['fuel_load']) if lap['fuel_load'] is not None else 0.0

        return jsonify(laps)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if cursor:
            try: cursor.close()
            except: pass
        if conn:
            try: conn.close()
            except: pass


@app.route('/api/session/<int:session_id>/tyre-degradation')
def get_tyre_degradation(session_id):
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT 
                l.lap_number,
                l.lap_time_ms / 1000.0 AS lap_time,
                l.lap_time_ms / 1000.0 AS avg_lap_time,
                l.tyre_compound,
                l.tyre_age,
                l.is_valid,
                MAX(CASE WHEN se.event_type = 'PitStop' THEN 1 ELSE 0 END) AS has_pit_stop
            FROM laps l
            LEFT JOIN strategy_events se ON l.lap_id = se.lap_id
                AND se.event_type = 'PitStop'
                AND (se.duration_sec IS NULL OR se.duration_sec >= 15)
            WHERE l.session_id = %s AND l.lap_time_ms > 0
            GROUP BY l.lap_id, l.lap_number, l.lap_time_ms, l.tyre_compound, l.tyre_age, l.is_valid
            ORDER BY l.lap_number
        """, (session_id,))
        deg = cursor.fetchall()
        for d in deg:
            d['lap_time'] = float(d['lap_time']) if d.get('lap_time') is not None else None
            d['avg_lap_time'] = float(d['avg_lap_time']) if d.get('avg_lap_time') is not None else d['lap_time']
        # Fuel-adjusted degradation: stint_delta is each lap's time relative
        # to its own stint's pace line (fuel burn removed).
        deg = detrend_laps(deg)
        return jsonify(deg)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if cursor:
            try: cursor.close()
            except: pass
        if conn:
            try: conn.close()
            except: pass




# How recent a captured lap must be to count as "live".  The top stat
# cards show genuine live telemetry only — game UDP capture, a live
# feed, or a same-day real-race import (import_f1_race stamps captured_at
# when the race ran today).  Older historical imports keep captured_at
# NULL and never qualify.
# Overridable via LIVE_WINDOW_MINUTES (e.g. 30 for long pauses).
def _live_window():
    try:
        minutes = float(os.environ.get('LIVE_WINDOW_MINUTES', '10'))
        if minutes <= 0:
            raise ValueError('must be positive')
        return datetime.timedelta(minutes=minutes)
    except (TypeError, ValueError):
        print(f"[WARNING] Invalid LIVE_WINDOW_MINUTES="
              f"{os.environ.get('LIVE_WINDOW_MINUTES')!r} — using default 10 minutes")
        return datetime.timedelta(minutes=10)


LIVE_WINDOW = _live_window()


@app.route('/api/latest-lap')
def get_latest_lap():
    # Optional ?driver=<CODE> shows that driver's latest live lap instead of
    # the most recent live lap in the whole database (dashboard selector).
    driver = request.args.get('driver', '').strip().upper()
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cutoff = (datetime.datetime.now() - LIVE_WINDOW).strftime('%Y-%m-%d %H:%M:%S')
        base_sql = """
            SELECT 
                l.lap_time_ms / 1000.0 AS lap_time,
                l.lap_number,
                l.tyre_compound,
                l.tyre_age,
                l.session_id,
                s.track_name
            FROM laps l
            JOIN sessions s ON l.session_id = s.session_id
        """
        if driver:
            cursor.execute(base_sql + """
                JOIN drivers d ON l.driver_id = d.driver_id
                WHERE d.driver_code = %s AND l.lap_time_ms > 0
                  AND l.captured_at >= %s
                ORDER BY l.lap_id DESC
                LIMIT 1
            """, (driver, cutoff))
        else:
            cursor.execute(base_sql + """
                WHERE l.lap_time_ms > 0 AND l.captured_at >= %s
                ORDER BY l.lap_id DESC
                LIMIT 1
            """, (cutoff,))
        lap = cursor.fetchone()
        if lap:
            lap['lap_time'] = float(lap['lap_time']) if lap['lap_time'] else None
            lap['live'] = True
            return jsonify(lap)
        # No lap captured in the live window: the frontend shows dashes.
        return jsonify({})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if cursor:
            try: cursor.close()
            except: pass
        if conn:
            try: conn.close()
            except: pass


# DASHBOARD DRIVER LIST
@app.route('/api/drivers/list')
def get_drivers_list():
    """Distinct drivers that have sessions in the database.

    Feeds the dashboard driver selector; returns per-driver session/lap
    counts and the most recent session date.
    """
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT
                d.driver_id,
                d.driver_code,
                d.driver_name,
                COUNT(DISTINCT s.session_id) AS sessions,
                COUNT(l.lap_id) AS laps,
                MAX(s.date) AS last_seen
            FROM drivers d
            JOIN sessions s ON s.driver_id = d.driver_id
            LEFT JOIN laps l ON l.session_id = s.session_id
            GROUP BY d.driver_id, d.driver_code, d.driver_name
            ORDER BY d.driver_code
        """)
        drivers = []
        for r in cursor.fetchall():
            drivers.append({
                "driver_id": r['driver_id'],
                "code": r['driver_code'],
                "name": r['driver_name'],
                "sessions": r['sessions'],
                "laps": r['laps'],
                "last_seen": r['last_seen'].strftime('%Y-%m-%d') if r['last_seen'] else None,
            })
        resp = jsonify({"drivers": drivers})
        resp.headers['Cache-Control'] = 'no-store'
        return resp
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if cursor:
            try: cursor.close()
            except: pass
        if conn:
            try: conn.close()
            except: pass


# PREDICTOR API
@app.route('/api/predict/options')
def get_predict_options():
    if not feature_names:
        return jsonify({"error": "Model not loaded"}), 500

    tyres  = []
    tracks = []

    for feature in feature_names:
        if feature.startswith('tyre_'):
            name = feature.replace('tyre_', '')
            if name not in ['age', 'load']:
                tyre_type = 'INTERMEDIATE' if name == 'Intermediate' else ('WET' if name == 'Wet' else 'DRY')
                tyres.append({"name": name, "type": tyre_type})
        elif feature.startswith('track_'):
            tracks.append(feature.replace('track_', ''))

    return jsonify({"tyres": tyres, "tracks": tracks})


@app.route('/api/predict', methods=['POST'])
def predict_lap():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    try:
        body          = request.get_json()
        tyre_age      = float(body['tyre_age'])
        lap_number    = int(body['lap_number'])
        fuel_load     = estimate_fuel_load(lap_number)
        tyre_compound = body['tyre_compound']
        track_name    = body['track_name']

        # Reject unseen tracks/tyres with a clear message instead of
        # silently predicting on an all-zero feature row.
        try:
            validate_model_inputs(tyre_compound, track_name, feature_names)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        input_data = construct_prediction_input(
            tyre_age=tyre_age,
            lap_number=lap_number,
            tyre_compound=tyre_compound,
            track_name=track_name,
            feature_names=feature_names
        )

        predicted_time = float(model.predict(input_data)[0])
        minutes = int(predicted_time // 60)
        seconds = predicted_time % 60

        return jsonify({
            "predicted_time": predicted_time,
            "formatted":      f"{minutes}:{seconds:06.3f}",
            "track":          track_name,
            "tyre_compound":  tyre_compound,
            "tyre_age":       tyre_age,
            "lap_number":     lap_number,
            "fuel_load":      fuel_load
        })
    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# STRATEGY ADVISOR API
def _stint_time(tyre, start_age, laps, track, start_lap_number):
    """Sum predicted lap times across a full stint.

    Predictions are detrended by the fuel-burn rate before summing so every
    scenario is evaluated on equal fuel footing.  The model's tyre_age
    coefficient conflates fuel burn (laps genuinely get faster as the car
    lightens) with tyre wear; fuel is burned identically by the stay-out and
    pit scenarios over the same remaining race laps, so crediting the
    stay-out scenario's higher ages with it would hand it a phantom
    |fuel_burn_rate| * laps_rem * cur_age seconds of advantage — the reason
    the old advisor always answered "Stay Out".  Detrending (adding
    |fuel_burn_rate| * age back to each prediction, since fuel_burn_rate <= 0)
    leaves only compound differences, pit loss and any genuine wear.
    """
    tf = f'tyre_{tyre}'
    tkf = f'track_{track}'
    if tf not in feature_names or tkf not in feature_names or laps <= 0:
        return 0.0
    total = 0.0
    for i in range(laps):
        age = start_age + i
        row = construct_prediction_input(
            tyre_age=age,
            lap_number=start_lap_number + i,
            tyre_compound=tyre,
            track_name=track,
            feature_names=feature_names
        )
        predicted = float(model.predict(row)[0])
        # Remove the fuel (non-wear) component of the age effect.
        total += predicted - fuel_burn_rate * age
    return total


def _reason(event, strategy, laps_rem, tyre_age, tyre, pit_loss):
    pit = strategy['pit_stops'] > 0
    if event == 'VSC':
        return (f"VSC cuts pit loss to {pit_loss}s. Fresh tyres worth it with {laps_rem} laps left." if pit
                else f"Tyres still usable (age {tyre_age}). Save stop for later.")
    if event == 'SafetyCar':
        return ("Safety Car = free pit stop. Take it NOW!" if pit
                else f"Only {laps_rem} laps left — not worth stopping.")
    if event == 'Rain':
        if 'Intermediate' in strategy['option'] or 'Wet' in strategy['option']:
            return "Track is wet — box for rain tyres immediately!"
        return "Rain just started. Track still has grip — monitor before committing."
    if event == 'Crash':
        return ("Incident likely brings VSC/SC — cheap pit window incoming." if pit
                else "Wait for VSC/SC confirmation before pitting.")
    if strategy['option'].startswith('Stay'):
        if tyre_age >= 18:
            return (f"Fuel-neutral check: after removing the fuel-burn effect "
                    f"baked into tyre_age, the model shows no degradation on "
                    f"{tyre} at age {tyre_age} — a fresh set would only add the "
                    f"{pit_loss}s pit loss.")
        return (f"Fresh tyres would gain less than the {pit_loss}s pit loss "
                f"with {laps_rem} laps to go.")
    return "Fastest strategy based on fuel-neutral model predictions."


@app.route('/api/strategy/analyze', methods=['POST'])
def analyze_strategy():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    try:
        body         = request.get_json()
        cur_lap      = int(body['current_lap'])
        total_laps   = int(body['total_laps'])
        cur_tyre     = body['current_tyre']
        cur_age      = int(body['current_age'])
        track        = body['track']
        event        = body['event_type']

        # Reject unseen tracks/tyres explicitly — otherwise every strategy
        # totals 0.0 and the advisor silently answers "Stay Out".
        try:
            validate_model_inputs(cur_tyre, track, feature_names)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        laps_rem  = max(1, total_laps - cur_lap)
        pit_loss  = 15 if event in ['VSC', 'SafetyCar'] else 25

        strategies = []

        # Stay out
        t = _stint_time(cur_tyre, cur_age, laps_rem, track, cur_lap)
        if t > 0:
            strategies.append({
                "option":      "Stay Out",
                "description": f"Continue on {cur_tyre} (age {cur_age})",
                "total_time":  t,
                "pit_stops":   0,
                "risk":        "Medium" if cur_age > 15 else "Low"
            })

        # Pit — same compound
        t = _stint_time(cur_tyre, 0, laps_rem, track, cur_lap) + pit_loss
        if t > pit_loss:
            strategies.append({
                "option":      f"Pit — Fresh {cur_tyre}",
                "description": f"New {cur_tyre} tyres (+{pit_loss}s pit)",
                "total_time":  t,
                "pit_stops":   1,
                "risk":        "Low"
            })

        # Pit — harder compound (Supports both Modern F1 and Legacy compound progressions)
        #
        # Only offered when the model actually shows tyre degradation.  The
        # fuel-neutral detrending treats the whole negative age slope as fuel,
        # so fuel_burn_rate < 0 means the cleaned data shows no net wear — and
        # a "more durable" harder compound has no durability benefit, only a
        # (usually slower) intercept.  fuel_burn_rate == 0 keeps the full wear
        # effect in the comparison, making the durability trade-off real.
        harder = {
            'Hypersoft': 'Ultrasoft',
            'Ultrasoft': 'Supersoft',
            'Supersoft': 'Soft',
            'Soft': 'Medium',
            'Medium': 'Hard'
        }
        if cur_tyre in harder and fuel_burn_rate == 0.0:
            alt = harder[cur_tyre]
            if f'tyre_{alt}' in feature_names and laps_rem > 8:
                t = _stint_time(alt, 0, laps_rem, track, cur_lap) + pit_loss
                if t > pit_loss:
                    strategies.append({
                        "option":      f"Pit — Switch to {alt}",
                        "description": f"More durable compound (+{pit_loss}s pit estimate)",
                        "total_time":  t,
                        "pit_stops":   1,
                        "risk":        "Medium"
                    })

        # Pit — softer compound
        softer = {
            'Hard': 'Medium',
            'Medium': 'Soft',
            'Soft': 'Supersoft',
            'Supersoft': 'Ultrasoft',
            'Ultrasoft': 'Hypersoft'
        }
        if cur_tyre in softer and laps_rem <= 15:
            alt = softer[cur_tyre]
            # Only recommend if compound exists in the trained model's feature set
            if f'tyre_{alt}' in feature_names:
                t = _stint_time(alt, 0, laps_rem, track, cur_lap) + pit_loss
                if t > pit_loss:
                    strategies.append({
                        "option":      f"Pit — Switch to {alt}",
                        "description": f"Attack mode — softer compound (+{pit_loss}s pit estimate)",
                        "total_time":  t,
                        "pit_stops":   1,
                        "risk":        "High"
                    })


        # Rain tyres
        if event == 'Rain':
            for rain in ['Intermediate', 'Wet']:
                if f'tyre_{rain}' in feature_names:
                    t = _stint_time(rain, 0, laps_rem, track, cur_lap) + pit_loss
                    if t > pit_loss:
                        strategies.append({
                            "option":      f"Pit — {rain}",
                            "description": f"Switch to {rain} for wet conditions",
                            "total_time":  t,
                            "pit_stops":   1,
                            "risk":        "Low"
                        })

        strategies.sort(key=lambda x: x['total_time'])

        best = strategies[0] if strategies else None
        recommendation = {
            "action": best['option'] if best else "No data",
            "reason": _reason(event, best, laps_rem, cur_age, cur_tyre, pit_loss) if best else "Not enough model data."
        }

        return jsonify({
            "event":             event,
            "current_situation": {"lap": cur_lap, "laps_remaining": laps_rem, "tyre": cur_tyre, "tyre_age": cur_age},
            "strategies":        strategies,
            "recommendation":    recommendation,
            "fuel_burn_rate":    fuel_burn_rate
        })

    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# DRIVER COMPARISON API (race data)
#
# The user flow: pick a season -> pick a track raced that season -> pick
# two or more drivers -> overlay their actual race laps on one chart.

def _sessions_on_track(cursor, year, track):
    """Best session per driver for (year, track) from the database.

    A driver can have several sessions on the same track in one season
    (re-imports create duplicate sessions).  For each driver we keep the
    session with the most timed laps, tie-broken by latest date, so the
    comparison always uses one well-defined race per driver.
    """
    cursor.execute("""
        SELECT d.driver_code, d.driver_name, s.session_id,
               s.session_type, s.date,
               COUNT(l.lap_id) AS laps
        FROM sessions s
        JOIN drivers d ON s.driver_id = d.driver_id
        LEFT JOIN laps l ON l.session_id = s.session_id AND l.lap_time_ms > 0
        WHERE YEAR(s.date) = %s AND s.track_name = %s
        GROUP BY d.driver_code, d.driver_name, s.session_id, s.session_type, s.date
        ORDER BY d.driver_code, s.date DESC
    """, (year, track))
    best = {}
    for row in cursor.fetchall():
        code = row['driver_code']
        key = (row['laps'], row['date'] or datetime.date.min, row['session_id'])
        cur = best.get(code)
        if cur is None or key > cur[0]:
            best[code] = (key, row)
    return {code: row for code, (_, row) in best.items()}


@app.route('/api/comparison/years')
def comparison_years():
    """Seasons that have timed laps in the database (newest first)."""
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT DISTINCT YEAR(s.date) AS year
            FROM sessions s
            JOIN laps l ON l.session_id = s.session_id
            WHERE s.date IS NOT NULL AND l.lap_time_ms > 0
            ORDER BY year DESC
        """)
        years = [r['year'] for r in cursor.fetchall()]
        return jsonify({"years": years})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if cursor:
            try: cursor.close()
            except: pass
        if conn:
            try: conn.close()
            except: pass


@app.route('/api/comparison/tracks')
def comparison_tracks():
    """Tracks with timed laps in a given season."""
    year = request.args.get('year', type=int)
    if not year:
        return jsonify({"error": "year query param required"}), 400
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT DISTINCT s.track_name
            FROM sessions s
            JOIN laps l ON l.session_id = s.session_id
            WHERE YEAR(s.date) = %s AND s.track_name IS NOT NULL
              AND l.lap_time_ms > 0
            ORDER BY s.track_name
        """, (year,))
        tracks = [r['track_name'] for r in cursor.fetchall()]
        return jsonify({"tracks": tracks})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if cursor:
            try: cursor.close()
            except: pass
        if conn:
            try: conn.close()
            except: pass


@app.route('/api/comparison/drivers')
def comparison_drivers():
    """Drivers (best session each) that raced a track in a season."""
    year = request.args.get('year', type=int)
    track = request.args.get('track', '').strip()
    if not year or not track:
        return jsonify({"error": "year and track query params required"}), 400
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        best = _sessions_on_track(cursor, year, track)
        drivers = []
        for code in sorted(best):
            row = best[code]
            drivers.append({
                "code": code,
                "name": row['driver_name'],
                "session_id": row['session_id'],
                "session_type": row['session_type'],
                "date": row['date'].strftime('%Y-%m-%d') if row['date'] else None,
                "laps": row['laps'],
            })
        return jsonify({"drivers": drivers})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if cursor:
            try: cursor.close()
            except: pass
        if conn:
            try: conn.close()
            except: pass


@app.route('/api/comparison/race')
def comparison_race():
    """Actual race laps for two or more drivers on a track in a season.

    drivers is a comma-separated list of driver codes (e.g. VER,LEC).
    Each driver's laps come from their best session on that track/season.
    """
    year = request.args.get('year', type=int)
    track = request.args.get('track', '').strip()
    codes = [c.strip().upper() for c in request.args.get('drivers', '').split(',') if c.strip()]
    if not year or not track or not codes:
        return jsonify({"error": "year, track and drivers query params required"}), 400
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        best = _sessions_on_track(cursor, year, track)
        result = {}
        missing = []
        for code in codes:
            if code not in best:
                missing.append(code)
                continue
            row = best[code]
            cursor.execute("""
                SELECT
                    l.lap_id,
                    l.lap_number,
                    l.lap_time_ms / 1000.0 AS lap_time,
                    l.tyre_compound,
                    l.tyre_age,
                    l.is_valid,
                    MAX(CASE WHEN se.event_type = 'PitStop' THEN 1 ELSE 0 END) AS has_pit_stop
                FROM laps l
                LEFT JOIN strategy_events se ON l.lap_id = se.lap_id
                WHERE l.session_id = %s AND l.lap_time_ms > 0
                GROUP BY l.lap_id, l.lap_number, l.lap_time_ms, l.tyre_compound,
                         l.tyre_age, l.is_valid
                ORDER BY l.lap_number
            """, (row['session_id'],))
            laps = []
            lap_ids = []
            for lap in cursor.fetchall():
                lap_ids.append(lap['lap_id'])
                laps.append({
                    "lap_id": lap['lap_id'],
                    "lap_number": int(lap['lap_number']),
                    "lap_time": float(lap['lap_time']),
                    "tyre_compound": lap['tyre_compound'],
                    "tyre_age": int(lap['tyre_age']) if lap['tyre_age'] is not None else None,
                    "is_valid": bool(lap['is_valid']),
                    "has_pit_stop": bool(lap['has_pit_stop']),
                })

            # Per-lap telemetry aggregates (speed / gear / RPM) for the
            # chart tooltip.  Both live capture and FastF1 imports (sampled
            # telemetry) populate the table; laps without rows simply carry
            # no 'telemetry' key.
            telemetry = {}
            if lap_ids:
                placeholders = ','.join(['%s'] * len(lap_ids))
                cursor.execute(f"""
                    SELECT t.lap_id,
                           AVG(t.speed) AS avg_speed,
                           MAX(t.speed) AS top_speed,
                           AVG(t.gear) AS avg_gear,
                           AVG(t.rpm) AS avg_rpm
                    FROM telemetry t
                    WHERE t.lap_id IN ({placeholders})
                    GROUP BY t.lap_id
                """, lap_ids)
                for t in cursor.fetchall():
                    telemetry[t['lap_id']] = t

            for lap in laps:
                lap_id = lap.pop('lap_id')
                agg = telemetry.get(lap_id)
                if agg is not None:
                    lap['telemetry'] = {
                        "avg_speed": int(round(float(agg['avg_speed']))) if agg['avg_speed'] is not None else None,
                        "top_speed": int(round(float(agg['top_speed']))) if agg['top_speed'] is not None else None,
                        "avg_gear": round(float(agg['avg_gear']), 1) if agg['avg_gear'] is not None else None,
                        "avg_rpm": int(round(float(agg['avg_rpm']))) if agg['avg_rpm'] is not None else None,
                    }
            result[code] = {
                "name": row['driver_name'],
                "session_id": row['session_id'],
                "session_type": row['session_type'],
                "date": row['date'].strftime('%Y-%m-%d') if row['date'] else None,
                "laps": laps,
            }
        payload = {"drivers": result, "year": year, "track": track}
        if missing:
            payload["missing"] = missing
        return jsonify(payload)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if cursor:
            try: cursor.close()
            except: pass
        if conn:
            try: conn.close()
            except: pass


# DRIVER COMPARISON API (models)
@app.route('/api/drivers')
def get_drivers():
    """List per-driver models available for head-to-head comparison."""
    try:
        # no-store: a retrain adds year models, and a browser must never
        # reuse a stale model list across page loads.
        resp = jsonify(driver_comparison.list_driver_models())
        resp.headers['Cache-Control'] = 'no-store'
        return resp
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/drivers/compare')
def compare_drivers_api():
    """Head-to-head: same (track, tyre, age) predicted by each driver's model.

    Optional ?year=<SEASON> compares the same-season per-driver-per-year
    models (apples-to-apples); when a driver has no model for that season
    the code falls back to their aggregate all-seasons model and the
    payload's summary.year_used is None.
    """
    try:
        code_a = request.args.get('driver_a', '').strip()
        code_b = request.args.get('driver_b', '').strip()
        if not code_a or not code_b:
            return jsonify({"error": "driver_a and driver_b query params required"}), 400
        year = request.args.get('year', type=int)
        if 'year' in request.args and year is None:
            return jsonify({"error": "year must be an integer season (e.g. 2021)"}), 400
        resp = jsonify(driver_comparison.compare_drivers(code_a, code_b, year=year))
        resp.headers['Cache-Control'] = 'no-store'
        return resp
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def clickable(url, text=None):
    if text is None:
        text = url
    return f"\033]8;;{url}\033\\{text}\033]8;;\033\\"


if __name__ == '__main__':
    # Windows consoles default to cp1252/cp437 which cannot encode the 🌐
    # banner emoji — force UTF-8 like predict_lap_times.py does.
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    print("=" * 60)
    print("F1 DIGITAL PIT WALL — Starting")
    print("=" * 60)
    print(f"\n🌐  {clickable('http://localhost:5000')}")
    print("\nPress Ctrl+C to stop\n")
    # Loopback only: the Werkzeug debug console is remote code execution if
    # this dev server is reachable off-box.  Serve on a network interface
    # via run_server.py (Waitress, debug off) instead.
    app.run(debug=True, host='127.0.0.1', port=5000)
