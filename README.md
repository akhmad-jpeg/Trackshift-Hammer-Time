# 🏎️ F1 Race Strategy & Telemetry Analytics Platform

> Real-time telemetry capture, race-data import, performance analysis, and lap-time prediction for Formula 1 strategy work.

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![MySQL](https://img.shields.io/badge/MySQL-8.0-orange.svg)](https://www.mysql.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-green.svg)](https://scikit-learn.org/)

---

## 📋 Overview

A complete data pipeline for Formula 1 strategy work: capture live telemetry from F1 2018 (UDP, Legacy format), import real race data via FastF1, store everything in a normalized MySQL database, and serve a Flask dashboard with machine-learning lap-time models, head-to-head driver comparisons, and a fuel-neutral strategy advisor.

**At a glance:**
- 🎮 Real-time UDP telemetry capture with correct F1 2018 Legacy packet parsing (all 20-car packet offsets verified against the official spec)
- 🔴 Live-data dashboard cards that show *only* genuine live streams — with a LIVE TELEMETRY / STANDBY indicator and a configurable freshness window
- 🏁 Real race data import (FastF1) with pit-stop event reconciliation and race-control extraction (SC / VSC / red flag)
- 🤖 A model hierarchy: global model, one model per driver, and one model per (driver, season) — powering an apples-to-apples driver comparison
- 📊 Flask dashboard: lap progression, fuel-adjusted tyre degradation, speed distribution, strategy events, race-lap overlays, and a model pace-gap chart
- 🎯 Fuel-neutral strategy advisor that reasons about stay-out vs. pit scenarios without overstating tyre wear
- 🖱️ One-click Windows launchers for every pipeline step — setup, import, train, serve, capture, compare
- 🧪 193 unit tests, all mocked against the DB so CI needs no MySQL

---

## ✨ Features

### 🎮 Live telemetry capture (F1 2017/2018 Legacy UDP)

- Listens on UDP port 20777 and parses the F1 2018 Legacy packet format — lap timer, speed, throttle/brake, gear, RPM, DRS, in-pits flag, and the live tyre-compound byte (mapped onto the same tyre ENUM the FastF1 importer uses).
- Detects lap boundaries from the in-game timer reset, stamps each lap `captured_at`, samples telemetry at ~1 Hz, and inserts laps/telemetry/strategy events through a background DB worker queue with fail-fast liveness checks.
- **Strategy-event detection** from completed-lap signals: PitStop (tyre change or in-pits flag + slow lap), SafetyCar / VSC (lap-time **and** full-lap speed gates, so traffic laps stay silent), and RedFlag — written to `strategy_events`.
- **Live-data heartbeat**: prints a liveness line every `--heartbeat <seconds>` (default 5) — `LIVE` with pkt/s, lap and tyre while streaming, `STALLED` with the exact silence duration when the game pauses or the feed dies.
- **Stream-drop alerts**: when the heartbeat flips to STALLED, the CLI beeps and prints a bold-red `[ALERT] LIVE DATA DROPPED`; recovery prints a green `RESUMED`. Color + beep are TTY-only (no escape garbage in redirected logs) and can be disabled with `--no-alert`.
- Configurable per run through the launcher prompt or CLI flags: track alias, starting tyre, weather, heartbeat interval.

### 🔴 Live dashboard cards

- The five top cards — **CURRENT LAST LAP, CURRENT FASTEST, CURRENT TRACK, TYRE COMPOUND, LAP NUMBER** — show *exclusively* live data: laps stamped `captured_at` by the capture loop (or a same-day real-race import). Historical races can never appear there.
- A pulsing **LIVE TELEMETRY** badge replaces a dim **STANDBY** while a stream is active; when nothing has been captured within the window, every card resets to dashes so a stale "last race" is never shown.
- Freshness window defaults to 10 minutes and is configurable via the `LIVE_WINDOW_MINUTES` environment variable.
- Optional per-driver filter: `?driver=HAM` shows that driver's latest live lap (used by the dashboard driver selector).
- The separate Session Info Bar keeps showing the stored session you're reviewing — live cards and stored-session review never mix.

### 🏁 Race data import (FastF1)

- Import any session (practice / qualifying / race) for any driver and season, with tyre-compound normalization, fuel-load estimation, and telemetry sampling.
- Extracts race-control messages (safety car / VSC / red flag) and reconciles pit-stop events with actual tyre changes.
- **Same-day imports count as live data**: a race imported on the day it ran is stamped `captured_at`, so the live cards can show a real race as it happens; historical imports stay `captured_at = NULL` and never qualify as live.
- Batch mode (`import_f1_dataset.py`) ingests a whole driver's career across seasons in one run.

### 📊 Dashboard

Four tabs, all backed by the MySQL data:

1. **DASHBOARD** — per-session lap-time progression, fuel-adjusted tyre-degradation chart (pit in/out laps marked, warm-up laps shown but excluded from the trend fit so the cold-tyre spike is visible), speed distribution, strategy events, recent-sessions sidebar with a per-driver filter.
2. **⚡ LAP PREDICTOR** — interactive prediction on the trained model with explicit rejection of unseen tracks/tyres.
3. **🎯 STRATEGY ADVISOR** — fuel-neutral analysis of Stay Out vs. Pit (same compound, harder, softer, or rain) given current lap, tyre age, and race event; returns a recommendation with a plain-English reason.
4. **🏁 DRIVER COMPARISON** — two parts:
   - *Compare on a track*: pick a season → track → two or more drivers and overlay their **actual race laps** on one chart, with pit in/out markers and per-lap telemetry tooltips (avg/top speed, gear, RPM).
   - *Model pace gap*: pick two drivers (and optionally a season) and predict both drivers' models on every shared (track, tyre) at tyre ages 1–20. The delta (A − B) per track is shown as a color-coded bar chart and a full table. A **statistical-significance verdict** labels the headline gap — *significant / suggestive / within model noise* — so a gap smaller than the models' ~1 s uncertainty is not over-read.

### 🤖 Machine-learning model hierarchy

- **Global model** (`ml_models/best_model.pkl`) trained on all clean race laps.
- **Per-driver models** (`ml_models/drivers/<code>/`) — one per driver with ≥ 60 clean laps, trained on that driver's own laps.
- **Per-driver-per-year models** (`ml_models/drivers/<code>/<year>/`) — same pipeline on a single season, so the intercept absorbs that year's driver + car + regulations. The dashboard's season selector uses these for **same-year, apples-to-apples** comparisons, falling back to the aggregate model when a driver lacks that season.
- Shared cleaning pipeline: stint warm-up laps (cold tyres / race start) and SC / VSC / red-flag laps are excluded, and pit in/out laps are dropped via the reconciled strategy events.
- Driver comparisons use only tracks **and** tyres both drivers' models cover — never extrapolated.

### 🧹 Data quality tooling

- `cleanup_pit_events.py` audits pit-event consistency (spurious/phantom/missing events, zero-ms-lap validity), dry-run by default, `--apply` to repair.
- `stint_analysis.py` detrends each stint's fuel curve so genuine tyre wear is visible as positive deltas.
- `benchmark_models.py` reproduces the model-selection benchmark (random split, stint-grouped, unseen-track contracts).

---

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- MySQL 8.0+
- F1 2018 game (only for live telemetry capture)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/akhmad-jpeg/Motorsports-telemetry.git
cd Motorsports-telemetry/f1-stategy-platform
```

2. **Install dependencies** (use a virtualenv)
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

3. **Set up the database**
```sql
CREATE DATABASE f1_strategy;
USE f1_strategy;
SOURCE database/schema.sql;
```

4. **Configure MySQL credentials** — the app reads `DB_HOST`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`, `DB_PORT` from the environment (`scripts/config.py`). Defaults are `localhost` / `root` / `f1_strategy` / `3306`, but the password has **no** default: it starts as the placeholder `CHANGE_ME` and the app refuses to connect until you set `DB_PASSWORD` (e.g. `set DB_PASSWORD=yourpassword` on Windows, or `export DB_PASSWORD=yourpassword` on Linux/macOS).

### Usage

**One-click launchers (recommended)** — double-click any file in `launchers/`; each one handles the Python/venv plumbing itself:

| Step | Launcher |
|---|---|
| Install dependencies into `.venv` | `launchers\01_setup_dependencies.bat` |
| Import one race (interactive) | `launchers\02_import_race.bat` |
| Batch-import races/seasons | `launchers\03_import_dataset.bat` |
| Audit / repair pit events | `launchers\04_audit_pit_events.bat` |
| Train the lap-time models | `launchers\05_train_model.bat` |
| Start the dashboard | `launchers\06_run_server.bat` |
| Live UDP telemetry capture | `launchers\07_capture_telemetry.bat` |
| Interactive lap-time prediction | `launchers\08_predict_lap_times.bat` |
| Performance analysis reports | `launchers\09_analyze_performance.bat` |
| Model-selection benchmark | `launchers\10_benchmark_models.bat` |
| Run the test suite | `launchers\11_run_tests.bat` |
| Compare two drivers head-to-head | `launchers\12_compare_drivers.bat` |

Run them in numbered order the first time (01 → 06). The full table lives in `launchers/README.md`.

The same steps are available as plain Python commands (e.g. on non-Windows or for scripting):

**Live capture (F1 2018, Legacy UDP on port 20777):**
```bash
python scripts/capture_telemetry.py --track Spa --tyre Soft --weather Dry --heartbeat 5
```

**Web dashboard:**
```bash
python scripts/run_server.py        # production (Waitress)
# or, for development:
python scripts/dashboard.py          # debug mode, binds 127.0.0.1 only
```
Open http://localhost:5000.

**Import a real race (FastF1):**
```bash
python scripts/import_f1_race.py
```

**Analysis reports (CLI):**
```bash
python scripts/analyze_performance.py
```

**Retrain the models (global + per-driver + per-driver-per-year):**
```bash
python scripts/ml_lap_predictions.py
```

**Compare two drivers head-to-head (same-year models with fallback):**
```bash
python scripts/driver_comparison.py --driver-a HAM --driver-b VER --year 2021
```

---

## 🗺️ Architecture

```
F1 2018 (UDP 20777, Legacy) ──► capture_telemetry.py ──┐
FastF1 race data            ──► import_f1_race.py    ──┤
                                                        ▼
                                             MySQL (f1_strategy)
                                                        │
  ┌──────────────┬──────────────┬───────────────┬───────┼────────────┐
  ▼              ▼              ▼               ▼       ▼            ▼
dashboard.py  analyze_performance  ml_lap_predictions.py   cleanup_pit_events
(Flask UI)    (charts + reports)   (global + per-driver +  (pit-event audit)
                                   per-driver-per-year)
                                        │
                          driver_comparison.py ◄── per-driver models
                          (head-to-head + significance)
                                        │
                          predict_lap_times.py
                          (strategy advisor)
```

Key modules in `scripts/`:

| File | Purpose |
|---|---|
| `capture_telemetry.py` | Live UDP capture; parses F1 2018 Legacy packets, detects laps + strategy events, stamps `captured_at`, heartbeat + stream-drop alerts |
| `import_f1_race.py` / `import_f1_dataset.py` | FastF1 race import; tyre compounds, pit events, race-control extraction, same-day-live stamping, batch mode |
| `cleanup_pit_events.py` | Audit + repair tool: purge spurious pit events, insert missing ones, re-validate lap validity (dry-run by default, `--apply` to write) |
| `stint_analysis.py` | Per-stint detrending so tyre wear is visible despite fuel burn (shared by dashboard + CLI) |
| `dashboard.py` / `run_server.py` | Flask web app (dashboard, predictor, strategy advisor, driver comparison) and its production entry point (Waitress, clickable localhost link) |
| `driver_comparison.py` | Head-to-head driver comparison (CLI + API): shared (track, tyre) predictions, same-year models with aggregate fallback, significance verdict, chart export |
| `analyze_performance.py` | CLI charts and summary reports |
| `ml_lap_predictions.py` / `feature_pipeline.py` | Model training (global + per-driver + per-driver-per-year) and feature engineering / vector alignment |
| `predict_lap_times.py` | Interactive lap-time prediction / strategy advisor CLI |
| `benchmark_models.py` | Reproducible model-selection benchmark (random split, stint-grouped, unseen-track contracts) |
| `fuel_estimation.py` | Fuel-load estimation from telemetry |

---

## 🧠 Model notes

- **Algorithm:** LinearRegression on `tyre_age` + tyre-compound and track one-hot features, selected over RandomForest/GradientBoosting in a head-to-head benchmark. The trees' apparent edge came from stint leakage, and their tyre-age response is jagged in exactly the region the pit decision lives.
- **Performance (within-track split):** the global model reports MAE ≈ 1.77 s, RMSE ≈ 3.3 s, R² ≈ 0.92. Note R² is dominated by track intercepts — the within-track precision that matters for strategy is ~1 s. Season-separated per-driver models are markedly tighter: HAM 2021 ≈ 1.19 s, VER 2021 ≈ 1.09 s, HAM/VER 2020 ≈ 0.99 / 1.08 s.
- **Unseen tracks are rejected, not extrapolated:** the predictor refuses tracks absent from training with a clear message (generalizing a track's speed from a single session is not possible — R² ≤ 0 on unseen tracks for every model tried).
- **Fuel-neutral advice:** stay-out vs pit comparisons detrend the fuel-burn slope baked into `tyre_age` (`fuel_burn_rate`, written at training time), so the advisor doesn't overstate tyre wear. The cleaned data shows no net degradation after warm-up laps are excluded, which is why the advisor generally favors staying out.
- **Model hierarchy:** one model per driver (≥ 60 clean laps) plus one per (driver, season) — the season selector in the dashboard compares same-year models (apples-to-apples) and falls back to the aggregate when a driver lacks that season.
- **Significance labeling:** driver comparisons run a paired t-test over the per-track deltas and label the headline gap *significant / suggestive / within model noise*, so a gap smaller than the models' uncertainty (the 2021 HAM–VER gap of ~0.19 s has p ≈ 0.26) is presented honestly.

---

## 🧪 Testing

```bash
python -m unittest discover -s tests
```

193 tests covering the packet parser (against the real spec byte offsets), race-control extraction, stint analysis (incl. pit-lap chart markers), pit-event reconciliation, importer behavior (incl. same-day-live stamping), the dashboard API (incl. the live-lap freshness window), driver comparison (incl. year-model fallback and significance), the strategy advisor, the server startup banner, and capture liveness (heartbeat, alerts, lap validity). All DB access is mocked; CI runs the same command on Python 3.13.

---

## 📂 Project structure

```
f1-stategy-platform/
├── database/schema.sql        # MySQL schema (incl. composite indexes)
├── scripts/                   # capture, import, analysis, models, dashboard, comparison
│   └── templates/dashboard.html
├── launchers/                 # one-click .bat files for every pipeline step
├── tests/                     # 193 unit tests
├── requirements.txt           # pinned dependencies
└── .github/workflows/ci.yml   # CI: install + run tests
```

Generated at runtime (gitignored): `f1_cache/` (FastF1 cache), `ml_models/` (trained artifacts — global model plus `drivers/<code>/` and `drivers/<code>/<year>/` subfolders), `analysis/` (CLI report output), `.venv/`.

---

## 🛠️ Tech stack

- Python 3.13, Flask + Waitress, MySQL
- pandas, matplotlib, scikit-learn, statsmodels, joblib
- FastF1 (real race data), mysql-connector-python

---

## 📧 Contact

**Ayaan Ahmad** — ayaanakhmad@gmail.com
