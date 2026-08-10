"""
F1 Telemetry + Weather Sensor Fusion Pipeline
Loads 2025 Monza race session via FastF1, extracts fastest-lap telemetry
for a chosen driver, fuses with weather data, and saves to /data.
"""

import os
import pathlib
import fastf1
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".cache"
DATA_DIR = ROOT / "data"
CACHE_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────
YEAR = 2025
GRAND_PRIX = "Italian"
SESSION_TYPE = "R"          # Race
DRIVER = "NOR"              # Lando Norris

# ── Enable caching ─────────────────────────────────────────────────────────
fastf1.Cache.enable_cache(str(CACHE_DIR))

# ── Load session ───────────────────────────────────────────────────────────
print(f"Loading {YEAR} {GRAND_PRIX} GP — session: {SESSION_TYPE} ...")
session = fastf1.get_session(YEAR, GRAND_PRIX, SESSION_TYPE)
session.load(telemetry=True, weather=True, laps=True)

# ── Fastest lap for chosen driver ──────────────────────────────────────────
print(f"Finding fastest lap for {DRIVER} ...")
driver_laps = session.laps.pick_drivers(DRIVER)
fastest_lap = driver_laps.pick_fastest()
print(f"  Fastest lap: lap {int(fastest_lap['LapNumber'])}  "
      f"time {fastest_lap['LapTime']}")

# ── Telemetry ───────────────────────────────────────────────────────────────
tel = fastest_lap.get_telemetry()
tel_cols = ["SessionTime", "Distance", "X", "Y", "Speed", "RPM", "Throttle", "Brake"]
# Brake may be bool or int; normalise to int
tel = tel[tel_cols].copy()
tel["Brake"] = tel["Brake"].astype(int)
tel = tel.rename(columns={"SessionTime": "SessionTime_tel"})
print(f"  Telemetry rows: {len(tel)}")

# ── Weather ─────────────────────────────────────────────────────────────────
weather = session.weather_data.copy()
# Weather timestamps are in SessionTime (timedelta)
weather_cols = ["Time", "TrackTemp", "AirTemp"]
weather = weather[weather_cols].rename(columns={"Time": "SessionTime_wx"})
print(f"  Weather rows: {len(weather)}")

# ── Align weather to telemetry via merge_asof ───────────────────────────────
# Both are timedelta-indexed; sort before merge
tel = tel.sort_values("SessionTime_tel").reset_index(drop=True)
weather = weather.sort_values("SessionTime_wx").reset_index(drop=True)

fused = pd.merge_asof(
    tel,
    weather,
    left_on="SessionTime_tel",
    right_on="SessionTime_wx",
    direction="nearest",
)
fused = fused.drop(columns=["SessionTime_wx"])

# Convert timedelta to total seconds for CSV readability
fused["SessionTime_s"] = fused["SessionTime_tel"].dt.total_seconds()
fused = fused.drop(columns=["SessionTime_tel"])

# Reorder
col_order = ["SessionTime_s", "Distance", "X", "Y", "Speed", "RPM",
             "Throttle", "Brake", "TrackTemp", "AirTemp"]
fused = fused[col_order]

print(f"  Fused rows: {len(fused)}  columns: {list(fused.columns)}")
print(fused.head(3).to_string())

# ── Save ────────────────────────────────────────────────────────────────────
out_path = DATA_DIR / f"{YEAR}_monza_{DRIVER}_fastest_lap_fused.csv"
fused.to_csv(out_path, index=False)
print(f"\nSaved → {out_path}")
