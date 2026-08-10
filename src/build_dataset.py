"""
Build multi-session telemetry dataset.

Sessions pulled:
  - 2025 Italian GP  (Monza)   : NOR + PIA  — teammate comparison, same car
  - 2025 Monaco GP              : NOR        — low-speed / technical circuit
  - 2025 Belgian GP  (Spa)     : NOR        — mixed high/low speed circuit

Output: data/telemetry_dataset.csv
"""

import pathlib
import sys
import pandas as pd

# Allow running from repo root or src/
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_pipeline import fetch_driver_lap_telemetry, DATA_DIR

SESSIONS = [
    # (year, gp,       session_type, driver)
    (2025, "Italian", "R", "NOR"),   # Monza — NOR
    (2025, "Italian", "R", "PIA"),   # Monza — PIA  (teammate)
    (2025, "Monaco",  "R", "NOR"),   # Monaco — low-speed
    (2025, "Belgian", "R", "NOR"),   # Spa — mixed
]

frames = []
for year, gp, stype, driver in SESSIONS:
    try:
        df = fetch_driver_lap_telemetry(year, gp, stype, driver)
        frames.append(df)
    except Exception as exc:
        print(f"  !! FAILED {year} {gp} {driver}: {exc}")

dataset = pd.concat(frames, ignore_index=True)

# ── Save ────────────────────────────────────────────────────────────────────
out_path = DATA_DIR / "telemetry_dataset.csv"
dataset.to_csv(out_path, index=False)
print(f"\nSaved → {out_path}  ({len(dataset):,} rows total)")

# ── Summary table ────────────────────────────────────────────────────────────
summary = (
    dataset.groupby(["Driver", "Year", "GrandPrix", "LapNumber", "LapTime"])
    .agg(
        Rows=("Distance", "count"),
        DistMax_m=("Distance", "max"),
        SpeedMax_kmh=("Speed", "max"),
        SpeedMean_kmh=("Speed", "mean"),
    )
    .reset_index()
)
summary["SpeedMax_kmh"]  = summary["SpeedMax_kmh"].round(1)
summary["SpeedMean_kmh"] = summary["SpeedMean_kmh"].round(1)
summary["DistMax_m"]     = summary["DistMax_m"].round(0).astype(int)

print("\n── Dataset Summary ─────────────────────────────────────────────────")
print(summary.to_string(index=False))
