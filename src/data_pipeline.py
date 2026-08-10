"""
F1 Telemetry + Weather Sensor Fusion Pipeline

Core reusable function: fetch_driver_lap_telemetry()
  - Loads a FastF1 session, finds a driver's fastest lap
  - Fuses telemetry (speed/RPM/throttle/brake/X/Y) with weather (nearest-aligned)
  - Returns a single DataFrame with metadata columns appended
"""

import pathlib
import fastf1
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".cache"
DATA_DIR = ROOT / "data"
CACHE_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# ── Enable caching (idempotent) ────────────────────────────────────────────
fastf1.Cache.enable_cache(str(CACHE_DIR))


def fetch_driver_lap_telemetry(
    year: int,
    gp: str,
    session_type: str,
    driver_code: str,
) -> pd.DataFrame:
    """
    Load the fastest lap for `driver_code` from the given session and return
    a fused DataFrame of telemetry + weather with metadata columns.

    Parameters
    ----------
    year         : e.g. 2025
    gp           : Grand Prix name recognised by FastF1, e.g. 'Italian'
    session_type : 'R', 'Q', 'FP1', etc.
    driver_code  : three-letter code, e.g. 'NOR'

    Returns
    -------
    pd.DataFrame with columns:
        Driver, Year, GrandPrix, LapNumber, LapTime,
        SessionTime_s, Distance, X, Y,
        Speed, RPM, Throttle, Brake,
        TrackTemp, AirTemp
    """
    print(f"\n── {year} {gp} GP [{session_type}] — {driver_code} ──")

    # Load session
    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=True, weather=True, laps=True)

    # Fastest lap
    fastest_lap = session.laps.pick_drivers(driver_code).pick_fastest()
    lap_number = int(fastest_lap["LapNumber"])
    lap_time = fastest_lap["LapTime"]
    lap_time_str = str(lap_time).split("days ")[-1]  # trim "0 days " prefix
    print(f"  Lap {lap_number}  |  {lap_time_str}")

    # ── Telemetry ────────────────────────────────────────────────────────────
    tel = fastest_lap.get_telemetry()
    tel = tel[["SessionTime", "Distance", "X", "Y",
               "Speed", "RPM", "Throttle", "Brake"]].copy()
    tel["Brake"] = tel["Brake"].astype(int)
    tel = tel.rename(columns={"SessionTime": "SessionTime_tel"})
    tel = tel.sort_values("SessionTime_tel").reset_index(drop=True)

    # ── Weather ──────────────────────────────────────────────────────────────
    weather = (
        session.weather_data[["Time", "TrackTemp", "AirTemp"]]
        .copy()
        .rename(columns={"Time": "SessionTime_wx"})
        .sort_values("SessionTime_wx")
        .reset_index(drop=True)
    )

    # ── Fuse via nearest-timestamp merge ─────────────────────────────────────
    fused = pd.merge_asof(
        tel, weather,
        left_on="SessionTime_tel",
        right_on="SessionTime_wx",
        direction="nearest",
    ).drop(columns=["SessionTime_wx"])

    fused["SessionTime_s"] = fused["SessionTime_tel"].dt.total_seconds()
    fused = fused.drop(columns=["SessionTime_tel"])

    # ── Attach metadata ──────────────────────────────────────────────────────
    fused.insert(0, "Driver",    driver_code)
    fused.insert(1, "Year",      year)
    fused.insert(2, "GrandPrix", gp)
    fused.insert(3, "LapNumber", lap_number)
    fused.insert(4, "LapTime",   lap_time_str)

    col_order = [
        "Driver", "Year", "GrandPrix", "LapNumber", "LapTime",
        "SessionTime_s", "Distance", "X", "Y",
        "Speed", "RPM", "Throttle", "Brake", "TrackTemp", "AirTemp",
    ]
    fused = fused[col_order]

    print(f"  Rows: {len(fused)}")
    return fused


# ── Standalone entrypoint (reproduces original NOR/Monza CSV) ───────────────
if __name__ == "__main__":
    df = fetch_driver_lap_telemetry(2025, "Italian", "R", "NOR")
    out = DATA_DIR / "2025_monza_NOR_fastest_lap_fused.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved → {out}")
