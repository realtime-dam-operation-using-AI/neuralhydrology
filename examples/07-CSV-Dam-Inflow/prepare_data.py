"""Build the ``hourlycsv`` data directory for the dam-inflow model comparison from the raw 202610151100 csv files.

Raw files (``data/YDDam/time_series/202610151100_YDDam_*.csv``):

* ``average_rf_{hourly,daily}_input_A.csv``   basin-average (Thiessen) rainfall           -> input set A
* ``rfs_{hourly,daily}_input_B.csv``          rainfall of the 7 gauging stations            -> input set B
* ``rfs_{hourly,daily}_input_C.csv``          A + B in one file                             -> input set C
* ``inflow_{hourly,daily}_target.csv``        dam inflow (m3/s)

Output layout (``data/YDDam_paper``, see :py:class:`neuralhydrology.datasetzoo.hourlycsv.HourlyCSV`)::

    time_series/input/hourly/YDDam_input.csv    rf_avg, rf_<station> x 7, rf_stn_mean, rf_avg_aux
    time_series/input/daily/YDDam_input.csv     same rainfall columns at daily resolution (MTS-LSTM only)
    time_series/target/hourly/YDDam_target.csv  inflow (m3/s), inflow_mm (mm/h, MC-LSTM only)
    time_series/target/daily/YDDam_target.csv   inflow (m3/s), inflow_mm (mm/d)

Run ``python prepare_data.py`` from this folder, or import :func:`build_dataset` from the notebooks.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE.parents[1] / "data" / "YDDam" / "time_series"
DATA_DIR = HERE.parents[1] / "data" / "YDDam_paper"
BASIN = "YDDam"
STAMP = "202610151100"

# Yongdam dam catchment area: sum of the 35 COSFIM sub-basin areas (specs/data/small_basins.csv) = 930.42 km2
BASIN_AREA_KM2 = 930.42
# m3/s -> mm/h : Q [m3/s] * 3600 [s/h] / (A [km2] * 1e6 [m2/km2]) * 1000 [mm/m]
M3S_TO_MMH = 3.6 / BASIN_AREA_KM2

STATIONS = ["9000028", "9000027", "9000023", "9000030", "9000029", "9000026", "9000025"]
STATION_COLS = [f"rf_{s}" for s in STATIONS]

INPUT_SETS = {
    "A": ["rf_avg"],
    "B": STATION_COLS,
    "C": ["rf_avg"] + STATION_COLS,
}
INPUT_SET_NAMES = {
    "A": "basin-average rainfall",
    "B": "station rainfall (7 gauges)",
    "C": "basin-average + station rainfall",
}


def read_raw(name: str) -> pd.DataFrame:
    """Read one raw csv (utf-8 with BOM) into a time-indexed DataFrame."""
    df = pd.read_csv(RAW_DIR / f"{STAMP}_{BASIN}_{name}.csv", encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def _rename_rainfall(df: pd.DataFrame) -> pd.DataFrame:
    """Map the raw column names to short, resolution-independent names (``rf_avg``, ``rf_<station>``)."""
    mapping = {}
    for col in df.columns:
        if col == "average_rainfall":
            mapping[col] = "rf_avg"
        elif "(" in col:
            station = col.split("(")[1].rstrip(")")
            # the daily file has a typo in one station id (90000230 instead of 9000023)
            if station == "90000230":
                station = "9000023"
            mapping[col] = f"rf_{station}"
    return df.rename(columns=mapping)


def load_inputs(frequency: str) -> pd.DataFrame:
    """Load input set C (A + B) at the given resolution and add the derived columns."""
    df = _rename_rainfall(read_raw(f"rfs_{frequency}_input_C"))
    df = df[["rf_avg"] + STATION_COLS]
    # arithmetic mean of the 7 stations: single mass input / extreme-label variable for the models that need one
    # scalar rainfall series in input set B (MC-LSTM, Exformer). Carries no information beyond set B.
    df["rf_stn_mean"] = df[STATION_COLS].mean(axis=1, skipna=True)
    # copy of rf_avg under a different name, so that MC-LSTM can use basin-average rainfall both as the
    # (un-normalized) mass input and as a (standardized) auxiliary gate input.
    df["rf_avg_aux"] = df["rf_avg"]
    return df


def load_targets(frequency: str) -> pd.DataFrame:
    """Load the inflow target at the given resolution and add the depth-unit version (mm/h or mm/d)."""
    df = read_raw(f"inflow_{frequency}_target").rename(columns={f"{frequency}_inflow": "inflow"})
    factor = M3S_TO_MMH * (24 if frequency == "daily" else 1)
    df["inflow_mm"] = df["inflow"] * factor
    return df[["inflow", "inflow_mm"]]


def check_consistency(verbose: bool = True) -> dict:
    """Cross-check the hourly and daily files (sum/mean of hourly values vs daily files)."""
    hourly_in, daily_in = load_inputs("hourly"), load_inputs("daily")
    hourly_t, daily_t = load_targets("hourly"), load_targets("daily")
    # the hourly stamps run from 01:00 to 24:00 (= next day 00:00); shift by one hour to aggregate per calendar day
    shifted = hourly_in.copy()
    shifted.index = shifted.index - pd.Timedelta(hours=1)
    daily_from_hourly = shifted[["rf_avg"] + STATION_COLS].resample("1D").sum(min_count=1)
    report = {}
    for col in ["rf_avg"] + STATION_COLS:
        pair = pd.concat([daily_from_hourly[col], daily_in[col]], axis=1).dropna()
        report[col] = float(pair.corr().iloc[0, 1])
    shifted_t = hourly_t.copy()
    shifted_t.index = shifted_t.index - pd.Timedelta(hours=1)
    pair = pd.concat([shifted_t["inflow"].resample("1D").mean(), daily_t["inflow"]], axis=1).dropna()
    report["inflow"] = float(pair.corr().iloc[0, 1])
    if verbose:
        print("correlation between daily file and daily aggregate of the hourly file:")
        for k, v in report.items():
            print(f"  {k:12s} {v:.4f}")
    return report


def build_dataset(data_dir: Path = DATA_DIR, verbose: bool = True) -> Path:
    """Write the csv files in the ``hourlycsv`` layout and return the data directory."""
    data_dir = Path(data_dir)
    files = {
        data_dir / "time_series" / "input" / "hourly" / f"{BASIN}_input.csv": load_inputs("hourly"),
        data_dir / "time_series" / "input" / "daily" / f"{BASIN}_input.csv": load_inputs("daily"),
        data_dir / "time_series" / "target" / "hourly" / f"{BASIN}_target.csv": load_targets("hourly"),
        data_dir / "time_series" / "target" / "daily" / f"{BASIN}_target.csv": load_targets("daily"),
    }
    for path, df in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, float_format="%.6g")
        if verbose:
            print(f"wrote {path.relative_to(data_dir.parent)}  rows={len(df):,}  cols={list(df.columns)}")
    (data_dir / f"{BASIN}_basin.txt").write_text(BASIN + "\n")
    return data_dir


if __name__ == "__main__":
    check_consistency()
    build_dataset()
