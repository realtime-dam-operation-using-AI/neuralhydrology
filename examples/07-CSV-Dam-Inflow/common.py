"""Shared helpers for the dam-inflow model comparison notebooks (examples/07-CSV-Dam-Inflow).

Everything that the per-model notebooks and the two comparison notebooks have in common lives here, so that all
models are trained, evaluated and plotted in exactly the same way:

* run management (configs, run directories, training, best epoch, timing),
* evaluation (hourly predictions for train/validation/test, metrics in m3/s for every model),
* flood-event / peak analysis,
* publication-style plotting (large fonts, consistent colours per model and line styles per input set),
* result persistence (``results/<model>_<set>/``) used by the comparison notebooks.
"""
from __future__ import annotations

import json
import logging
import pickle
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import yaml

from neuralhydrology.evaluation import metrics as nh_metrics
from neuralhydrology.modelzoo import get_model
from neuralhydrology.utils.config import Config

from make_configs import COMMON, MODEL_ORDER, MODEL_SPECS
from prepare_data import BASIN, BASIN_AREA_KM2, DATA_DIR, INPUT_SET_NAMES, INPUT_SETS, M3S_TO_MMH, STATION_COLS

HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE / "configs"
RUNS_DIR = HERE / "runs"
RESULTS_DIR = HERE / "results"
FIGURES_DIR = HERE / "figures"

PERIODS = ("train", "validation", "test")
PERIOD_RANGE = {
    "train": (COMMON["train_start_date"], COMMON["train_end_date"]),
    "validation": (COMMON["validation_start_date"], COMMON["validation_end_date"]),
    "test": (COMMON["test_start_date"], COMMON["test_end_date"]),
}
PERIOD_LABEL = {p: f"{p} ({a[-4:]}-{b[-4:]})" for p, (a, b) in PERIOD_RANGE.items()}
SETS = list(INPUT_SETS)
MODEL_NAME = {m: s["name"] for m, s in MODEL_SPECS.items()}
NAME_TO_MODEL = {v: k for k, v in MODEL_NAME.items()}

# metrics reported in the tables (all computed from hourly series in m3/s)
MAIN_METRICS = ["NSE", "KGE", "RMSE", "MAE", "PBIAS", "Pearson-r", "Alpha-NSE", "Beta-NSE", "FHV", "FMS", "FLV",
                "Peak-Timing", "Peak-MAPE", "Missed-Peaks"]
METRIC_IDEAL = {"NSE": 1, "KGE": 1, "RMSE": 0, "MAE": 0, "PBIAS": 0, "Pearson-r": 1, "Alpha-NSE": 1, "Beta-NSE": 0,
                "FHV": 0, "FMS": 0, "FLV": 0, "Peak-Timing": 0, "Peak-MAPE": 0, "Missed-Peaks": 0}
HIGHER_IS_BETTER = {"NSE", "KGE", "Pearson-r"}

LOGGER = logging.getLogger(__name__)


# ====================================================================================================================
# plotting style
# ====================================================================================================================
_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
            "#17becf", "#000080"]
MODEL_COLOR = {m: _PALETTE[i % len(_PALETTE)] for i, m in enumerate(MODEL_ORDER)}
SET_STYLE = {"A": "-", "B": "--", "C": ":"}
SET_COLOR = {"A": "#1f77b4", "B": "#d62728", "C": "#2ca02c"}
OBS_COLOR = "black"


def set_plot_style() -> None:
    """Large, publication-friendly fonts and consistent figure defaults (call once per notebook)."""
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "lines.linewidth": 1.6,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.unicode_minus": False,
        "font.family": "DejaVu Sans",
    })


def save_fig(fig: plt.Figure, name: str, formats: Iterable[str] = ("png",)) -> List[Path]:
    """Save a figure below ``figures/`` (300 dpi) and return the written paths."""
    FIGURES_DIR.mkdir(exist_ok=True)
    paths = []
    for fmt in formats:
        path = FIGURES_DIR / f"{name}.{fmt}"
        fig.savefig(path)
        paths.append(path)
    return paths


# ====================================================================================================================
# configs and runs
# ====================================================================================================================
def experiment_name(model: str, input_set: str) -> str:
    return f"{model}_{input_set}"


def config_path(model: str, input_set: str) -> Path:
    return CONFIG_DIR / f"{experiment_name(model, input_set)}.yml"


def load_config_dict(model: str, input_set: str) -> dict:
    with open(config_path(model, input_set)) as fp:
        return yaml.safe_load(fp)


def model_specific_config(model: str, input_set: str) -> dict:
    """The keys of a run config that differ from the shared ``COMMON`` block (what the notebook should show)."""
    cfg = load_config_dict(model, input_set)
    return {k: v for k, v in cfg.items() if k not in COMMON or COMMON[k] != v}


def _log_lines(run_dir: Path) -> List[tuple]:
    """(timestamp, message) tuples of output.log."""
    lines = []
    pattern = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+: (.*)$")
    for line in (Path(run_dir) / "output.log").read_text().splitlines():
        match = pattern.match(line)
        if match:
            lines.append((datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S"), match.group(2)))
    return lines


def training_history(run_dir: Path) -> pd.DataFrame:
    """Epoch-wise training loss, validation loss, validation metrics and epoch duration (s) parsed from output.log.

    For multi-frequency models the per-frequency metrics (e.g. ``NSE_1h``) are kept and the hourly values are
    additionally exposed under the plain name (``NSE``), so that all models can be plotted the same way.
    """
    train_re = re.compile(r"Epoch (\d+) average loss: avg_loss: ([0-9.eE+-]+|nan)")
    valid_re = re.compile(r"Epoch (\d+) average validation loss: ([0-9.eE+-]+|nan)(.*)")
    metric_re = re.compile(r"([A-Za-z][\w\-]*): (-?[0-9.eE+-]+|nan)")
    rows: Dict[int, dict] = {}
    prev_time = None
    for stamp, msg in _log_lines(run_dir):
        m = train_re.match(msg)
        if m:
            epoch = int(m.group(1))
            rows.setdefault(epoch, {"epoch": epoch})
            rows[epoch]["train_loss"] = float(m.group(2))
            rows[epoch]["epoch_time_s"] = (stamp - prev_time).total_seconds() if prev_time else np.nan
            prev_time = stamp
            continue
        m = valid_re.match(msg)
        if m:
            epoch = int(m.group(1))
            rows.setdefault(epoch, {"epoch": epoch})
            rows[epoch]["val_loss"] = float(m.group(2))
            for key, val in metric_re.findall(m.group(3)):
                if key != "avg_loss":
                    rows[epoch][key] = float(val)
            prev_time = stamp
            continue
        prev_time = stamp
    if not rows:
        raise RuntimeError(f"no epoch information found in {run_dir}/output.log")
    history = pd.DataFrame(sorted(rows.values(), key=lambda r: r["epoch"])).set_index("epoch")
    for metric in ("NSE", "KGE"):
        if metric not in history.columns and f"{metric}_1h" in history.columns:
            history[metric] = history[f"{metric}_1h"]
    return history


def best_epoch(run_dir: Path) -> int:
    """Epoch with the lowest validation loss (early stopping does not restore the best weights by itself)."""
    return int(training_history(run_dir)["val_loss"].idxmin())


def early_stopping_info(run_dir: Path) -> dict:
    lines = _log_lines(run_dir)
    triggered = next((msg for _, msg in lines if msg.startswith("Early stopping triggered")), None)
    return {"early_stopped": triggered is not None, "message": triggered}


def is_completed(run_dir: Path) -> bool:
    """A run counts as complete if it stopped early or reached the configured number of epochs."""
    try:
        history = training_history(run_dir)
        cfg = yaml.safe_load((Path(run_dir) / "config.yml").read_text())
    except (RuntimeError, FileNotFoundError):
        return False
    n_epochs = int(history.index.max())
    if not list(Path(run_dir).glob(f"model_epoch{n_epochs:03d}.pt")):
        return False
    return early_stopping_info(run_dir)["early_stopped"] or n_epochs >= int(cfg["epochs"])


def find_run_dir(model: str, input_set: str, completed_only: bool = True) -> Optional[Path]:
    """Most recent run directory of an experiment (``runs/<model>_<set>_<DDMM>_<HHMMSS>``)."""
    name = experiment_name(model, input_set)
    candidates = sorted(RUNS_DIR.glob(f"{name}_[0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    for candidate in candidates:
        if not completed_only or is_completed(candidate):
            return candidate
    return None


def reset_logging() -> None:
    """Remove the root logger handlers so that a second ``start_run`` in the same kernel logs to its own file."""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        handler.close()


def train(model: str, input_set: str, force: bool = False, in_process: bool = True) -> Path:
    """Train one (model, input set) combination unless a completed run already exists.

    ``in_process=True`` calls ``start_run`` inside the current Python process (shows progress bars in notebooks);
    ``in_process=False`` runs ``nh-run train`` as a subprocess (used by ``train_all.py``).
    """
    existing = find_run_dir(model, input_set)
    if existing is not None and not force:
        print(f"[{experiment_name(model, input_set)}] completed run found, skipping training: {existing.name}")
        return existing
    cfg_file = config_path(model, input_set)
    t0 = time.time()
    if in_process:
        from neuralhydrology.nh_run import start_run
        reset_logging()
        start_run(config_file=cfg_file)
    else:
        env = {"MPLBACKEND": "Agg"}
        import os
        env = {**os.environ, **env}
        subprocess.run([sys.executable, "-m", "neuralhydrology.nh_run", "train", "--config-file", str(cfg_file)],
                       cwd=HERE, env=env, check=True)
    run_dir = find_run_dir(model, input_set, completed_only=False)
    print(f"[{experiment_name(model, input_set)}] training finished in {(time.time() - t0) / 60:.1f} min -> {run_dir.name}")
    return run_dir


def count_parameters(run_dir: Path) -> int:
    cfg = Config(Path(run_dir) / "config.yml")
    cfg.update_config({"device": "cpu"})
    model = get_model(cfg)
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# ====================================================================================================================
# evaluation
# ====================================================================================================================
def _results_file(run_dir: Path, period: str, epoch: int) -> Path:
    return Path(run_dir) / period / f"model_epoch{epoch:03d}" / f"{period}_results.p"


def evaluate_period(run_dir: Path, period: str, epoch: int, force: bool = False) -> float:
    """Run ``nh-run evaluate`` for one period/epoch if the result file does not exist yet. Returns wall time (s)."""
    target = _results_file(run_dir, period, epoch)
    if target.exists() and not force:
        return float("nan")
    from neuralhydrology.nh_run import eval_run
    reset_logging()
    t0 = time.time()
    eval_run(run_dir=Path(run_dir), period=period, epoch=epoch)
    return time.time() - t0


def load_predictions(run_dir: Path, period: str, epoch: int) -> pd.DataFrame:
    """Hourly observed / simulated inflow (m3/s) of one period as a DataFrame with columns ``obs`` and ``sim``.

    Handles the three output layouts that occur in this study:

    * single frequency, ``predict_last_n = 1``: one prediction per hour (``time_step = 0``),
    * single frequency, ``predict_last_n > 1`` (PeakFocus): the last step of every hourly window is used, like the
      library does for its own metrics,
    * two frequencies (MTS-LSTM, ODE-LSTM): one sample per day with 24 hourly predictions, stacked to a continuous
      hourly series.

    MC-LSTM predicts ``inflow_mm`` (mm/h); it is converted back to m3/s so that every model is compared in the same unit.
    """
    with open(_results_file(run_dir, period, epoch), "rb") as fp:
        results = pickle.load(fp)
    cfg = yaml.safe_load((Path(run_dir) / "config.yml").read_text())
    target = cfg["target_variables"][0]
    freq = "1h"
    ds = results[BASIN][freq]["xr"]
    predict_last_n = cfg["predict_last_n"]
    if isinstance(predict_last_n, dict):  # multi-frequency: stack the last 24 hourly predictions of every day
        n_steps = int(predict_last_n[freq])
        sub = ds.isel(time_step=slice(-n_steps, None))
        stacked = sub.stack(datetime=["date", "time_step"])
        dates = pd.to_datetime(stacked["date"].values) + pd.to_timedelta(stacked["time_step"].values, unit="h")
        obs = pd.Series(stacked[f"{target}_obs"].values, index=dates)
        sim = pd.Series(stacked[f"{target}_sim"].values, index=dates)
    else:
        sub = ds.isel(time_step=-1)
        dates = pd.to_datetime(sub["date"].values)
        obs = pd.Series(sub[f"{target}_obs"].values, index=dates)
        sim = pd.Series(sub[f"{target}_sim"].values, index=dates)
    df = pd.DataFrame({"obs": obs, "sim": sim}).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.index.name = "date"
    if target == "inflow_mm":
        df = df / M3S_TO_MMH
    # negative predictions are physically impossible (the library does the same before computing its metrics)
    df["sim"] = df["sim"].clip(lower=0)
    return df


def _to_xarray(series: pd.Series) -> xr.DataArray:
    return xr.DataArray(series.values.astype(np.float64), coords={"date": series.index.values}, dims=["date"])


def compute_metrics(obs: pd.Series, sim: pd.Series) -> Dict[str, float]:
    """All library metrics at hourly resolution plus MAE and PBIAS. NaNs in obs or sim are ignored pairwise."""
    mask = obs.notna() & sim.notna()
    obs, sim = obs[mask], sim[mask]
    values = nh_metrics.calculate_all_metrics(_to_xarray(obs), _to_xarray(sim), resolution="1h")
    values["MAE"] = float(np.mean(np.abs(sim.values - obs.values)))
    values["PBIAS"] = float(100.0 * (sim.values - obs.values).sum() / obs.values.sum())
    return {k: float(v) for k, v in values.items()}


def yearly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, group in df.groupby(df.index.year):
        mask = group["obs"].notna() & group["sim"].notna()
        if mask.sum() < 24:
            continue
        o, s = _to_xarray(group["obs"][mask]), _to_xarray(group["sim"][mask])
        rows.append({"year": year, "NSE": nh_metrics.nse(o, s), "KGE": nh_metrics.kge(o, s),
                     "obs_max": group["obs"].max(), "sim_max": group["sim"].max(),
                     "obs_mean": group["obs"].mean(), "sim_mean": group["sim"].mean()})
    return pd.DataFrame(rows).set_index("year")


def monthly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for month, group in df.groupby(df.index.month):
        mask = group["obs"].notna() & group["sim"].notna()
        o, s = _to_xarray(group["obs"][mask]), _to_xarray(group["sim"][mask])
        rows.append({"month": month, "NSE": nh_metrics.nse(o, s), "PBIAS": 100 * (s.values - o.values).sum() / o.values.sum(),
                     "obs_mean": group["obs"].mean(), "sim_mean": group["sim"].mean()})
    return pd.DataFrame(rows).set_index("month")


# ====================================================================================================================
# flood events / peaks
# ====================================================================================================================
def detect_events(obs: pd.Series, n_events: int = 15, separation_hours: int = 72, min_peak: float = 0.0) -> List[pd.Timestamp]:
    """Timestamps of the ``n_events`` largest independent observed peaks (at least ``separation_hours`` apart)."""
    candidates = obs.dropna().sort_values(ascending=False)
    events: List[pd.Timestamp] = []
    for stamp, value in candidates.items():
        if value < min_peak:
            break
        if all(abs((stamp - e).total_seconds()) > separation_hours * 3600 for e in events):
            events.append(stamp)
        if len(events) == n_events:
            break
    return sorted(events)


def event_table(df: pd.DataFrame, events: List[pd.Timestamp], search_hours: int = 24, window_before_h: int = 48,
                window_after_h: int = 96) -> pd.DataFrame:
    """Peak magnitude / timing / volume errors for a list of observed peak times.

    * ``sim_peak``: maximum of the simulation within +-``search_hours`` of the observed peak
    * ``timing_error_h``: time of the simulated peak minus time of the observed peak (positive = late)
    * ``peak_error_pct``: (sim_peak - obs_peak) / obs_peak * 100
    * ``volume_error_pct``: relative error of the event volume over [-window_before_h, +window_after_h]
    * ``event_NSE``: NSE over the same event window
    """
    rows = []
    for peak in events:
        search = df.loc[peak - pd.Timedelta(hours=search_hours):peak + pd.Timedelta(hours=search_hours)]
        if search["sim"].isna().all():
            continue
        window = df.loc[peak - pd.Timedelta(hours=window_before_h):peak + pd.Timedelta(hours=window_after_h)].dropna()
        sim_peak_time = search["sim"].idxmax()
        obs_peak = float(df.loc[peak, "obs"])
        sim_peak = float(search.loc[sim_peak_time, "sim"])
        o, s = _to_xarray(window["obs"]), _to_xarray(window["sim"])
        rows.append({
            "obs_peak_time": peak,
            "obs_peak": obs_peak,
            "sim_peak": sim_peak,
            "sim_peak_time": sim_peak_time,
            "peak_error_pct": (sim_peak - obs_peak) / obs_peak * 100,
            "timing_error_h": (sim_peak_time - peak).total_seconds() / 3600,
            "volume_error_pct": (window["sim"].sum() - window["obs"].sum()) / window["obs"].sum() * 100,
            "event_NSE": float(nh_metrics.nse(o, s)),
        })
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.set_index("obs_peak_time")
    return table


def event_summary(table: pd.DataFrame) -> Dict[str, float]:
    """Aggregate statistics of an event table (used in the summary json and the peak comparison notebook)."""
    if table.empty:
        return {}
    return {
        "n_events": int(len(table)),
        "peak_error_mean_pct": float(table["peak_error_pct"].mean()),
        "peak_error_median_pct": float(table["peak_error_pct"].median()),
        "peak_abs_error_mean_pct": float(table["peak_error_pct"].abs().mean()),
        "timing_error_mean_h": float(table["timing_error_h"].mean()),
        "timing_abs_error_mean_h": float(table["timing_error_h"].abs().mean()),
        "timing_within_3h_frac": float((table["timing_error_h"].abs() <= 3).mean()),
        "peak_within_20pct_frac": float((table["peak_error_pct"].abs() <= 20).mean()),
        "volume_error_mean_pct": float(table["volume_error_pct"].mean()),
        "event_NSE_median": float(table["event_NSE"].median()),
    }


# ====================================================================================================================
# full pipeline for one (model, input set): evaluate every period, compute metrics, persist
# ====================================================================================================================
def result_dir(model: str, input_set: str) -> Path:
    return RESULTS_DIR / experiment_name(model, input_set)


def evaluate_run(model: str, input_set: str, run_dir: Optional[Path] = None, n_events: int = 15,
                 force: bool = False) -> dict:
    """Evaluate the best epoch on train / validation / test, store predictions + summary, return the summary."""
    run_dir = Path(run_dir) if run_dir is not None else find_run_dir(model, input_set)
    if run_dir is None:
        raise FileNotFoundError(f"no completed run for {experiment_name(model, input_set)}")
    out_dir = result_dir(model, input_set)
    summary_file = out_dir / "summary.json"
    if summary_file.exists() and not force:
        summary = json.loads(summary_file.read_text())
        if summary.get("run_dir") == run_dir.name:
            return summary
    out_dir.mkdir(parents=True, exist_ok=True)

    history = training_history(run_dir)
    epoch = best_epoch(run_dir)
    stop_info = early_stopping_info(run_dir)
    preds, period_metrics, eval_times = [], {}, {}
    for period in PERIODS:
        eval_times[period] = evaluate_period(run_dir, period, epoch)
        df = load_predictions(run_dir, period, epoch)
        df["period"] = period
        preds.append(df)
        period_metrics[period] = compute_metrics(df["obs"], df["sim"])
    predictions = pd.concat(preds)
    predictions.to_csv(out_dir / "predictions.csv.gz", float_format="%.4f")
    history.to_csv(out_dir / "history.csv")

    test = predictions[predictions["period"] == "test"]
    events = detect_events(test["obs"], n_events=n_events)
    table = event_table(test, events)
    table.to_csv(out_dir / "events_test.csv")

    summary = {
        "model": model,
        "model_name": MODEL_NAME[model],
        "nh_model": MODEL_SPECS[model]["nh_model"],
        "input_set": input_set,
        "input_set_name": INPUT_SET_NAMES[input_set],
        "dynamic_inputs": load_config_dict(model, input_set)["dynamic_inputs"],
        "run_dir": run_dir.name,
        "n_parameters": count_parameters(run_dir),
        "epochs_trained": int(history.index.max()),
        "best_epoch": int(epoch),
        "best_val_loss": float(history.loc[epoch, "val_loss"]),
        "early_stopped": bool(stop_info["early_stopped"]),
        "epoch_time_mean_s": float(history["epoch_time_s"].iloc[1:].mean()) if len(history) > 1 else float(history["epoch_time_s"].mean()),
        "train_time_total_s": float(history["epoch_time_s"].sum()),
        "train_time_to_best_s": float(history.loc[:epoch, "epoch_time_s"].sum()),
        "eval_time_test_s": eval_times["test"],
        "metrics": period_metrics,
        "events_test": event_summary(table),
    }
    summary_file.write_text(json.dumps(summary, indent=2, default=float))
    return summary


def load_summary(model: str, input_set: str) -> Optional[dict]:
    path = result_dir(model, input_set) / "summary.json"
    return json.loads(path.read_text()) if path.exists() else None


def load_saved_predictions(model: str, input_set: str) -> pd.DataFrame:
    path = result_dir(model, input_set) / "predictions.csv.gz"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "date"
    return df


def load_saved_history(model: str, input_set: str) -> pd.DataFrame:
    return pd.read_csv(result_dir(model, input_set) / "history.csv", index_col=0)


def load_saved_events(model: str, input_set: str) -> pd.DataFrame:
    df = pd.read_csv(result_dir(model, input_set) / "events_test.csv", index_col=0, parse_dates=True)
    df["sim_peak_time"] = pd.to_datetime(df["sim_peak_time"])
    return df


def collect_results(models: Iterable[str] = MODEL_ORDER, sets: Iterable[str] = SETS) -> pd.DataFrame:
    """Long table with one row per (model, set, period) and one column per metric, plus run statistics."""
    rows = []
    for model in models:
        for input_set in sets:
            summary = load_summary(model, input_set)
            if summary is None:
                continue
            for period, values in summary["metrics"].items():
                rows.append({
                    "model": model, "model_name": summary["model_name"], "set": input_set, "period": period,
                    "n_parameters": summary["n_parameters"], "epochs_trained": summary["epochs_trained"],
                    "best_epoch": summary["best_epoch"], "early_stopped": summary["early_stopped"],
                    "epoch_time_mean_s": summary["epoch_time_mean_s"], "train_time_total_s": summary["train_time_total_s"],
                    "train_time_to_best_s": summary["train_time_to_best_s"], "eval_time_test_s": summary["eval_time_test_s"],
                    **values, **{f"event_{k}": v for k, v in summary["events_test"].items()},
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["model"] = pd.Categorical(df["model"], categories=[m for m in MODEL_ORDER if m in set(df["model"])], ordered=True)
        df["model_name"] = pd.Categorical(df["model_name"], categories=[MODEL_NAME[m] for m in df["model"].cat.categories], ordered=True)
    return df


def metric_table(results: pd.DataFrame, period: str, metrics: Iterable[str] = ("NSE", "KGE", "RMSE", "PBIAS", "FHV", "Peak-Timing")) -> pd.DataFrame:
    """Wide table (rows = model, columns = metric x input set) for one period."""
    sub = results[results["period"] == period]
    table = sub.pivot_table(index="model_name", columns="set", values=list(metrics), observed=True)
    return table.reindex(columns=pd.MultiIndex.from_product([list(metrics), SETS]))


# ====================================================================================================================
# data access for plots (rainfall for hyetographs)
# ====================================================================================================================
def load_rainfall(columns: Iterable[str] = ("rf_avg",)) -> pd.DataFrame:
    path = DATA_DIR / "time_series" / "input" / "hourly" / f"{BASIN}_input.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df[list(columns)]


def load_observed_inflow() -> pd.Series:
    path = DATA_DIR / "time_series" / "target" / "hourly" / f"{BASIN}_target.csv"
    return pd.read_csv(path, index_col=0, parse_dates=True)["inflow"]


# ====================================================================================================================
# plotting
# ====================================================================================================================
def plot_learning_curves(histories: Dict[str, pd.DataFrame], title: str, labels: Optional[Dict[str, str]] = None,
                         colors: Optional[Dict[str, str]] = None, styles: Optional[Dict[str, str]] = None,
                         best_epochs: Optional[Dict[str, int]] = None, figsize=(18, 5)) -> plt.Figure:
    """Training loss, validation loss and validation NSE per epoch; one line per key of ``histories``."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for key, hist in histories.items():
        label = labels[key] if labels else key
        color = colors[key] if colors else None
        style = styles[key] if styles else "-"
        kw = dict(label=label, color=color, ls=style, marker="o", ms=3)
        axes[0].plot(hist.index, hist["train_loss"], **kw)
        axes[1].plot(hist.index, hist["val_loss"], **kw)
        if "NSE" in hist:
            axes[2].plot(hist.index, hist["NSE"], **kw)
        if best_epochs and key in best_epochs:
            axes[1].axvline(best_epochs[key], color=color, ls=":", lw=1.2)
    axes[0].set_ylabel("Training loss (MSE, standardized)")
    axes[1].set_ylabel("Validation loss (MSE, standardized)")
    axes[2].set_ylabel("Validation NSE")
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[2].set_ylim(max(-0.2, axes[2].get_ylim()[0]), 1.0)
    for ax in axes:
        ax.set_xlabel("Epoch")
    axes[0].legend(loc="best")
    fig.suptitle(title, fontsize=17)
    fig.tight_layout()
    return fig


def _add_rain_axis(ax: plt.Axes, rain: pd.Series, color: str = "#4c72b0", label: str = "Rainfall (mm/h)", frac: float = 0.3):
    """Inverted rainfall bars on a twin axis at the top of ``ax``."""
    ax2 = ax.twinx()
    width = (rain.index[1] - rain.index[0]) if len(rain) > 1 else pd.Timedelta("1h")
    ax2.bar(rain.index, rain.values, width=width, color=color, alpha=0.6, label="Rainfall")
    ax2.set_ylim(rain.max() * (1 / frac) if rain.max() > 0 else 1, 0)
    ax2.set_ylabel(label)
    ax2.grid(False)
    return ax2


def plot_hydrograph(obs: pd.Series, sims: Dict[str, pd.Series], rain: Optional[pd.Series] = None,
                    start=None, end=None, title: str = "", colors: Optional[Dict[str, str]] = None,
                    styles: Optional[Dict[str, str]] = None, ax: Optional[plt.Axes] = None,
                    figsize=(18, 6), obs_lw: float = 1.8, sim_lw: float = 1.5, ylabel: str = "Inflow (m³/s)") -> plt.Axes:
    """Observed vs simulated hydrograph(s), optionally with a hyetograph on top."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    sl = slice(start, end)
    ax.plot(obs.loc[sl].index, obs.loc[sl], color=OBS_COLOR, lw=obs_lw, label="Observed", zorder=2)
    for key, sim in sims.items():
        ax.plot(sim.loc[sl].index, sim.loc[sl], lw=sim_lw, label=key, zorder=3,
                color=colors.get(key) if colors else None, ls=styles.get(key, "-") if styles else "-", alpha=0.9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    if rain is not None:
        _add_rain_axis(ax, rain.loc[sl])
    ax.legend(loc="upper right")
    return ax


def plot_scatter(obs: pd.Series, sim: pd.Series, title: str = "", ax: Optional[plt.Axes] = None, log: bool = True,
                 color: str = "#1f77b4", figsize=(6, 6)) -> plt.Axes:
    """Observed vs simulated scatter with the 1:1 line and NSE / KGE / PBIAS annotation."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    mask = obs.notna() & sim.notna()
    o, s = obs[mask], sim[mask]
    values = compute_metrics(o, s)
    if log:
        o_plot, s_plot = o.clip(lower=0.1), s.clip(lower=0.1)
        ax.set_xscale("log")
        ax.set_yscale("log")
    else:
        o_plot, s_plot = o, s
    ax.scatter(o_plot, s_plot, s=4, alpha=0.25, color=color, rasterized=True)
    lim = (0.1 if log else 0, max(o_plot.max(), s_plot.max()) * 1.1)
    ax.plot(lim, lim, color="black", lw=1.2, ls="--", label="1:1")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Observed inflow (m³/s)")
    ax.set_ylabel("Simulated inflow (m³/s)")
    ax.set_title(title)
    ax.text(0.03, 0.97, f"NSE = {values['NSE']:.3f}\nKGE = {values['KGE']:.3f}\nPBIAS = {values['PBIAS']:.1f}%",
            transform=ax.transAxes, va="top", ha="left", fontsize=13,
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
    ax.set_aspect("equal", adjustable="box")
    return ax


def plot_fdc(obs: pd.Series, sims: Dict[str, pd.Series], title: str = "Flow duration curve", colors=None, styles=None,
             ax: Optional[plt.Axes] = None, figsize=(9, 6)) -> plt.Axes:
    """Flow duration curves (exceedance probability vs discharge, log y)."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    def fdc(series: pd.Series):
        values = np.sort(series.dropna().values)[::-1]
        prob = np.arange(1, len(values) + 1) / (len(values) + 1) * 100
        return prob, values

    p, v = fdc(obs)
    ax.plot(p, np.clip(v, 0.1, None), color=OBS_COLOR, lw=2.2, label="Observed")
    for key, sim in sims.items():
        p, v = fdc(sim)
        ax.plot(p, np.clip(v, 0.1, None), lw=1.6, label=key, color=colors.get(key) if colors else None,
                ls=styles.get(key, "-") if styles else "-")
    ax.set_yscale("log")
    ax.set_xlabel("Exceedance probability (%)")
    ax.set_ylabel("Inflow (m³/s)")
    ax.set_title(title)
    ax.axvspan(0, 2, color="red", alpha=0.07)
    ax.axvspan(70, 100, color="blue", alpha=0.07)
    ax.text(1, ax.get_ylim()[1] * 0.5, "FHV", color="red", ha="left", fontsize=12)
    ax.text(85, ax.get_ylim()[1] * 0.5, "FLV", color="blue", ha="center", fontsize=12)
    ax.legend(loc="lower left")
    return ax


def plot_metric_bars(results: pd.DataFrame, metric: str, period: str = "test", ax: Optional[plt.Axes] = None,
                     figsize=(16, 6), title: Optional[str] = None) -> plt.Axes:
    """Grouped bars (one group per model, one bar per input set) for one metric and period."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    sub = results[results["period"] == period]
    table = sub.pivot_table(index="model_name", columns="set", values=metric, observed=True).reindex(columns=SETS)
    x = np.arange(len(table))
    width = 0.26
    for i, input_set in enumerate(SETS):
        if input_set not in table:
            continue
        bars = ax.bar(x + (i - 1) * width, table[input_set], width, label=f"Set {input_set}", color=SET_COLOR[input_set],
                      alpha=0.85, edgecolor="black", lw=0.5)
        for bar, value in zip(bars, table[input_set]):
            if np.isfinite(value):
                text = f"{value:,.0f}" if abs(value) >= 100 else f"{value:.2f}"
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), text, ha="center",
                        va="bottom" if value >= 0 else "top", fontsize=9, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(table.index, rotation=30, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} — {PERIOD_LABEL[period]}")
    if metric in METRIC_IDEAL:
        ax.axhline(METRIC_IDEAL[metric], color="black", lw=1, ls="--")
    ax.legend(loc="best")
    return ax


def plot_heatmap(table: pd.DataFrame, title: str = "", fmt: str = "{:.2f}", cmap: str = "RdYlGn",
                 vmin: Optional[float] = None, vmax: Optional[float] = None, ax: Optional[plt.Axes] = None,
                 figsize=(10, 8), cbar_label: str = "") -> plt.Axes:
    """Annotated heatmap of a 2-D table (rows x columns)."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    values = table.values.astype(float)
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(table.shape[1]))
    ax.set_xticklabels([str(c) for c in table.columns], rotation=30, ha="right")
    ax.set_yticks(range(table.shape[0]))
    ax.set_yticklabels([str(i) for i in table.index])
    ax.grid(False)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, fmt.format(values[i, j]), ha="center", va="center", fontsize=11, color="black")
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(cbar_label)
    ax.set_title(title)
    return ax


def narrative_for_run(summary: dict) -> str:
    """Rule-based Korean interpretation of one run's metrics (train/validation/test) for the notebooks."""
    m = summary["metrics"]
    t, v, tr = m["test"], m["validation"], m["train"]
    ev = summary["events_test"]
    lines = [f"**{summary['model_name']} / 입력세트 {summary['input_set']} ({summary['input_set_name']})**"]
    lines.append(f"- 테스트 기간 NSE {t['NSE']:.3f}, KGE {t['KGE']:.3f}, RMSE {t['RMSE']:.1f} m³/s, PBIAS {t['PBIAS']:+.1f}%.")
    gap = tr["NSE"] - t["NSE"]
    if gap > 0.1:
        lines.append(f"- 학습 NSE({tr['NSE']:.3f})와 테스트 NSE의 차이가 {gap:.2f}로 커서 **과적합 또는 기간 간 수문특성 차이**가 의심됩니다.")
    elif gap < 0.03:
        lines.append(f"- 학습({tr['NSE']:.3f})·검증({v['NSE']:.3f})·테스트 NSE가 비슷해 **일반화가 안정적**입니다.")
    else:
        lines.append(f"- 학습 NSE {tr['NSE']:.3f}, 검증 NSE {v['NSE']:.3f}: 테스트 기간에서 성능이 다소 낮아지지만 큰 과적합 징후는 없습니다.")
    if abs(t["PBIAS"]) > 10:
        lines.append(f"- 전체 유량을 {'과대' if t['PBIAS'] > 0 else '과소'} 추정하는 경향(PBIAS {t['PBIAS']:+.1f}%)이 뚜렷합니다.")
    fhv = t["FHV"]
    lines.append(f"- 고유량 구간(FHV {fhv:+.1f}%)은 {'과소' if fhv < -5 else '과대' if fhv > 5 else '거의 편향 없이'} 재현되고, "
                 f"저유량 구간(FLV {t['FLV']:+.1f}%)은 {'크게 과대' if t['FLV'] > 30 else '과대' if t['FLV'] > 5 else '과소' if t['FLV'] < -5 else '편향 없이'} 재현됩니다.")
    if ev:
        lines.append(f"- 테스트 기간 상위 {ev['n_events']}개 홍수사건: 첨두 크기 오차 중앙값 {ev['peak_error_median_pct']:+.1f}%, "
                     f"평균 절대 첨두오차 {ev['peak_abs_error_mean_pct']:.1f}%, 첨두 시각 오차 평균 {ev['timing_error_mean_h']:+.1f} h "
                     f"(±3 h 이내 {ev['timing_within_3h_frac']*100:.0f}%, 크기 ±20% 이내 {ev['peak_within_20pct_frac']*100:.0f}%).")
    stopped = "early stopping" if summary["early_stopped"] and summary["epochs_trained"] < COMMON["epochs"] else f"최대 {COMMON['epochs']} epoch 도달"
    lines.append(f"- 학습: {summary['epochs_trained']} epoch ({stopped}), "
                 f"최적 epoch {summary['best_epoch']}, epoch당 평균 {summary['epoch_time_mean_s']:.0f} s, "
                 f"총 학습시간 {summary['train_time_total_s']/60:.1f} min, 파라미터 {summary['n_parameters']:,}개.")
    return "\n".join(lines)
