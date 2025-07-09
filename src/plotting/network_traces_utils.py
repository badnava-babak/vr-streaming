"""
network_traces_utils.py

Helpers for loading and analysing 4 G / 5 G network-link traces—including
header-less, two-column TSV/CSV files like:

    1.0   109.0
    2.0    82.0
    3.0    88.0
    ...

The first column is **timestamp (s)**, the second is **throughput (Mbit/s)**.

New in v2.0
-----------
* `load_scenario_files()` – load **multiple files** (one per scenario) into an
  ordered dict `{scenario_name: DataFrame}`.
* `multi_stats()` – tabulate basic stats for all scenarios.
* `plot_multi_traces()` – overlay throughput-time curves for comparison.

Public API
----------
load_net_trace(path)           -> DataFrame (canonical columns)
concat_traces(paths)           -> DataFrame (stacked + trace_id)
load_scenario_files(paths)     -> OrderedDict[str, DataFrame]
basic_net_stats(df)            -> Series summary (single trace)
multi_stats(odict)             -> DataFrame summary (all scenarios)
plot_net_trace(df, …)          -> time-series plot
plot_multi_traces(odict, …)    -> overlayed time-series
plot_throughput_cdf(df, …)     -> empirical CDF plot
"""
from __future__ import annotations

import pathlib
from collections import OrderedDict
from typing import Sequence, Mapping

import pandas as pd
import scipy.io as sio
import numpy as np

from src.plotting.viewport import compute_tx_times


# -------------------------------------------------------------
# INTERNAL NORMALISER (unchanged)
# -------------------------------------------------------------

def _normalise_df(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Timestamp(s)": "time_s",
        "timestamp_s": "time_s",
        "time_s": "time_s",
        "time_sec": "time_s",
        "time_ms": "time_ms",
        "Timestamp(ms)": "time_ms",
        "timestamp_ms": "time_ms",
        "t_ms": "time_ms",
        "t": "time_ms",
        "Throughput(Mbps)": "throughput_Mbps",
        "throughput_Mbps": "throughput_Mbps",
        "rate_Mbps": "throughput_Mbps",
        "thr": "throughput_Mbps",
        "Throughput": "throughput_Mbps",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # unnamed two-column case → assume timestamp, throughput
    if list(df.columns) == [0, 1]:
        df = df.rename(columns={0: "time_s", 1: "throughput_Mbps"})

    # convert time
    if "time_ms" not in df.columns:
        if "time_s" in df.columns:
            df["time_ms"] = (df["time_s"].astype(float) * 1000).astype(int)
            df = df.drop(columns=["time_s"])
        else:
            raise ValueError("Missing timestamp column.")

    if "throughput_Mbps" not in df.columns:
        raise ValueError("Missing throughput column.")

    df = df.sort_values("time_ms").reset_index(drop=True)
    return df


# -------------------------------------------------------------
# FILE LOADERS (CSV / MAT) – unchanged except header=None fall-back
# -------------------------------------------------------------

def _load_csv(path: pathlib.Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, header=None, delim_whitespace=True)
    except Exception:
        df = pd.read_csv(path, delim_whitespace=True)
    if df.shape[1] == 1:
        df['time_s'] = range(1, len(df) + 1)
        df = df.rename(columns={0: "throughput_Mbps"})
    elif df.shape[1] == 2 and list(df.columns) == [0, 1]:
        # bare numbers separated by space or tab
        df = df.rename(columns={0: "time_s", 1: "throughput_Mbps"})

    return _normalise_df(df)


def _load_mat(path: pathlib.Path) -> pd.DataFrame:
    raw = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    time_key = next((k for k in raw.keys() if k.lower() in {"time_ms", "timestamp_ms", "t_ms", "t", "timestamp_s", "time_s"}), None)
    thr_key = next((k for k in raw.keys() if k.lower() in {"throughput_mbps", "rate_mbps", "thr", "throughput", "throughput(mbps)"}), None)
    if time_key is None or thr_key is None:
        raise ValueError("MAT missing timestamp / throughput fields.")
    df = pd.DataFrame({time_key: raw[time_key].flatten(), thr_key: raw[thr_key].flatten()})
    return _normalise_df(df)


def load_net_trace(path: str | pathlib.Path) -> pd.DataFrame:
    path = pathlib.Path(path)
    return _load_csv(path)
    # if path.suffix.lower() in {".csv", ".txt", ".tsv"}:
    #     pass
    # elif path.suffix.lower() == ".mat":
    #     return _load_mat(path)
    # else:
    #     raise ValueError(f"Unsupported file type: {path.suffix}")


# -------------------------------------------------------------
# BATCH HELPERS
# -------------------------------------------------------------

def concat_traces(paths: Sequence[str | pathlib.Path]) -> pd.DataFrame:
    dfs = []
    for i, p in enumerate(paths):
        df = load_net_trace(p).copy()
        df["trace_id"] = i
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def load_scenario_files(paths: Mapping[str, str | pathlib.Path]) -> OrderedDict[str, pd.DataFrame]:
    """Load multiple scenario-trace files given a dict *{name: path}*.

    Returns an OrderedDict preserving the insertion order.
    """
    odict = OrderedDict()
    for name, p in paths.items():
        odict[name] = load_net_trace(p)
    return odict


# -------------------------------------------------------------
# STATS
# -------------------------------------------------------------

def basic_net_stats(df: pd.DataFrame):
    return pd.Series({
        "duration_s": (df["time_ms"].iloc[-1] - df["time_ms"].iloc[0]) / 1000,
        "mean_Mbps": df["throughput_Mbps"].mean(),
        "median_Mbps": df["throughput_Mbps"].median(),
        "p95_Mbps": df["throughput_Mbps"].quantile(0.95),
        "min_Mbps": df["throughput_Mbps"].min(),
        "max_Mbps": df["throughput_Mbps"].max(),
    })


def multi_stats(odict: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame({k: basic_net_stats(v) for k, v in odict.items()}).T


# -------------------------------------------------------------
# PLOTTING
# -------------------------------------------------------------

def plot_net_trace(df: pd.DataFrame, *, title: str | None = None, vline_every: float | None = None, **plot_kwargs):
    t = df["time_ms"] / 1000
    thr = df["throughput_Mbps"]
    plt.figure(figsize=(10, 4))
    plt.plot(t, thr, **plot_kwargs)
    plt.xlabel("Time [s]")
    plt.ylabel("Throughput [Mbit/s]")
    if vline_every:
        for x in np.arange(0, t.max(), vline_every):
            plt.axvline(x, linestyle="--", alpha=0.3)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_multi_traces(odict: Mapping[str, pd.DataFrame], *, vline_every: float | None = None):
    plt.figure(figsize=(10, 4))
    for name, df in odict.items():
        t = df["time_ms"] / 1000
        plt.plot(t, df["throughput_Mbps"], label=name)
    if vline_every:
        xmax = max(df["time_ms"].max() for df in odict.values()) / 1000
        for x in np.arange(0, xmax, vline_every):
            plt.axvline(x, linestyle="--", alpha=0.3)
    plt.xlabel("Time [s]")
    plt.ylabel("Throughput [Mbit/s]")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_throughput_cdf(df: pd.DataFrame, *, title: str | None = None):
    thr = np.sort(df["throughput_Mbps"].values)
    p = np.arange(1, len(thr) + 1) / len(thr)
    plt.figure(figsize=(5, 4))
    plt.plot(thr, p)
    plt.xlabel("Throughput [Mbit/s]")
    plt.ylabel("Empirical CDF")
    if title:
        plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()




def plot_tx_times_per_qp(
        seg_bitrate_Mbps: np.ndarray,  # shape (7, T) from build_fov_traces()['bitrate']
        net_df,  # DataFrame from load_net_trace()
        qp_labels=None,  # list of 7 display labels, e.g. [22,27,…]
        seg_duration_s: float = 1.0,  # 30 frames ÷ 30 fps
        vline_every: int | None = None,
        **plot_kwargs):
    """
    Simulate and plot the download time of every 1-s video segment for all
    seven QP levels.

    Parameters
    ----------
    seg_bitrate_Mbps : (7, T) ndarray
        Per-segment FoV-weighted bitrate (Mbit/s) returned by build_fov_traces().
    net_df : pandas.DataFrame
        Throughput log with columns  time_ms  and  throughput_Mbps.
    qp_labels : list[int], optional
        Display labels; defaults to [0, 1, 2, 3, 4, 5, 6] if omitted.
    seg_duration_s : float, default 1.0
        Duration of a video segment in seconds.
    vline_every : int, optional
        Draw dashed vertical markers every *k* segments.
    **plot_kwargs
        Forwarded to `plt.plot` for styling (alpha, linewidth…).
    """
    if qp_labels is None:
        qp_labels = list(range(seg_bitrate_Mbps.shape[0]))

    T = seg_bitrate_Mbps.shape[1]
    x = np.arange(T)

    plt.figure(figsize=(10, 4))
    for q in range(7):
        tx = compute_tx_times(seg_bitrate_Mbps[q], net_df,
                              seg_duration_s=seg_duration_s)
        plt.plot(x, tx, label=f"QP {qp_labels[q]}", **plot_kwargs)

    if vline_every:
        for x0 in range(0, T, vline_every):
            plt.axvline(x0, linestyle="--", alpha=0.3)

    plt.xlabel("Segment index")
    plt.ylabel("Download time [s]")
    plt.title("Segment download time vs. QP level")
    plt.legend()
    plt.tight_layout()
    plt.show()


import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------- #
# Helper: plot time-aligned statistics of multiple traces
# ------------------------------------------------------------------------- #
def plot_time_stats(trace_paths,
                    bin_size_s=1.0,
                    align_start=True,
                    max_duration_s=None,
                    title=None,
                    color_mean="tab:blue",
                    color_band="lightblue",
                    color_p="tab:orange",
                    color_median="tab:green"):
    """
    Plot mean ± std, median, p5 and p95 throughput over time.

    Parameters
    ----------
    trace_paths   : list[str | pathlib.Path]
    bin_size_s    : float, resampling interval (seconds)
    align_start   : bool
        True  → time axis starts at 0 for every trace (good for drive tests).
        False → keep absolute timestamps (requires traces recorded in sync).
    max_duration_s: float | None
        Trim all traces to this duration (after alignment) if given.
    """
    # ---- 1. load & resample each trace to common grid ----------------------
    series = []
    for p in trace_paths:
        df = load_net_trace(p)
        # convert to seconds offset
        if align_start:
            offset = (df["time_ms"] - df["time_ms"].iloc[0]) / 1_000.0
        else:
            offset = df["time_ms"] / 1_000.0
        s = pd.Series(df["throughput_Mbps"].values, index=offset)
        # resample to regular grid
        s_reg = s
        # s_reg = (s
        #          .sort_index()
        #          .resample(f"{bin_size_s}S")
        #          .mean()
        #          .interpolate("linear"))
        if max_duration_s is not None:
            s_reg = s_reg.iloc[: int(max_duration_s // bin_size_s) + 1]
        series.append(s_reg)

    # align by union of all time stamps
    aligned = pd.concat(series, axis=1)
    aligned.columns = [pathlib.Path(p).stem for p in trace_paths]

    # ---- 2. compute stats across traces -----------------------------------
    mean = aligned.mean(axis=1)
    std = aligned.std(axis=1)
    p05 = aligned.quantile(0.05, axis=1)
    p95 = aligned.quantile(0.95, axis=1)
    median = aligned.quantile(0.5, axis=1)

    t = aligned.index.values.astype(float)

    # ---- 3. plotting -------------------------------------------------------
    plt.figure(figsize=(10, 4))
    # mean ± std band
    plt.fill_between(t, mean - std, mean + std,
                     color=color_band, alpha=0.4,
                     label="±1 σ")
    # mean
    plt.plot(t, mean, color=color_mean, linewidth=2.0, label="mean")
    # p5 / p95
    plt.plot(t, p05, color=color_p, linestyle="--", label="p5 / p95")
    plt.plot(t, p95, color=color_p, linestyle="--")
    # median
    plt.plot(t, median, color=color_median, linewidth=1.5, label="median")

    plt.xlabel("Time [s]" if align_start else "Timestamp [s]")
    plt.ylabel("Throughput [Mbit/s]")
    if title:
        plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return pd.DataFrame({
        "mean": mean, "std": std, "median": median,
        "p05": p05, "p95": p95
    })
