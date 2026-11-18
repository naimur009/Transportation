#!/usr/bin/env python3
"""
Step 5d -- Plot disruption results (delay characterization + functionality
degradation), Illinois test run.

DATASET NOTE (updated to match the Illinois test run's naming convention):
-----------------------------------------------------------------------
The earlier draft of this script plotted fort-to-port disruption results
from a single hardcoded STRAHNET/STRACNET pair of networks. Those network
labels and default paths don't exist in this pipeline. This script has
been updated to match `7_1_random_disruption.py` / `7_2_targeted_disruption.py`
/ `7_3_adaptive_disruption.py`'s own output convention:

  * Default paths now point at `reports/highway_*` and `reports/railway_*`
    (matching those three scripts' own `NETWORK_LABELS = ["highway",
    "railway"]` and per-network output filenames), not
    `reports/strahnet_*` / `reports/stracnet_*`.
  * "Fort-to-port paths" in the axis/row labeling has become
    "county-to-county routes" -- the underlying facility set is now a
    single county-node list `C` (see `4_graph_smoothing.py`'s
    `C_county_nodes`), not two disjoint fort/port sets. This is a
    terminology-only change here: this script never reads O/D or C
    directly, it only plots the `n_unaffected` / `n_delayed` /
    `n_disconnected` / `n_pairs` / `f_G` columns that 7_1/7_2/7_3 already
    computed (correctly, over the unique C(|C|, 2) county-to-county pairs
    -- see those scripts' own docstrings), so no plotting math changes.
  * The Railway column's random-disruption CSV now also auto-detects a
    default path (`reports/railway_random_disruption_trials.csv`),
    matching the auto-detect behavior already given to
    `--rail-targeted-csv` / `--rail-adaptive-csv` -- previously only an
    explicit `--rail-results-csv` would enable a railway random panel.
  * Railway plotting is enabled automatically if ANY of the default
    railway files exist on disk (matching the "both networks by default"
    convention used by every other script in this pipeline), in addition
    to being enabled by passing any `--rail-*` flag explicitly.

Everything else -- the grid layout, the summarizing/aggregation math, the
error-bar logic -- is UNCHANGED from the previous version of this script.

Layout
------
  * A grid: rows = {Targeted, Adaptive, Random} (only rows with at least
    one network's data are drawn), columns = {Highway, Railway} (only
    columns with at least one row's data are drawn).
  * Each panel is a LINE plot with markers -- THREE separate lines
    (unaffected / delayed / disconnected), not a stacked area -- colored
    blue / orange / red respectively.
  * The Random row additionally draws a vertical error bar at every point
    (+/- 1 std across the 100 trials/intensity), since random disruption
    is stochastic. Targeted and Adaptive have no error bars -- both are
    deterministic.
  * Both axes run 0-100%: x-axis is disruption intensity (0-50% by
    default), y-axis is percent of county-to-county routes in each
    category (0-100%), not a 0-1 proportion.

Fig. "functionality" has the same row/column structure: Targeted and
Adaptive each a single deterministic line (Adaptive will typically drop
faster than Targeted at the same intensity, since it reacts to new
bottlenecks created by its own earlier removals), Random a mean line +/-
1 std band.

Input
-----
Random disruption trial CSVs (one row per (intensity, trial), stochastic):
    --results-csv       PATH   highway random trials (required)
    --rail-results-csv  PATH   railway random trials (optional, auto-detected)
Required columns: intensity_pct, trial_idx, f_G, n_unaffected, n_delayed,
n_disconnected, n_pairs.

Targeted disruption CSVs (one row per intensity, deterministic, optional):
    --targeted-csv       PATH   highway targeted results
    --rail-targeted-csv  PATH   railway targeted results

Adaptive disruption CSVs (one row per intensity, deterministic, optional):
    --adaptive-csv       PATH   highway adaptive results
    --rail-adaptive-csv  PATH   railway adaptive results
Required columns (same as Targeted): intensity_pct, n_unaffected,
n_delayed, n_disconnected, n_pairs, and either f_G or functionality.

Output
------
    reports/fig3_delay_characterization_random.png
    reports/fig4_functionality_degradation.png
    reports/fig5_lcc_degradation.png
(Filenames kept from the previous version so nothing downstream that
references them breaks. Despite the "_random" suffix, Fig. 3's file now
also contains the targeted/adaptive rows whenever that data is supplied --
the suffix is legacy naming, not a description of the current content.
Fig. 5 is new: largest-connected-component ratio vs. intensity, same
Targeted/Adaptive/Random x network grid as Fig. 3/4, but only drawn for
runs whose CSV actually has the lcc_ratio/n_components columns -- an
older CSV from before that metric was added gets a "data not available"
panel instead of crashing the whole figure, and if NO input has LCC data
at all, Fig. 5 is skipped entirely with an explanatory message.)

Usage
-----
    # highway only, random only
    python 8_generate_diagram.py

    # highway, random + targeted -> 2-row x 1-column grid
    python 8_generate_diagram.py --targeted-csv reports/highway_targeted_disruption.csv

    # highway, all three methods -> 3-row x 1-column grid
    python 8_generate_diagram.py \\
        --targeted-csv reports/highway_targeted_disruption.csv \\
        --adaptive-csv reports/highway_adaptive_disruption.csv

    # full layout: highway + railway, all three methods -> 3x2 grid
    # (if 7_1/7_2/7_3 were already run for both networks with their own
    # defaults, this needs no flags at all -- railway is auto-detected)
    python 8_generate_diagram.py \\
        --targeted-csv reports/highway_targeted_disruption.csv \\
        --adaptive-csv reports/highway_adaptive_disruption.csv \\
        --rail-results-csv reports/railway_random_disruption_trials.csv \\
        --rail-targeted-csv reports/railway_targeted_disruption.csv \\
        --rail-adaptive-csv reports/railway_adaptive_disruption.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_RESULTS_CSV = Path("reports/highway_random_disruption_trials.csv")
DEFAULT_TARGETED_CSV = Path("reports/highway_targeted_disruption.csv")
DEFAULT_ADAPTIVE_CSV = Path("reports/highway_adaptive_disruption.csv")

DEFAULT_RAIL_RESULTS_CSV = Path("reports/railway_random_disruption_trials.csv")
DEFAULT_RAIL_TARGETED_CSV = Path("reports/railway_targeted_disruption.csv")
DEFAULT_RAIL_ADAPTIVE_CSV = Path("reports/railway_adaptive_disruption.csv")

DEFAULT_OUTPUT_DIR = Path("reports")

FIG3_FILENAME = "fig3_delay_characterization_random.png"
FIG4_FILENAME = "fig4_functionality_degradation.png"
FIG5_FILENAME = "fig5_lcc_degradation.png"

# Category colors: blue / orange / red for unaffected / delayed / disconnected.
COLOR_UNAFFECTED = "#1f77b4"
COLOR_DELAYED = "#ff7f0e"
COLOR_DISCONNECTED = "#d62728"
COLOR_FUNCTIONALITY = "#2f2f2f"
COLOR_LCC = "#0f6e56"

RANDOM_REQUIRED_COLS = {
    "intensity_pct", "trial_idx", "f_G",
    "n_unaffected", "n_delayed", "n_disconnected", "n_pairs",
}
# Shared by Targeted and Adaptive -- both are one-row-per-intensity,
# deterministic, no trial_idx.
DETERMINISTIC_REQUIRED_COLS = {
    "intensity_pct",
    "n_unaffected", "n_delayed", "n_disconnected", "n_pairs",
}

# Row order top-to-bottom. Only row types with at least one network's data
# actually get drawn (see plot_fig3_delay_characterization /
# plot_fig4_functionality) -- this just fixes the order when more than one
# is present.
ROW_TYPE_ORDER = ("targeted", "adaptive", "random")
ROW_LABEL_FIG3 = {
    "targeted": "Route Categorization\n(Targeted)",
    "adaptive": "Route Categorization\n(Adaptive)",
    "random": "Route Categorization\n(Random)",
}
ROW_LABEL_FIG4 = {
    "targeted": "Functionality\n(Targeted)",
    "adaptive": "Functionality\n(Adaptive)",
    "random": "Functionality\n(Random)",
}
ROW_LABEL_FIG5 = {
    "targeted": "LCC Ratio\n(Targeted)",
    "adaptive": "LCC Ratio\n(Adaptive)",
    "random": "LCC Ratio\n(Random)",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_random_results(path: Path) -> pd.DataFrame:
    """Load a random-disruption trials CSV (one row per (intensity, trial))."""
    if not path.exists():
        raise SystemExit(
            f"Random disruption results not found: {path}\n"
            f"Run 7_1_random_disruption.py first -- this script only "
            f"plots its output, it does not run any simulation itself."
        )

    df = pd.read_csv(path)
    missing = RANDOM_REQUIRED_COLS - set(df.columns)
    if missing:
        raise SystemExit(
            f"{path} is missing expected column(s) {sorted(missing)} -- "
            f"is this really random-disruption trial output, and from a "
            f"version matching this script?"
        )

    print(f"Loaded {len(df):,} trial rows from {path} "
          f"({df['intensity_pct'].nunique()} intensities, "
          f"{df.groupby('intensity_pct').size().min()}-"
          f"{df.groupby('intensity_pct').size().max()} trials/intensity)")
    return df


def load_deterministic_results(path: Path, label: str) -> pd.DataFrame:
    """
    Load an optional deterministic (one-row-per-intensity) disruption CSV
    for overlay comparison. Used for BOTH Targeted (7_2) and Adaptive
    (7_3) results -- their CSVs have the identical shape.

    `label` is only used in error messages (e.g. "Targeted", "Adaptive").
    """
    if not path.exists():
        raise SystemExit(f"{label} disruption CSV given but not found: {path}")

    df = pd.read_csv(path)

    # Backward/forward compatibility: some outputs in this repo have used
    # either `f_G` or `functionality` for the same metric.
    if "f_G" not in df.columns and "functionality" in df.columns:
        df = df.rename(columns={"functionality": "f_G"})

    missing = DETERMINISTIC_REQUIRED_COLS - set(df.columns)
    if missing:
        raise SystemExit(f"{path} is missing expected column(s) {sorted(missing)}.")
    if "f_G" not in df.columns:
        raise SystemExit(
            f"{path} is missing functionality metric column: expected either 'f_G' "
            f"or 'functionality'."
        )

    print(f"Loaded {len(df):,} {label.lower()}-disruption rows from {path}")
    return df


# ---------------------------------------------------------------------------
# Summarizing
# ---------------------------------------------------------------------------

def summarize_random_by_intensity(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trial-level rows into per-intensity mean/std summaries.

    LCC ratio and component count are OPTIONAL columns (added to
    7_1_random_disruption.py after this script was first written) -- if
    they're present, mean/std are computed for them exactly like every
    other metric; if a CSV predates that addition, they're silently
    skipped rather than raising, so this function still works on older
    trial files. Callers that need to know whether LCC data actually made
    it into the summary should check for the 'lcc_mean' column.
    """
    df = df.copy()
    df["pct_unaffected"] = 100 * df["n_unaffected"] / df["n_pairs"]
    df["pct_delayed"] = 100 * df["n_delayed"] / df["n_pairs"]
    df["pct_disconnected"] = 100 * df["n_disconnected"] / df["n_pairs"]
    df["pct_f_G"] = 100 * df["f_G"]

    has_lcc = "lcc_ratio" in df.columns
    has_components = "n_components" in df.columns
    if has_lcc:
        df["pct_lcc"] = 100 * df["lcc_ratio"]

    agg_kwargs = dict(
        f_G_mean=("pct_f_G", "mean"),
        f_G_std=("pct_f_G", "std"),
        unaffected_mean=("pct_unaffected", "mean"),
        unaffected_std=("pct_unaffected", "std"),
        delayed_mean=("pct_delayed", "mean"),
        delayed_std=("pct_delayed", "std"),
        disconnected_mean=("pct_disconnected", "mean"),
        disconnected_std=("pct_disconnected", "std"),
        n_trials=("trial_idx", "count"),
    )
    if has_lcc:
        agg_kwargs["lcc_mean"] = ("pct_lcc", "mean")
        agg_kwargs["lcc_std"] = ("pct_lcc", "std")
    if has_components:
        agg_kwargs["components_mean"] = ("n_components", "mean")
        agg_kwargs["components_std"] = ("n_components", "std")

    summary = df.groupby("intensity_pct").agg(**agg_kwargs).reset_index().sort_values("intensity_pct")

    std_cols = ["f_G_std", "unaffected_std", "delayed_std", "disconnected_std"]
    if has_lcc:
        std_cols.append("lcc_std")
    if has_components:
        std_cols.append("components_std")
    for col in std_cols:
        summary[col] = summary[col].fillna(0.0)

    if 0 not in set(summary["intensity_pct"]):
        baseline_trials = int(summary["n_trials"].max()) if len(summary) else 0
        baseline_row = {
            "intensity_pct": 0,
            "f_G_mean": 100.0,
            "f_G_std": 0.0,
            "unaffected_mean": 100.0,
            "unaffected_std": 0.0,
            "delayed_mean": 0.0,
            "delayed_std": 0.0,
            "disconnected_mean": 0.0,
            "disconnected_std": 0.0,
            "n_trials": baseline_trials,
        }
        if has_lcc:
            # Undisrupted network is one fully connected piece -> LCC ratio = 100%.
            baseline_row["lcc_mean"] = 100.0
            baseline_row["lcc_std"] = 0.0
        if has_components:
            baseline_row["components_mean"] = 1.0
            baseline_row["components_std"] = 0.0
        baseline_row = pd.DataFrame([baseline_row])
        summary = pd.concat([baseline_row, summary], ignore_index=True)

    summary = summary.sort_values("intensity_pct").reset_index(drop=True)
    return summary


def summarize_deterministic_by_intensity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Same percent conversion as the random summary, no trial aggregation
    needed -- both Targeted and Adaptive are deterministic, one row per
    intensity already.

    lcc_ratio is OPTIONAL (see summarize_random_by_intensity's docstring
    for why) -- carried through as lcc_pct when present, skipped otherwise.
    """
    df = df.copy().sort_values("intensity_pct")
    df["unaffected"] = 100 * df["n_unaffected"] / df["n_pairs"]
    df["delayed"] = 100 * df["n_delayed"] / df["n_pairs"]
    df["disconnected"] = 100 * df["n_disconnected"] / df["n_pairs"]
    df["f_G_pct"] = 100 * df["f_G"]

    has_lcc = "lcc_ratio" in df.columns
    if has_lcc:
        df["lcc_pct"] = 100 * df["lcc_ratio"]

    if 0 not in set(df["intensity_pct"]):
        baseline_row = {
            "intensity_pct": 0,
            "f_G_pct": 100.0,
            "unaffected": 100.0,
            "delayed": 0.0,
            "disconnected": 0.0,
        }
        if has_lcc:
            baseline_row["lcc_pct"] = 100.0
        df = pd.concat([pd.DataFrame([baseline_row]), df], ignore_index=True, sort=False)

    df = df.sort_values("intensity_pct").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Fig. 3 -- delay characterization (line + marker, per-category, w/ error bars)
# ---------------------------------------------------------------------------

def _draw_delay_panel(ax, summary: pd.DataFrame, deterministic: bool) -> None:
    """
    Draw one Fig. 3 panel: three lines (unaffected/delayed/disconnected).
    Error bars (+/- 1 std) only when the data is stochastic (random);
    plain lines when deterministic (targeted OR adaptive).
    """
    import matplotlib.ticker as mticker

    x = summary["intensity_pct"]
    categories = [
        ("unaffected", COLOR_UNAFFECTED, "unaffected"),
        ("delayed", COLOR_DELAYED, "delayed"),
        ("disconnected", COLOR_DISCONNECTED, "disconnected"),
    ]

    for key, color, label in categories:
        if deterministic:
            y = summary[key]
            ax.plot(x, y, "-o", color=color, markersize=3, linewidth=1.2, label=label)
        else:
            y = summary[f"{key}_mean"]
            yerr = summary[f"{key}_std"]
            ax.errorbar(
                x, y, yerr=yerr, fmt="-o", color=color, markersize=3,
                linewidth=1.2, elinewidth=0.8, capsize=0, label=label,
            )

    max_x = max(50, int(np.ceil(float(x.max()) / 5.0) * 5))
    ax.set_ylim(-2, 102)
    ax.set_xlim(0, max_x)
    ax.set_xticks(np.arange(0, max_x + 1, 5))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}%"))
    ax.grid(alpha=0.3)
    ax.legend(loc="center right", fontsize=8, framealpha=0.9)


def plot_fig3_delay_characterization(panels: dict, out_path: Path) -> None:
    """
    Reproduce Fig. 3's grid: rows = {Targeted, Adaptive, Random} (only rows
    with at least one network's data are drawn), columns = networks
    provided (Highway and/or Railway), in that order.

    `panels` shape:
        {
            "Highway": {"targeted": df_or_None, "adaptive": df_or_None, "random": df},
            "Railway": {"targeted": df_or_None, "adaptive": df_or_None, "random": df_or_None},
        }
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    networks = [
        name for name, d in panels.items()
        if any(d.get(rt) is not None for rt in ROW_TYPE_ORDER)
    ]
    if not networks:
        raise ValueError("No data provided for Fig. 3 -- nothing to plot.")

    row_types = [
        rt for rt in ROW_TYPE_ORDER
        if any(panels[n].get(rt) is not None for n in networks)
    ]

    n_rows, n_cols = len(row_types), len(networks)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6.5 * n_cols, 4.2 * n_rows),
        squeeze=False, sharex=True,
    )

    for r, row_type in enumerate(row_types):
        for c, network in enumerate(networks):
            ax = axes[r][c]
            summary = panels[network].get(row_type)
            if summary is None:
                ax.axis("off")
                ax.text(
                    0.5, 0.5, f"{network} {row_type} data\nnot available",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, style="italic", color="gray",
                )
                continue
            _draw_delay_panel(ax, summary, deterministic=(row_type != "random"))
            if r == 0:
                ax.set_title(network, fontsize=13)
            if c == 0:
                ax.set_ylabel(ROW_LABEL_FIG3[row_type])
            if r == n_rows - 1:
                ax.set_xlabel("Disruption Intensity")

    fig.suptitle(
        "Impacts of the disruption based on delay characterization",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Fig. 4 -- network functionality f(G)
# ---------------------------------------------------------------------------

def _draw_functionality_panel(ax, summary: pd.DataFrame, deterministic: bool) -> None:
    """
    Draw one Fig. 4 panel: f(G) vs. intensity, single series.
    Targeted/Adaptive -> plain line. Random -> mean line +/- 1 std.
    """
    import matplotlib.ticker as mticker

    x = summary["intensity_pct"]
    if deterministic:
        ax.plot(x, summary["f_G_pct"], "-", color=COLOR_FUNCTIONALITY, linewidth=1.4)
    else:
        ax.errorbar(
            x, summary["f_G_mean"], yerr=summary["f_G_std"],
            fmt="-", color=COLOR_FUNCTIONALITY, linewidth=1.2,
            elinewidth=0.7, capsize=0,
        )

    max_x = max(50, int(np.ceil(float(x.max()) / 5.0) * 5))
    ax.set_ylim(-2, 102)
    ax.set_xlim(0, max_x)
    ax.set_xticks(np.arange(0, max_x + 1, 5))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}%"))
    ax.grid(alpha=0.3)


def plot_fig4_functionality(panels: dict, out_path: Path) -> None:
    """Reproduce Fig. 4's grid, same row/column logic as Fig. 3."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    networks = [
        name for name, d in panels.items()
        if any(d.get(rt) is not None for rt in ROW_TYPE_ORDER)
    ]
    if not networks:
        raise ValueError("No data provided for Fig. 4 -- nothing to plot.")

    row_types = [
        rt for rt in ROW_TYPE_ORDER
        if any(panels[n].get(rt) is not None for n in networks)
    ]

    n_rows, n_cols = len(row_types), len(networks)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6.5 * n_cols, 3.8 * n_rows),
        squeeze=False, sharex=True,
    )

    for r, row_type in enumerate(row_types):
        for c, network in enumerate(networks):
            ax = axes[r][c]
            summary = panels[network].get(row_type)
            if summary is None:
                ax.axis("off")
                ax.text(
                    0.5, 0.5, f"{network} {row_type} data\nnot available",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, style="italic", color="gray",
                )
                continue
            _draw_functionality_panel(ax, summary, deterministic=(row_type != "random"))
            if r == 0:
                ax.set_title(network, fontsize=13)
            if c == 0:
                ax.set_ylabel(ROW_LABEL_FIG4[row_type])
            if r == n_rows - 1:
                ax.set_xlabel("Disruption Intensity")

    fig.suptitle("Degradation of network functionality against different disruptions", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Fig. 5 -- largest connected component (LCC) ratio, structural fragmentation
# ---------------------------------------------------------------------------

def _lcc_summary_or_none(summary: Optional[pd.DataFrame], deterministic: bool):
    """Return `summary` unchanged if it actually carries LCC data, else None.

    Distinguishes two different reasons a Fig. 5 panel might be empty:
    the underlying disruption run doesn't exist at all (summary is
    already None, same as Fig. 3/4), vs. the run exists but predates the
    LCC/component-count columns (summary is present but missing
    'lcc_mean'/'lcc_pct'). Both render the same "not available" panel,
    but only the second case is worth a console warning -- the first is
    already explained by Fig. 3/4's own "no such CSV" messaging.
    """
    if summary is None:
        return None
    lcc_col = "lcc_pct" if deterministic else "lcc_mean"
    if lcc_col not in summary.columns:
        return None
    return summary


def build_lcc_panels(panels: dict) -> dict:
    """Derive the Fig. 5 (LCC) panel dict from the already-built Fig. 3/4
    panels dict, keeping only summaries that actually carry LCC data.

    A disruption run from before the LCC/component-tracking update (e.g.
    an older 7_3_adaptive_disruption.py CSV) has a valid summary for
    Fig. 3/4 but no lcc_ratio column -- that combination is dropped here
    with a printed note, rather than crashing Fig. 5 or silently plotting
    nothing with no explanation.
    """
    lcc_panels: dict = {}
    for network, rows in panels.items():
        lcc_rows = {}
        for row_type, summary in rows.items():
            deterministic = row_type in ("targeted", "adaptive")
            filtered = _lcc_summary_or_none(summary, deterministic)
            if summary is not None and filtered is None:
                print(
                    f"  NOTE: {network} {row_type} results have no LCC/component "
                    f"columns (this run predates that metric) -- skipping its "
                    f"Fig. 5 panel."
                )
            lcc_rows[row_type] = filtered
        lcc_panels[network] = lcc_rows
    return lcc_panels


def _draw_lcc_panel(ax, summary: pd.DataFrame, deterministic: bool) -> None:
    """
    Draw one Fig. 5 panel: LCC ratio vs. intensity, single series.
    Targeted/Adaptive -> plain line. Random -> mean line +/- 1 std.

    LCC ratio is a STRUCTURAL metric (how much of the network is still
    one connected piece), distinct from Fig. 4's functionality (a
    FUNCTIONAL metric: how much slower/less reachable are the routes that
    still exist) -- the two can diverge, e.g. a network can stay 100%
    connected (LCC ratio = 100%) while functionality drops because
    surviving routes got longer, or LCC can crater while functionality
    still looks moderate because most county pairs happen to sit in the
    surviving largest piece.
    """
    import matplotlib.ticker as mticker

    x = summary["intensity_pct"]
    if deterministic:
        ax.plot(x, summary["lcc_pct"], "-", color=COLOR_LCC, linewidth=1.4)
    else:
        ax.errorbar(
            x, summary["lcc_mean"], yerr=summary["lcc_std"],
            fmt="-", color=COLOR_LCC, linewidth=1.2,
            elinewidth=0.7, capsize=0,
        )

    max_x = max(50, int(np.ceil(float(x.max()) / 5.0) * 5))
    ax.set_ylim(-2, 102)
    ax.set_xlim(0, max_x)
    ax.set_xticks(np.arange(0, max_x + 1, 5))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}%"))
    ax.grid(alpha=0.3)


def plot_fig5_lcc_degradation(panels: dict, out_path: Path) -> None:
    """Reproduce the Fig. 3/4 row/column grid for LCC ratio: rows =
    {Targeted, Adaptive, Random} (only rows with at least one network's
    LCC data are drawn), columns = networks with LCC data provided.

    `panels` shape is the output of build_lcc_panels() -- same shape as
    Fig. 3/4's panels dict, but with any (network, row_type) combination
    lacking LCC data already replaced with None.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    networks = [
        name for name, d in panels.items()
        if any(d.get(rt) is not None for rt in ROW_TYPE_ORDER)
    ]
    if not networks:
        raise ValueError("No LCC data provided for Fig. 5 -- nothing to plot.")

    row_types = [
        rt for rt in ROW_TYPE_ORDER
        if any(panels[n].get(rt) is not None for n in networks)
    ]

    n_rows, n_cols = len(row_types), len(networks)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6.5 * n_cols, 3.8 * n_rows),
        squeeze=False, sharex=True,
    )

    for r, row_type in enumerate(row_types):
        for c, network in enumerate(networks):
            ax = axes[r][c]
            summary = panels[network].get(row_type)
            if summary is None:
                ax.axis("off")
                ax.text(
                    0.5, 0.5, f"{network} {row_type} LCC data\nnot available",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, style="italic", color="gray",
                )
                continue
            _draw_lcc_panel(ax, summary, deterministic=(row_type != "random"))
            if r == 0:
                ax.set_title(network, fontsize=13)
            if c == 0:
                ax.set_ylabel(ROW_LABEL_FIG5[row_type])
            if r == n_rows - 1:
                ax.set_xlabel("Disruption Intensity")

    fig.suptitle(
        "Structural fragmentation under disruption (largest connected component)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot disruption results (delay characterization and "
                     "functionality degradation) for the HIGHWAY and "
                     "RAILWAY county-to-county disruption sweeps, with an "
                     "optional Adaptive row alongside Targeted and Random."
    )
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV,
                         help=f"Highway random-disruption trials CSV (default: {DEFAULT_RESULTS_CSV})")
    parser.add_argument(
        "--targeted-csv", type=Path, default=None,
        help=f"Optional highway targeted-disruption CSV. If omitted, "
             f"auto-detects {DEFAULT_TARGETED_CSV}.",
    )
    parser.add_argument(
        "--adaptive-csv", type=Path, default=None,
        help=f"Optional highway adaptive-disruption CSV (from "
             f"7_3_adaptive_disruption.py). If omitted, auto-detects "
             f"{DEFAULT_ADAPTIVE_CSV}.",
    )
    parser.add_argument(
        "--rail-results-csv", type=Path, default=None,
        help=f"Optional railway random-disruption trials CSV. If omitted, "
             f"auto-detects {DEFAULT_RAIL_RESULTS_CSV} when railway "
             f"plotting is enabled.",
    )
    parser.add_argument(
        "--rail-targeted-csv", type=Path, default=None,
        help=f"Optional railway targeted-disruption CSV. If omitted, "
             f"auto-detects {DEFAULT_RAIL_TARGETED_CSV} when railway plotting is enabled.",
    )
    parser.add_argument(
        "--rail-adaptive-csv", type=Path, default=None,
        help=f"Optional railway adaptive-disruption CSV. If omitted, "
             f"auto-detects {DEFAULT_RAIL_ADAPTIVE_CSV} when railway plotting is enabled.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                         help=f"Where to save the two PNGs (default: {DEFAULT_OUTPUT_DIR})")
    return parser.parse_args()


def _resolve_optional_csv(cli_path: Optional[Path], default_path: Path, label: str) -> Optional[Path]:
    """Resolve an optional CSV path with auto-detect and graceful skip."""
    if cli_path is not None:
        if cli_path.is_file() and cli_path.stat().st_size > 0:
            return cli_path
        if cli_path.exists():
            print(f"{label} CSV is empty or not a regular file at {cli_path} -- "
                  f"skipping {label} plot.")
            return None
        print(f"{label} CSV not found at {cli_path} -- skipping {label} plot.")
        return None

    if default_path.is_file() and default_path.stat().st_size > 0:
        print(f"Auto-detected {label} CSV: {default_path}")
        return default_path

    if default_path.exists():
        print(f"{label} CSV is empty or not a regular file at {default_path} -- "
              f"skipping {label} plot.")
        return None

    print(f"No {label} CSV found -- skipping {label} plot.")
    return None


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("STEP 5d -- Plotting disruption results")
    print("=" * 70)

    panels_fig3: dict = {}
    panels_fig4: dict = {}

    # --- Highway (random required; targeted/adaptive optional) ---
    highway_random_df = load_random_results(args.results_csv)
    highway_random_summary = summarize_random_by_intensity(highway_random_df)

    highway_targeted_summary = None
    highway_targeted_path = _resolve_optional_csv(args.targeted_csv, DEFAULT_TARGETED_CSV, "Highway targeted")
    if highway_targeted_path is not None:
        highway_targeted_df = load_deterministic_results(highway_targeted_path, "Targeted")
        highway_targeted_summary = summarize_deterministic_by_intensity(highway_targeted_df)

    highway_adaptive_summary = None
    highway_adaptive_path = _resolve_optional_csv(args.adaptive_csv, DEFAULT_ADAPTIVE_CSV, "Highway adaptive")
    if highway_adaptive_path is not None:
        highway_adaptive_df = load_deterministic_results(highway_adaptive_path, "Adaptive")
        highway_adaptive_summary = summarize_deterministic_by_intensity(highway_adaptive_df)

    panels_fig3["Highway"] = {
        "targeted": highway_targeted_summary,
        "adaptive": highway_adaptive_summary,
        "random": highway_random_summary,
    }
    panels_fig4["Highway"] = dict(panels_fig3["Highway"])

    # --- Railway (optional, but auto-enabled if any default railway file
    # exists -- matching the "both networks by default" convention used
    # by 7_1/7_2/7_3, on top of always honoring an explicit --rail-* flag) ---
    rail_requested = (
        args.rail_results_csv is not None
        or args.rail_targeted_csv is not None
        or args.rail_adaptive_csv is not None
        or DEFAULT_RAIL_RESULTS_CSV.exists()
        or DEFAULT_RAIL_TARGETED_CSV.exists()
        or DEFAULT_RAIL_ADAPTIVE_CSV.exists()
    )
    if rail_requested:
        rail_random_summary = None
        rail_results_path = _resolve_optional_csv(
            args.rail_results_csv, DEFAULT_RAIL_RESULTS_CSV, "Railway random"
        )
        if rail_results_path is not None:
            rail_random_df = load_random_results(rail_results_path)
            rail_random_summary = summarize_random_by_intensity(rail_random_df)

        rail_targeted_summary = None
        rail_targeted_path = _resolve_optional_csv(
            args.rail_targeted_csv, DEFAULT_RAIL_TARGETED_CSV, "Railway targeted"
        )
        if rail_targeted_path is not None:
            rail_targeted_df = load_deterministic_results(rail_targeted_path, "Targeted")
            rail_targeted_summary = summarize_deterministic_by_intensity(rail_targeted_df)

        rail_adaptive_summary = None
        rail_adaptive_path = _resolve_optional_csv(
            args.rail_adaptive_csv, DEFAULT_RAIL_ADAPTIVE_CSV, "Railway adaptive"
        )
        if rail_adaptive_path is not None:
            rail_adaptive_df = load_deterministic_results(rail_adaptive_path, "Adaptive")
            rail_adaptive_summary = summarize_deterministic_by_intensity(rail_adaptive_df)

        if any(s is not None for s in (rail_random_summary, rail_targeted_summary, rail_adaptive_summary)):
            panels_fig3["Railway"] = {
                "targeted": rail_targeted_summary,
                "adaptive": rail_adaptive_summary,
                "random": rail_random_summary,
            }
            panels_fig4["Railway"] = dict(panels_fig3["Railway"])
        else:
            print("No railway CSVs actually found -- plotting Highway only.")
    else:
        print("No railway CSVs given or auto-detected -- plotting Highway only.")

    plot_fig3_delay_characterization(panels_fig3, args.output_dir / FIG3_FILENAME)
    plot_fig4_functionality(panels_fig4, args.output_dir / FIG4_FILENAME)

    panels_fig5 = build_lcc_panels(panels_fig3)
    any_lcc_data = any(
        summary is not None
        for rows in panels_fig5.values()
        for summary in rows.values()
    )
    if any_lcc_data:
        plot_fig5_lcc_degradation(panels_fig5, args.output_dir / FIG5_FILENAME)
    else:
        print(
            "\nNo LCC/component data found in any input CSV -- skipping the LCC "
            "degradation figure. Re-run 7_1_random_disruption.py / "
            "7_2_targeted_disruption.py (and 7_3, once updated) to pick up the "
            "LCC/component-count columns, then re-run this script."
        )

    print("\nDone.")


if __name__ == "__main__":
    main()