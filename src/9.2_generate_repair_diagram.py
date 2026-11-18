#!/usr/bin/env python3
"""Plot repair results using the disruption-diagram layout."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

NETWORK_LABELS = ["highway", "railway"]
STRATEGIES = ["targeted", "adaptive", "random"]
PATHS = {(n, "random"): Path(f"reports/{n}_random_repair_summary.csv") for n in NETWORK_LABELS}
PATHS.update({(n, "targeted"): Path(f"reports/{n}_targeted_repair.csv") for n in NETWORK_LABELS})
PATHS.update({(n, "adaptive"): Path(f"reports/{n}_adaptive_repair.csv") for n in NETWORK_LABELS})

COLOR_UNAFFECTED = "#1f77b4"
COLOR_DELAYED = "#ff7f0e"
COLOR_DISCONNECTED = "#d62728"
COLOR_FUNCTIONALITY = "#2f2f2f"
COLOR_LCC = "#0f6e56"

TOTAL_NETWORK_KM = {
    "highway": 28949.219,
    "railway": 15894.330,
}
DEFAULT_REPAIR_RATE_KM_DAY = 50.0


def _column(df: pd.DataFrame, base: str, random: bool):
    if random and f"{base}_mean" in df:
        return df[f"{base}_mean"], df.get(f"{base}_std", 0.0)
    return df[base], 0.0


def load(
    network: str,
    strategy: str,
    repair_rates: dict[str, float] | None = None,
) -> pd.DataFrame | None:
    path = PATHS[(network, strategy)]
    if not path.is_file() or path.stat().st_size == 0:
        return None
    raw = pd.read_csv(path)
    random = strategy == "random"
    x_name = "cumulative_pct_of_total_mean" if random and "cumulative_pct_of_total_mean" in raw else "cumulative_pct_of_total"
    if "step" in raw:
        step_name = "step"
    elif "n_repaired" in raw:
        step_name = "n_repaired"
    else:
        return None

    rates = repair_rates or {}
    rate_km_day = rates.get(network, DEFAULT_REPAIR_RATE_KM_DAY)
    total_km = TOTAL_NETWORK_KM.get(network, 28949.219)

    if "repaired_length_km" in raw:
        repaired_km = raw["repaired_length_km"].to_numpy(dtype=float)
    elif "cumulative_pct_of_total_mean" in raw:
        repaired_km = (raw["cumulative_pct_of_total_mean"].to_numpy(dtype=float) / 100.0) * total_km
    else:
        repaired_km = (raw[x_name].to_numpy(dtype=float) / 100.0) * total_km

    days = repaired_km / rate_km_day

    result = pd.DataFrame({
        "pct_repaired": raw[x_name],
        "n_repaired": raw[step_name],
        "repaired_length_km": repaired_km,
        "days": days,
    })
    for base in ("frac_unaffected", "frac_delayed", "frac_disconnected", "functionality", "lcc_ratio"):
        if base not in raw and f"{base}_mean" not in raw:
            return None
        mean, std = _column(raw, base, random)
        result[base] = 100 * mean
        result[f"{base}_std"] = 100 * std
    result = result.sort_values("pct_repaired").reset_index(drop=True)
    if 0.0 not in set(result["pct_repaired"]):
        baseline = {
            "pct_repaired": 0.0,
            "n_repaired": 0.0,
            "repaired_length_km": 0.0,
            "days": 0.0,
        }
        for base in ("frac_unaffected", "functionality", "lcc_ratio"):
            baseline[base] = 100.0
            baseline[f"{base}_std"] = 0.0
        for base in ("frac_delayed", "frac_disconnected"):
            baseline[base] = 0.0
            baseline[f"{base}_std"] = 0.0
        result = pd.concat([pd.DataFrame([baseline]), result], ignore_index=True)
    return result


def _available_panels(repair_rates: dict[str, float] | None = None) -> dict:
    panels = {}
    for network in NETWORK_LABELS:
        for strategy in STRATEGIES:
            data = load(network, strategy, repair_rates)
            if data is not None:
                panels[(network, strategy)] = data
    if not panels:
        raise SystemExit("No repair-analysis CSVs found -- run the repair scripts first.")
    return panels


def _grid(panels: dict):
    networks = [n for n in NETWORK_LABELS if any((n, s) in panels for s in STRATEGIES)]
    strategies = [s for s in STRATEGIES if any((n, s) in panels for n in networks)]
    return networks, strategies


def _format_axis(ax, x_max: float) -> None:
    import matplotlib.ticker as mticker
    import numpy as np
    x_max = max(50, int(np.ceil(x_max / 5.0) * 5))
    ax.set_xlim(0, x_max)
    ax.set_xticks(np.arange(0, x_max + 1, 5))
    ax.set_ylim(-2, 102)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}%"))
    ax.grid(alpha=0.3)


def _plot_grid(panels: dict, metric: str, output: Path, title: str, ylabel: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    networks, strategies = _grid(panels)
    height = 4.2 if metric == "delay" else 3.8
    fig, axes = plt.subplots(len(strategies), len(networks), figsize=(6.5 * len(networks), height * len(strategies)), squeeze=False, sharex=True)
    for row, strategy in enumerate(strategies):
        for col, network in enumerate(networks):
            ax = axes[row][col]
            data = panels.get((network, strategy))
            if data is None:
                ax.axis("off")
                continue
            x = data["pct_repaired"]
            random = strategy == "random"
            if metric == "delay":
                series = [("frac_unaffected", COLOR_UNAFFECTED, "unaffected"), ("frac_delayed", COLOR_DELAYED, "delayed"), ("frac_disconnected", COLOR_DISCONNECTED, "disconnected")]
            else:
                color = COLOR_FUNCTIONALITY if metric == "functionality" else COLOR_LCC
                series = [(metric, color, ylabel)]
            for name, color, label in series:
                ax.plot(x, data[name], "-", color=color, linewidth=0.8, label=label)
            _format_axis(ax, float(x.max()))
            if row == 0:
                ax.set_title(network.capitalize(), fontsize=13)
            if col == 0:
                if metric == "delay":
                    row_label = "Route Categorization"
                elif metric == "functionality":
                    row_label = "Functionality"
                else:
                    row_label = "LCC Ratio"
                ax.set_ylabel(f"{row_label}\n({strategy.capitalize()})")
            if row == len(strategies) - 1:
                ax.set_xlabel("% of network length repaired")
            if metric == "delay":
                ax.legend(loc="center right", fontsize=8, framealpha=0.9)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


def _recovery_milestones(data: pd.DataFrame, count: int = 10) -> pd.DataFrame:
    """Reduce a repair trajectory to `count` labelled milestones.

    Milestones are spaced evenly along the arc length of the curve (in
    normalised time / performance space), so every plotted step covers a
    comparable stretch of the trajectory. This keeps fast-rising, flat, and
    slow-crawl regions all represented, instead of letting a long plateau
    consume most of the milestones. Interior points that repeat the previous
    performance level are dropped.
    """
    import numpy as np

    if len(data) <= count:
        return data.copy()
    x_name = "days" if "days" in data else ("n_repaired" if "n_repaired" in data else "pct_repaired")
    x = data[x_name].to_numpy(dtype=float)
    y = data["functionality"].clip(0, 100).to_numpy(dtype=float)
    x_span = float(x[-1] - x[0])
    if x_span <= 0:
        x_span = 1.0
    segments = np.hypot(np.abs(np.diff(x)) / x_span, np.abs(np.diff(y)) / 100.0)
    cumulative = np.concatenate(([0.0], np.cumsum(segments)))
    total = float(cumulative[-1])
    if total <= 0:
        return data.iloc[[0, len(data) - 1]].copy()
    targets = np.linspace(0.0, total, count)
    indices = sorted({int(np.searchsorted(cumulative, target, side="left")) for target in targets})
    indices[0], indices[-1] = 0, len(data) - 1
    ordered = sorted(set(indices))
    levels = y[ordered]
    keep = [0]
    for position in range(1, len(ordered) - 1):
        if levels[position] != levels[keep[-1]]:
            keep.append(position)
    keep.append(len(ordered) - 1)
    return data.iloc[[ordered[position] for position in keep]].copy()


def _days_axis_limit_and_tick(x_max: float) -> tuple[int, int]:
    """Pick round x_limit and tick interval in days."""
    import numpy as np

    for step in (30, 50, 25, 20, 15, 10, 60):
        n_ticks = x_max / step
        if 4.0 <= n_ticks <= 7.0:
            chosen_step = step
            break
    else:
        chosen_step = 50
    limit = int(np.ceil((x_max * 1.03) / chosen_step)) * chosen_step
    return limit, chosen_step


def _axis_limit(x_max: float) -> int:
    """Round the x limit up to a multiple of 1000 with headroom past the data."""
    import numpy as np

    return max(1000, int(np.ceil((x_max + 1.0) / 1000.0) * 1000))


def _tick_step(x_limit: float) -> int:
    """Pick a round tick interval that yields about five x-axis ticks."""
    import numpy as np

    raw = x_limit / 5.0
    magnitude = 10 ** int(np.floor(np.log10(raw)))
    for factor in (1, 2, 2.5, 5, 10):
        if factor * magnitude >= raw:
            return int(factor * magnitude)
    return int(10 * magnitude)


def _extend_to_limit(x, performance, x_limit):
    """Append a final plateau point so the last step is visible on the axis."""
    import numpy as np

    if x[-1] < x_limit:
        x = np.append(x, x_limit)
        performance = np.append(performance, performance[-1])
    return x, performance


def _plot_prf_recovery(
    data: pd.DataFrame,
    output: Path,
    network: str,
    strategy: str,
) -> None:
    """Create the two-panel stepped performance-response figure used in the paper."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    import numpy as np

    milestones = _recovery_milestones(data)
    tick_x = milestones["days"].to_numpy()
    performance = milestones["functionality"].clip(0, 100).to_numpy()
    service_loss = 100 - performance
    x_max = max(1.0, float(data["days"].max()))
    x_limit, tick_step = _days_axis_limit_and_tick(x_max)
    x, performance = _extend_to_limit(tick_x, performance, x_limit)

    teal = "#0f9f98"
    orange = "#f26b00"
    charcoal = "#222222"
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.7), sharey=True)

    ax = axes[0]
    ax.step(x, performance, where="post", color=charcoal, linewidth=1.8)
    ax.fill_between(
        x, performance, 100, step="post", facecolor="#d9d9d9",
        edgecolor="#555555", hatch="////", linewidth=0.5, alpha=0.8,
    )
    ax.plot([0, x_limit], [100, 100], "--", color="#777777", linewidth=0.9)
    ax.annotate(
        "disruptive event", xy=(tick_x[0], performance[0]), xytext=(tick_x[0], 108),
        ha="center", va="bottom", fontsize=9,
        arrowprops={"arrowstyle": "-|>", "color": charcoal, "lw": 0.9},
    )
    ax.set_title("(a) Performance response", fontsize=11)
    ax.set_xlabel("Recovery time (days)")
    ax.set_ylabel("Performance response (%)")
    ax.text(x_limit * 0.70, 101.5, "normal level", fontsize=9)
    ax.text(x_limit * 0.55, max(8, performance[-1] * 0.45), "R", fontsize=11)
    ax.text(x_limit * 0.55, (100 + performance[-1]) / 2, "S", fontsize=11)

    ax = axes[1]
    ax.fill_between(x, 0, performance, step="post", color=teal, alpha=0.98)
    ax.fill_between(x, performance, 100, step="post", color=orange, alpha=0.98)
    ax.plot([0, x_limit], [100, 100], "--", color="#555555", linewidth=0.9)
    ax.set_title("(b) Recovery and service loss", fontsize=11)
    ax.set_xlabel("Recovery time (days)")
    ax.text(x_limit * 0.55, max(8, performance[-1] * 0.45), "R", color="white", fontsize=11)
    ax.text(x_limit * 0.55, (100 + performance[-1]) / 2, "S", color="white", fontsize=11)
    ax.legend(
        handles=[
            Patch(facecolor=teal, label="R: recovered performance"),
            Patch(facecolor=orange, label="S: service loss"),
        ],
        loc="lower right", fontsize=8, frameon=False,
    )

    for axis in axes:
        axis.set_xlim(0, x_limit)
        axis.set_ylim(0, 112)
        axis.set_xticks(np.arange(0, x_limit + 1, tick_step))
        axis.grid(axis="y", alpha=0.2)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.suptitle(
        f"Performance response function of {network.capitalize()} network "
        f"under {strategy.capitalize()} repair",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


def _plot_prf_page(
    panels: dict[tuple[str, str], pd.DataFrame],
    output: Path,
    network: str,
) -> None:
    """Create one page containing the targeted, adaptive, and random PRF rows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    import numpy as np

    available = [panels[(network, s)] for s in STRATEGIES if (network, s) in panels]
    if not available:
        print(f"Skipping {network}: no repair CSVs found.")
        return
    x_max = max(float(data["days"].max()) for data in available)
    x_limit, tick_step = _days_axis_limit_and_tick(x_max)

    teal, orange, charcoal = "#0f9f98", "#f26b00", "#222222"
    fig, axes = plt.subplots(3, 2, figsize=(11, 12), sharey=True)
    for row, strategy in enumerate(STRATEGIES):
        data = panels.get((network, strategy))
        if data is None:
            axes[row, 0].axis("off")
            axes[row, 1].axis("off")
            continue
        milestones = _recovery_milestones(data)
        tick_x = milestones["days"].to_numpy()
        performance = milestones["functionality"].clip(0, 100).to_numpy()
        x, performance = _extend_to_limit(tick_x, performance, x_limit)

        left, right = axes[row]
        left.step(x, performance, where="post", color=charcoal, linewidth=1.2)
        left.fill_between(
            x, performance, 100, step="post", facecolor="#d9d9d9",
            edgecolor="#555555", hatch="////", linewidth=0.35, alpha=0.8,
        )
        left.plot([0, x_limit], [100, 100], "--", color="#777777", linewidth=0.7)
        left.set_title(f"({chr(97 + row * 2)}) {strategy.capitalize()} repair: response", fontsize=10)
        left.set_ylabel("Performance response (%)")
        left.text(x_limit * 0.70, 101.5, "normal level", fontsize=8)
        left.text(x_limit * 0.55, max(8, performance[-1] * 0.4), "R", fontsize=9)
        left.text(x_limit * 0.55, (100 + performance[-1]) / 2, "S", fontsize=9)

        right.fill_between(x, 0, performance, step="post", color=teal)
        right.fill_between(x, performance, 100, step="post", color=orange)
        right.plot([0, x_limit], [100, 100], "--", color="#555555", linewidth=0.7)
        right.set_title(f"({chr(98 + row * 2)}) {strategy.capitalize()} repair: R and S", fontsize=10)
        right.text(x_limit * 0.55, max(8, performance[-1] * 0.4), "R", color="white", fontsize=9)
        right.text(x_limit * 0.55, (100 + performance[-1]) / 2, "S", color="white", fontsize=9)
        if row == 0:
            right.legend(
                handles=[
                    Patch(facecolor=teal, label="R: recovered performance"),
                    Patch(facecolor=orange, label="S: service loss"),
                ],
                loc="lower right", fontsize=7, frameon=False,
            )

        for axis in (left, right):
            axis.set_xlim(0, x_limit)
            axis.set_ylim(0, 112)
            axis.set_xticks(np.arange(0, x_limit + 1, tick_step))
            axis.set_xlabel("Recovery time (days)")
            axis.grid(axis="y", alpha=0.2)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

    fig.suptitle(
        f"Performance response function of {network.capitalize()} network "
        "under different repair strategies",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


def _plot_network_comparison(
    panels: dict[tuple[str, str], pd.DataFrame],
    output: Path,
) -> None:
    """Direct comparison between Highway and Railway recovery curves across strategies."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    color_highway = "#1f77b4"
    color_railway = "#d62728"

    max_days = 0.0
    for strat in STRATEGIES:
        for net in NETWORK_LABELS:
            d = panels.get((net, strat))
            if d is not None:
                max_days = max(max_days, float(d["days"].max()))

    x_limit, tick_step = _days_axis_limit_and_tick(max_days)

    for idx, strategy in enumerate(STRATEGIES):
        ax = axes[idx]
        for net, col, lbl in [("highway", color_highway, "Highway"), ("railway", color_railway, "Railway")]:
            data = panels.get((net, strategy))
            if data is None:
                continue
            ms = _recovery_milestones(data)
            tick_x = ms["days"].to_numpy()
            perf = ms["functionality"].clip(0, 100).to_numpy()
            x, y = _extend_to_limit(tick_x, perf, x_limit)
            ax.step(x, y, where="post", color=col, linewidth=2.0, label=lbl)

        ax.plot([0, x_limit], [100, 100], "--", color="#777777", linewidth=0.8)
        ax.set_title(f"{strategy.capitalize()} repair", fontsize=11)
        ax.set_xlabel("Recovery time (days)")
        if idx == 0:
            ax.set_ylabel("Functionality recovery (%)")
            ax.legend(loc="lower right", framealpha=0.9)
        ax.set_xlim(0, x_limit)
        ax.set_ylim(-2, 105)
        ax.set_xticks(np.arange(0, x_limit + 1, tick_step))
        ax.grid(alpha=0.3)

    fig.suptitle("Highway vs Railway functionality recovery comparison (in days)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate repair-analysis diagrams.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--repair-rate-km-day",
        type=float,
        default=DEFAULT_REPAIR_RATE_KM_DAY,
        help="Unified daily repair rate in km/day (default: 50.0).",
    )
    parser.add_argument(
        "--highway-repair-rate",
        type=float,
        default=None,
        help="Daily repair rate for Highway in km/day (overrides unified rate).",
    )
    parser.add_argument(
        "--railway-repair-rate",
        type=float,
        default=None,
        help="Daily repair rate for Railway in km/day (overrides unified rate).",
    )
    args = parser.parse_args()

    repair_rates = {
        "highway": args.highway_repair_rate or args.repair_rate_km_day,
        "railway": args.railway_repair_rate or args.repair_rate_km_day,
    }

    panels = _available_panels(repair_rates)
    output_dir = args.output_dir
    _plot_grid(panels, "delay", output_dir / "fig6_repair_delay_characterization.png", "Impacts of the repair based on delay characterization", "Route percentage")
    _plot_grid(panels, "functionality", output_dir / "fig7_repair_functionality_recovery.png", "Recovery of network functionality against different repairs", "Functionality")
    _plot_grid(panels, "lcc_ratio", output_dir / "fig8_repair_lcc_recovery.png", "Structural recovery under repair (largest connected component)", "LCC Ratio")
    for network in NETWORK_LABELS:
        _plot_prf_page(
            panels,
            output_dir / f"fig9_{network}_all_strategies_prf_recovery.png",
            network,
        )
    _plot_network_comparison(
        panels,
        output_dir / "fig9_highway_vs_railway_comparison_days.png",
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
