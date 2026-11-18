#!/usr/bin/env python3
"""
Step 6b -- Targeted repair (reverse of Step 5b targeted disruption).

Replays 7_2_targeted_disruption.py's own fixed descending-betweenness
removal order up to --intensity% (default 50%, matching the repair
phase's starting point) to reconstruct the damaged network, loading
7_2's cached centrality pickle if present
(data/analysis/<network>_targeted_centrality.pkl) rather than
recomputing it.

Repair order: the removed edges are restored in DESCENDING ORIGINAL
BETWEENNESS. Because targeted disruption itself removed edges in
descending centrality order in the first place, sorting the removed set
again by descending centrality reproduces that exact same order -- so
the single most critical corridor that was removed is also the first
one restored. This matches "restore what matters most, first."

Output
------
    reports/<network>_targeted_repair.csv    (one row per repair step)
    data/analysis/<network>_targeted_repair.pkl

Usage
-----
    python 9_2_targeted_repair.py                  # BOTH networks
    python 9_2_targeted_repair.py --network highway
"""
from __future__ import annotations

import argparse
import importlib.util
import pickle
import time
from pathlib import Path

import pandas as pd


_COMMON_UTILITIES_PATH = Path(__file__).with_name("8.0_common_repair_utilities.py")
_COMMON_UTILITIES_SPEC = importlib.util.spec_from_file_location(
    "common_repair_utilities", _COMMON_UTILITIES_PATH
)
if _COMMON_UTILITIES_SPEC is None or _COMMON_UTILITIES_SPEC.loader is None:
    raise ImportError(f"Unable to load repair utilities from {_COMMON_UTILITIES_PATH}")
rc = importlib.util.module_from_spec(_COMMON_UTILITIES_SPEC)
_COMMON_UTILITIES_SPEC.loader.exec_module(rc)


def run_one_network(network: str, args: argparse.Namespace) -> bool:
    graph_path = rc.SMOOTHED_GRAPH_PATH[network]
    baseline_path = rc.BASELINE_MATRIX_PATH[network]
    centrality_path = rc.TARGETED_CENTRALITY_PATH[network]

    print()
    print("#" * 70)
    print(f"# NETWORK: {network}")
    print("#" * 70)

    if not graph_path.exists():
        print(f"  WARNING: smoothed graph not found, skipping this network:\n    {graph_path}")
        return False

    rc._print_banner(f"STEP 6b -- TARGETED REPAIR [{network}]")
    G, C = rc.load_graph(graph_path)
    baseline = rc.load_baseline(baseline_path, G, C)
    total_km = rc.total_network_length_km(G)
    print(f"Total intact network length: {total_km:,.1f} km")

    cached = rc.load_cached_targeted_centrality(centrality_path, G)
    if cached is not None:
        print(f"Loaded cached centrality from: {centrality_path}")

    print(f"\nReplaying targeted disruption to {args.intensity}% to "
          f"reconstruct the damaged network...")
    G_disrupted, removed_edges = rc.build_targeted_disrupted_state(
        G, C, args.intensity, centrality=cached
    )
    if not removed_edges:
        print("  Nothing was removed at this intensity -- nothing to repair.")
        return False

    output_csv = Path(f"reports/{network}_targeted_repair.csv")
    output_pickle = Path(f"data/analysis/{network}_targeted_repair.pkl")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_pickle.parent.mkdir(parents=True, exist_ok=True)

    G_current = G_disrupted.copy()
    rows = []
    repaired_km = 0.0
    n_total = len(removed_edges)

    metrics0 = rc.evaluate_state(G_current, C, baseline, total_km)
    rows.append({
        "step": 0, "n_repaired": 0, "n_remaining": n_total,
        "repaired_length_km": 0.0, "cumulative_pct_of_total": 0.0,
        "last_repaired_edge": None, "last_repair_centrality": None,
        **metrics0, "elapsed_s": 0.0,
    })

    try:
        from tqdm import tqdm
        edge_iter = tqdm(list(enumerate(removed_edges, start=1)), desc="Targeted repair", unit="edge")
    except ImportError:
        edge_iter = list(enumerate(removed_edges, start=1))
        print(f"Restoring {n_total} edge(s) in descending original-centrality order "
              f"(install tqdm for a progress bar)...")

    t0 = time.monotonic()
    for step, (edge_id, data, score) in edge_iter:
        t_step = time.monotonic()
        rc.restore_edge_by_id(G_current, edge_id, data)
        repaired_km += float(data[rc.LENGTH_ATTR])

        metrics = rc.evaluate_state(G_current, C, baseline, total_km)
        rows.append({
            "step": step, "n_repaired": step, "n_remaining": n_total - step,
            "repaired_length_km": repaired_km,
            "cumulative_pct_of_total": 100 * repaired_km / total_km,
            "last_repaired_edge": repr(edge_id),
            "last_repair_centrality": score,
            **metrics,
            "elapsed_s": time.monotonic() - t_step,
        })

    elapsed = time.monotonic() - t0
    print(f"\nTargeted repair completed in {elapsed:,.1f}s ({n_total:,} edges restored).")

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv} ({len(df):,} rows)")

    with open(output_pickle, "wb") as f:
        pickle.dump(df, f)
    print(f"Saved: {output_pickle}")

    print("\nEndpoint check:")
    print(f"  Before any repair : LCC ratio={rows[0]['lcc_ratio']:.4f}  functionality={rows[0]['functionality']:.4f}")
    print(f"  Fully repaired    : LCC ratio={rows[-1]['lcc_ratio']:.4f}  functionality={rows[-1]['functionality']:.4f}")

    return True


def parse_args():
    p = argparse.ArgumentParser(description="Targeted repair: reverse of targeted disruption.")
    p.add_argument("--network", choices=sorted(rc.NETWORK_LABELS), default=None,
                    help="Restrict to one network. If omitted, BOTH highway and railway run.")
    p.add_argument("--intensity", type=float, default=rc.DEFAULT_REPAIR_START_INTENSITY,
                    help="Disruption intensity to start repair from (default: 50).")
    return p.parse_args()


def main():
    args = parse_args()
    networks = [args.network] if args.network else list(rc.NETWORK_LABELS)
    print(f"Running Step 6b (targeted repair) for: {', '.join(networks)}")

    ran, skipped = [], []
    for network in networks:
        ok = run_one_network(network, args)
        (ran if ok else skipped).append(network)

    print()
    rc._print_banner("ALL NETWORKS COMPLETE")
    print(f"Ran successfully: {ran or '(none)'}")
    if skipped:
        print(f"Skipped: {skipped}")
    if not ran:
        raise SystemExit(
            "\nNo network was actually processed -- every requested "
            "smoothed graph pickle was missing. Run 4_graph_smoothing.py "
            "first."
        )


if __name__ == "__main__":
    main()