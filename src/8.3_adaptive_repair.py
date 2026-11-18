#!/usr/bin/env python3
"""
Step 6c -- Adaptive repair (reverse of Step 5c adaptive disruption).

Replays 7_3_adaptive_disruption.py's batch-adaptive removal procedure up
to --intensity% (default 50%) to reconstruct the damaged network, then
reverses it using a BATCH-ADAPTIVE strategy:

Instead of recomputing betweenness after every single edge restoration
(which is extremely expensive), the repair is done in batches of ~1% of
total network length:

  1. Build a "candidate" graph = current (partially repaired) graph PLUS
     every still-removed edge added back.
  2. Compute county-to-county edge betweenness centrality on that
     candidate graph ONCE, restricted to sources=C, targets=C -- the
     same subset measure 6_edge_betweenness.py / 7_3 use.
  3. Rank ALL still-removed edges by their centrality in the candidate
     graph (descending).
  4. Restore edges from that ranking until ~1% of total network length
     has been restored in this batch.
  5. Recompute centrality (go back to step 1) for the next batch.
  6. Repeat until every removed edge is restored.

This reduces the number of betweenness computations from N (one per edge)
to roughly 50 (one per 1%-of-network batch), giving a massive speedup
while retaining the adaptive character: the repairer re-evaluates which
edges matter most as the network recovers.

Output
------
    reports/<network>_adaptive_repair.csv
    data/analysis/<network>_adaptive_repair.pkl

Usage
-----
    python 8_3_adaptive_repair.py                  # BOTH networks
    python 8_3_adaptive_repair.py --network highway
"""
from __future__ import annotations

import argparse
import importlib.util
import pickle
import time
from pathlib import Path

import networkx as nx
import pandas as pd


_COMMON_UTILITIES_PATH = Path(__file__).with_name("8.0_common_repair_utilities.py")
_COMMON_UTILITIES_SPEC = importlib.util.spec_from_file_location(
    "common_repair_utilities", _COMMON_UTILITIES_PATH
)
if _COMMON_UTILITIES_SPEC is None or _COMMON_UTILITIES_SPEC.loader is None:
    raise ImportError(f"Unable to load repair utilities from {_COMMON_UTILITIES_PATH}")
rc = importlib.util.module_from_spec(_COMMON_UTILITIES_SPEC)
_COMMON_UTILITIES_SPEC.loader.exec_module(rc)


def rank_remaining_for_repair(G_current, remaining, C):
    """Compute centrality on a candidate graph (current + all remaining
    edges restored) and return the remaining edges sorted by centrality
    descending.

    Returns a list of (index_into_remaining, edge_id, data, score) tuples,
    sorted from highest to lowest centrality.
    """
    G_candidate = G_current.copy()
    for edge_id, data in remaining:
        rc.restore_edge_by_id(G_candidate, edge_id, data)

    centrality = nx.edge_betweenness_centrality_subset(
        G_candidate, sources=C, targets=C, weight=rc.WEIGHT_ATTR, normalized=False,
    )

    # Build list of (original_index, edge_id, data, score) for all remaining
    indexed = []
    for i, (edge_id, data) in enumerate(remaining):
        score = centrality.get(edge_id, 0.0)
        indexed.append((i, edge_id, data, score))

    # Sort descending by centrality score
    indexed.sort(key=lambda x: (-x[3], tuple(repr(y) for y in x[1])))
    return indexed


def run_one_network(network: str, args: argparse.Namespace) -> bool:
    graph_path = rc.SMOOTHED_GRAPH_PATH[network]
    baseline_path = rc.BASELINE_MATRIX_PATH[network]

    print()
    print("#" * 70)
    print(f"# NETWORK: {network}")
    print("#" * 70)

    if not graph_path.exists():
        print(f"  WARNING: smoothed graph not found, skipping this network:\n    {graph_path}")
        return False

    rc._print_banner(f"STEP 6c -- ADAPTIVE REPAIR (BATCH-RECOMPUTE) [{network}]")
    G, C = rc.load_graph(graph_path)
    baseline = rc.load_baseline(baseline_path, G, C)
    total_km = rc.total_network_length_km(G)
    print(f"Total intact network length: {total_km:,.1f} km")

    print(f"\nReplaying adaptive disruption to {args.intensity}% to "
          f"reconstruct the damaged network...")
    G_disrupted, removed_edges_with_score = rc.build_adaptive_disrupted_state(
        G, C, args.intensity
    )
    remaining = [(e, d) for e, d, _score in removed_edges_with_score]
    n_total = len(remaining)
    if n_total == 0:
        print("  Nothing was removed at this intensity -- nothing to repair.")
        return False

    # Compute total length of removed edges (for batch sizing)
    total_removed_km = sum(float(d[rc.LENGTH_ATTR]) for _, d in remaining)
    # Each batch restores ~1% of total network length
    batch_km_target = total_km * 0.01
    n_batches_est = max(1, int(total_removed_km / batch_km_target) + 1)

    print(f"\n  Removed edges to restore: {n_total:,} ({total_removed_km:,.1f} km)")
    print(f"  Batch size: ~{batch_km_target:,.1f} km per ranking recompute")
    print(f"  Estimated betweenness computations: ~{n_batches_est}")

    output_csv = Path(f"reports/{network}_adaptive_repair.csv")
    output_pickle = Path(f"data/analysis/{network}_adaptive_repair.pkl")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_pickle.parent.mkdir(parents=True, exist_ok=True)

    G_current = G_disrupted.copy()
    rows = []
    repaired_km = 0.0

    metrics0 = rc.evaluate_state(G_current, C, baseline, total_km)
    rows.append({
        "step": 0, "n_repaired": 0, "n_remaining": n_total,
        "repaired_length_km": 0.0, "cumulative_pct_of_total": 0.0,
        "last_repaired_edge": None, "last_repair_centrality": None,
        **metrics0, "elapsed_s": 0.0,
    })

    try:
        from tqdm import tqdm
        bar = tqdm(total=n_total, desc="Adaptive repair", unit="edge")
    except ImportError:
        bar = None
        print(f"Restoring {n_total} edge(s) adaptively (install tqdm for a progress bar)...")

    t0 = time.monotonic()
    step = 0
    batch_num = 0

    while remaining:
        # -- BATCH-ADAPTIVE: recompute ranking ONCE per batch --
        batch_num += 1
        t_rank = time.monotonic()
        msg = (f"\n  Batch {batch_num}: recomputing betweenness on candidate graph "
               f"({G_current.number_of_edges():,} edges + {len(remaining):,} candidates)...")
        if bar is not None:
            bar.write(msg)
        else:
            print(msg)

        ranked = rank_remaining_for_repair(G_current, remaining, C)
        rank_time = time.monotonic() - t_rank

        msg = f"  Ranking done in {rank_time:.1f}s -- restoring edges from this ranking..."
        if bar is not None:
            bar.write(msg)
        else:
            print(msg)

        # Restore edges from this ranking until we've restored ~1% of
        # total network km, or we run out of ranked edges.
        batch_km = 0.0
        restored_indices = set()  # Track which original indices we restore

        for _orig_idx, edge_id, data, score in ranked:
            if batch_km >= batch_km_target and remaining:
                # We've restored enough for this batch; recompute ranking
                break

            # Find and remove this edge from remaining by identity
            # (we track by original index, but remaining may have shifted)
            found_idx = None
            for ri, (re_id, _re_data) in enumerate(remaining):
                if re_id == edge_id:
                    found_idx = ri
                    break
            if found_idx is None:
                continue  # Already restored in this batch (shouldn't happen)

            t_step = time.monotonic()
            rc.restore_edge_by_id(G_current, edge_id, data)
            edge_km = float(data[rc.LENGTH_ATTR])
            repaired_km += edge_km
            batch_km += edge_km
            remaining.pop(found_idx)
            step += 1

            metrics = rc.evaluate_state(G_current, C, baseline, total_km)
            rows.append({
                "step": step, "n_repaired": step, "n_remaining": len(remaining),
                "repaired_length_km": repaired_km,
                "cumulative_pct_of_total": 100 * repaired_km / total_km,
                "last_repaired_edge": repr(edge_id),
                "last_repair_centrality": score,
                **metrics,
                "elapsed_s": time.monotonic() - t_step,
            })

            if bar is not None:
                bar.set_postfix(lcc=f"{metrics['lcc_ratio']:.3f}", f_G=f"{metrics['functionality']:.3f}")
                bar.update(1)
            elif step % 10 == 0 or not remaining:
                print(f"  {step}/{n_total} repaired | LCC={metrics['lcc_ratio']:.4f} | "
                      f"f(G)={metrics['functionality']:.4f}")

    if bar is not None:
        bar.close()

    elapsed = time.monotonic() - t0
    print(f"\nAdaptive repair completed in {elapsed:,.1f}s ({n_total:,} edges restored, "
          f"{batch_num} betweenness recomputations).")

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
    p = argparse.ArgumentParser(
        description="Adaptive repair (batch-recompute): reverse of adaptive "
                    "disruption. Recomputes betweenness once per ~1%% batch "
                    "instead of after every single edge restoration."
    )
    p.add_argument("--network", choices=sorted(rc.NETWORK_LABELS), default=None,
                    help="Restrict to one network. If omitted, BOTH highway and railway run.")
    p.add_argument("--intensity", type=float, default=rc.DEFAULT_REPAIR_START_INTENSITY,
                    help="Disruption intensity to start repair from (default: 50).")
    return p.parse_args()


def main():
    args = parse_args()
    networks = [args.network] if args.network else list(rc.NETWORK_LABELS)
    print(f"Running Step 6c (adaptive repair) for: {', '.join(networks)}")

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