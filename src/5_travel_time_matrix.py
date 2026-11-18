#!/usr/bin/env python3
"""
Step 4a -- County-to-county travel-time matrix (Illinois test run).

DATASET NOTE (updated to match the Illinois test run's naming convention):
-----------------------------------------------------------------------
The earlier draft of this script computed a fort-to-port matrix T
(|O| forts x |D| ports) from a single hardcoded STRAHNET graph
(`data/graph/smoothing/strahnet_smoothed.pkl`, a
`{"graph", "O_fort_nodes", "D_port_nodes"}` dict). Neither those two
disjoint facility sets nor that graph path exist in this pipeline. This
script has been rewritten to match what this project actually produces:

  * Facility set: ONE set of COUNTY nodes, `C`, not two disjoint sets
    `O`/`D`. Traced directly to `3_snap_counties.py`'s bookkeeping
    (`C = set(county_nodes)`) and `4_graph_smoothing.py`'s own output
    dict shape (`{"graph": G_smoothed, "C_county_nodes": sorted(C)}`).
    Since there is only one facility type, this is a county-to-county
    matrix on an UNDIRECTED graph: `t(c1, c2) == t(c2, c1)`, so this
    script computes the SYMMETRIC full matrix (for convenient
    `matrix.loc[a, b]` lookups either way round) but reports/logs only
    the `N = C(|C|, 2)` unique unordered pairs, matching
    `3_snap_counties.py`'s own "N = C(|C|, 2) possible county-to-county
    routes" bookkeeping -- not the |O| x |D| cross-product framing.

  * Graph input: traced directly to `4_graph_smoothing.py`'s own output
    convention:
        out_graph_path = output_graph_dir / f"{network}_smoothed.pkl"
    i.e. `data/graph/smoothing/highway_smoothed.pkl` and
    `.../railway_smoothed.pkl` -- BOTH highway and railway, not a single
    STRAHNET graph. Following the "both networks by default" convention
    already established by every other script in this pipeline (2_1
    through 4_graph_smoothing.py), `--network` is OPTIONAL: omitting it
    computes the matrix for BOTH graphs automatically, one after
    another, each producing its own output pickle and its own long-format
    CSV. A missing smoothed-graph pickle for one network prints a clear
    warning and skips just that network.

Everything else -- the actual algorithm -- is UNCHANGED from the
original script and still implements the paper's method (Section
2.3.1): for each origin node, run ONE `nx.single_source_dijkstra` call
(weight = `travel_time_min`) rather than one Dijkstra call per pair, then
read off the times to every other facility node from that single result.
This is far cheaper than an all-pairs loop of individual Dijkstra calls.

Since the underlying graph is undisrupted and fully connected (validated
at the end of the smoothing step), every t(c1, c2) computed here should
be finite. Under later disruption simulations, some entries will become
infinite (no path exists) -- that's expected there, not here.

Paths
-----
Input (per network):
    data/graph/smoothing/<network>_smoothed.pkl
        (4_graph_smoothing.py's own output_graph_dir/f"{network}_smoothed.pkl",
        a {"graph": nx.MultiGraph, "C_county_nodes": [...]} dict)

Output (per network):
    data/analysis/<network>_travel_time_matrix.pkl
        (dict: {"matrix": pandas.DataFrame (index=columns=county nodes,
         values=travel_time_min, symmetric), "C_county_nodes": [...]})
    reports/<network>_county_travel_time_long.csv
        (one row per UNIQUE county-to-county pair, i.e. C(|C|, 2) rows --
        not the full symmetric matrix, since (c1, c2) and (c2, c1) are the
        same undirected route)

Usage
-----
    # NEW DEFAULT: computes the matrix for BOTH highway and railway graphs
    python 5_travel_time_matrix.py

    # restrict to just one network
    python 5_travel_time_matrix.py --network highway
    python 5_travel_time_matrix.py --network railway

    # override paths (single-network mode only)
    python 5_travel_time_matrix.py --network highway --input path/to/other.pkl
"""

from __future__ import annotations

import argparse
import csv
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import pandas as pd

# ---------------------------------------------------------------------------
# Paths -- kept in sync by hand with 4_graph_smoothing.py's own output
# convention (output_graph_dir / f"{network}_smoothed.pkl").
# ---------------------------------------------------------------------------

SMOOTHED_INPUT_DIR = Path("data/graph/smoothing")   # 4_graph_smoothing.py's own OUTPUT_GRAPH_DIR
ANALYSIS_OUTPUT_DIR = Path("data/analysis")
REPORTS_DIR = Path("reports")

NETWORK_LABELS = ["highway", "railway"]

NETWORK_INPUT_FILENAMES = {
    network: f"{network}_smoothed.pkl" for network in NETWORK_LABELS
}

# HARDCODED DEFAULT NETWORK -- same pattern as every other script in this
# pipeline. Only used in single-file mode when --input is given without
# --network (belt-and-suspenders; --input requires --network, see
# parse_args, but this stays as a safety fallback).
HARDCODED_NETWORK = "highway"

WEIGHT_ATTR = "travel_time_min"

NodeId = Tuple[float, float]


def _print_banner(title: str) -> None:
    """Section-banner style matching the rest of this pipeline (2_2, 2_3, 2_7, 2_8, 3, 4)."""
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_smoothed_graph(path: Path) -> Tuple[nx.MultiGraph, List[NodeId]]:
    """Load a smoothed graph pickle and return (graph, county_nodes).

    Handles the {"graph": G, "C_county_nodes": C} dict shape that
    4_graph_smoothing.py actually writes -- NOT the old
    {"O_fort_nodes", "D_port_nodes"} shape this script used to expect.
    """
    if not path.exists():
        raise SystemExit(f"Input not found: {path}")

    with open(path, "rb") as f:
        obj = pickle.load(f)

    if not (isinstance(obj, dict) and "graph" in obj):
        raise SystemExit(
            f"{path} is not the expected {{'graph', 'C_county_nodes'}} dict "
            f"produced by 4_graph_smoothing.py -- refusing to guess."
        )

    G = obj["graph"]
    C = list(obj.get("C_county_nodes", []))

    if not C:
        raise SystemExit(
            f"Loaded graph but C_county_nodes ({len(C)}) is empty -- nothing "
            f"to compute a travel-time matrix over."
        )

    missing_C = [n for n in C if not G.has_node(n)]
    if missing_C:
        raise SystemExit(
            f"{len(missing_C)} county node(s) from C are not present in the "
            f"graph -- the input file is inconsistent. Re-run "
            f"4_graph_smoothing.py and check its 'all county nodes "
            f"preserved' validation line."
        )

    n_pairs = len(C) * (len(C) - 1) // 2
    print(f"Loaded graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    print(f"County nodes (C): {len(C):,}   "
          f"unique county-to-county routes N = C(|C|, 2) = {n_pairs:,}")
    return G, C


def facility_name_lookup(G: nx.MultiGraph, nodes: List[NodeId]) -> Dict[NodeId, str]:
    """Best-effort human-readable label per node, from the 'facility_names'
    node attribute set by 3_snap_counties.py. Falls back to the coordinate
    itself if no name is recorded."""
    lookup = {}
    for n in nodes:
        names = G.nodes.get(n, {}).get("facility_names", [])
        lookup[n] = "; ".join(names) if names else f"({n[0]:.6f}, {n[1]:.6f})"
    return lookup


# ---------------------------------------------------------------------------
# Travel-time matrix
# ---------------------------------------------------------------------------

def build_travel_time_matrix(
    G: nx.MultiGraph, C: List[NodeId]
) -> Tuple[pd.DataFrame, List[Tuple[NodeId, NodeId]]]:
    """
    Run single_source_dijkstra ONCE per county node (not once per pair),
    then read off the travel time to every other county from that single
    result.

    Because the graph is undirected and there is only one facility set,
    the resulting matrix is SYMMETRIC (t(c1, c2) == t(c2, c1)) with a zero
    diagonal. The full C x C matrix is still built (convenient for
    `matrix.loc[a, b]` lookups in either direction), but `unreachable`
    only records each unordered pair once (using the fixed order of `C`,
    i < j) so a caller doesn't double-count a single missing route as two.

    Returns (matrix, unreachable_pairs) where matrix is a DataFrame
    indexed and columned by county node, values in travel_time_min (or
    float('inf') for any pair with no path -- should not happen on a
    connected, undisrupted graph, but never silently hidden if it does).
    """
    rows: Dict[NodeId, Dict[NodeId, float]] = {}
    unreachable: List[Tuple[NodeId, NodeId]] = []

    start = time.monotonic()
    for i, c1 in enumerate(C, start=1):
        distances, _paths = nx.single_source_dijkstra(G, c1, weight=WEIGHT_ATTR)
        row = {}
        for j, c2 in enumerate(C):
            if c1 == c2:
                row[c2] = 0.0
                continue
            t = distances.get(c2, float("inf"))
            row[c2] = t
            if t == float("inf") and C.index(c1) < j:
                unreachable.append((c1, c2))
        rows[c1] = row

        if i % 100 == 0 or i == len(C):
            elapsed = time.monotonic() - start
            rate = i / elapsed if elapsed > 0 else 0.0
            print(f"  {i:,}/{len(C):,} counties processed ({rate:,.1f}/s, {elapsed:,.1f}s elapsed)")

    matrix = pd.DataFrame.from_dict(rows, orient="index")
    matrix = matrix.reindex(index=C, columns=C)  # guarantee full C x C shape/order
    return matrix, unreachable


def summarize_matrix(matrix: pd.DataFrame, unreachable: List[Tuple[NodeId, NodeId]]) -> None:
    n = matrix.shape[0]
    n_pairs = n * (n - 1) // 2

    print("\n" + "=" * 70)
    print("TRAVEL-TIME MATRIX SUMMARY")
    print("=" * 70)
    print(f"Shape: {n:,} x {n:,} counties -- {n_pairs:,} unique county-to-county routes "
          f"(symmetric matrix, zero diagonal)")

    if unreachable:
        print(f"\n*** WARNING: {len(unreachable):,} county-to-county pair(s) have NO path "
              f"(infinite travel time). On a graph that passed the smoothing step's "
              f"connectivity check, this should not happen -- investigate before "
              f"trusting this matrix. First few: ***")
        for c1, c2 in unreachable[:10]:
            print(f"    county {c1} -> county {c2}")
    else:
        print("All county-to-county pairs are reachable (no infinite entries) -- expected "
              "for an undisrupted, connected graph.")

    # Only the upper triangle (unique pairs, excluding the zero diagonal)
    # feeds the summary stats -- otherwise every route would be double
    # counted (once each direction) plus n zero self-distances.
    import numpy as np
    iu = np.triu_indices(n, k=1)
    upper_values = matrix.values[iu]
    finite = upper_values[upper_values != float("inf")]

    if len(finite):
        print(f"\nTravel time (minutes) across all finite, unique county-to-county pairs:")
        print(f"  min:    {finite.min():,.1f}")
        print(f"  mean:   {finite.mean():,.1f}")
        print(f"  median: {pd.Series(finite).median():,.1f}")
        print(f"  max:    {finite.max():,.1f}")


def write_long_format_csv(
    matrix: pd.DataFrame, path: Path, county_names: Dict[NodeId, str]
) -> None:
    """Write ONE row per unique (unordered) county-to-county pair.

    Unlike the old fort x port cross-product (where every entry was a
    distinct route), a county-to-county matrix on an undirected graph is
    symmetric -- (c1, c2) and (c2, c1) are the same route. Only the upper
    triangle (i < j, using the matrix's own row order) is written, so the
    CSV has exactly C(|C|, 2) rows, matching 3_snap_counties.py's own N
    bookkeeping.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    nodes = list(matrix.index)
    n = len(nodes)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "county1_lon", "county1_lat", "county1_name",
            "county2_lon", "county2_lat", "county2_name",
            "travel_time_min",
        ])
        rows_written = 0
        for i in range(n):
            c1 = nodes[i]
            for j in range(i + 1, n):
                c2 = nodes[j]
                t = matrix.loc[c1, c2]
                writer.writerow([
                    c1[0], c1[1], county_names.get(c1, ""),
                    c2[0], c2[1], county_names.get(c2, ""),
                    t,
                ])
                rows_written += 1
    print(f"Long-format travel-time CSV saved to: {path} ({rows_written:,} unique pair rows)")


# ---------------------------------------------------------------------------
# Per-network run
# ---------------------------------------------------------------------------

def run_one_network(
    network: str,
    input_path: Path,
    analysis_output_dir: Path,
    reports_dir: Path,
) -> bool:
    """Run Step 4a end to end for a single network.

    Returns True on success. Returns False (without raising) if the input
    pickle doesn't exist, so a multi-network run can skip this network and
    continue with the other one instead of aborting entirely -- matching
    the "both by default, skip what's missing" convention used everywhere
    else in this pipeline (2_1 through 4_graph_smoothing.py).
    """
    print()
    print("#" * 70)
    print(f"# NETWORK: {network}")
    print("#" * 70)

    if not input_path.exists():
        print(
            f"  WARNING: input graph pickle not found, skipping this network:\n"
            f"    {input_path}\n"
            f"  (Run 4_graph_smoothing.py for this network first -- its "
            f"output_graph_dir/f'{{network}}_smoothed.pkl' is exactly what "
            f"this path points at -- or pass --input directly.)"
        )
        return False

    _print_banner(f"STEP 4a -- COUNTY-TO-COUNTY TRAVEL-TIME MATRIX [{network}]")
    print(f"Input:  {input_path}")
    out_matrix_path = analysis_output_dir / f"{network}_travel_time_matrix.pkl"
    out_long_csv_path = reports_dir / f"{network}_county_travel_time_long.csv"
    print(f"Output: {out_matrix_path}")
    print(f"Report: {out_long_csv_path}")

    G, C = load_smoothed_graph(input_path)

    print(f"\nRunning single_source_dijkstra from each of {len(C):,} counties...")
    matrix, unreachable = build_travel_time_matrix(G, C)

    summarize_matrix(matrix, unreachable)

    analysis_output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_matrix_path, "wb") as f:
        pickle.dump({"matrix": matrix, "C_county_nodes": C}, f)
    print(f"\nSaved travel-time matrix to: {out_matrix_path}")

    county_names = facility_name_lookup(G, C)
    write_long_format_csv(matrix, out_long_csv_path, county_names)

    _print_banner(f"DONE [{network}]")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 4a: compute the county-to-county travel-time matrix for "
            "the smoothed HIGHWAY and RAILWAY graphs. Runs BOTH networks "
            "automatically unless --network restricts it to one."
        )
    )
    parser.add_argument(
        "--network",
        choices=sorted(NETWORK_LABELS),
        default=None,
        help=(
            "Restrict the run to just this one network. If omitted, BOTH "
            "highway and railway are processed automatically (matching "
            "every other script in this pipeline)."
        ),
    )
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Override the input graph pickle path (default: "
             f"{SMOOTHED_INPUT_DIR}/<network>_smoothed.pkl -- "
             "4_graph_smoothing.py's own output). Requires --network (an "
             "explicit path only makes sense for a single network at a "
             "time).",
    )
    parser.add_argument(
        "--analysis-output-dir", type=Path, default=ANALYSIS_OUTPUT_DIR,
        help=f"Folder for the output matrix pickle(s) (default: {ANALYSIS_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--reports-dir", type=Path, default=REPORTS_DIR,
        help=f"Folder for the long-format CSV report(s) (default: {REPORTS_DIR}).",
    )
    args = parser.parse_args()

    if args.input is not None and args.network is None:
        raise SystemExit(
            "--input requires --network (an explicit input path only "
            "makes sense for a single network at a time)."
        )

    return args


def main() -> None:
    args = parse_args()

    if args.network:
        networks_to_run = [args.network]
    else:
        # NEW DEFAULT: both networks, in a fixed/deterministic order --
        # matches every other script in this pipeline's "both by default"
        # convention.
        networks_to_run = list(NETWORK_LABELS)

    print(f"Running Step 4a (travel-time matrix) for: {', '.join(networks_to_run)}")

    ran = []
    skipped = []

    for network in networks_to_run:
        input_path = (
            args.input
            if args.input is not None
            else SMOOTHED_INPUT_DIR / NETWORK_INPUT_FILENAMES[network]
        )

        ok = run_one_network(
            network=network,
            input_path=input_path,
            analysis_output_dir=args.analysis_output_dir,
            reports_dir=args.reports_dir,
        )

        (ran if ok else skipped).append(network)

    print()
    _print_banner("ALL NETWORKS COMPLETE")
    print(f"Ran successfully: {ran or '(none)'}")
    if skipped:
        print(f"Skipped (input graph pickle not found): {skipped}")
    if not ran:
        raise SystemExit(
            "\nNo network was actually processed -- every requested input "
            "graph pickle was missing. Run 4_graph_smoothing.py first, or "
            "pass --network + --input directly."
        )

    print("\nNext step: edge betweenness centrality restricted to "
          "county-to-county pairs (nx.edge_betweenness_centrality_subset "
          f"with sources=C, targets=C, weight='{WEIGHT_ATTR}') -- ranks "
          "every road/rail segment by how many county-to-county shortest "
          "paths depend on it.")


if __name__ == "__main__":
    main()