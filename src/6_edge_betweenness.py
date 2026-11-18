#!/usr/bin/env python3
"""
Step 4b -- Edge betweenness centrality restricted to county-to-county
pairs (Illinois test run).

DATASET NOTE (updated to match the Illinois test run's naming convention):
-----------------------------------------------------------------------
The earlier draft of this script ranked edges by fort-to-port betweenness
(sources=O, targets=D) from a single hardcoded STRAHNET graph
(`data/graph/smoothing/strahnet_smoothed.pkl`, a
`{"graph", "O_fort_nodes", "D_port_nodes"}` dict). Neither those two
disjoint facility sets nor that graph path exist in this pipeline. This
script has been rewritten to match what this project actually produces:

  * Facility set: ONE set of COUNTY nodes, `C`, not two disjoint sets
    `O`/`D`. Traced directly to `3_snap_counties.py`'s bookkeeping
    (`C = set(county_nodes)`) and `4_graph_smoothing.py`'s own output
    dict shape (`{"graph": G_smoothed, "C_county_nodes": sorted(C)}`).
    `nx.edge_betweenness_centrality_subset` takes `sources` and `targets`
    as two separate arguments; here BOTH are passed the same set `C`
    (county-to-county, not county-to-itself-excluded specially --
    networkx already contributes zero for a source/target pair that is
    the same node, since Dijkstra from a node to itself is a zero-length,
    zero-edge path).

  * Graph input: traced directly to `4_graph_smoothing.py`'s own output
    convention:
        out_graph_path = output_graph_dir / f"{network}_smoothed.pkl"
    i.e. `data/graph/smoothing/highway_smoothed.pkl` and
    `.../railway_smoothed.pkl` -- BOTH highway and railway, not a single
    STRAHNET graph. Following the "both networks by default" convention
    already established by every other script in this pipeline (2_1
    through `5_travel_time_matrix.py`), `--network` is OPTIONAL: omitting
    it computes centrality for BOTH graphs automatically, one after
    another, each producing its own output pickle and its own ranked CSV.
    A missing smoothed-graph pickle for one network prints a clear
    warning and skips just that network.

Everything else -- the actual algorithm -- is UNCHANGED from the
original script and still implements the paper's modified betweenness
measure (Section 2.3.2):

    c_B(e) = sum over c1, c2 in C of [ sigma(c1, c2 | e) / sigma(c1, c2) ]

  where sigma(c1, c2)   = number of shortest paths from county c1 to
                           county c2
        sigma(c1, c2|e) = number of those shortest paths that pass
                           through edge e

This is standard edge betweenness centrality (computed via Brandes'
algorithm under the hood), EXCEPT the sum only runs over county-to-county
pairs instead of every pair of nodes in the whole graph -- both the
paper's stated methodology (this pipeline only cares about
county-to-county routes, not arbitrary point-to-point traffic) and a
large efficiency win versus the full all-pairs calculation.

networkx has this exact restricted variant built in --
`nx.edge_betweenness_centrality_subset(G, sources=C, targets=C, weight=...)`
-- so this script does not hand-roll Brandes' algorithm; it calls that
function directly.

A high c_B(e) means many county-to-county fastest routes depend on that
one road/rail segment -- these are the network's bottlenecks, and are
exactly the edges the later robustness/resilience disruption simulations
should be checking against (e.g. "what happens if we knock out the
top-N highest centrality edges").

NOTE on absolute scale: networkx's edge_betweenness_centrality_subset
(like the standard edge_betweenness_centrality) applies a uniform 0.5
scaling factor to every edge's score on an UNDIRECTED graph, since
undirected shortest paths get counted in both directions internally.
This is a constant multiplier applied identically to every edge, so it
never affects RELATIVE ranking -- which is all that matters for
identifying bottlenecks -- but don't be surprised if the raw numbers
look like exactly half of a naive hand-calculation.

Paths
-----
Input (per network):
    data/graph/smoothing/<network>_smoothed.pkl
        (4_graph_smoothing.py's own output_graph_dir/f"{network}_smoothed.pkl",
        a {"graph": nx.MultiGraph, "C_county_nodes": [...]} dict)

Output (per network):
    data/analysis/<network>_edge_betweenness.pkl
        (dict: {"centrality": {(u,v,key): score, ...}, "C_county_nodes": [...]})
    reports/<network>_edge_betweenness_ranked.csv
        (every edge, sorted by centrality descending -- the report to
         actually look at)

Usage
-----
    # NEW DEFAULT: computes centrality for BOTH highway and railway graphs
    python 6_edge_betweenness.py

    # restrict to just one network
    python 6_edge_betweenness.py --network highway
    python 6_edge_betweenness.py --network railway

    # override paths (single-network mode only)
    python 6_edge_betweenness.py --network highway --input path/to/other.pkl
"""

from __future__ import annotations

import argparse
import csv
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx

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

# How many top edges to print directly to the console for a quick look,
# beyond what's in the full CSV.
TOP_N_PREVIEW = 20

NodeId = Tuple[float, float]
EdgeKey = Tuple[NodeId, NodeId, int]


def _print_banner(title: str) -> None:
    """Section-banner style matching the rest of this pipeline (2_2, 2_3, 2_7, 2_8, 3, 4, 5)."""
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
            f"to compute centrality over."
        )

    print(f"Loaded graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges "
          f"({'multigraph' if G.is_multigraph() else 'simple graph'})")
    print(f"County nodes (C): {len(C):,}")
    return G, C


def facility_name_lookup(G: nx.MultiGraph, nodes: List[NodeId]) -> Dict[NodeId, str]:
    """Best-effort human-readable label per node, from the 'facility_names'
    node attribute set by 3_snap_counties.py."""
    lookup = {}
    for n in nodes:
        names = G.nodes.get(n, {}).get("facility_names", [])
        lookup[n] = "; ".join(names) if names else ""
    return lookup


# ---------------------------------------------------------------------------
# Betweenness centrality
# ---------------------------------------------------------------------------

def compute_county_betweenness(
    G: nx.MultiGraph, C: List[NodeId]
) -> Dict:
    """
    Restricted edge betweenness centrality via networkx's own
    `edge_betweenness_centrality_subset` -- Brandes' algorithm under the
    hood, summed only over sources=C, targets=C (county-to-county pairs)
    rather than all node pairs in the whole graph. A source/target pair
    that is the same county contributes nothing (a zero-length path has
    no edges to attribute centrality to), so passing the same set `C` for
    both `sources` and `targets` correctly restricts the sum to distinct
    county-to-county routes without any special-casing here.

    networkx's subset variant does NOT natively support MultiGraph edge
    keys in its return value the same way plain edge_betweenness_centrality
    does on a simple Graph -- it keys results by (u, v) node pairs. Since
    this pipeline's graph is a MultiGraph (parallel edges are possible,
    see 4_graph_smoothing.py's own docstring on why), this function
    converts to a plain Graph ONLY for the centrality calculation, first
    collapsing any parallel (u, v) edges down to the single FASTEST one
    (min travel_time_min) -- centrality is a property of the fastest
    route, so a slower parallel edge between the same two nodes can never
    carry any of the shortest-path traffic and correctly contributes zero
    centrality regardless. The original MultiGraph (with all edges intact)
    is what every other script in this pipeline continues to use; this
    collapse is local to this calculation only.

    Returns a dict keyed by (u, v, key) -- the ORIGINAL MultiGraph edge
    identity -- mapping to that edge's centrality score. A parallel edge
    that was discarded as slower during the collapse gets an explicit
    score of 0.0 rather than being silently absent from the output.
    """
    print("\nCollapsing parallel edges to their fastest instance for the centrality "
          "calculation (does not modify the graph used elsewhere in the pipeline)...")

    G_simple = nx.Graph()
    fastest_key: Dict[Tuple[NodeId, NodeId], int] = {}
    for u, v, k, data in G.edges(keys=True, data=True):
        pair = (u, v) if u <= v else (v, u)
        t = data.get(WEIGHT_ATTR, float("inf"))
        if pair not in G_simple.edges or t < G_simple.get_edge_data(*pair, {}).get(WEIGHT_ATTR, float("inf")):
            G_simple.add_edge(u, v, **data)
            fastest_key[pair] = k

    n_parallel_discarded = G.number_of_edges() - G_simple.number_of_edges()
    if n_parallel_discarded:
        print(f"  {n_parallel_discarded:,} slower parallel edge(s) excluded from the "
              f"calculation (they carry zero shortest-path traffic by construction).")

    n_pairs = len(C) * (len(C) - 1) // 2
    print(f"Running edge_betweenness_centrality_subset over {len(C):,} counties "
          f"x {len(C):,} counties ({n_pairs:,} unique county-to-county pair(s))...")
    start = time.monotonic()
    raw_centrality = nx.edge_betweenness_centrality_subset(
        G_simple, sources=C, targets=C, weight=WEIGHT_ATTR
    )
    elapsed = time.monotonic() - start
    print(f"  Done in {elapsed:,.1f}s.")

    # Map back onto every ORIGINAL (u, v, key) edge identity in the MultiGraph.
    centrality: Dict[EdgeKey, float] = {}
    for u, v, k in G.edges(keys=True):
        pair = (u, v) if u <= v else (v, u)
        if fastest_key.get(pair) == k:
            score = raw_centrality.get(pair, raw_centrality.get((pair[1], pair[0]), 0.0))
        else:
            score = 0.0  # a slower parallel edge -- never on a shortest path
        centrality[(u, v, k)] = score

    return centrality


def summarize_centrality(
    centrality: Dict[EdgeKey, float], G: nx.MultiGraph, top_n: int
) -> List[Tuple[EdgeKey, float]]:
    ranked = sorted(centrality.items(), key=lambda kv: -kv[1])

    print("\n" + "=" * 70)
    print("EDGE BETWEENNESS CENTRALITY SUMMARY")
    print("=" * 70)
    print(f"Total edges scored: {len(ranked):,}")

    nonzero = [s for _, s in ranked if s > 0]
    zero_count = len(ranked) - len(nonzero)
    print(f"Edges with centrality > 0 (used by at least one county-to-county "
          f"shortest path): {len(nonzero):,}")
    print(f"Edges with centrality == 0 (never on any county-to-county fastest "
          f"route): {zero_count:,}")

    if nonzero:
        import statistics
        print(f"  min (nonzero): {min(nonzero):.6f}")
        print(f"  mean (nonzero): {statistics.mean(nonzero):.6f}")
        print(f"  max: {max(nonzero):.6f}")

    print(f"\nTop {top_n} highest-centrality edges (the network's biggest "
          f"county-to-county bottlenecks):")
    for (u, v, k), score in ranked[:top_n]:
        length_km = G.get_edge_data(u, v, k).get("length_km", float("nan"))
        print(f"  {score:.4f}   {u} -- {v}   (length_km={length_km:.2f})")

    return ranked


def write_ranked_csv(
    ranked: List[Tuple[EdgeKey, float]], G: nx.MultiGraph, path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "u_lon", "u_lat", "v_lon", "v_lat", "edge_key",
            "betweenness_centrality", "length_km", "travel_time_min", "edge_type",
        ])
        for rank, ((u, v, k), score) in enumerate(ranked, start=1):
            data = G.get_edge_data(u, v, k) or {}
            writer.writerow([
                rank, u[0], u[1], v[0], v[1], k,
                score, data.get("length_km"), data.get("travel_time_min"),
                data.get("edge_type"),
            ])
    print(f"\nFull ranked edge list saved to: {path} ({len(ranked):,} rows)")


# ---------------------------------------------------------------------------
# Per-network run
# ---------------------------------------------------------------------------

def run_one_network(
    network: str,
    input_path: Path,
    analysis_output_dir: Path,
    reports_dir: Path,
    top_n_preview: int,
) -> bool:
    """Run Step 4b end to end for a single network.

    Returns True on success. Returns False (without raising) if the input
    pickle doesn't exist, so a multi-network run can skip this network and
    continue with the other one instead of aborting entirely -- matching
    the "both by default, skip what's missing" convention used everywhere
    else in this pipeline (2_1 through 5_travel_time_matrix.py).
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

    _print_banner(f"STEP 4b -- COUNTY-TO-COUNTY EDGE BETWEENNESS [{network}]")
    print(f"Input:  {input_path}")
    out_centrality_path = analysis_output_dir / f"{network}_edge_betweenness.pkl"
    out_ranked_csv_path = reports_dir / f"{network}_edge_betweenness_ranked.csv"
    print(f"Output: {out_centrality_path}")
    print(f"Report: {out_ranked_csv_path}")

    G, C = load_smoothed_graph(input_path)

    centrality = compute_county_betweenness(G, C)
    ranked = summarize_centrality(centrality, G, top_n_preview)

    analysis_output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_centrality_path, "wb") as f:
        pickle.dump({"centrality": centrality, "C_county_nodes": C}, f)
    print(f"Saved full centrality dict to: {out_centrality_path}")

    write_ranked_csv(ranked, G, out_ranked_csv_path)

    _print_banner(f"DONE [{network}]")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 4b: compute county-to-county edge betweenness centrality "
            "for the smoothed HIGHWAY and RAILWAY graphs. Runs BOTH "
            "networks automatically unless --network restricts it to one."
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
        help=f"Folder for the output centrality pickle(s) (default: {ANALYSIS_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--reports-dir", type=Path, default=REPORTS_DIR,
        help=f"Folder for the ranked-edge CSV report(s) (default: {REPORTS_DIR}).",
    )
    parser.add_argument(
        "--top-n-preview", type=int, default=TOP_N_PREVIEW,
        help=f"How many top edges to print to console per network (default: {TOP_N_PREVIEW}).",
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

    print(f"Running Step 4b (edge betweenness) for: {', '.join(networks_to_run)}")

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
            top_n_preview=args.top_n_preview,
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

    print("\nNext step: robustness/resilience disruption modeling -- targeted "
          "removal (highest-centrality edges first) vs. random removal, "
          "measuring how the county-to-county travel-time matrix degrades as "
          "edges are removed.")


if __name__ == "__main__":
    main()