#!/usr/bin/env python3
"""
Step 3 -- Graph smoothing (paper Section 2.2.4), for HIGHWAY and RAILWAY
(Illinois test run).

DATASET NOTE (updated to match the Illinois test run's naming convention):
-----------------------------------------------------------------------
The earlier draft of this script was written against a single hardcoded
STRAHNET graph (`data/graph/snap_facility/strahnet_with_facilities.pkl`,
a `{"graph", "O_fort_nodes", "D_port_nodes"}` dict) produced by the old
fort/port version of `3_snap_facility.py`. This pipeline snaps COUNTY
CENTROIDS, not forts/seaports, onto TWO separate graphs (highway and
railway), via `3_snap_counties.py`. This script has been updated to
match:

  * Input traced directly to `3_snap_counties.py`'s own output
    convention:
        out_graph_path = output_graph_dir / f"{network}_with_counties.pkl"
    i.e. `data/graph/snap_counties/highway_with_counties.pkl` and
    `.../railway_with_counties.pkl` -- a `{"graph": nx.MultiGraph,
    "C_county_nodes": [...]}` dict, not the old
    `{"O_fort_nodes", "D_port_nodes"}` shape.
  * "Protected" nodes (never collapsed by smoothing, regardless of
    degree) are now the single county-node set `C`, not the union of two
    disjoint fort/port sets `O | D`.
  * The attribute-fallback scan (used only if the input pickle isn't the
    expected dict shape) now looks for `"county"` in a node's
    `facility_types` set, matching `3_snap_counties.py`'s own
    `existing_types.add("county")` -- not the old `"fort"`/`"port"`
    markers, which this pipeline never writes.
  * Following the "both networks by default" convention already
    established by every other script in this pipeline (`2_1` through
    `3_snap_counties.py`): `--network` is OPTIONAL. Omitting it smooths
    BOTH highway and railway automatically, one after another, each with
    its own validation block, its own output graph, and its own report
    CSV. A missing input pickle for one network prints a clear warning
    and skips just that network rather than aborting the whole run.

Everything else -- the actual smoothing algorithm -- is UNCHANGED from
the original script:

Takes a facility-snapped graph and collapses every degree-2 node that is
NOT a county node into a single merged edge between its two neighbors.
This does not change network topology or any county-to-county shortest
path -- it only removes redundant "shape vertices" that GIS line-drawing
left behind, so the graph is dramatically smaller and faster for the
network-analysis steps that follow (travel-time matrix, betweenness
centrality, disruption simulations).

Reference numbers from the paper (STRAHNET, national scale): before
smoothing, 95,172 nodes / 96,368 edges, 97% degree-2. After smoothing,
2,849 nodes (2.9%) / 4,045 edges (4.2%). This Illinois test run is a much
smaller extract, so expect a large *proportional* drop but not
necessarily those exact absolute numbers; if you see only a mild
reduction, something's off (see the sanity checks at the end of each
network's run).

What gets merged and how
-------------------------
For every non-county node of degree 2 whose two incident edges go to two
DISTINCT neighbors (i.e. it's a genuine pass-through point on a chain,
not a self-loop or a two-edges-to-the-same-neighbor loop), this script:

  1. Removes the node's two edges and the node itself.
  2. Adds ONE new edge directly between its two neighbors, with:
       - length_km          = sum of the two edges' length_km
       - travel_time_min    = sum of the two edges' travel_time_min
       - edge_type          = "real" if both inputs were "real", otherwise
                               the sorted, "+"-joined set of distinct types
                               seen (e.g. "auto_bridge+real") so a smoothed
                               edge that swallowed a gap-fill or manual
                               connector is never silently relabeled "real"
       - any boolean flag attribute (is_gap_fill, is_manual, ...) is OR'd
         across the two inputs, so a smoothed edge is flagged as such if
         EITHER contributing edge was
       - any other/unknown attribute is copied from whichever of the two
         edges has more attributes set, with a one-time warning the first
         time this happens (so an unexpected new attribute doesn't
         silently vanish without being noticed)

The graph is a MultiGraph, so the new merged edge is always added as a
new parallel edge -- it never overwrites/collides with an existing edge
between the same node pair.

Repeats until no more non-county degree-2 nodes remain (removing one node
can occasionally make a neighboring chain newly smoothable, though in
practice this pipeline's node degrees are stable enough that it usually
converges in a single sweep).

Paths
-----
Input (per network):
    data/graph/snap_counties/<network>_with_counties.pkl
        (3_snap_counties.py's own output path/shape)

Output (per network):
    data/graph/smoothing/<network>_smoothed.pkl
        (same dict shape, graph replaced with the smoothed version)
    reports/<network>_smoothing_report.csv
        (one row per interior chain that got smoothed away)

Usage
-----
    # NEW DEFAULT: smooths BOTH highway and railway graphs
    python 4_graph_smoothing.py

    # restrict to just one network
    python 4_graph_smoothing.py --network highway
    python 4_graph_smoothing.py --network railway

    # override paths (single-network mode only)
    python 4_graph_smoothing.py --network highway --input path/to/other.pkl
"""

from __future__ import annotations

import argparse
import csv
import pickle
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

# ---------------------------------------------------------------------------
# Paths -- kept in sync by hand with 3_snap_counties.py's own output
# convention (output_graph_dir / f"{network}_with_counties.pkl").
# ---------------------------------------------------------------------------

SNAPPED_INPUT_DIR = Path("data/graph/snap_counties")   # 3_snap_counties.py's own output_graph_dir
OUTPUT_GRAPH_DIR = Path("data/graph/smoothing")
REPORTS_DIR = Path("reports")

NETWORK_LABELS = ["highway", "railway"]

NETWORK_INPUT_FILENAMES = {
    network: f"{network}_with_counties.pkl" for network in NETWORK_LABELS
}

# HARDCODED DEFAULT NETWORK -- same pattern as every other script in this
# pipeline. Only used in single-file mode when --input is given without
# --network (belt-and-suspenders; --input requires --network, see
# parse_args, but this stays as a safety fallback).
HARDCODED_NETWORK = "highway"

# Attributes summed when two edges are merged into one.
SUM_ATTRS = ("length_km", "travel_time_min")

# Attribute treated as the "class" of an edge; merged by set-union of
# distinct values rather than summed. Missing on an edge -> treated as "real"
# (matches 2_7_create_final_weighted_graph.py's / 3_snap_counties.py's own
# `edge_data.get("edge_type", "real")`).
TYPE_ATTR = "edge_type"
DEFAULT_TYPE_VALUE = "real"

NodeId = Tuple[float, float]


def _print_banner(title: str) -> None:
    """Section-banner style matching the rest of this pipeline (2_2, 2_3, 2_7, 2_8, 3)."""
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_snapped_graph(path: Path) -> Tuple[nx.MultiGraph, Set[NodeId]]:
    """
    Load the county-snapped pickle and return (graph, county_nodes).

    Handles the {"graph": G, "C_county_nodes": C} dict shape that
    3_snap_counties.py actually writes. Falls back to treating the pickle
    as a bare graph (with county nodes read off a "facility_types" node
    attribute, if present) only if it isn't that dict shape, so this
    script still degrades gracefully against a hand-modified input.
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict) and "graph" in obj:
        G = obj["graph"]
        C = set(obj.get("C_county_nodes", []))
        if not C:
            print("  WARNING: loaded dict had no C_county_nodes -- "
                  "falling back to scanning node attributes for facility markers.")
            C = _facility_nodes_from_attrs(G)
    else:
        G = obj
        print("  WARNING: input pickle was not the expected "
              "{'graph','C_county_nodes'} dict -- treating it as a bare graph "
              "and scanning node attributes for facility markers instead.")
        C = _facility_nodes_from_attrs(G)

    if not G.is_multigraph():
        print("  NOTE: loaded graph is a plain nx.Graph, not nx.MultiGraph. "
              "Converting to MultiGraph so smoothing can never silently drop "
              "a parallel edge.")
        G = nx.MultiGraph(G)

    print(f"Loaded graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    print(f"County nodes (C): {len(C):,}")
    return G, C


def _facility_nodes_from_attrs(G: nx.Graph) -> Set[NodeId]:
    """Fallback: read county membership off a 'facility_types' node attribute.

    Matches 3_snap_counties.py's own `existing_types.add("county")` marker
    -- not the old "fort"/"port" markers this pipeline never writes.
    """
    C = set()
    for n, data in G.nodes(data=True):
        types = data.get("facility_types", ())
        if "county" in types:
            C.add(n)
    return C


# ---------------------------------------------------------------------------
# Edge-attribute merging
# ---------------------------------------------------------------------------

_warned_unknown_attrs: Set[str] = set()


def merge_edge_data(d1: dict, d2: dict) -> dict:
    """
    Combine two edges' attribute dicts into the single merged edge's attrs.

    - SUM_ATTRS ("length_km", "travel_time_min"): added together.
    - TYPE_ATTR ("edge_type"): union of distinct values, "+"-joined and
      sorted, collapsing to a single value if both sides agree. Missing on
      either side is treated as DEFAULT_TYPE_VALUE ("real"), matching how
      the upstream snap-facility script itself defaults a missing edge_type.
    - Any attribute starting with "is_" (e.g. is_gap_fill, is_manual): OR'd,
      so the merged edge is flagged True if either contributing edge was.
    - Anything else: copied from d1, falling back to d2 if d1 doesn't have
      it. The first time an attribute falls into this catch-all, a one-time
      warning is printed so an unexpected attribute never silently
      disappears without being noticed.
    """
    merged: dict = {}

    for attr in SUM_ATTRS:
        merged[attr] = d1.get(attr, 0.0) + d2.get(attr, 0.0)

    t1 = d1.get(TYPE_ATTR, DEFAULT_TYPE_VALUE)
    t2 = d2.get(TYPE_ATTR, DEFAULT_TYPE_VALUE)
    merged[TYPE_ATTR] = t1 if t1 == t2 else "+".join(sorted({t1, t2}))

    all_keys = set(d1) | set(d2)
    for key in all_keys - set(SUM_ATTRS) - {TYPE_ATTR}:
        if key.startswith("is_"):
            merged[key] = bool(d1.get(key, False)) or bool(d2.get(key, False))
        else:
            if key not in _warned_unknown_attrs:
                print(f"  NOTE: edge attribute '{key}' isn't a known sum/type/is_-flag "
                      f"attribute -- carrying it over from whichever edge has it, "
                      f"first edge wins on conflict. (This note prints once per attribute name.)")
                _warned_unknown_attrs.add(key)
            merged[key] = d1[key] if key in d1 else d2[key]

    return merged


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def _incident_edges(G: nx.MultiGraph, n: NodeId):
    """Return this node's incident (neighbor, key, data) triples, normalized
    so `neighbor` is always the OTHER endpoint regardless of edge direction
    as networkx happens to store it."""
    out = []
    for u, v, k, data in G.edges(n, keys=True, data=True):
        neighbor = v if u == n else u
        out.append((neighbor, k, data))
    return out


def find_smoothable(G: nx.MultiGraph, n: NodeId, protected: Set[NodeId]) -> Optional[Tuple]:
    """
    If node `n` is currently a genuine pass-through point, return
    (neighbor1, key1, data1, neighbor2, key2, data2). Otherwise return None.

    Excluded even at degree 2:
      - protected (county) nodes -- never removed, regardless of degree.
      - self-loops (a neighbor equal to n itself).
      - two parallel edges to the SAME neighbor (a length-2 loop through n,
        not a pass-through chain -- collapsing it would delete a real cycle
        rather than simplify a chain).
    """
    if n in protected:
        return None
    if G.degree(n) != 2:
        return None

    incident = _incident_edges(G, n)
    if len(incident) != 2:
        return None  # e.g. a self-loop alone already contributes degree 2

    (n1, k1, d1), (n2, k2, d2) = incident
    if n1 == n or n2 == n:
        return None  # self-loop
    if n1 == n2:
        return None  # parallel edges to the same neighbor -- a loop, not a chain

    return n1, k1, d1, n2, k2, d2


def smooth_graph(
    G: nx.MultiGraph, protected: Set[NodeId]
) -> Tuple[nx.MultiGraph, List[dict]]:
    """
    Repeatedly collapse non-county degree-2 nodes until none remain.

    Uses a work queue seeded with every current degree-2 candidate, rather
    than a single pass over `G.nodes()`, because removing a node can (in
    principle) leave a neighbor newly eligible if its own degree was
    already 2 through a different pairing. Each popped node is re-checked
    against the LIVE graph before acting, so a node that's already been
    consumed (or whose degree changed) is silently skipped rather than
    double-processed.

    Returns the same graph `G` (mutated in place) and a list of per-merge
    log rows for the report CSV.
    """
    queue = deque(
        n for n in list(G.nodes())
        if n not in protected and G.degree(n) == 2
    )
    seen_in_queue = set(queue)
    log_rows: List[dict] = []
    merges = 0

    while queue:
        n = queue.popleft()
        seen_in_queue.discard(n)
        if not G.has_node(n):
            continue

        result = find_smoothable(G, n, protected)
        if result is None:
            continue

        n1, k1, d1, n2, k2, d2 = result
        merged_data = merge_edge_data(d1, d2)

        G.remove_edge(n1, n, k1)
        G.remove_edge(n, n2, k2)
        G.remove_node(n)
        G.add_edge(n1, n2, **merged_data)

        merges += 1
        log_rows.append({
            "removed_node_lon": n[0], "removed_node_lat": n[1],
            "neighbor1_lon": n1[0], "neighbor1_lat": n1[1],
            "neighbor2_lon": n2[0], "neighbor2_lat": n2[1],
            "merged_length_km": merged_data.get("length_km"),
            "merged_travel_time_min": merged_data.get("travel_time_min"),
            "merged_edge_type": merged_data.get(TYPE_ATTR),
        })

        for candidate in (n1, n2):
            if (
                candidate not in protected
                and G.has_node(candidate)
                and G.degree(candidate) == 2
                and candidate not in seen_in_queue
            ):
                queue.append(candidate)
                seen_in_queue.add(candidate)

    print(f"Smoothing complete: {merges:,} node(s) removed and merged.")
    return G, log_rows


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def total_length_and_time(G: nx.MultiGraph) -> Tuple[float, float]:
    length = sum(d.get("length_km", 0.0) for _, _, d in G.edges(data=True))
    time = sum(d.get("travel_time_min", 0.0) for _, _, d in G.edges(data=True))
    return length, time


def validate(
    G_before: nx.MultiGraph, G_after: nx.MultiGraph,
    protected: Set[NodeId],
) -> None:
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    n_before, e_before = G_before.number_of_nodes(), G_before.number_of_edges()
    n_after, e_after = G_after.number_of_nodes(), G_after.number_of_edges()
    print(f"Nodes: {n_before:,} -> {n_after:,}  ({100 * n_after / n_before:.1f}% remaining)")
    print(f"Edges: {e_before:,} -> {e_after:,}  ({100 * e_after / e_before:.1f}% remaining)")

    connected_before = nx.is_connected(G_before)
    connected_after = nx.is_connected(G_after)
    print(f"Connected before: {connected_before}   after: {connected_after}")
    if connected_before != connected_after:
        print("  *** WARNING: connectivity changed during smoothing -- this should "
              "be impossible if only degree-2 non-facility nodes were removed. "
              "Do not trust this output; investigate before proceeding. ***")

    missing_protected = [p for p in protected if not G_after.has_node(p)]
    if missing_protected:
        print(f"  *** WARNING: {len(missing_protected)} county node(s) were "
              f"REMOVED during smoothing -- this must never happen. Investigate "
              f"the protected-node check. ***")
    else:
        print(f"All {len(protected):,} county nodes preserved.")

    len_before, time_before = total_length_and_time(G_before)
    len_after, time_after = total_length_and_time(G_after)
    len_diff = abs(len_before - len_after)
    time_diff = abs(time_before - time_after)
    print(f"Total length_km: {len_before:,.1f} -> {len_after:,.1f}  (diff: {len_diff:.6f})")
    print(f"Total travel_time_min: {time_before:,.1f} -> {time_after:,.1f}  (diff: {time_diff:.6f})")
    if len_diff > 1e-3 or time_diff > 1e-3:
        print("  *** WARNING: total length/time changed by more than floating-point "
              "noise -- smoothing should be purely additive. Investigate. ***")
    else:
        print("Totals conserved (as expected -- smoothing only concatenates, never "
              "changes, the underlying lengths/times).")


# ---------------------------------------------------------------------------
# Per-network run
# ---------------------------------------------------------------------------

def run_one_network(
    network: str,
    input_path: Path,
    output_graph_dir: Path,
    reports_dir: Path,
) -> bool:
    """Run Step 3 (smoothing) end to end for a single network.

    Returns True on success. Returns False (without raising) if the input
    pickle doesn't exist, so a multi-network run can skip this network and
    continue with the other one instead of aborting entirely -- matching
    the "both by default, skip what's missing" convention used everywhere
    else in this pipeline (2_1 through 3_snap_counties.py).
    """
    print()
    print("#" * 70)
    print(f"# NETWORK: {network}")
    print("#" * 70)

    if not input_path.exists():
        print(
            f"  WARNING: input graph pickle not found, skipping this network:\n"
            f"    {input_path}\n"
            f"  (Run 3_snap_counties.py for this network first -- its "
            f"output_graph_dir/f'{{network}}_with_counties.pkl' is exactly "
            f"what this path points at -- or pass --input directly.)"
        )
        return False

    _print_banner(f"STEP 3 -- GRAPH SMOOTHING [{network}]")
    print(f"Input:  {input_path}")
    out_graph_path = output_graph_dir / f"{network}_smoothed.pkl"
    out_report_path = reports_dir / f"{network}_smoothing_report.csv"
    print(f"Output: {out_graph_path}")
    print(f"Report: {out_report_path}")

    G, C = load_snapped_graph(input_path)
    protected = C

    G_before = G.copy()

    print("\nRunning smoothing pass...")
    G_smoothed, log_rows = smooth_graph(G, protected)

    validate(G_before, G_smoothed, protected)

    output_graph_dir.mkdir(parents=True, exist_ok=True)
    with open(out_graph_path, "wb") as f:
        pickle.dump(
            {"graph": G_smoothed, "C_county_nodes": sorted(C)},
            f,
        )
    print(f"\nSaved smoothed graph + county node list to: {out_graph_path}")

    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(out_report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "removed_node_lon", "removed_node_lat",
            "neighbor1_lon", "neighbor1_lat",
            "neighbor2_lon", "neighbor2_lat",
            "merged_length_km", "merged_travel_time_min", "merged_edge_type",
        ])
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"Per-merge smoothing report saved to: {out_report_path} ({len(log_rows):,} rows)")

    _print_banner(f"DONE [{network}]")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 3: smooth away non-county degree-2 nodes on the "
            "county-snapped HIGHWAY and RAILWAY graphs. Runs BOTH networks "
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
             f"{SNAPPED_INPUT_DIR}/<network>_with_counties.pkl -- "
             "3_snap_counties.py's own output). Requires --network (an "
             "explicit path only makes sense for a single network at a "
             "time).",
    )
    parser.add_argument(
        "--output-graph-dir", type=Path, default=OUTPUT_GRAPH_DIR,
        help=f"Folder for the output graph pickle(s) (default: {OUTPUT_GRAPH_DIR}).",
    )
    parser.add_argument(
        "--reports-dir", type=Path, default=REPORTS_DIR,
        help=f"Folder for the per-merge CSV report(s) (default: {REPORTS_DIR}).",
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
        # matches 2_1 through 3_snap_counties.py's "both by default" convention.
        networks_to_run = list(NETWORK_LABELS)

    print(f"Running Step 3 (smoothing) for: {', '.join(networks_to_run)}")

    ran = []
    skipped = []

    for network in networks_to_run:
        input_path = (
            args.input
            if args.input is not None
            else SNAPPED_INPUT_DIR / NETWORK_INPUT_FILENAMES[network]
        )

        ok = run_one_network(
            network=network,
            input_path=input_path,
            output_graph_dir=args.output_graph_dir,
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
            "graph pickle was missing. Run 3_snap_counties.py first, or "
            "pass --network + --input directly."
        )

    print("\nNext step: network analysis (travel-time matrix via Dijkstra over all "
          "county-to-county pairs, then edge betweenness centrality restricted to "
          "county-to-county routes).")


if __name__ == "__main__":
    main()