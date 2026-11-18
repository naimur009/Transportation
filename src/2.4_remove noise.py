#!/usr/bin/env python3
"""
Step 1.5 -- Component-level noise review for HIGHWAY/RAILWAY.

Purpose
-------
`prepare` : Load the current graph (Step 1.4's post-gap-bridging pickle),
            split it into connected components, and export ONE FEATURE PER
            COMPONENT to a review GeoJSON. Each feature carries a
            `component_id`, size/length stats, and a `decision` field
            defaulting to false. Also writes a sidecar pickle mapping
            component_id -> exact node list, so the `apply` step never has
            to re-derive component membership from scratch (which would
            risk drifting if the graph changes between prepare and apply).

`apply`   : Reload the same graph, read back the reviewed GeoJSON, and DELETE
            every node belonging to a component whose `decision` was flipped
            to true (i.e. "this is noise, drop it"). Writes a denoised graph
            pickle plus a CSV audit log of exactly what was removed and why,
            and prints a full before/after connected-component report.

This is the manual-judgment counterpart to "keep only the largest
component". Use this instead of / in addition to "largest only" when a
network legitimately has more than one real component (e.g. islands,
genuinely separate spurs) and you want to decide component-by-component
instead of automatically keeping just #1.

This script picks up directly where Step 1.4 (2.3_gap_repair.py) leaves
off: its output, `data/graph/gap_bridging/<network>_bridged.pkl`, is this
script's default input. Node IDs are still the raw, unrounded (lon, lat)
tuples that convention runs through Steps 1.2-1.4 (rounding is deferred to
the later "build final graph" step, and is untouched by this script).

QGIS review workflow (same pattern as the manual gap-repair loop)
----------------------------------------------------------------
1. Load Esri World Imagery (bottom) -> <network>_clean.geojson or the noded
   file (middle) -> the component review GeoJSON this script produces (top).
2. Style the review layer's line color/width by `n_nodes` or `length_km` so
   large real components and tiny noise slivers are visually distinct.
3. Open the attribute table, sort by `length_km` ascending (small/suspicious
   components first).
4. Enable editing (pencil icon). For each row, zoom to it, look at the
   imagery underneath, and decide:
       - genuinely disconnected but real (island, isolated spur, real border
         gap) -> leave `decision` = false
       - digitizing noise / stray sliver / clearly not part of the network
         -> set `decision` = true
   Write your reasoning in the `notes` field either way -- that field is
   carried straight through to the apply step's audit log.
5. Save edits.

Usage
-----
    python 2_4_remove_noise.py --network highway prepare
    python 2_4_remove_noise.py --network railway prepare
    python 2_4_remove_noise.py --network highway apply

    # override paths if your graph pickle isn't in the default location
    python 2_4_remove_noise.py --network highway prepare \\
        --graph-pickle data/graph/gap_bridging/highway_bridged.pkl

    # both networks in one go (default when --network is omitted)
    python 2_4_remove_noise.py prepare
    python 2_4_remove_noise.py apply

Input graph assumptions
------------------------
This script expects the pickled graph produced by Step 1.4
(`2_3_gap_repair.py`), which is an `networkx.MultiGraph`, not a plain
`Graph`:
  - Node IDs are raw (lon, lat) float tuples (the convention used
    throughout this pipeline, e.g. `2_1_build_naive_graph.py` /
    `2_3_gap_repair.py`'s preliminary graph -- NOT yet rounded to 6dp;
    that only happens in the later final-graph-build step).
  - Because the input is a MultiGraph, two nodes may be joined by more
    than one parallel edge (Step 1.4 preserves ~40 distinct source lines
    that share the same (start, end) coordinate pair rather than
    collapsing them). Every function below that sums edge lengths or
    counts edge attributes iterates `G.edges(data=True)` /
    `sub.edges(data=True)` directly (not `G.get_edge_data(u, v)`, which
    on a MultiGraph returns a dict-of-dicts keyed by edge key, not a flat
    attribute dict) so parallel edges are counted correctly rather than
    silently collapsed to one.
  - Edges MAY carry `length_km`, `is_gap_fill` attributes (all optional --
    `2_3_gap_repair.py` sets both on every edge it creates, but this
    script computes `length_km` itself via geodesic distance between the
    two node coordinates if it's ever missing, so it still works against
    an older or hand-built graph with no weights).

Defaults
--------
Input graph pickle defaults to Step 1.4's own output path/naming
(`data/graph/gap_bridging/<network>_bridged.pkl`,
`NETWORK_FILENAMES = {"highway": ..., "railway": ...}`) so this script
requires no path edits to run directly against a fresh 2_3_gap_repair.py
output -- confirm/override with `--graph-pickle` if your layout differs.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

import networkx as nx

try:
    from geopy.distance import geodesic
    _HAVE_GEOPY = True
except ImportError:
    _HAVE_GEOPY = False

Node = Tuple[float, float]  # (lon, lat), raw/unrounded

# ---------------------------------------------------------------------------
# Defaults -- matches Step 1.4's (2_3_gap_repair.py) own output convention:
# BRIDGED_OUTPUT_DIR = data/graph/gap_bridging/<network>_bridged.pkl,
# NETWORK_FILENAMES = {"highway": ..., "railway": ...}.
# ---------------------------------------------------------------------------
NETWORK_FILENAMES = {
    "highway": "highway_clean.geojson",
    "railway": "railway_clean.geojson",
}

BRIDGED_INPUT_DIR = Path("data/graph/gap_bridging")  # Step 1.4's own output dir
DEFAULT_GRAPH_PICKLE = {
    network: BRIDGED_INPUT_DIR / f"{network}_bridged.pkl"
    for network in NETWORK_FILENAMES
}

REVIEW_DIR = Path("data/graph/component_review")
DEFAULT_REVIEW_GEOJSON = lambda network: REVIEW_DIR / f"{network}_component_review.geojson"
DEFAULT_SIDECAR = lambda network: REVIEW_DIR / f"{network}_component_review_sidecar.pkl"
DEFAULT_OUTPUT_PICKLE = lambda network: REVIEW_DIR / f"{network}_denoised.pkl"
DEFAULT_LOG_CSV = lambda network: REVIEW_DIR / f"{network}_component_noise_log.csv"

# HARDCODED DEFAULT NETWORK -- same pattern as Steps 1.3/1.4. Only used when
# neither --network nor a positional/explicit path narrows the run.
HARDCODED_NETWORK = "highway"


def _print_banner(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def _edge_length_km(data: dict, u: Node, v: Node) -> float:
    """Return one edge's length in km from its already-retrieved attribute
    dict, using the `length_km` attribute if present, otherwise computing
    geodesic distance from the raw (lon, lat) node coordinates. Falls back
    to a flat-degree approximation if geopy isn't installed.

    IMPORTANT: this takes the edge's attribute `data` dict directly (as
    yielded by `G.edges(data=True)` / `sub.edges(data=True)`) rather than
    re-fetching it via `G.get_edge_data(u, v)`. On a MultiGraph,
    `get_edge_data(u, v)` returns a dict keyed by edge key (i.e. a dict of
    dicts, one per parallel edge) rather than a single flat attribute dict
    -- calling `.get("length_km")` on that would silently fail to find the
    key and fall through to the geodesic estimate for every parallel edge,
    and would also make it impossible to sum each parallel edge's own
    length separately. Iterating `edges(data=True)` avoids that ambiguity
    entirely: every call site already has the correct per-edge dict in
    hand.
    """
    if "length_km" in data:
        return float(data["length_km"])
    if _HAVE_GEOPY:
        return geodesic((u[1], u[0]), (v[1], v[0])).km
    # crude fallback: 1 degree ~= 111 km, ignores latitude compression --
    # only used if geopy truly isn't available, purely so the script still
    # runs rather than crashing.
    dx, dy = (u[0] - v[0]), (u[1] - v[1])
    return ((dx ** 2 + dy ** 2) ** 0.5) * 111.0


def compute_components_with_ids(G: nx.MultiGraph) -> List[Tuple[str, FrozenSet[Node]]]:
    """Split G into connected components and assign each a DETERMINISTIC id.

    Ordering rule: sort components by (-length_km, min_node) and id them
    C0000, C0001, ... in that order. This is deterministic given a fixed
    graph, independent of networkx's internal traversal order, so re-running
    `prepare` against an unchanged graph always reproduces the same ids --
    critical since `apply` re-derives nothing and instead trusts the
    sidecar, but the ids in the review GeoJSON must still make sense to a
    human reviewing it twice.

    Works unchanged against an `nx.MultiGraph`: `sub.edges(data=True)`
    yields one tuple per parallel edge (not collapsed to one per node
    pair), so summing `_edge_length_km` over it correctly totals every
    source line's length even when several lines share the same (start,
    end) coordinate pair.

    Returns:
        List of (component_id, frozenset_of_nodes), sorted by descending
        length_km (largest/most-likely-real component first, so a reviewer
        sorting by that column sees noise candidates cluster at the bottom).
    """
    raw_components = list(nx.connected_components(G))

    scored = []
    for comp in raw_components:
        sub = G.subgraph(comp)
        length_km = sum(
            _edge_length_km(d, u, v) for u, v, d in sub.edges(data=True)
        )
        min_node = min(comp)  # tuple comparison -- fully deterministic
        scored.append((length_km, min_node, frozenset(comp)))

    scored.sort(key=lambda row: (-row[0], row[1]))

    return [
        (f"C{idx:04d}", comp)
        for idx, (_length_km, _min_node, comp) in enumerate(scored)
    ]


def build_component_stats(G: nx.MultiGraph, comp: FrozenSet[Node]) -> dict:
    """Compute the summary stats stored as GeoJSON properties for one component."""
    sub = G.subgraph(comp)
    n_bridge = sum(1 for _, _, d in sub.edges(data=True) if d.get("is_gap_fill"))
    length_km = sum(_edge_length_km(d, u, v) for u, v, d in sub.edges(data=True))
    return {
        "n_nodes": sub.number_of_nodes(),
        "n_edges": sub.number_of_edges(),
        "n_bridge_edges": n_bridge,
        "length_km": round(length_km, 3),
    }


# ---------------------------------------------------------------------------
# Final connected-component report -- printed at the end of both `prepare`
# (as an initial-state snapshot) and `apply` (as the final post-removal
# result), mirroring the report Step 1.4 (2_3_gap_repair.py) prints after
# gap bridging.
# ---------------------------------------------------------------------------

def report_connected_components(
    G: nx.MultiGraph, network: str, label: str, top_n: int = 5
) -> Dict[str, object]:
    """Print (and return) a connected-component report for the given graph.

    Reports total node/edge counts, total connected-component count,
    whether the graph is a single fully-connected component, and a table
    of the `top_n` largest components by node count -- so it's immediately
    visible whether any remaining fragmentation is "one dominant component
    plus a handful of tiny stragglers" (usually fine) or something worth a
    second look.

    Args:
        G: The graph to report on.
        network: "highway" or "railway" -- used only in the printed banner.
        label: Short description of when this report was taken (e.g.
            "BEFORE component review" or "AFTER component review (apply)").
        top_n: How many of the largest components to list individually.

    Returns:
        A dict summary of the printed numbers.
    """
    total_nodes = G.number_of_nodes()
    total_edges = G.number_of_edges()
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    n_components = len(components)
    is_fully_connected = n_components == 1

    top_components = []
    for rank, comp in enumerate(components[:top_n], start=1):
        sub = G.subgraph(comp)
        top_components.append(
            {"rank": rank, "nodes": sub.number_of_nodes(), "edges": sub.number_of_edges()}
        )

    largest_nodes = top_components[0]["nodes"] if top_components else 0
    largest_edges = top_components[0]["edges"] if top_components else 0
    largest_pct = (100 * largest_nodes / total_nodes) if total_nodes else 0.0

    _print_banner(f"CONNECTED-COMPONENT REPORT -- {network.upper()} [{label}]")
    print(f"Total nodes:              {total_nodes:,}")
    print(f"Total edges:              {total_edges:,}")
    print(f"Connected components:     {n_components:,}")
    if is_fully_connected:
        print("Fully connected:          YES -- exactly 1 component.")
    else:
        print(
            f"Fully connected:          NO -- {n_components:,} components remain."
        )
    print(
        f"Largest component:        {largest_nodes:,} nodes, {largest_edges:,} edges "
        f"({largest_pct:.2f}% of all nodes)"
    )

    if n_components > 1:
        print(f"\nTop {min(top_n, n_components)} component(s) by node count:")
        print(f"  {'rank':>4}  {'nodes':>10}  {'edges':>10}")
        for c in top_components:
            print(f"  {c['rank']:>4}  {c['nodes']:>10,}  {c['edges']:>10,}")
        if n_components > top_n:
            print(f"  ... plus {n_components - top_n:,} smaller component(s) not listed.")

    return {
        "network": network,
        "label": label,
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "n_components": n_components,
        "is_fully_connected": is_fully_connected,
        "largest_component_nodes": largest_nodes,
        "largest_component_edges": largest_edges,
        "largest_component_pct_of_nodes": largest_pct,
        "top_components": top_components,
    }


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------

def cmd_prepare(args: argparse.Namespace) -> None:
    _print_banner(f"STEP 1.5 -- prepare component review [{args.network}]")

    graph_path: Path = args.graph_pickle
    print(f"Loading graph: {graph_path}")
    if not graph_path.exists():
        raise SystemExit(
            f"Graph pickle not found: {graph_path}\n"
            f"Pass the real path explicitly with --graph-pickle, e.g. the "
            f"output of your latest 2_3_gap_repair.py run "
            f"(data/graph/gap_bridging/{args.network}_bridged.pkl)."
        )
    with open(graph_path, "rb") as f:
        G: nx.MultiGraph = pickle.load(f)

    print(f"Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    report_connected_components(G, args.network, label="BEFORE component review")

    components = compute_components_with_ids(G)
    print(f"\nConnected components: {len(components):,}")

    args.review_geojson.parent.mkdir(parents=True, exist_ok=True)

    features = []
    sidecar: Dict[str, List[Node]] = {}

    for comp_id, comp_nodes in components:
        stats = build_component_stats(G, comp_nodes)
        sub = G.subgraph(comp_nodes)

        # Geometry: one line segment per edge in the component, as a
        # MultiLineString -- gives you the component's real shape in QGIS,
        # not just an abstract node cloud. `sub.edges()` on a MultiGraph
        # yields one (u, v) pair per parallel edge too, so a component with
        # duplicate parallel lines still draws every one of them.
        coords = [[list(u), list(v)] for u, v in sub.edges()]
        if not coords:
            # isolated single node with no edges -- draw as a degenerate
            # zero-length line so it still shows up as a point-like feature
            (only_node,) = comp_nodes
            coords = [[list(only_node), list(only_node)]]

        features.append({
            "type": "Feature",
            "properties": {
                "component_id": comp_id,
                **stats,
                "decision": False,  # true = noise, delete it. Default false.
                "notes": "",
            },
            "geometry": {
                "type": "MultiLineString",
                "coordinates": coords,
            },
        })
        sidecar[comp_id] = sorted(comp_nodes)

    geojson_obj = {"type": "FeatureCollection", "features": features}
    with open(args.review_geojson, "w") as f:
        json.dump(geojson_obj, f)
    print(f"Wrote review GeoJSON: {args.review_geojson}  ({len(features)} components)")

    with open(args.sidecar, "wb") as f:
        pickle.dump({
            "graph_path": str(graph_path),
            "network": args.network,
            "n_nodes_at_export": G.number_of_nodes(),
            "n_edges_at_export": G.number_of_edges(),
            "components": sidecar,  # component_id -> sorted list of nodes
        }, f)
    print(f"Wrote sidecar (exact node membership): {args.sidecar}")

    _print_banner("Top 10 smallest components by length_km (likely noise candidates)")
    smallest = sorted(features, key=lambda ft: ft["properties"]["length_km"])[:10]
    for ft in smallest:
        p = ft["properties"]
        print(f"  {p['component_id']}: {p['n_nodes']} nodes, "
              f"{p['length_km']} km, {p['n_edges']} edges")

    print(
        "\nNext: open the review GeoJSON in QGIS (see the docstring at the top "
        "of this script for layer order and workflow), set `decision` = true "
        "for components that are noise, save edits, then run:\n"
        f"  python {Path(__file__).name} --network {args.network} apply"
    )


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------

def cmd_apply(args: argparse.Namespace) -> None:
    _print_banner(f"STEP 1.5 -- apply component review [{args.network}]")

    graph_path: Path = args.graph_pickle
    print(f"Loading graph: {graph_path}")
    if not graph_path.exists():
        raise SystemExit(f"Graph pickle not found: {graph_path}")
    with open(graph_path, "rb") as f:
        G: nx.MultiGraph = pickle.load(f)
    print(f"Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    print(f"Loading sidecar: {args.sidecar}")
    if not args.sidecar.exists():
        raise SystemExit(
            f"Sidecar not found: {args.sidecar}\n"
            f"Run the `prepare` subcommand first."
        )
    with open(args.sidecar, "rb") as f:
        sidecar = pickle.load(f)

    if sidecar["n_nodes_at_export"] != G.number_of_nodes() or \
       sidecar["n_edges_at_export"] != G.number_of_edges():
        print(
            "  WARNING: current graph's node/edge count differs from the "
            "graph this review was prepared against "
            f"({sidecar['n_nodes_at_export']:,} nodes / "
            f"{sidecar['n_edges_at_export']:,} edges at export time vs. "
            f"{G.number_of_nodes():,} / {G.number_of_edges():,} now). "
            "The component decisions below may no longer correspond to this "
            "graph's actual structure. Re-run `prepare` and redo the review "
            "if you're not certain why this changed."
        )

    print(f"Loading reviewed GeoJSON: {args.review_geojson}")
    if not args.review_geojson.exists():
        raise SystemExit(f"Review GeoJSON not found: {args.review_geojson}")
    with open(args.review_geojson) as f:
        review = json.load(f)

    sidecar_components: Dict[str, List[Node]] = sidecar["components"]
    reviewed_ids = {
        ft["properties"]["component_id"] for ft in review["features"]
    }
    missing_from_sidecar = reviewed_ids - set(sidecar_components)
    if missing_from_sidecar:
        raise SystemExit(
            f"Review GeoJSON references component id(s) not present in the "
            f"sidecar: {sorted(missing_from_sidecar)}. Don't hand-edit "
            f"component_id values -- re-run `prepare` if the graph changed."
        )

    to_remove: List[Tuple[str, str]] = []  # (component_id, notes)
    for ft in review["features"]:
        props = ft["properties"]
        decision = props.get("decision", False)
        # QGIS booleans sometimes round-trip through GeoJSON as the strings
        # "true"/"false" or ints 0/1 depending on driver -- normalize.
        is_noise = decision in (True, "true", "True", 1, "1")
        if is_noise:
            to_remove.append((props["component_id"], props.get("notes", "")))

    print(f"Components marked as noise (decision=true): {len(to_remove)} "
          f"of {len(reviewed_ids)}")

    report_before = report_connected_components(G, args.network, label="BEFORE apply")

    if not to_remove:
        print("\nNothing marked for removal -- graph is unchanged. Exiting.")
        report_connected_components(G, args.network, label="AFTER apply (no changes)")
        return

    args.log_csv.parent.mkdir(parents=True, exist_ok=True)
    log_rows = []
    total_nodes_removed = 0
    total_edges_removed = 0
    total_length_removed = 0.0

    for comp_id, notes in to_remove:
        nodes = sidecar_components[comp_id]
        present = [n for n in nodes if n in G]
        absent = len(nodes) - len(present)
        if absent:
            print(
                f"  WARNING: {comp_id}: {absent} of {len(nodes)} nodes no "
                f"longer present in the graph (already removed by an "
                f"earlier step?). Removing whatever's still there."
            )
        sub = G.subgraph(present)
        n_edges = sub.number_of_edges()
        length_km = sum(_edge_length_km(d, u, v) for u, v, d in sub.edges(data=True))

        G.remove_nodes_from(present)

        total_nodes_removed += len(present)
        total_edges_removed += n_edges
        total_length_removed += length_km
        log_rows.append({
            "component_id": comp_id,
            "n_nodes_removed": len(present),
            "n_edges_removed": n_edges,
            "length_km_removed": round(length_km, 3),
            "notes": notes,
        })
        print(f"  Removed {comp_id}: {len(present)} nodes, {n_edges} edges, "
              f"{length_km:.2f} km -- {notes or '(no notes)'}")

    with open(args.log_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["component_id", "n_nodes_removed", "n_edges_removed",
                           "length_km_removed", "notes"]
        )
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"\nAudit log written to: {args.log_csv}")

    print(f"\nNodes removed:  {total_nodes_removed:,}")
    print(f"Edges removed:  {total_edges_removed:,}")
    print(f"Length removed: {total_length_removed:,.2f} km")

    args.output_pickle.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_pickle, "wb") as f:
        pickle.dump(G, f)
    print(f"\nSaved denoised graph to: {args.output_pickle}")

    # ------------------------------------------------------------------
    # FINAL REPORT -- the last thing this command prints: how many
    # connected components remain now, compared to before this run.
    # ------------------------------------------------------------------
    report_after = report_connected_components(G, args.network, label="AFTER apply (final)")

    _print_banner(f"BEFORE / AFTER SUMMARY -- {args.network.upper()}")
    print(
        f"  Connected components: {report_before['n_components']:,} -> "
        f"{report_after['n_components']:,}"
    )
    print(
        f"  Fully connected:       "
        f"{'YES' if report_before['is_fully_connected'] else 'NO'} -> "
        f"{'YES' if report_after['is_fully_connected'] else 'NO'}"
    )
    print(f"  Nodes:                 {report_before['total_nodes']:,} -> {report_after['total_nodes']:,}")
    print(f"  Edges:                 {report_before['total_edges']:,} -> {report_after['total_edges']:,}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def resolve_networks(args: argparse.Namespace) -> List[str]:
    """Decide which network(s) to run, following the same "both by default"
    convention as Steps 1.2-1.4: if --network wasn't passed, run both
    highway and railway (in a fixed, deterministic order) instead of
    requiring the flag every time."""
    if args.network:
        return [args.network]
    return sorted(NETWORK_FILENAMES.keys())


def build_paths_for_network(args: argparse.Namespace, network: str) -> argparse.Namespace:
    """Return a shallow copy of `args` with every per-network default path
    resolved for `network`, without mutating the shared parsed args object
    (needed because a single invocation may loop over both networks)."""
    resolved = argparse.Namespace(**vars(args))
    resolved.network = network
    resolved.graph_pickle = args.graph_pickle or DEFAULT_GRAPH_PICKLE[network]
    resolved.review_geojson = args.review_geojson or DEFAULT_REVIEW_GEOJSON(network)
    resolved.sidecar = args.sidecar or DEFAULT_SIDECAR(network)
    if args.command == "apply":
        resolved.output_pickle = args.output_pickle or DEFAULT_OUTPUT_PICKLE(network)
        resolved.log_csv = args.log_csv or DEFAULT_LOG_CSV(network)
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 1.5: review connected components in QGIS "
                    "and delete the ones marked as noise. Runs BOTH "
                    "highway and railway automatically unless --network "
                    "restricts it to one."
    )
    parser.add_argument(
        "--network", choices=sorted(NETWORK_FILENAMES.keys()), default=None,
        help="Restrict the run to just this one network. If omitted, BOTH "
             "highway and railway are processed automatically.",
    )
    parser.add_argument(
        "--graph-pickle", type=Path, default=None,
        help="Path to the input graph pickle. Defaults to Step 1.4's own "
             "output path per --network "
             "(data/graph/gap_bridging/<network>_bridged.pkl) -- override "
             "this if your layout differs. Only usable in single-network "
             "mode (i.e. together with --network).",
    )
    parser.add_argument(
        "--review-geojson", type=Path, default=None,
        help="Path to the component review GeoJSON (default: "
             "data/graph/component_review/<network>_component_review.geojson).",
    )
    parser.add_argument(
        "--sidecar", type=Path, default=None,
        help="Path to the sidecar pickle (default: "
             "data/graph/component_review/<network>_component_review_sidecar.pkl).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("prepare", help="Export components for QGIS review.")

    apply_parser = subparsers.add_parser(
        "apply", help="Delete components marked as noise (decision=true)."
    )
    apply_parser.add_argument(
        "--output-pickle", type=Path, default=None,
        help="Where to save the denoised graph (default: "
             "data/graph/component_review/<network>_denoised.pkl).",
    )
    apply_parser.add_argument(
        "--log-csv", type=Path, default=None,
        help="Where to save the removal audit log (default: "
             "data/graph/component_review/<network>_component_noise_log.csv).",
    )

    args = parser.parse_args()

    if args.graph_pickle is not None and args.network is None:
        raise SystemExit(
            "--graph-pickle requires --network (an explicit input path only "
            "makes sense for a single network at a time)."
        )

    return args


def main() -> None:
    args = parse_args()

    networks_to_run = resolve_networks(args)
    print(f"Running Step 1.5 ({args.command}) for: {', '.join(networks_to_run)}")

    for network in networks_to_run:
        print()
        print("#" * 70)
        print(f"# NETWORK: {network}")
        print("#" * 70)

        net_args = build_paths_for_network(args, network)

        if args.command == "prepare":
            cmd_prepare(net_args)
        elif args.command == "apply":
            cmd_apply(net_args)

    print()
    _print_banner("ALL NETWORKS COMPLETE")
    print(f"Ran: {', '.join(networks_to_run)}")


if __name__ == "__main__":
    main()