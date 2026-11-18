#!/usr/bin/env python3
"""
Step 2 -- Snap COUNTY CENTROIDS onto the final HIGHWAY and RAILWAY graphs
(Illinois test run).

DATASET NOTE (updated to match the Illinois test run's naming convention):
-----------------------------------------------------------------------
The earlier draft of this script (3_snap_facility.py / "3.1_snap_facilities.py")
snapped forts (`ppp_installations_clean.geojson`) and seaports
(`seaports_clean.geojson`) onto a single hardcoded STRAHNET graph
(`data/graph/final/strahnet_final_largest_graph.pkl`). Neither of those
facility datasets, nor that graph path, exist in this pipeline. This
script has been rewritten to match what this project actually produces:

  * Facility input: COUNTY CENTROIDS, not forts/seaports. Traced directly
    to 1_cleanDataset.py's own output convention:
        OUTPUT_FILENAMES["county_centroids"] = "county_centroids_clean.geojson"
        CLEANED_DIR = data/cleaned
    i.e. data/cleaned/county_centroids_clean.geojson -- not independently
    guessed. That file has one point per county (`facility_type` ==
    "COUNTY_CENTROIDS", per `clean_facility_dataset()`), a stable
    `phase1_record_id`, and whatever original source attribute columns
    the county layer shipped with (name column not standardized upstream,
    so this script probes a short list of common Census county-name
    fields -- see `NAME_COL_CANDIDATES` below -- falling back to
    `phase1_record_id` if none match).

  * Graph input: BOTH "highway" and "railway", not a single STRAHNET
    graph. Traced directly to 2_8_keep_the_largest_element.py's own
    output convention:
        out_path = final_dir / f"{network}_final_largest.pkl"
    i.e. data/graph/final/highway_final_largest.pkl and
    data/graph/final/railway_final_largest.pkl -- exactly the files that
    script's own docstring says are "Ready for Step 2 (facility
    snapping)." Following the "both networks by default" convention
    already established by every other script in this pipeline (2_1
    through 2_8), --network is OPTIONAL: omitting it snaps counties onto
    BOTH graphs automatically, one after another, each producing its own
    output graph and its own CSV report. A missing graph pickle for one
    network prints a clear warning and skips just that network.

  * Facility bookkeeping: the original fort/port version tracked two
    disjoint sets (O = fort-nodes, D = port-nodes) for an O x D
    fort-to-port travel-time matrix. There is only one facility type here
    (counties), so this script tracks a single set C = unique
    county-nodes per network, and reports N = C(|C|, 2) possible
    county-to-county route pairs for that network's later travel-time
    matrix (unordered pairs -- county A to county B and county B to
    county A are the same trip on an undirected graph).

Everything else -- the actual snapping algorithm -- is UNCHANGED from the
original script and still implements the paper's method (Section 2.2.3,
"Identify Origins and Destinations"):

  1. For each facility, find the geometrically closest point on the graph
     G using Euclidean distance -- searching BOTH existing nodes AND the
     interiors of edges, not just nearest-node.
  2. If that closest point IS an existing node -> label the node as a
     county node.
  3. If the closest point falls in the MIDDLE of an edge -> split that
     edge into two segments at that point, insert a new node there, and
     label the new node as the county node.
  4. If two distinct counties snap to the same point -> that one node
     represents both (no duplicate nodes created).
  5. Bookkeeping: C = set of unique county-nodes for N = |C| choose 2
     county-to-county routes for the later travel-time matrix.

CAVEAT: the input graph's edges are stored as straight lines between two
rounded (lon, lat) node coordinates -- the original curved road/rail
geometry was not preserved through earlier pipeline steps. This script
therefore finds the nearest point on the STRAIGHT CHORD between each
edge's endpoints, not the true curved pavement/track. Reasonable
approximation given what's available; adapt to snap against real
LineString geometry if you have it.

Paths
-----
Input (per network):
    data/graph/final/<network>_final_largest.pkl
        (2_8_keep_the_largest_element.py's own output path)

Input (shared across both networks):
    data/cleaned/county_centroids_clean.geojson
        (1_cleanDataset.py's own OUTPUT_FILENAMES["county_centroids"])

Output (per network):
    data/graph/snap_facility/<network>_with_counties.pkl
    reports/<network>_county_snap_report.csv

Usage
-----
    # NEW DEFAULT: snaps counties onto BOTH highway and railway graphs
    python 3_snap_counties.py

    # restrict to just one network
    python 3_snap_counties.py --network highway
    python 3_snap_counties.py --network railway

    # override paths (single-network mode only)
    python 3_snap_counties.py --network highway --graph-pickle path/to/other.pkl
    python 3_snap_counties.py --network highway --counties-geojson path/to/other.geojson
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
from shapely.strtree import STRtree

# ---------------------------------------------------------------------------
# Paths -- kept in sync by hand with 1_cleanDataset.py's OUTPUT_FILENAMES
# and 2_8_keep_the_largest_element.py's own final-graph output convention.
# ---------------------------------------------------------------------------

FINAL_GRAPH_INPUT_DIR = Path("data/graph/final")     # 2_8's own final_dir
NETWORK_GRAPH_FILENAMES = {
    "highway": "highway_final_largest.pkl",
    "railway": "railway_final_largest.pkl",
}

# 1_cleanDataset.py's own CLEANED_DIR / OUTPUT_FILENAMES["county_centroids"] --
# a single shared file, not per-network.
COUNTY_CENTROIDS_PATH = Path("data/cleaned/county_centroids_clean.geojson")

OUTPUT_GRAPH_DIR = Path("data/graph/snap_counties")
REPORTS_DIR = Path("reports")

# HARDCODED DEFAULT NETWORK -- same pattern as every other script in this
# pipeline (2_1 through 2_8). Only used as --network's implicit fallback
# when an override path is given without --network (belt-and-suspenders;
# overrides here require --network explicitly, see parse_args).
HARDCODED_NETWORK = "highway"

# county_centroids_clean.geojson's name column was never standardized by
# 1_cleanDataset.py (clean_facility_dataset() only adds `facility_type` and
# `phase1_record_id` -- the original source attribute names are preserved
# as-is). These are the common Census county-name field candidates, tried
# in order; falls back to `phase1_record_id` if none match.
NAME_COL_CANDIDATES = [
    "namelsad", "name", "county_name", "countyname", "co_name",
    "name10", "namelsad10", "geoid", "countyfp",
]

METRIC_CRS = "EPSG:5070"     # matches the rest of the pipeline (meters)
GEOGRAPHIC_CRS = "EPSG:4326"
SAME_NODE_TOLERANCE_M = 1.0  # snap-to-existing-node if within this distance
ROUND_DECIMALS = 6           # matches step 1.5/2.7's permanent node-ID rounding
FAR_SNAP_WARNING_M = 50_000  # flag counties snapped from farther than this


def _print_banner(title: str) -> None:
    """Section-banner style matching the rest of this pipeline (2_2, 2_3, 2_7, 2_8)."""
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_graph(path: Path) -> nx.MultiGraph:
    with open(path, "rb") as f:
        G = pickle.load(f)
    print(f"Loaded graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges "
          f"({'multigraph' if G.is_multigraph() else 'simple graph'})")
    return G


def load_counties(path: Path):
    if not path.exists():
        raise SystemExit(
            f"County centroids file not found: {path}\n"
            f"This should be 1_cleanDataset.py's own "
            f"OUTPUT_FILENAMES['county_centroids'] output -- run Phase 1 "
            f"first, or pass --counties-geojson explicitly."
        )
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise SystemExit(f"{path} has no CRS set -- refusing to guess.")
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(GEOGRAPHIC_CRS)

    name_col = next((c for c in NAME_COL_CANDIDATES if c in gdf.columns), None)
    if name_col is None:
        if "phase1_record_id" in gdf.columns:
            name_col = "phase1_record_id"
            print(
                f"  NOTE: none of {NAME_COL_CANDIDATES} found in {path}; "
                f"falling back to 'phase1_record_id' as the display name."
            )
        else:
            raise SystemExit(
                f"Could not find a name column in {path}. Tried "
                f"{NAME_COL_CANDIDATES} plus 'phase1_record_id'; found "
                f"columns: {list(gdf.columns)}. Edit NAME_COL_CANDIDATES."
            )
    print(f"Loaded {len(gdf):,} county centroids from {path} (name column: '{name_col}')")
    return gdf, name_col


# ---------------------------------------------------------------------------
# Building a searchable edge index (straight chords, in meters)
# ---------------------------------------------------------------------------

def build_edge_index(G: nx.MultiGraph):
    """
    Build a list of (u, v, key, line_m) for every non-self-loop edge, where
    line_m is that edge's straight-line chord as a shapely LineString in
    EPSG:5070 (meters), plus an STRtree over those geometries for fast
    nearest-neighbor queries.
    """
    lines_deg = []
    edge_keys = []
    for u, v, k in G.edges(keys=True):
        if u == v:
            continue  # skip self-loops -- nothing meaningful to snap to
        lines_deg.append(LineString([u, v]))
        edge_keys.append((u, v, k))

    print(f"Reprojecting {len(lines_deg):,} edge chords to {METRIC_CRS} for nearest-point search...")
    lines_m = gpd.GeoSeries(lines_deg, crs=GEOGRAPHIC_CRS).to_crs(METRIC_CRS).tolist()

    records = [(u, v, k, line_m) for (u, v, k), line_m in zip(edge_keys, lines_m)]
    tree = STRtree([r[3] for r in records])
    print(f"Built spatial index over {len(records):,} edge chords.")
    return records, tree


def find_nearest_edge(pt_m: Point, records, tree) -> Tuple[int, float]:
    idx = tree.nearest(pt_m)
    line_m = records[idx][3]
    dist_m = pt_m.distance(line_m)
    return idx, dist_m


# ---------------------------------------------------------------------------
# Core snapping logic (unchanged from the original fort/port version)
# ---------------------------------------------------------------------------

def round_coord(pt) -> Tuple[float, float]:
    return (round(pt[0], ROUND_DECIMALS), round(pt[1], ROUND_DECIMALS))


def snap_one_facility(G: nx.MultiGraph, pt_m: Point, records, tree) -> Tuple[Tuple[float, float], float, str]:
    """Snap a single county centroid onto the graph. Returns (node_id, distance_m, method)."""
    idx, dist_m = find_nearest_edge(pt_m, records, tree)
    u, v, k, line_m = records[idx]

    if G.get_edge_data(u, v, k) is None:
        # stale reference -- this original edge was already split by an
        # earlier county snap in this same run. Re-locate among the
        # edges now touching u or v rather than trusting the frozen index.
        candidates = []
        for a, b, kk, d in list(G.edges(u, keys=True, data=True)) + list(G.edges(v, keys=True, data=True)):
            if a == b:
                continue
            cand_line_deg = LineString([a, b])
            cand_line_m = gpd.GeoSeries([cand_line_deg], crs=GEOGRAPHIC_CRS).to_crs(METRIC_CRS).iloc[0]
            candidates.append((a, b, kk, cand_line_m))
        if not candidates:
            raise RuntimeError(f"Could not re-locate stale edge near {u}/{v} -- graph may be malformed.")
        dists = [pt_m.distance(c[3]) for c in candidates]
        best = int(np.argmin(dists))
        u, v, k, line_m = candidates[best]
        dist_m = dists[best]

    nearest_on_line_m = nearest_points(line_m, pt_m)[0]

    u_m = Point(gpd.GeoSeries([Point(u)], crs=GEOGRAPHIC_CRS).to_crs(METRIC_CRS).iloc[0])
    v_m = Point(gpd.GeoSeries([Point(v)], crs=GEOGRAPHIC_CRS).to_crs(METRIC_CRS).iloc[0])
    dist_to_u = nearest_on_line_m.distance(u_m)
    dist_to_v = nearest_on_line_m.distance(v_m)

    if dist_to_u <= SAME_NODE_TOLERANCE_M:
        return u, dist_m, "reused_existing_node"
    if dist_to_v <= SAME_NODE_TOLERANCE_M:
        return v, dist_m, "reused_existing_node"

    # --- split the edge at nearest_on_line_m ---
    total_len_m = line_m.length
    frac = dist_to_u / total_len_m if total_len_m > 0 else 0.0
    frac = min(max(frac, 0.0), 1.0)

    split_pt_deg = gpd.GeoSeries([nearest_on_line_m], crs=METRIC_CRS).to_crs(GEOGRAPHIC_CRS).iloc[0]
    new_node = round_coord((split_pt_deg.x, split_pt_deg.y))

    if new_node == u or new_node == v or G.has_node(new_node):
        return new_node, dist_m, "reused_existing_node (post-rounding collision)"

    edge_data = dict(G.get_edge_data(u, v, k))
    length_km = edge_data.get("length_km", 0.0)
    travel_time_min = edge_data.get("travel_time_min", 0.0)
    edge_type = edge_data.get("edge_type", "real")

    length_km_1 = length_km * frac
    length_km_2 = length_km * (1 - frac)
    time_1 = travel_time_min * frac
    time_2 = travel_time_min * (1 - frac)

    G.remove_edge(u, v, k)
    G.add_edge(u, new_node, length_km=length_km_1, travel_time_min=time_1, edge_type=edge_type)
    G.add_edge(new_node, v, length_km=length_km_2, travel_time_min=time_2, edge_type=edge_type)

    return new_node, dist_m, "split_edge"


def snap_county_set(G, gdf, name_col, records, tree, report_rows) -> List[Tuple[float, float]]:
    gdf_m = gdf.to_crs(METRIC_CRS)
    node_ids = []
    for (_, row_deg), (_, row_m) in zip(gdf.iterrows(), gdf_m.iterrows()):
        pt_deg = row_deg.geometry
        pt_m = row_m.geometry
        node, dist_m, method = snap_one_facility(G, pt_m, records, tree)

        if not G.has_node(node):
            G.add_node(node)
        existing_types = G.nodes[node].get("facility_types", set())
        existing_types.add("county")
        G.nodes[node]["facility_types"] = existing_types
        existing_names = G.nodes[node].get("facility_names", [])
        existing_names.append(row_deg[name_col])
        G.nodes[node]["facility_names"] = existing_names

        node_ids.append(node)
        report_rows.append([
            "county", row_deg[name_col], pt_deg.x, pt_deg.y,
            node[0], node[1], round(dist_m, 2), method,
        ])

    return node_ids


# ---------------------------------------------------------------------------
# Per-network run
# ---------------------------------------------------------------------------

def run_one_network(
    network: str,
    graph_path: Path,
    counties_gdf: gpd.GeoDataFrame,
    county_name_col: str,
    output_graph_dir: Path,
    reports_dir: Path,
) -> bool:
    """Snap county centroids onto one network's final graph.

    Returns True on success. Returns False (without raising) if the input
    graph pickle doesn't exist, so a multi-network run can skip this
    network and continue with the other one instead of aborting entirely
    -- matching the "both by default, skip what's missing" convention used
    everywhere else in this pipeline (2_1 through 2_8).
    """
    print()
    print("#" * 70)
    print(f"# NETWORK: {network}")
    print("#" * 70)

    if not graph_path.exists():
        print(
            f"  WARNING: input graph pickle not found, skipping this network:\n"
            f"    {graph_path}\n"
            f"  (Run 2_8_keep_the_largest_element.py for this network first -- "
            f"its final_dir/f'{{network}}_final_largest.pkl' is exactly what "
            f"this path points at -- or pass --graph-pickle directly.)"
        )
        return False

    _print_banner(f"STEP 2 -- SNAP COUNTY CENTROIDS [{network}]")
    print(f"Loading graph from: {graph_path}")
    G = load_graph(graph_path)

    print("\nBuilding edge spatial index...")
    records, tree = build_edge_index(G)

    report_rows: list = []

    print(f"\nSnapping {len(counties_gdf):,} county centroid point(s)...")
    county_nodes = snap_county_set(G, counties_gdf, county_name_col, records, tree, report_rows)

    C = sorted(set(county_nodes))
    n_pairs = len(C) * (len(C) - 1) // 2

    _print_banner(f"COUNTY SNAPPING SUMMARY -- {network.upper()}")
    print(f"County points in:         {len(counties_gdf):,}")
    print(f"Unique county nodes C:    {len(C):,}  (collisions: {len(counties_gdf) - len(C):,})")
    print(f"N = C(|C|, 2) possible county-to-county routes: {n_pairs:,}")
    print(f"Graph after snapping: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    print(f"Still connected: {nx.is_connected(nx.Graph(G))}")

    split_count = sum(1 for r in report_rows if r[-1] == "split_edge")
    reused_count = len(report_rows) - split_count
    print(f"\nSplit an existing edge: {split_count:,} counties")
    print(f"Reused an existing node: {reused_count:,} counties")

    output_graph_dir.mkdir(parents=True, exist_ok=True)
    out_graph_path = output_graph_dir / f"{network}_with_counties.pkl"
    with open(out_graph_path, "wb") as f:
        pickle.dump({"graph": G, "C_county_nodes": C}, f)
    print(f"\nSaved graph + county node list to: {out_graph_path}")

    reports_dir.mkdir(parents=True, exist_ok=True)
    out_report_path = reports_dir / f"{network}_county_snap_report.csv"
    with open(out_report_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "facility_kind", "facility_name", "orig_lon", "orig_lat",
            "snapped_node_lon", "snapped_node_lat", "snap_distance_m", "method",
        ])
        writer.writerows(report_rows)
    print(f"Per-county snap report saved to: {out_report_path}")

    large_snaps = [r for r in report_rows if r[6] > FAR_SNAP_WARNING_M]
    if large_snaps:
        print(f"\nWARNING: {len(large_snaps)} counties snapped from more than "
              f"{FAR_SNAP_WARNING_M/1000:.0f} km away -- worth a look (data issue?):")
        for r in sorted(large_snaps, key=lambda r: -r[6])[:15]:
            print(f"  {r[0]:6s} {r[1]!r:40s} snap_distance_m={r[6]:,.0f}")

    _print_banner(f"DONE [{network}]")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 2: snap county centroids onto the final HIGHWAY and "
            "RAILWAY graphs. Runs BOTH networks automatically unless "
            "--network restricts it to one."
        )
    )
    parser.add_argument(
        "--network",
        choices=sorted(NETWORK_GRAPH_FILENAMES.keys()),
        default=None,
        help=(
            "Restrict the run to just this one network. If omitted, BOTH "
            "highway and railway are processed automatically (matching "
            "every other script in this pipeline)."
        ),
    )
    parser.add_argument(
        "--graph-pickle", type=Path, default=None,
        help="Override the input graph pickle path (default: "
             f"{FINAL_GRAPH_INPUT_DIR}/<network>_final_largest.pkl -- "
             "2_8_keep_the_largest_element.py's own output). Requires "
             "--network (an explicit path only makes sense for a single "
             "network at a time).",
    )
    parser.add_argument(
        "--counties-geojson", type=Path, default=COUNTY_CENTROIDS_PATH,
        help=f"Path to the county centroids file (default: {COUNTY_CENTROIDS_PATH} "
             "-- 1_cleanDataset.py's own OUTPUT_FILENAMES['county_centroids']). "
             "Shared across both networks.",
    )
    parser.add_argument(
        "--output-graph-dir", type=Path, default=OUTPUT_GRAPH_DIR,
        help=f"Folder for the output graph pickle(s) (default: {OUTPUT_GRAPH_DIR}).",
    )
    parser.add_argument(
        "--reports-dir", type=Path, default=REPORTS_DIR,
        help=f"Folder for the per-county CSV snap report(s) (default: {REPORTS_DIR}).",
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

    if args.network:
        networks_to_run = [args.network]
    else:
        # NEW DEFAULT: both networks, in a fixed/deterministic order --
        # matches 2_1 through 2_8's "both by default" convention.
        networks_to_run = sorted(NETWORK_GRAPH_FILENAMES.keys())

    print(f"Running Step 2 (county snapping) for: {', '.join(networks_to_run)}")

    counties_gdf, county_name_col = load_counties(args.counties_geojson)

    ran = []
    skipped = []

    for network in networks_to_run:
        graph_path = (
            args.graph_pickle
            if args.graph_pickle is not None
            else FINAL_GRAPH_INPUT_DIR / NETWORK_GRAPH_FILENAMES[network]
        )

        ok = run_one_network(
            network=network,
            graph_path=graph_path,
            counties_gdf=counties_gdf,
            county_name_col=county_name_col,
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
            "graph pickle was missing. Run 2_8_keep_the_largest_element.py "
            "first, or pass --network + --graph-pickle directly."
        )


if __name__ == "__main__":
    main()