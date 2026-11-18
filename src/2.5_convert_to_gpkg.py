#!/usr/bin/env python3
"""
Step 1.4.5 (prep) -- Export the reviewed graph (Step 1.5's denoised .pkl
output) to a GeoPackage that's actually safe to review/edit in QGIS.

DATASET NOTE (updated to match the Illinois test run's naming convention):
-----------------------------------------------------------------------
Earlier drafts of this script were written against the old
strahnet/stracnet naming. The pipeline has since moved to "highway" /
"railway" labels throughout (see 1_cleanDataset.py's OUTPUT_FILENAMES,
2_2_fix_toucing_and_crossing.py's NODED_OUTPUT_DIR, 2_3_gap_repair.py's
BRIDGED_OUTPUT_DIR, and 2_4_remove_noise.py's REVIEW_DIR /
DEFAULT_OUTPUT_PICKLE). This script has been updated to match:

  * NETWORK_FILENAMES now maps "highway"/"railway" -> their respective
    *_noded.geojson filenames (Step 1.3's output), matching the pattern
    used by every later script for resolving the noded source geometry.
  * The default input graph pickle is Step 1.5's (2_4_remove_noise.py)
    OWN output -- data/graph/component_review/<network>_denoised.pkl --
    not Step 1.4's raw bridged pickle. This script's job is to produce a
    QGIS-reviewable GeoPackage of the graph as it stands AFTER the
    component-level noise review, so it should read the denoised graph.
    If you genuinely want to export the pre-denoise bridged graph instead
    (e.g. to review noise-removal decisions themselves), point
    --bridged-pkl at data/graph/gap_bridging/<network>_bridged.pkl
    explicitly.

If 2_4_remove_noise.py's output naming changes again, update
NETWORK_FILENAMES / DENOISED_INPUT_DIR below to match -- keep them in sync
by hand, there is no automatic link between the scripts.

Why this script exists
-----------------------
The graph only ever stores two endpoint coordinates per edge
(`G.add_edge(coords[0], coords[-1], ...)`). It never keeps a line's
intermediate vertices -- fine for graph math (shortest path, connectivity),
but if you naively rebuild geometry as `LineString([u, v])` for every edge,
REAL road/rail edges get drawn as a straight line between their two
endpoints instead of following the actual curve/interchange/corner. That's
the "blue line cuts across the map" problem.

This script fixes that at export time:
  * For a real edge (came from a source line in the noded GeoJSON), it
    looks up and uses that line's FULL original geometry (every vertex).
  * For a synthetic edge (`is_gap_fill=True` -- automatic 400m bridge, or a
    manual connector you already added), there IS no source geometry, so a
    straight line between the two nodes is drawn -- and that's correct,
    because a straight connector really is what you're proposing there.

Every feature also gets an `edge_type` field (`real` / `auto_bridge` /
`manual`) so you can style/filter them separately in QGIS (e.g. red for
real, blue for auto_bridge, a third color for manual), and so you know
exactly which lines are safe to review as "does this gap look real" vs.
which are already-confirmed source road/rail geometry.

Usage
-----
    python 2_5_convert_to_gpkg.py --network highway
    python 2_5_convert_to_gpkg.py --network railway
    python 2_5_convert_to_gpkg.py --network highway --bridged-pkl path/to/other.pkl

    # both networks in one go (default -- no --network needed)
    python 2_5_convert_to_gpkg.py

Input
-----
    data/graph/component_review/<network>_denoised.pkl
        Step 1.5's (2_4_remove_noise.py) `apply` output -- the post-noise-
        review graph. This is the default input.

    data/graph/touching_crossing/<network>_noded.geojson
        Step 1.3's output, used to look up each real edge's full-vertex
        geometry.

Output
------
    data/graph/qgis_review/<network>_review.gpkg
    (single layer "edges", EPSG:4326, ready to open directly in QGIS)
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, Tuple

import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString

# ---------------------------------------------------------------------------
# Paths -- kept in sync by hand with the scripts that actually produce these
# files (2_2_fix_toucing_and_crossing.py's NODED_OUTPUT_DIR and
# 2_4_remove_noise.py's REVIEW_DIR / DEFAULT_OUTPUT_PICKLE).
# ---------------------------------------------------------------------------

NODED_INPUT_DIR = Path("data/graph/touching_crossing")     # Step 1.3's own output dir
DENOISED_INPUT_DIR = Path("data/graph/component_review")   # Step 1.5's own output dir (apply's --output-pickle default)
OUTPUT_DIR = Path("data/graph/qgis_review")

NETWORK_FILENAMES = {
    "highway": "highway_noded.geojson",
    "railway": "railway_noded.geojson",
}

# HARDCODED DEFAULT NETWORK -- same pattern as Steps 1.3/1.4/1.5. Only used
# if you want a single-network default; --network with no value still runs
# both by default (see main()).
HARDCODED_NETWORK = "highway"


def build_source_geometry_lookup(noded_gdf: gpd.GeoDataFrame) -> Dict[Tuple, list]:
    """Map (start, end) node-coordinate pairs -> list of full-vertex geometries.

    A list (not a single geometry) because duplicate (start, end) pairs are
    possible (parallel carriageways etc., see Step 1.4's MultiGraph note) --
    if two source lines share the same pair, both geometries are kept and
    handed out in order as matching graph edges are found, so neither one
    is silently dropped or duplicated.

    Keyed on the SORTED pair (frozenset-like tuple) so direction doesn't
    matter -- the graph doesn't care about edge direction and a source line
    could have its coords in either order relative to how the graph stored it.
    """
    lookup: Dict[Tuple, list] = {}
    for geom in noded_gdf.geometry:
        if geom is None or geom.geom_type != "LineString":
            continue
        coords = list(geom.coords)
        if len(coords) < 2:
            continue
        key = tuple(sorted([coords[0], coords[-1]]))
        lookup.setdefault(key, []).append(geom)
    return lookup


def export_graph(graph_pkl: Path, noded_geojson: Path, out_path: Path) -> None:
    print(f"Loading graph: {graph_pkl}")
    if not graph_pkl.exists():
        raise SystemExit(
            f"Graph pickle not found: {graph_pkl}\n"
            f"Default input is Step 1.5's denoised output "
            f"(data/graph/component_review/<network>_denoised.pkl). Run "
            f"2_4_remove_noise.py's `prepare`/`apply` cycle first, or pass "
            f"--bridged-pkl to point at a different graph pickle directly "
            f"(e.g. the raw Step 1.4 output in data/graph/gap_bridging/ if "
            f"you haven't done a noise-review pass yet)."
        )
    with open(graph_pkl, "rb") as f:
        G = pickle.load(f)
    print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges "
          f"({'MultiGraph' if G.is_multigraph() else 'Graph'})")

    print(f"Loading noded source geometry: {noded_geojson}")
    if not noded_geojson.exists():
        raise SystemExit(
            f"Noded GeoJSON not found: {noded_geojson}\n"
            f"This should be Step 1.3's (2_2_fix_toucing_and_crossing.py) "
            f"output -- run that step first, or pass --noded-geojson "
            f"explicitly."
        )
    noded_gdf = gpd.read_file(noded_geojson)
    lookup = build_source_geometry_lookup(noded_gdf)

    rows = []
    n_real = n_auto_bridge = n_manual = n_missing_source = 0

    edge_iter = G.edges(data=True, keys=True) if G.is_multigraph() else \
        ((u, v, None, d) for u, v, d in G.edges(data=True))

    for u, v, key, data in edge_iter:
        is_gap_fill = data.get("is_gap_fill", False)
        is_manual = data.get("is_manual", False)

        if not is_gap_fill:
            # Real edge -- try to recover the full-vertex source geometry.
            src_key = tuple(sorted([u, v]))
            candidates = lookup.get(src_key)
            if candidates:
                geom = candidates.pop(0)  # consume one match per real edge
                if not candidates:
                    del lookup[src_key]
                edge_type = "real"
                n_real += 1
            else:
                # Shouldn't normally happen for a non-gap-fill edge, but
                # don't silently fabricate a straight line and pretend it's
                # real -- fall back to a straight line AND flag it clearly.
                geom = LineString([u, v])
                edge_type = "real_MISSING_SOURCE_GEOM"
                n_missing_source += 1
        else:
            geom = LineString([u, v])
            edge_type = "manual" if is_manual else "auto_bridge"
            if is_manual:
                n_manual += 1
            else:
                n_auto_bridge += 1

        rows.append({
            "geometry": geom,
            "edge_type": edge_type,
            "node_start": str(u),
            "node_end": str(v),
            "length_km": data.get("length_km"),
            "travel_time_min": data.get("travel_time_min"),
        })

    out_gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")

    print(f"\nEdge breakdown:")
    print(f"  real (full source geometry):      {n_real:,}")
    print(f"  auto_bridge (400m gap fills):      {n_auto_bridge:,}")
    print(f"  manual (hand-added connectors):    {n_manual:,}")
    if n_missing_source:
        print(f"  *** real_MISSING_SOURCE_GEOM:      {n_missing_source:,}  "
              f"(non-gap-fill edges with no matching line in the noded "
              f"GeoJSON -- investigate before trusting these; they were "
              f"exported as straight lines and flagged so they're easy to "
              f"filter/find in QGIS.) ***")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_gdf.to_file(out_path, layer="edges", driver="GPKG")
    print(f"\nSaved: {out_path}  (layer 'edges')")
    print("Open in QGIS and style by 'edge_type': real / auto_bridge / manual "
          "/ real_MISSING_SOURCE_GEOM get different colors, so straight-line "
          "bridge candidates are the only thing you're visually judging "
          "against imagery.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network",
        choices=sorted(NETWORK_FILENAMES),
        default=None,
        help=(
            "Restrict the run to just this one network (highway or "
            "railway). If omitted, BOTH networks are processed "
            "automatically, one after the other -- matching the "
            "'both by default' convention used by every other script in "
            "this pipeline. --bridged-pkl / --noded-geojson / --output "
            "only make sense for a single network, so they can't be "
            "combined with an omitted --network."
        ),
    )
    parser.add_argument("--bridged-pkl", type=Path, default=None,
                         help="Override path to the input graph pickle "
                              "(default: "
                              f"{DENOISED_INPUT_DIR}/<network>_denoised.pkl "
                              "-- Step 1.5's post-noise-review output). "
                              "Only valid with a single --network.")
    parser.add_argument("--noded-geojson", type=Path, default=None,
                         help="Override path to the noded .geojson (default: "
                              f"{NODED_INPUT_DIR}/<network>_noded.geojson). "
                              "Only valid with a single --network.")
    parser.add_argument("--output", type=Path, default=None,
                         help="Override output .gpkg path (default: "
                              f"{OUTPUT_DIR}/<network>_review.gpkg). "
                              "Only valid with a single --network.")
    args = parser.parse_args()

    if args.network is None:
        if args.bridged_pkl or args.noded_geojson or args.output:
            raise SystemExit(
                "--bridged-pkl/--noded-geojson/--output require a single "
                "--network (highway or railway) -- they can't be combined "
                "with the default 'run both networks' mode."
            )
        networks = sorted(NETWORK_FILENAMES)  # ["highway", "railway"]
    else:
        networks = [args.network]

    print(f"Running Step 1.4.5 (prep) export for: {', '.join(networks)}")

    ran = []
    skipped = []

    for i, network in enumerate(networks):
        if len(networks) > 1:
            print(f"\n{'#' * 70}\n# NETWORK {i + 1}/{len(networks)}: {network}\n{'#' * 70}")

        graph_pkl = args.bridged_pkl or (DENOISED_INPUT_DIR / f"{network}_denoised.pkl")
        noded_geojson = args.noded_geojson or (NODED_INPUT_DIR / NETWORK_FILENAMES[network])
        out_path = args.output or (OUTPUT_DIR / f"{network}_review.gpkg")

        if len(networks) > 1 and not graph_pkl.exists():
            # In multi-network default mode, a missing network's upstream
            # file (e.g. railway not yet run through Step 1.5) shouldn't
            # abort the whole run -- skip it loudly and keep going.
            print(f"  SKIPPING {network}: graph pkl not found at {graph_pkl}")
            skipped.append(network)
            continue

        export_graph(graph_pkl, noded_geojson, out_path)
        ran.append(network)

    print()
    print("=" * 70)
    print("ALL NETWORKS COMPLETE")
    print("=" * 70)
    print(f"Ran successfully: {ran or '(none)'}")
    if skipped:
        print(f"Skipped (input file not found): {skipped}")
    if not ran:
        raise SystemExit(
            "\nNo network was actually processed -- every requested graph "
            "pickle was missing. Run 2_4_remove_noise.py's prepare/apply "
            "cycle first, or pass --bridged-pkl directly."
        )


if __name__ == "__main__":
    main()