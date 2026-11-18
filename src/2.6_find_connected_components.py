#!/usr/bin/env python3

"""
Step 1.6 -- Find Connected Components

INPUT
-----
QGIS-reviewed GPKG (this is Step 1.4.5's / 2_5_convert_to_gpkg.py's own
output -- path and filename convention taken directly from that script's
`OUTPUT_DIR` and `out_path = OUTPUT_DIR / f"{network}_review.gpkg"`, not
reconstructed independently):

    data/graph/qgis_review/<network>_review.gpkg

    e.g. data/graph/qgis_review/highway_review.gpkg
         data/graph/qgis_review/railway_review.gpkg

OUTPUT
------
    data/graph/connected_components/
        <network>_connected_components.gpkg
        <network>_connected_components.geojson

PURPOSE
-------
Only prepare connected-component data.

This script does NOT:
    - make noise decisions
    - create a decision field
    - create a notes field
    - remove components
    - apply any manual review

Each connected component becomes one MultiLineString feature.

DATASET NOTE (updated to match the Illinois test run's naming convention):
-----------------------------------------------------------------------
Earlier drafts of this script were written against the old
strahnet/stracnet naming, pointing at
data/graph/qgis_review/{strahnet,stracnet}_review.gpkg. The pipeline has
since moved to "highway"/"railway" labels throughout (see
1_cleanDataset.py's OUTPUT_FILENAMES, 2_2_fix_toucing_and_crossing.py's
NODED_OUTPUT_DIR, 2_3_gap_repair.py's BRIDGED_OUTPUT_DIR,
2_4_remove_noise.py's REVIEW_DIR / DEFAULT_OUTPUT_PICKLE, and
2_5_convert_to_gpkg.py's OUTPUT_DIR / NETWORK_FILENAMES). This script has
been updated to match:

  * DEFAULT_INPUT_GPKG now maps "highway"/"railway" -> their respective
    <network>_review.gpkg paths under data/graph/qgis_review/, matching
    2_5_convert_to_gpkg.py's OUTPUT_DIR (`data/graph/qgis_review`) and its
    output filename pattern (`f"{network}_review.gpkg"`) exactly -- not a
    separately-guessed path.
  * --network now accepts {"highway", "railway"} instead of
    {"strahnet", "stracnet"}.
  * Following the "both networks by default" convention already
    established in every other script in this pipeline
    (2_1_build_naive_graph.py, 2_2_fix_toucing_and_crossing.py,
    2_3_gap_repair.py, 2_4_remove_noise.py, 2_5_convert_to_gpkg.py),
    --network is now OPTIONAL: omitting it runs both highway and railway
    automatically, one after the other, each in its own banner section. A
    missing input GPKG for one network prints a clear warning and skips
    just that network rather than aborting the whole run. Pass --network
    to restrict the run to just one.

If 2_5_convert_to_gpkg.py's output naming changes again, update
DEFAULT_INPUT_GPKG below to match -- keep them in sync by hand, there is
no automatic link between the scripts.

Usage
-----
    # NEW DEFAULT: runs both highway and railway automatically
    python src/2_6_find_connected_components.py

    # restrict to just one network
    python src/2_6_find_connected_components.py --network highway
    python src/2_6_find_connected_components.py --network railway

    # or point at a specific GPKG directly (single-network mode, requires
    # --network to label the run and name the output files)
    python src/2_6_find_connected_components.py --network highway \\
        --input data/graph/qgis_review/highway_review.gpkg
"""


from __future__ import annotations

import argparse
from pathlib import Path
from typing import FrozenSet, List, Tuple

import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, MultiLineString


# ============================================================
# DEFAULT PATHS
# ============================================================
#
# Sourced directly from 2_5_convert_to_gpkg.py's own output convention:
#     OUTPUT_DIR = Path("data/graph/qgis_review")
#     out_path = args.output or (OUTPUT_DIR / f"{network}_review.gpkg")
# i.e. data/graph/qgis_review/highway_review.gpkg and
# data/graph/qgis_review/railway_review.gpkg -- NOT independently guessed.

QGIS_REVIEW_INPUT_DIR = Path("data/graph/qgis_review")  # 2_5_convert_to_gpkg.py's own OUTPUT_DIR

NETWORK_LABELS = ["highway", "railway"]

DEFAULT_INPUT_GPKG = {
    network: QGIS_REVIEW_INPUT_DIR / f"{network}_review.gpkg"
    for network in NETWORK_LABELS
}

OUTPUT_DIR = Path(
    "data/graph/connected_components"
)

# HARDCODED DEFAULT NETWORK -- same pattern as Steps 1.3/1.4/1.5/1.4.5.
# Only used in single-file mode when --input is given without --network.
HARDCODED_NETWORK = "highway"


Node = Tuple[float, float]


# ============================================================
# PRINT
# ============================================================

def print_banner(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# EDGE LENGTH
# ============================================================

def edge_length_km(
    G: nx.Graph,
    u: Node,
    v: Node,
    data: dict,
) -> float:

    # Prefer length already stored in GPKG.
    if "length_km" in data:
        try:
            return float(data["length_km"])
        except (TypeError, ValueError):
            pass

    # Otherwise calculate from coordinates.
    try:
        from geopy.distance import geodesic

        return geodesic(
            (u[1], u[0]),
            (v[1], v[0]),
        ).km

    except ImportError:

        dx = u[0] - v[0]
        dy = u[1] - v[1]

        return (
            (dx ** 2 + dy ** 2) ** 0.5
        ) * 111.0


# ============================================================
# BUILD GRAPH FROM GPKG
# ============================================================

def build_graph_from_gpkg(
    gdf: gpd.GeoDataFrame,
) -> nx.Graph:
    """
    Convert GPKG line features into a NetworkX graph.

    Each line:
        first coordinate  -> node
        last coordinate   -> node
        line               -> edge
    """

    G = nx.Graph()

    for _, row in gdf.iterrows():

        geom = row.geometry

        if geom is None or geom.is_empty:
            continue

        # ----------------------------------------------------
        # LineString
        # ----------------------------------------------------

        if geom.geom_type == "LineString":

            coords = list(geom.coords)

            if len(coords) < 2:
                continue

            u = (
                float(coords[0][0]),
                float(coords[0][1]),
            )

            v = (
                float(coords[-1][0]),
                float(coords[-1][1]),
            )

            attributes = row.drop(
                labels=["geometry"]
            ).to_dict()

            G.add_edge(
                u,
                v,
                **attributes,
            )

        # ----------------------------------------------------
        # MultiLineString
        # ----------------------------------------------------

        elif geom.geom_type == "MultiLineString":

            for line in geom.geoms:

                coords = list(line.coords)

                if len(coords) < 2:
                    continue

                u = (
                    float(coords[0][0]),
                    float(coords[0][1]),
                )

                v = (
                    float(coords[-1][0]),
                    float(coords[-1][1]),
                )

                attributes = row.drop(
                    labels=["geometry"]
                ).to_dict()

                G.add_edge(
                    u,
                    v,
                    **attributes,
                )

    return G


# ============================================================
# FIND COMPONENTS
# ============================================================

def compute_components_with_ids(
    G: nx.Graph,
) -> List[Tuple[str, FrozenSet[Node]]]:
    """
    Find connected components.

    Deterministic ordering:

        1. total length descending
        2. minimum node coordinate

    IDs:

        C0000
        C0001
        C0002
        ...
    """

    raw_components = list(
        nx.connected_components(G)
    )

    scored = []

    for component in raw_components:

        sub = G.subgraph(component)

        total_length = 0.0

        for u, v, data in sub.edges(data=True):

            total_length += edge_length_km(
                G,
                u,
                v,
                data,
            )

        min_node = min(component)

        scored.append(
            (
                total_length,
                min_node,
                frozenset(component),
            )
        )

    scored.sort(
        key=lambda x: (
            -x[0],
            x[1],
        )
    )

    return [
        (
            f"C{idx:04d}",
            component,
        )
        for idx, (
            _length,
            _min_node,
            component,
        ) in enumerate(scored)
    ]


# ============================================================
# COMPONENT STATISTICS
# ============================================================

def build_component_stats(
    G: nx.Graph,
    component: FrozenSet[Node],
) -> dict:

    sub = G.subgraph(component)

    n_bridge = sum(
        1
        for _, _, data in sub.edges(data=True)
        if data.get("is_gap_fill")
    )

    n_manual = sum(
        1
        for _, _, data in sub.edges(data=True)
        if data.get("is_manual")
    )

    length_km = sum(
        edge_length_km(
            G,
            u,
            v,
            data,
        )
        for u, v, data in sub.edges(data=True)
    )

    return {
        "n_nodes": sub.number_of_nodes(),
        "n_edges": sub.number_of_edges(),
        "n_bridge_edges": n_bridge,
        "n_manual_edges": n_manual,
        "length_km": round(
            length_km,
            3,
        ),
    }


# ============================================================
# COMPONENT GEOMETRY
# ============================================================

def build_component_geometry(
    G: nx.Graph,
    component: FrozenSet[Node],
) -> MultiLineString:

    sub = G.subgraph(component)

    lines = []

    for u, v in sub.edges():

        lines.append(
            LineString([
                (u[0], u[1]),
                (v[0], v[1]),
            ])
        )

    # Handle isolated node.
    if not lines:

        node = next(iter(component))

        lines.append(
            LineString([
                (node[0], node[1]),
                (node[0], node[1]),
            ])
        )

    return MultiLineString(lines)


# ============================================================
# PREPARE
# ============================================================

def prepare(
    network: str,
    input_gpkg: Path,
    output_gpkg: Path,
    output_geojson: Path,
) -> bool:
    """
    Run Step 1.6 end to end for a single network.

    Returns True on success. Returns False (without raising) if the input
    GPKG doesn't exist, so a multi-network run can skip this network and
    continue with the other one instead of aborting entirely.
    """

    print_banner(
        f"STEP 1.6 -- CONNECTED COMPONENT FINDING [{network}]"
    )

    print(f"Input GPKG:")
    print(f"  {input_gpkg}")

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not input_gpkg.exists():

        print(
            f"\n  WARNING: input GPKG not found, skipping this network:\n"
            f"    {input_gpkg}\n"
            f"  (Run 2_5_convert_to_gpkg.py for this network first -- its "
            f"OUTPUT_DIR/f'{{network}}_review.gpkg' is exactly what this "
            f"path points at -- or pass --input directly.)"
        )
        return False

    # --------------------------------------------------------
    # Read GPKG
    # --------------------------------------------------------

    print_banner("READING GPKG")

    gdf = gpd.read_file(
        input_gpkg
    )

    print(
        f"Features : {len(gdf):,}"
    )

    print(
        f"CRS      : {gdf.crs}"
    )

    print(
        "Geometry :"
    )

    print(
        gdf.geometry.geom_type.value_counts()
    )

    # --------------------------------------------------------
    # Ensure geographic coordinates
    # --------------------------------------------------------

    if gdf.crs is None:

        raise SystemExit(
            f"ERROR: Input GPKG has no CRS ({input_gpkg})."
        )

    # Graph node IDs are expected to be lon/lat.
    if gdf.crs.to_epsg() != 4326:

        print(
            "\nReprojecting input to EPSG:4326..."
        )

        gdf = gdf.to_crs(
            "EPSG:4326"
        )

    # --------------------------------------------------------
    # Build graph
    # --------------------------------------------------------

    print_banner(
        "BUILDING GRAPH"
    )

    G = build_graph_from_gpkg(
        gdf
    )

    print(
        f"Nodes : {G.number_of_nodes():,}"
    )

    print(
        f"Edges : {G.number_of_edges():,}"
    )

    # --------------------------------------------------------
    # Components
    # --------------------------------------------------------

    print_banner(
        "FINDING CONNECTED COMPONENTS"
    )

    components = compute_components_with_ids(
        G
    )

    print(
        f"Connected components : "
        f"{len(components):,}"
    )

    # --------------------------------------------------------
    # Prepare output features
    # --------------------------------------------------------

    features = []

    for component_id, component_nodes in components:

        stats = build_component_stats(
            G,
            component_nodes,
        )

        geometry = build_component_geometry(
            G,
            component_nodes,
        )

        features.append({
            "component_id": component_id,
            **stats,
            "geometry": geometry,
        })

    component_gdf = gpd.GeoDataFrame(
        features,
        geometry="geometry",
        crs="EPSG:4326",
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_gpkg.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # GeoJSON
    # --------------------------------------------------------

    print_banner(
        "SAVING GEOJSON"
    )

    component_gdf.to_file(
        output_geojson,
        driver="GeoJSON",
    )

    print(
        f"Saved:\n"
        f"  {output_geojson}"
    )

    # --------------------------------------------------------
    # GPKG
    # --------------------------------------------------------

    print_banner(
        "SAVING GPKG"
    )

    if output_gpkg.exists():
        output_gpkg.unlink()

    component_gdf.to_file(
        output_gpkg,
        layer="connected_components",
        driver="GPKG",
    )

    print(
        f"Saved:\n"
        f"  {output_gpkg}"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_banner(
        "TOP 10 COMPONENTS"
    )

    for _, row in component_gdf.head(10).iterrows():

        print(
            f"{row['component_id']}: "
            f"{row['n_nodes']:,} nodes | "
            f"{row['n_edges']:,} edges | "
            f"{row['length_km']:.3f} km"
        )

    print_banner(f"DONE [{network}]")

    return True


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Step 1.6: find connected components from a QGIS-reviewed "
            "GPKG (2_5_convert_to_gpkg.py's own output). Runs BOTH "
            "highway and railway automatically unless --network restricts "
            "it to one."
        )
    )

    parser.add_argument(
        "--network",
        choices=NETWORK_LABELS,
        default=None,
        help=(
            "Restrict the run to just this one network. If omitted, BOTH "
            "highway and railway are processed automatically (matching "
            "every other script in this pipeline)."
        ),
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Path to a specific input GPKG. Optional -- if omitted, "
            f"resolved from --network via {QGIS_REVIEW_INPUT_DIR}/"
            "<network>_review.gpkg (2_5_convert_to_gpkg.py's own output "
            "path). If you pass this without --network, it's treated as "
            f"single-file mode labeled {HARDCODED_NETWORK!r}."
        ),
    )

    args = parser.parse_args()

    if args.input is not None and args.network is None:
        # Single-file mode still needs a label for the output filenames --
        # fall back to HARDCODED_NETWORK rather than requiring the flag.
        args.network = HARDCODED_NETWORK

    return args


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    if args.network:
        networks_to_run = [args.network]
    else:
        # NEW DEFAULT: both networks, in a fixed/deterministic order --
        # matches 2_1 through 2_5's "both by default" convention.
        networks_to_run = list(NETWORK_LABELS)

    print(f"Running Step 1.6 for: {', '.join(networks_to_run)}")

    ran = []
    skipped = []

    for network in networks_to_run:

        print()
        print("#" * 70)
        print(f"# NETWORK: {network}")
        print("#" * 70)

        input_gpkg = (
            args.input
            if args.input is not None
            else DEFAULT_INPUT_GPKG[network]
        )

        output_gpkg = (
            OUTPUT_DIR
            / f"{network}_connected_components.gpkg"
        )

        output_geojson = (
            OUTPUT_DIR
            / f"{network}_connected_components.geojson"
        )

        ok = prepare(
            network=network,
            input_gpkg=input_gpkg,
            output_gpkg=output_gpkg,
            output_geojson=output_geojson,
        )

        (ran if ok else skipped).append(network)

    print()
    print_banner("ALL NETWORKS COMPLETE")
    print(f"Ran successfully: {ran or '(none)'}")
    if skipped:
        print(f"Skipped (input GPKG not found): {skipped}")
    if not ran:
        raise SystemExit(
            "\nNo network was actually processed -- every requested input "
            "GPKG was missing. Run 2_5_convert_to_gpkg.py first, or pass "
            "--input directly."
        )


if __name__ == "__main__":
    main()