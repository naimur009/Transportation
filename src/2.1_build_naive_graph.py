"""
naive_graph_diagnostic.py

DIAGNOSTIC ONLY -- Step 1.2 of the "build a connected graph" pipeline.

Builds a "naive" graph straight from raw, unrounded line endpoints so you can
see, with your own numbers, how fragmented a cleaned road/rail line dataset
is BEFORE any touching/crossing/gap fixes are applied.

This graph is not used anywhere else in the pipeline. It gets thrown away
right after these numbers (and optionally a plot) are produced. A LOW
component count here is not success -- it means this diagnostic isn't doing
its job (see the "exactly 1 component" handling below).

DATASET NOTE (updated to match 1_cleanDataset.py, Illinois test run):
-----------------------------------------------------------------------
1_cleanDataset.py now writes its Phase 1 outputs as:
    data/cleaned/highway_clean.geojson
    data/cleaned/railway_clean.geojson
    data/cleaned/county_centroids_clean.geojson   (not used by this script)

This is a rename from the earlier strahnet_clean.geojson /
stracnet_clean.geojson convention. This script has been updated to match:
the two network labels are now "highway" and "railway", and the default
filenames point at highway_clean.geojson / railway_clean.geojson. If your
1_cleanDataset.py's OUTPUT_FILENAMES dict changes again, update
NETWORK_FILENAMES below to match -- keep the two in sync by hand, there is
no automatic link between the two scripts.

Usage:
    python naive_graph_diagnostic.py [input_geojson] [--network {highway,railway}]
                                      [--data-dir DIR] [--output-dir DIR] [--no-plot]

BOTH NETWORKS BY DEFAULT: if you don't pass a positional input_geojson AND you
don't pass --network, this script runs the full diagnostic for BOTH
highway_clean.geojson and railway_clean.geojson in one go (one after the
other), each getting its own printed summary and its own fragmentation PNG
(naive_fragmentation_highway.png / naive_fragmentation_railway.png). A
missing/unreadable file for one network prints a clear warning and does NOT
stop the other network's run.

Pass --network to restrict the run to just one network, or pass a positional
input_geojson to point at an arbitrary file directly (in which case only that
one file is processed, labeled by whatever --network you also passed, or
"highway" if you didn't).

Examples:
    # NEW DEFAULT: runs both highway and railway automatically
    python src/2.1_build_naive_graph.py

    # restrict to just one network
    python src/2.1_build_naive_graph.py --network highway
    python src/2.1_build_naive_graph.py --network railway

    # or point at any file directly, overriding the defaults above
    python src/2.1_build_naive_graph.py data/cleaned/highway_clean.geojson
"""

import argparse
import os
import time

import geopandas as gpd
import networkx as nx

# Defaults matching this project's layout: Transportation/data/cleaned/*.geojson
# These filenames match OUTPUT_FILENAMES in 1_cleanDataset.py -- update both
# together if that script's output naming changes again.
DEFAULT_DATA_DIR = "data/cleaned"
NETWORK_FILENAMES = {
    "highway": "highway_clean.geojson",
    "railway": "railway_clean.geojson",
}


def _progress_iter(iterable, total, desc, min_interval=0.2):
    """
    Lightweight progress reporter with no required dependency.

    Uses tqdm if it's installed (nicer bar); otherwise falls back to a plain
    in-place percentage counter that updates at most every `min_interval`
    seconds, so it's cheap even over thousands/millions of iterations.
    """
    try:
        from tqdm import tqdm
        yield from tqdm(iterable, total=total, desc=desc, unit="it")
        return
    except ImportError:
        pass

    last_print = 0.0
    for i, item in enumerate(iterable, 1):
        yield item
        now = time.monotonic()
        if now - last_print >= min_interval or i == total:
            pct = 100 * i / total if total else 100
            print(f"\r{desc}: {i:,}/{total:,} ({pct:5.1f}%)", end="", flush=True)
            last_print = now
    print()  # move off the in-place progress line when done


def load_and_validate(path: str) -> gpd.GeoDataFrame:
    """
    Load the GeoJSON and confirm CRS / geometry types before anything else.

    Does NOT modify the data -- this is purely a sanity check so you don't
    build a naive graph on top of an already-broken assumption.
    """
    gdf = gpd.read_file(path)

    print(f"Loaded {len(gdf)} rows from {path}")
    print(f"CRS: {gdf.crs}")
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        print(
            "  WARNING: expected EPSG:4326 (plain lat/lon degrees). Every step in "
            "this phase assumes lat/lon input -- reproject or investigate before "
            "trusting the numbers below."
        )

    geom_counts = gdf.geometry.geom_type.value_counts()
    print("Geometry types:")
    print(geom_counts.to_string())

    bad_types = set(geom_counts.index) - {"LineString", "MultiLineString"}
    if bad_types:
        print(
            f"  WARNING: found unexpected geometry type(s) {sorted(bad_types)}. "
            "That's a Phase 3 (cleaning) problem, not something this diagnostic "
            "fixes -- go check the upstream output before trusting the counts below."
        )

    return gdf


def build_naive_graph(gdf: gpd.GeoDataFrame) -> nx.Graph:
    """
    Build the naive graph: one edge per line, using that line's two endpoint
    coordinates EXACTLY as they appear in the file (no rounding, no snapping,
    no reprojecting) as the two node IDs.

    Diagnostic only -- the returned graph is never merged into, or reused by,
    any later step in the pipeline.
    """
    exploded = gdf.explode(index_parts=False).reset_index(drop=True)
    total = len(exploded)
    print(f"Building naive graph from {total:,} exploded line rows...")

    G = nx.Graph()
    skipped = 0
    # itertuples is materially faster than iterrows over 90k+ rows
    geoms = _progress_iter(exploded.geometry, total, desc="Adding edges")
    for geom in geoms:
        if geom is None or geom.geom_type != "LineString":
            skipped += 1
            continue
        coords = list(geom.coords)
        if len(coords) < 2:
            skipped += 1
            continue
        start = coords[0]   # raw, unrounded (x, y) tuple -- used as node ID
        end = coords[-1]    # raw, unrounded (x, y) tuple -- used as node ID
        G.add_edge(start, end)

    if skipped:
        print(
            f"  NOTE: skipped {skipped} row(s) with missing/invalid/non-LineString "
            "geometry after exploding."
        )

    return G


def summarize(G: nx.Graph, expected_min_components: int = 100) -> int:
    """Print the diagnostic summary and return the component count."""
    n_components = nx.number_connected_components(G)

    print()
    print("=" * 60)
    print("NAIVE GRAPH DIAGNOSTIC -- Step 1.2 (diagnostic only, not for reuse)")
    print("=" * 60)
    print(f"Nodes:                 {G.number_of_nodes():,}")
    print(f"Edges:                 {G.number_of_edges():,}")
    print(f"Connected components:  {n_components:,}")
    print("=" * 60)

    if n_components == 1:
        print(
            "RESULT: exactly 1 component.\n"
            "That is NOT the expected outcome for a naive, unrounded conversion --\n"
            "a raw government line file should fragment into hundreds of pieces\n"
            "(order of magnitude: one published reference case saw 700+ components\n"
            "from ~92,800 raw linestrings). Getting 1 here almost certainly means the\n"
            "input file was already snapped, rounded, or otherwise pre-processed\n"
            "upstream -- which means this diagnostic isn't testing what it's supposed\n"
            "to test. Do not treat this as a success. Go check the Phase 3 / cleaning\n"
            "output that produced this file."
        )
    elif n_components < expected_min_components:
        print(
            f"RESULT: {n_components} components. That's fewer than the rough hundreds-\n"
            "order ballpark you'd expect from a raw, unrounded conversion of a large\n"
            "national dataset. Not automatically wrong (a small dataset naturally has\n"
            "fewer components), but treat a suspiciously low number with the same\n"
            "suspicion as the 1-component case -- double-check nothing upstream already\n"
            "rounded or snapped coordinates."
        )
    else:
        print(
            f"RESULT: {n_components} components -- heavily fragmented, as expected for\n"
            "a naive unrounded conversion. This is the baseline number Step 1.3\n"
            "(touching/crossing fix) and Step 1.4 (gap fix) should dramatically reduce."
        )

    return n_components


def plot_components(G: nx.Graph, output_path: str) -> None:
    """
    Plot each connected component in a distinct color and save as a PNG.

    Renders everything with ONE matplotlib LineCollection instead of calling
    geopandas' .plot() once per component. For datasets with hundreds/thousands
    of components (e.g. a raw HIGHWAY file), per-component .plot() calls each
    carry fixed overhead that adds up to a very long, silent hang -- a single
    bulk LineCollection avoids that and renders in seconds instead.
    """
    import matplotlib
    matplotlib.use("Agg")  # headless-safe backend
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    components = list(nx.connected_components(G))
    n_components = len(components)
    n_edges = G.number_of_edges()
    print(f"Building fragmentation plot: {n_components:,} components, "
          f"{n_edges:,} edges total...")

    cmap = plt.cm.tab20
    segments = []
    colors = []
    comp_iter = _progress_iter(components, n_components, desc="Preparing segments")
    for i, comp in enumerate(comp_iter):
        color = cmap(i % 20)
        for u, v in G.subgraph(comp).edges():
            # Node IDs may be 3D (x, y, z) tuples when the source geometry is
            # LineString Z (e.g. the HWY2025 highway layer). LineCollection
            # only accepts 2D (x, y) vertices, so drop any Z component here.
            # This affects plotting only -- graph node identity elsewhere
            # (including the connected-component count above) still uses
            # the full raw coordinate tuple, Z included.
            segments.append((u[:2], v[:2]))
            colors.append(color)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.add_collection(LineCollection(segments, colors=colors, linewidths=1))
    ax.autoscale()
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"Naive graph -- {n_components:,} disconnected pieces")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    print("Rendering and saving figure...")
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved fragmentation plot to {output_path}")


def run_one_network(input_path: str, output_dir: str, no_plot: bool, label: str) -> bool:
    """
    Run the full naive-graph diagnostic (load -> build -> summarize -> plot)
    for a single input file.

    `label` (e.g. "highway"/"railway") is used only to:
      - print a clear section banner so multi-network output isn't ambiguous
        about which numbers belong to which network, and
      - name the fragmentation PNG so a two-network run doesn't have the
        second network silently overwrite the first's plot.

    Returns True on success, False if the input file was missing (so the
    caller can skip it and continue with any other requested network
    instead of aborting the whole run).
    """
    print()
    print("#" * 70)
    print(f"# NETWORK: {label}")
    print("#" * 70)

    if not os.path.exists(input_path):
        print(
            f"  WARNING: input file not found, skipping this network:\n"
            f"    {input_path}\n"
            f"  (If this network genuinely isn't part of your dataset, this "
            f"is expected -- otherwise pass a path directly or check "
            f"--data-dir.)"
        )
        return False

    gdf = load_and_validate(input_path)
    G_naive = build_naive_graph(gdf)
    summarize(G_naive)

    if not no_plot:
        plot_path = os.path.join(output_dir, f"naive_fragmentation_{label}.png")
        plot_components(G_naive, plot_path)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic only: build a naive graph from raw line endpoints "
                     "and report how fragmented the input dataset is. Runs BOTH "
                     "highway and railway automatically unless --network or a "
                     "positional input_geojson restricts it to one."
    )
    parser.add_argument(
        "input_geojson",
        nargs="?",
        default=None,
        help=(
            "Path to a specific cleaned line GeoJSON. Optional -- if omitted, "
            f"BOTH {DEFAULT_DATA_DIR}/highway_clean.geojson and "
            f"{DEFAULT_DATA_DIR}/railway_clean.geojson are processed "
            "automatically (unless --network restricts this to one). If you "
            "pass this, only that one file is processed."
        ),
    )
    parser.add_argument(
        "--network",
        choices=sorted(NETWORK_FILENAMES),
        default=None,
        help=(
            "Restrict the run to just this one network. If omitted (and no "
            "positional input_geojson is given), BOTH highway and railway "
            "are processed automatically."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"Folder to look in for <network>_clean.geojson files (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/diagnostics",
        help="Where to save the optional fragmentation plot(s) (default: outputs/diagnostics)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating the fragmentation PNG(s)",
    )
    args = parser.parse_args()

    if args.input_geojson:
        # Explicit file given -- process only that one, labeled by whatever
        # --network was also passed (defaulting to "highway" if not).
        networks_to_run = [(args.network or "highway", args.input_geojson)]
    elif args.network:
        # Restricted to one named network via its default path.
        networks_to_run = [
            (args.network, os.path.join(args.data_dir, NETWORK_FILENAMES[args.network]))
        ]
    else:
        # NEW DEFAULT: both networks, in a fixed/deterministic order.
        networks_to_run = [
            (network, os.path.join(args.data_dir, filename))
            for network, filename in sorted(NETWORK_FILENAMES.items())
        ]

    print(
        f"Running naive-graph diagnostic for: "
        f"{', '.join(label for label, _ in networks_to_run)}"
    )

    results = {}
    for label, input_path in networks_to_run:
        results[label] = run_one_network(
            input_path=input_path,
            output_dir=args.output_dir,
            no_plot=args.no_plot,
            label=label,
        )

    ran = [label for label, ok in results.items() if ok]
    skipped = [label for label, ok in results.items() if not ok]

    print()
    print("=" * 70)
    print("ALL NETWORKS COMPLETE")
    print("=" * 70)
    print(f"Ran successfully: {ran or '(none)'}")
    if skipped:
        print(f"Skipped (input file not found): {skipped}")
    if not ran:
        raise SystemExit(
            "\nNo network was actually processed -- every requested input file "
            "was missing. Check --data-dir, or pass a path directly, e.g.:\n"
            f"  python {os.path.basename(__file__)} data/cleaned/highway_clean.geojson"
        )

    print(
        "\nDone. This graph is diagnostic-only -- nothing here is merged, rounded, "
        "or carried forward. Proceed to Step 1.3 (fix touching & crossing) next."
    )


if __name__ == "__main__":
    main()