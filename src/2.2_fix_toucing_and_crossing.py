#!/usr/bin/env python3
"""
Step 1.3 — Fix touching & crossing points in cleaned HIGHWAY/RAILWAY line data.

What this script does
----------------------
Takes the output of the upstream cleaning phase (a GeoJSON of LineString /
MultiLineString rows in EPSG:4326) and repairs two kinds of broken topology
that plain coordinate rounding cannot fix:

  * Touching (T-intersection): one line's *endpoint* lands in the *middle*
    of another line.
  * Crossing (X-intersection):  two lines' *middles* cross each other.

In both cases the break point is not a vertex of either line, so rounding
endpoints does nothing to it. The fix is to hand the full collection of
lines to `shapely.ops.unary_union`, which finds every such point and splits
both lines there, producing a fully "noded" line collection where every
topological connection is now a shared endpoint.

This script only performs Step 1.3. It stops after saving the noded
GeoDataFrame; building the preliminary graph and finding dangling nodes is
Step 1.4 and is explicitly out of scope here.

Usage
-----
    python 2_2_fix_touching_and_crossing.py
    python 2_2_fix_touching_and_crossing.py --network railway --data-dir data/cleaned
    python 2_2_fix_touching_and_crossing.py some/other/file.geojson --network highway

BOTH NETWORKS BY DEFAULT: if you don't pass a positional input_geojson AND
you don't pass --network, this script runs Step 1.3 for BOTH highway and
railway automatically, one after the other, each getting its own printed
banner/summary and its own output file
(graph/touching_crossing/highway_noded.geojson and
.../railway_noded.geojson). A missing/unreadable clean input for one
network prints a clear warning and skips just that network -- it does not
abort the other network's run. Pass --network to restrict the run to just
one network, or pass a positional input_geojson to point at a specific file
directly (single-file mode, labeled by whatever --network you also passed,
or HARDCODED_NETWORK if you didn't).

Input
-----
    data/cleaned/<network>_clean.geojson   (e.g. data/cleaned/highway_clean.geojson)

Output
------
    graph/touching_crossing/<network>_noded.geojson

NOTE ON PROVENANCE
------------------
This script matches the CLI/style conventions of the actual
`2.1_build_naive_graph.py` (now confirmed against its real source, not
reconstructed): a positional optional `input_geojson`, `--network
{highway,railway}`, `--data-dir` (default `data/cleaned`), `--output-dir`
(default `outputs/diagnostics`), a `NETWORK_FILENAMES` mapping of
`{"highway": "highway_clean.geojson", "railway": "railway_clean.geojson"}`,
a `_progress_iter` helper (tqdm if available, else a plain in-place
percent counter) for long loops, `#`-banner prints per network section,
a `load_and_validate()` function, and a guarded `main()` that runs BOTH
networks by default. These names/defaults are taken directly from
`2.1_build_naive_graph.py` and `1_cleanDataset.py`'s `OUTPUT_FILENAMES`
(highway_clean.geojson / railway_clean.geojson), replacing this project's
earlier strahnet/stracnet naming.
"""

from __future__ import annotations

import argparse
import math
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

import geopandas as gpd
import networkx as nx
import shapely
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

# ---------------------------------------------------------------------------
# Conventions reconstructed from 2.1_build_naive_graph.py's described shape.
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = Path("data/cleaned")            # where <network>_clean.geojson input lives
DEFAULT_OUTPUT_DIR = Path("outputs/diagnostics")   # diagnostic logs only (e.g. discarded-geometry log)
NODED_OUTPUT_DIR = Path("data/graph/touching_crossing") # where <network>_noded.geojson is saved

# NOTE: DEFAULT_DATA_DIR (input) and NODED_OUTPUT_DIR (output) are
# deliberately different folders: inputs stay in data/cleaned/ alongside
# the other *_clean.geojson files, while this step's noded output goes into
# graph/touching_crossing/ — the first stage of a dedicated graph/ tree that
# later steps (1.4 gap-bridging, 1.5 final weighted graph, etc.) are
# expected to add their own subfolders under.

NETWORK_FILENAMES = {
    "highway": "highway_clean.geojson",
    "railway": "railway_clean.geojson",
}

# ---------------------------------------------------------------------------
# HARDCODED DEFAULT NETWORK
# ---------------------------------------------------------------------------
# Edit this value directly when you want to switch which network this
# script runs against by default (e.g. "highway" or "railway"). This is
# used as --network's default, so you can just run the script with no
# flags at all. You can still override it from the command line with
# `--network <other>` any time without touching this file.
HARDCODED_NETWORK = "highway"


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def _print_banner(title: str) -> None:
    """Print a section banner, matching the section-header print style used
    throughout 2.1_build_naive_graph.py."""
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def _progress_iter(iterable: Iterable, total: int, label: str, every: int = 5000) -> Iterator:
    """Wrap an iterable with periodic progress printing for long-running loops.

    Mirrors the `_progress_iter` helper used in 2.1_build_naive_graph.py:
    prints a status line every `every` items (and always for the final item)
    so long loops over tens of thousands of rows are never silently
    unresponsive.

    Args:
        iterable: The iterable to wrap.
        total: Total number of items, used for the "i/total" print and to
            guarantee a final progress line.
        label: Short label identifying which loop this is, shown in the
            printed status line.
        every: Print a status line every this many items.

    Yields:
        Each item from `iterable`, unchanged.
    """
    start = time.monotonic()
    for i, item in enumerate(iterable, start=1):
        yield item
        if i % every == 0 or i == total:
            elapsed = time.monotonic() - start
            rate = i / elapsed if elapsed > 0 else 0.0
            print(f"    [{label}] {i:,}/{total:,} ({rate:,.0f}/s, {elapsed:,.1f}s elapsed)")


def _get_shapely_major_version() -> int:
    """Return the installed shapely major version number (e.g. 2 for '2.0.4').

    Falls back to assuming version 2 (and printing a warning) if the version
    string can't be parsed, rather than crashing on something this
    performance check treats as advisory, not load-bearing.
    """
    try:
        return int(shapely.__version__.split(".")[0])
    except (AttributeError, ValueError, IndexError):
        print("  WARNING: could not determine installed shapely version; assuming >= 2.0.")
        return 2


# ---------------------------------------------------------------------------
# Load + validate (reused/adapted from Step 1.2's load_and_validate())
# ---------------------------------------------------------------------------

def load_and_validate(path: Path) -> gpd.GeoDataFrame:
    """Load a GeoJSON line file and validate it against Step 1's assumptions.

    Reused/adapted from the Step 1.2 (2.1_build_naive_graph.py) validation
    logic: confirms the file exists, confirms the CRS is EPSG:4326, confirms
    every geometry is a LineString or MultiLineString, and prints a row
    count / geometry-type breakdown.

    Args:
        path: Path to the input GeoJSON file.

    Returns:
        The loaded GeoDataFrame, unmodified.

    Raises:
        FileNotFoundError: If `path` does not exist.
        ValueError: If the CRS is not EPSG:4326, or a geometry type other
            than LineString/MultiLineString is present.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
            f"Hint: run this script from the project root directory (the "
            f"directory that contains 'data/'), or pass --data-dir / the "
            f"input_geojson path explicitly."
        )

    gdf = gpd.read_file(path)

    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        raise ValueError(
            f"Expected CRS EPSG:4326, got {gdf.crs} for {path}. "
            f"Every step in this pipeline assumes lat/lon degrees input; if "
            f"this file is in a different CRS, that's a Phase 3 (upstream "
            f"cleaning) problem, not a Step 1.3 problem."
        )

    geom_types = set(gdf.geometry.geom_type.unique())
    allowed = {"LineString", "MultiLineString"}
    if not geom_types.issubset(allowed):
        bad = sorted(geom_types - allowed)
        raise ValueError(
            f"Unexpected geometry type(s) {bad} in {path}. Only "
            f"LineString/MultiLineString are valid input to this step; "
            f"anything else is a Phase 3 (upstream cleaning) problem, not "
            f"a Step 1.3 problem."
        )

    print(f"Loaded {len(gdf):,} rows from {path}")
    print(f"  CRS: {gdf.crs}")
    print(f"  Geometry types: {dict(gdf.geometry.geom_type.value_counts())}")
    return gdf


def explode_to_linestrings(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Explode MultiLineString rows into individual LineString rows.

    Matches Step 1.2's explode call exactly (`explode(index_parts=False)`
    followed by `reset_index(drop=True)`) so the resulting row count is
    directly comparable to Step 1.2's naive-graph edge count.

    Args:
        gdf: GeoDataFrame of LineString/MultiLineString geometries.

    Returns:
        A new GeoDataFrame where every row is a single LineString.
    """
    return gdf.explode(index_parts=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Normalization of unary_union's return value
# ---------------------------------------------------------------------------

def _normalize_to_linestrings(geom: BaseGeometry) -> Tuple[List, Counter]:
    """Flatten a `unary_union` result into a flat list of LineStrings.

    `unary_union` can hand back one of three shapes; this function handles
    all three explicitly so nothing crashes and nothing silently vanishes:

      * LineString        -> wrapped in a single-element list.
      * MultiLineString    -> expanded via `.geoms`.
      * GeometryCollection -> only LineString members are kept; every other
        member is counted by type and reported back to the caller so it can
        be logged. Nothing is dropped without being counted.

    Args:
        geom: The geometry returned by `shapely.ops.unary_union`.

    Returns:
        A tuple of (list of LineStrings, Counter of discarded geometry
        types -> count). The Counter is empty if nothing was discarded.

    Raises:
        ValueError: If `unary_union` returns something other than one of
            the three documented shapes (would indicate an unexpected
            shapely behavior change worth investigating, not silently
            working around).
    """
    discarded: Counter = Counter()

    if geom.geom_type == "LineString":
        return [geom], discarded

    if geom.geom_type == "MultiLineString":
        return list(geom.geoms), discarded

    if geom.geom_type == "GeometryCollection":
        lines = []
        for part in geom.geoms:
            if part.geom_type == "LineString":
                lines.append(part)
            else:
                discarded[part.geom_type] += 1
        if discarded:
            print(
                f"  WARNING: unary_union returned a GeometryCollection with "
                f"non-LineString parts that were discarded: {dict(discarded)}. "
                f"This typically means two input lines touch at a single "
                f"point that also surfaces as an isolated Point in the union "
                f"result (or a degenerate zero-length overlap)."
            )
        return lines, discarded

    raise ValueError(
        f"Unexpected top-level geometry type from unary_union: "
        f"{geom.geom_type!r}. Expected LineString, MultiLineString, or "
        f"GeometryCollection."
    )


# ---------------------------------------------------------------------------
# The union / noding step itself, with scale-aware performance handling
# ---------------------------------------------------------------------------

def _union_batch(geoms: List, label: str) -> BaseGeometry:
    """Run a single `unary_union` call over `geoms`, with start/elapsed timing
    printed around it so the process is never silently unresponsive."""
    print(f"  Running unary_union on {len(geoms):,} geometries ({label})...")
    start = time.monotonic()
    result = unary_union(geoms)
    elapsed = time.monotonic() - start
    print(f"  Done in {elapsed:,.1f}s.")
    return result


def node_lines(
    gdf: gpd.GeoDataFrame, batch_size: Optional[int] = None
) -> Tuple[List, Counter]:
    """Run `unary_union` over every geometry in `gdf` to node touching/crossing
    points, then normalize the result into a flat list of LineStrings.

    Performance note (Step 1.3 at ~97,000+ input rows):
        A single `unary_union` call is, on shapely >= 2.0, a vectorized call
        into GEOS that is already internally spatially indexed for noding —
        this is the fastest reliable path for the vast majority of
        real-world runs at this scale, and is what the governing spec's own
        example code does. On shapely < 2.0 (the older cascaded-union
        implementation), the same call can be substantially slower and more
        memory-hungry at this row count, so this function detects the
        installed version and prints an explicit warning in that case rather
        than silently assuming performance is fine.

        A `batch_size` escape hatch is provided for constrained
        environments: geometries are unioned in chunks (each chunk's
        progress is tracked with `_progress_iter`), then the batch results
        are unioned together in one final pass. This is a
        memory/checkpointing optimization, not an algorithmic complexity
        win — the final union-of-batches step is still required for
        correctness across batch boundaries, and batching is off by default
        because a single call is both correct and, on shapely >= 2.0, fast
        enough for the common case.

    Args:
        gdf: Exploded (single-part) LineString GeoDataFrame.
        batch_size: If set and `len(gdf) > batch_size`, union in chunks of
            this size before a final union-of-batches pass. If None
            (default), union everything in a single call.

    Returns:
        A tuple of (flat list of LineStrings, Counter of any discarded
        non-LineString geometry types, keyed by a short reason string).
    """
    raw_geoms = gdf.geometry.tolist()
    total = len(raw_geoms)

    geoms: List = []
    skipped_null_or_empty = 0
    for geom in _progress_iter(raw_geoms, total=total, label="pre-flight geometry scan"):
        if geom is None or geom.is_empty:
            skipped_null_or_empty += 1
            continue
        geoms.append(geom)

    discarded: Counter = Counter()
    if skipped_null_or_empty:
        discarded["null_or_empty_input"] = skipped_null_or_empty
        print(
            f"  WARNING: skipped {skipped_null_or_empty:,} null/empty "
            f"geometries before union. These would otherwise risk a cryptic "
            f"GEOS-level error inside a single large unary_union call over "
            f"{total:,} geometries."
        )

    shapely_major = _get_shapely_major_version()
    print(f"  Detected shapely major version: {shapely_major}")
    if shapely_major < 2:
        print(
            "  WARNING: shapely < 2.0 detected. unary_union over ~97k+ line "
            "geometries is known to be substantially slower and more "
            "memory-hungry on Shapely 1.x's cascaded-union implementation "
            "than on Shapely 2.x's vectorized GEOS calls. Strongly consider "
            "upgrading shapely, or pass --batch-size to checkpoint progress, "
            "before running this against full-scale data."
        )

    if batch_size and len(geoms) > batch_size:
        n_batches = math.ceil(len(geoms) / batch_size)
        print(f"  Batching union: {len(geoms):,} geometries into {n_batches} "
              f"batch(es) of up to {batch_size:,}")
        batch_results = []
        for bi in _progress_iter(range(n_batches), total=n_batches, label="union batches", every=1):
            batch = geoms[bi * batch_size:(bi + 1) * batch_size]
            batch_results.append(_union_batch(batch, label=f"batch {bi + 1}/{n_batches}"))
        noded = _union_batch(batch_results, label="final union-of-batches (required for correctness across batch boundaries)")
    else:
        noded = _union_batch(geoms, label="single call")

    lines, post_union_discarded = _normalize_to_linestrings(noded)
    discarded.update(post_union_discarded)
    return lines, discarded


def fix_touching_and_crossing(
    gdf: gpd.GeoDataFrame, batch_size: Optional[int] = None
) -> Tuple[gpd.GeoDataFrame, Counter, int]:
    """Full Step 1.3 pipeline: explode -> unary_union -> normalize -> rebuild.

    Args:
        gdf: Input GeoDataFrame (LineString/MultiLineString, any part-count).
        batch_size: Passed through to `node_lines`; see its docstring.

    Returns:
        A tuple of (noded GeoDataFrame with the original CRS, Counter of
        discarded geometry counts, number of exploded lines fed into the
        union call — i.e. the "before" count for the before/after summary).
    """
    exploded = explode_to_linestrings(gdf)
    lines, discarded = node_lines(exploded, batch_size=batch_size)
    noded_gdf = gpd.GeoDataFrame(geometry=lines, crs=gdf.crs)
    return noded_gdf, discarded, len(exploded)


def idempotency_check(noded_gdf: gpd.GeoDataFrame, batch_size: Optional[int]) -> None:
    """Re-run the noding function on its own output and confirm nothing changes.

    This is a correctness gate, not a diagnostic: if the line count changes
    on a second pass, unresolved touching/crossing points remain, and the
    pipeline must not proceed to Step 1.4 with that output.

    Args:
        noded_gdf: The already-noded GeoDataFrame from `fix_touching_and_crossing`.
        batch_size: Passed through to the re-run, for consistency with
            however the first pass was configured.

    Raises:
        RuntimeError: If the second pass produces a different line count
            than the first.
    """
    print("\nRunning idempotency check (re-noding the noded output)...")
    twice_gdf, _discarded_twice, _n_before_twice = fix_touching_and_crossing(
        noded_gdf, batch_size=batch_size
    )
    if len(twice_gdf) != len(noded_gdf):
        raise RuntimeError(
            f"IDEMPOTENCY CHECK FAILED: re-running the noding function on "
            f"its own output changed the line count "
            f"({len(noded_gdf):,} -> {len(twice_gdf):,}). This means "
            f"unresolved touching/crossing points remain in the supposedly "
            f"noded output. Refusing to continue — do not hand this output "
            f"to Step 1.4."
        )
    print(
        f"Idempotency check passed: {len(noded_gdf):,} lines -> "
        f"{len(twice_gdf):,} lines (unchanged)."
    )


# ---------------------------------------------------------------------------
# Connected-component report
# ---------------------------------------------------------------------------

def build_endpoint_graph(gdf: gpd.GeoDataFrame) -> nx.Graph:
    """Build a graph from raw (unrounded) line endpoints.

    This deliberately mirrors Step 1.2's naive-graph construction
    (`build_naive_graph` in `2.1_build_naive_graph.py`): each row's first
    and last coordinate become the two node IDs for one edge, with no
    rounding or snapping. Using the exact same construction here means the
    component counts reported by this step are directly comparable to Step
    1.2's naive-graph baseline -- any drop in component count between the
    two is attributable to Step 1.3's touching/crossing fix alone, not to a
    difference in how the graph was built.

    This graph is diagnostic only, same as Step 1.2's -- it is not reused
    by Step 1.4 or any later step.

    Args:
        gdf: A single-part (exploded) LineString GeoDataFrame.

    Returns:
        An undirected NetworkX graph keyed by raw (x, y[, z]) endpoint
        tuples.
    """
    G = nx.Graph()
    skipped = 0
    for geom in gdf.geometry:
        if geom is None or geom.geom_type != "LineString":
            skipped += 1
            continue
        coords = list(geom.coords)
        if len(coords) < 2:
            skipped += 1
            continue
        G.add_edge(coords[0], coords[-1])

    if skipped:
        print(
            f"  NOTE: skipped {skipped} row(s) with missing/invalid/"
            f"non-LineString geometry while building the component-count graph."
        )

    return G


def report_connected_components(
    gdf: gpd.GeoDataFrame, label: str
) -> Tuple[int, int, int]:
    """Print a connected-component summary for a line GeoDataFrame.

    Args:
        gdf: A single-part (exploded) LineString GeoDataFrame.
        label: Short description printed alongside the numbers (e.g.
            "BEFORE Step 1.3 (naive endpoints)" or
            "AFTER Step 1.3 (noded endpoints)").

    Returns:
        A tuple of (node count, edge count, connected-component count).
    """
    G = build_endpoint_graph(gdf)
    n_components = nx.number_connected_components(G)

    print(f"  [{label}]")
    print(f"    Nodes (unique raw endpoints): {G.number_of_nodes():,}")
    print(f"    Edges:                        {G.number_of_edges():,}")
    print(f"    Connected components:         {n_components:,}")

    return G.number_of_nodes(), G.number_of_edges(), n_components


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments, matching 2.1_build_naive_graph.py's
    described CLI shape.

    NOTE: `--network` defaults to None (not HARDCODED_NETWORK). When it's
    None AND no positional `input_geojson` was given, `main()` runs BOTH
    highway and railway automatically -- see the module docstring's "BOTH
    NETWORKS BY DEFAULT" section. Pass `--network` explicitly any time you
    want to restrict a run to just one network.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Step 1.3: fix touching & crossing points in cleaned "
            "HIGHWAY/RAILWAY line data via shapely.ops.unary_union. "
            "Runs BOTH highway and railway automatically unless --network "
            "or a positional input_geojson restricts it to one."
        )
    )
    parser.add_argument(
        "input_geojson",
        nargs="?",
        default=None,
        help=(
            "Path to a specific input GeoJSON. Optional -- if omitted, BOTH "
            "highway and railway are processed automatically via "
            "--data-dir/NETWORK_FILENAMES (unless --network restricts this "
            "to one). If you pass this, only that one file is processed."
        ),
    )
    parser.add_argument(
        "--network",
        choices=sorted(NETWORK_FILENAMES.keys()),
        default=None,
        help=(
            "Restrict the run to just this one network. If omitted (and no "
            "positional input_geojson is given), BOTH highway and railway "
            "are processed automatically. Also determines the output "
            "filename (graph/touching_crossing/<network>_noded.geojson) "
            "when a positional input_geojson is given without --network -- "
            f"in that case it falls back to HARDCODED_NETWORK ({HARDCODED_NETWORK!r})."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory holding <network>_clean.geojson input files (default: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            f"Directory for diagnostic artifacts, e.g. a discarded-geometry "
            f"log if anything gets dropped during normalization "
            f"(default: {DEFAULT_OUTPUT_DIR}). The primary noded GeoJSON "
            f"always goes to {NODED_OUTPUT_DIR}/, not this directory."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Optional: chunk the union into batches of this many "
            "geometries, then do a final union-of-batches pass. Off by "
            "default (single call), which is both correct and the fastest "
            "path on shapely>=2.0. Use this as an escape hatch on "
            "memory-constrained machines."
        ),
    )
    return parser.parse_args()


def resolve_input_path(network: str, args: argparse.Namespace) -> Path:
    """Resolve the input GeoJSON path for one network run.

    If a positional `input_geojson` was given, it takes precedence over
    `--data-dir` lookup (used only in single-file mode). Otherwise resolved
    from `--data-dir` + `NETWORK_FILENAMES[network]`.
    """
    if args.input_geojson:
        return Path(args.input_geojson)
    filename = NETWORK_FILENAMES[network]
    return args.data_dir / filename


def run_one_network(network: str, args: argparse.Namespace) -> bool:
    """Run Step 1.3 end to end for a single network.

    Returns True on success. Returns False (without raising) if the input
    file doesn't exist, so a multi-network run can skip this network and
    continue with the other one instead of aborting entirely.
    """
    input_path = resolve_input_path(network, args)

    print()
    print("#" * 70)
    print(f"# NETWORK: {network}")
    print("#" * 70)

    if not input_path.exists():
        print(
            f"  WARNING: input file not found, skipping this network:\n"
            f"    {input_path}\n"
            f"  (If this network genuinely isn't part of your dataset, this "
            f"is expected -- otherwise pass a path directly or check "
            f"--data-dir.)"
        )
        return False

    _print_banner("STEP 1.3 — Fix touching & crossing")
    print(f"Input:       {input_path}")
    print(f"Network:     {network}")
    print(f"Diagnostics: {args.output_dir}")
    out_preview = NODED_OUTPUT_DIR / f"{network}_noded.geojson"
    print(f"Will save:   {out_preview}")
    if NODED_OUTPUT_DIR.resolve() == input_path.resolve().parent:
        print(
            "             (same folder as the input — that's fine, the "
            "filenames differ so nothing gets overwritten)"
        )

    _print_banner("Loading and validating input")
    gdf = load_and_validate(input_path)

    _print_banner("Exploding + noding (unary_union)")
    noded_gdf, discarded, n_before = fix_touching_and_crossing(
        gdf, batch_size=args.batch_size
    )
    n_after = len(noded_gdf)

    _print_banner("Before/after summary")
    print(f"Lines before noding (post-explode): {n_before:,}")
    print(f"Lines after noding:                 {n_after:,}")
    if n_after >= n_before:
        print(
            "An increase is EXPECTED and CORRECT: lines get cut into more, "
            "shorter pieces wherever they touch or cross something else."
        )
    else:
        print(
            "WARNING: line count DECREASED after noding. This is unusual — "
            "investigate before trusting this output (it can happen if "
            "many exact-duplicate input lines collapsed together, but that "
            "should be confirmed against the source data, not assumed)."
        )

    if discarded:
        log_path = args.output_dir / f"{network}_discarded_geometry_log.txt"
        with open(log_path, "w") as f:
            f.write(
                "Geometry discarded during Step 1.3 normalization "
                "(none of this silently vanished — see counts below):\n"
            )
            for reason, count in discarded.items():
                f.write(f"  {reason}: {count}\n")
        print(f"Discarded-geometry log written to: {log_path}")
    else:
        print("No geometry was discarded during normalization.")

    _print_banner("Idempotency check")
    idempotency_check(noded_gdf, batch_size=args.batch_size)

    _print_banner("Saving output")
    out_path = NODED_OUTPUT_DIR / f"{network}_noded.geojson"
    noded_gdf.to_file(out_path, driver="GeoJSON")
    print(f"Saved noded output to: {out_path}")

    _print_banner("Connected component report")
    exploded = explode_to_linestrings(gdf)
    _, _, components_before = report_connected_components(
        exploded, label="BEFORE Step 1.3 (naive endpoints, pre-noding)"
    )
    _, _, components_after = report_connected_components(
        noded_gdf, label="AFTER Step 1.3 (noded endpoints)"
    )
    delta = components_before - components_after
    print(f"  Component count change: {components_before:,} -> {components_after:,} "
          f"({'-' if delta >= 0 else '+'}{abs(delta):,})")
    if components_after > 1:
        print(
            "  NOTE: >1 component remaining is EXPECTED here -- fixing touching/"
            "crossing points does not close real gaps between separate lines. "
            "Closing the remaining gaps to reach a single fully connected "
            "component is Step 1.4's job, not this step's."
        )
    elif components_after == 1:
        print(
            "  NOTE: exactly 1 component after Step 1.3 alone would be unusual "
            "for a large national dataset -- double check this wasn't already "
            "gap-repaired upstream before trusting this as a real result."
        )

    _print_banner("Step 1.3 complete")
    print(
        f"Next step: 1.4 — build a preliminary graph from '{out_path}' "
        f"(raw, unrounded endpoints as node IDs) and find dangling "
        f"(degree-1) nodes."
    )

    return True


def main() -> None:
    """Entry point.

    Runs BOTH highway and railway automatically unless a positional
    `input_geojson` or `--network` restricts the run to a single network
    (see the module docstring's "BOTH NETWORKS BY DEFAULT" section).
    """
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    NODED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.input_geojson:
        # Explicit file given -- single-file mode, labeled by --network if
        # given, else HARDCODED_NETWORK.
        networks_to_run = [args.network or HARDCODED_NETWORK]
    elif args.network:
        # Restricted to one named network via its default path.
        networks_to_run = [args.network]
    else:
        # NEW DEFAULT: both networks, in a fixed/deterministic order.
        networks_to_run = sorted(NETWORK_FILENAMES.keys())

    print(f"Running Step 1.3 for: {', '.join(networks_to_run)}")

    results = {network: run_one_network(network, args) for network in networks_to_run}

    ran = [n for n, ok in results.items() if ok]
    skipped = [n for n, ok in results.items() if not ok]

    print()
    print("=" * 70)
    print("ALL NETWORKS COMPLETE")
    print("=" * 70)
    print(f"Ran successfully: {ran or '(none)'}")
    if skipped:
        print(f"Skipped (input file not found): {skipped}")
    if not ran:
        raise SystemExit(
            "\nNo network was actually processed -- every requested input "
            "file was missing. Check --data-dir, or pass a path directly."
        )


if __name__ == "__main__":
    main()