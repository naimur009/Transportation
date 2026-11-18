#!/usr/bin/env python3
"""
Step 1.4.1-1.4.4 -- Preliminary graph, dangling nodes, gap threshold, and
iterative automatic gap bridging for cleaned/noded HIGHWAY/RAILWAY line
data.

What this script does
----------------------
Picks up where Step 1.3 (2_2_fix_toucing_and_crossing.py) leaves off. Takes
the noded GeoJSON it produced (every touching/crossing point already split)
and:

  1.4.1 Builds a preliminary `networkx.Graph` the same simple way as the
        Step 1.2 naive graph -- each line's two raw, UNROUNDED endpoint
        coordinates become that edge's two node IDs. Coordinate rounding is
        explicitly deferred to Step 1.5; this script never rounds node IDs.
  1.4.2 Finds every degree-1 ("dangling") node in that preliminary graph --
        a loose end where only one line touches, meaning the road just
        stops (a real dead end, or a data gap).
  1.4.3 Fixes a gap threshold theta = 400m (see GAP_THRESHOLD_METERS below
        for the paper's own justification), with an optional off-by-default
        sweep from 50m-1000m for anyone who wants to reproduce that curve.
  1.4.4 Repeatedly bridges dangling nodes that are within theta of their
        nearest other dangling node, re-checking after every pass (closing
        one gap can expose new dangling ends elsewhere) until a full pass
        adds zero new bridges, or a pass cap is hit.

Step 1.4.5 (manual review of larger/rarer gaps via connected-component
comparison, QGIS review, manual connector logging) is explicitly OUT OF
SCOPE for this script -- see `2_4_component_gap_review.py`. See the "what's
next" line at the end of a run.

PATCH (Step 1.4.5 prerequisite fix)
------------------------------------
`build_preliminary_graph` below now ALSO stores `length_km` (true length,
reprojected to EPSG:5070) and `source_line_id` (the row index in the noded
GeoDataFrame) on every edge, in addition to the `length_deg` it always
stored. `2_4_component_gap_review.py` needs a trustworthy per-edge length
in km to total up component lengths, and recovering that later by matching
a graph edge back to a source line via its (start, end) coordinate pair
alone is ambiguous for the ~40 pairs shared by two distinct source lines
(see "Duplicate endpoint pairs" below) -- exactly the case `MultiGraph` was
introduced to handle correctly. Computing `length_km` here, once, while the
real geometry is still in hand, avoids that ambiguity. Bridge edges added
later by `bridge_gaps_once` already carried `length_km` from the start, so
every edge in the final `<network>_bridged.pkl` -- original line or
auto-bridge -- now has one. If you have an existing `<network>_bridged.pkl`
that predates this patch, regenerate it by re-running this script; edges
built with the old code have no `length_km` and Step 1.4.5 will warn loudly
about undercounted component lengths if it finds any.

This script deliberately mirrors the console-output style, CLI shape, and
helper-function conventions already established in:
  - src/2_1_build_naive_graph.py           (Step 1.2 -- naive graph diagnostic)
  - src/2_2_fix_toucing_and_crossing.py    (Step 1.3 -- touching/crossing fix)
so that reading this script's output feels like a natural continuation of
those two, not a stylistically different tool. `_print_banner` is copied
verbatim from Step 1.3's script. `load_and_validate` is reused (imported),
not reimplemented, with one additional check layered on top (see
`load_and_validate_noded` below).

Usage
-----
    python 2_3_build_prelim_graph_and_bridge_gaps.py
    python 2_3_build_prelim_graph_and_bridge_gaps.py --network railway
    python 2_3_build_prelim_graph_and_bridge_gaps.py some/other/noded.geojson --network highway
    python 2_3_build_prelim_graph_and_bridge_gaps.py --theta-m 350 --max-passes 15
    python 2_3_build_prelim_graph_and_bridge_gaps.py --sweep-theta   # opt-in, off by default

BOTH NETWORKS BY DEFAULT: if you don't pass a positional input_geojson AND
you don't pass --network, this script runs Steps 1.4.1-1.4.4 for BOTH
highway and railway automatically, one after the other, each with its own
banner, its own naive-graph baseline comparison, its own bridging speed
(HIGHWAY=55mph, RAILWAY=49mph), and its own output pickle
(gap_bridging/highway_bridged.pkl / .../railway_bridged.pkl). A missing
noded input for one network prints a clear warning and skips just that
network -- it does not abort the other network's run. Pass --network to
restrict the run to just one network, or pass a positional input_geojson to
point at a specific file directly (single-file mode, labeled by whatever
--network you also passed, or HARDCODED_NETWORK if you didn't).

Input
-----
    data/graph/touching_crossing/<network>_noded.geojson
    (the exact path Step 1.3 itself writes to -- read straight out of that
    script's own NODED_OUTPUT_DIR constant, not guessed)

    For the naive-graph baseline comparison (1.4.1's required "explicitly
    compared against Step 1.2" output), this script also loads and runs
    Step 1.2's own build_naive_graph() against the *original* cleaned input,
    i.e. data/cleaned/<network>_clean.geojson -- it does not invent or
    reuse a stale logged number.

Output
------
    data/graph/gap_bridging/<network>_bridged.pkl

    A pickle of the post-bridging networkx.MultiGraph (NOT a plain Graph --
    see the "Duplicate endpoint pairs" note below), not a GeoJSON. Unlike
    Steps 1.2/1.3, this step's output is graph-shaped (it now carries
    synthetic bridge edges with no matching source geometry, plus edge
    attributes like `is_gap_fill`) rather than a pure line collection --
    GeoJSON has no clean native way to represent that, so a graph pickle is
    used instead. No existing intermediate-graph serialization precedent
    was found elsewhere in this codebase at the time this script was
    written; if one is later established, prefer it instead.

    Any downstream script that loads this pickle (Step 1.4.5's manual-gap
    review, in particular) should treat it as an `nx.MultiGraph`: iterating
    `G.edges(data=True)` may yield the same (u, v) pair more than once if
    two distinct source lines share both endpoints, and `G.degree()`
    correctly counts that as degree 2, not 1.

Duplicate endpoint pairs (why this script uses MultiGraph, not Graph)
----------------------------------------------------------------------
    A plain `nx.Graph` allows only one edge per (u, v) node pair -- a second
    `add_edge(u, v, ...)` call for the same pair silently overwrites the
    first, with no error or warning. In this dataset, ~40 distinct
    (start, end) coordinate pairs are each shared by two separate source
    lines (parallel carriageways, or two lines terminating at the same
    junction); under a plain `Graph`, one line's length/travel-time in each
    such pair was silently discarded, and about 60% of those pairs had
    meaningfully different lengths (not just redundant duplicates). This
    script builds `G_prelim` as an `nx.MultiGraph` specifically so every
    source line survives as its own edge. All graph operations used here
    (`degree()`, `has_edge()`, `add_edge()`, `number_connected_components()`,
    `is_connected()`) support MultiGraph without modification.
"""

from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Conventions inherited directly from Step 1.3 (2_2_fix_toucing_and_crossing.py)
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = Path("data/cleaned")                      # original <network>_clean.geojson lives here
NODED_INPUT_DIR = Path("data/graph/touching_crossing")       # Step 1.3's own NODED_OUTPUT_DIR -- this step's primary input
BRIDGED_OUTPUT_DIR = Path("data/graph/gap_bridging")          # this step's own output subfolder, sibling to touching_crossing/
DEFAULT_OUTPUT_DIR = Path("outputs/diagnostics")              # diagnostic logs only (e.g. dangling-node CSV, theta-sweep plot)

NETWORK_FILENAMES = {
    "highway": "highway_clean.geojson",
    "railway": "railway_clean.geojson",
}

# ---------------------------------------------------------------------------
# HARDCODED DEFAULT NETWORK -- same pattern as Step 1.3
# ---------------------------------------------------------------------------
# Edit this value directly when you want to switch which network this
# script runs against by default. Still overridable per-run with
# `--network <other>` without touching this file.
HARDCODED_NETWORK = "highway"

# ---------------------------------------------------------------------------
# 1.4.3 -- Gap threshold theta
# ---------------------------------------------------------------------------
# Per the governing paper (Linkov et al.): a parameter sweep of theta from
# 50m to 1000m (in 50m steps) was performed. Any theta smaller than 400m
# missed too many true gaps and left the graph overly fragmented; any theta
# larger than 400m started wrongly connecting road segments that are not
# actually connected in the real world. 400m was the smallest value that did
# not produce excess fragmentation, so it was adopted as the fixed threshold.
# (The paper also notes this rule alone cannot catch every gap -- one real
# missing segment was as long as 130km, found only by manual map
# cross-referencing. That larger, rarer-gap hunt is Step 1.4.5, explicitly
# out of scope for this script.)
GAP_THRESHOLD_METERS = 500

# Default bridging speed is network-aware, not a single hardcoded value:
# HIGHWAY uses 55 mph as its default (matches HIGHWAY_DEFAULT_SPEED_MPH in
# 1_cleanDataset.py); RAILWAY uses a flat 49 mph for every edge (including
# bridge edges), matching RAILWAY_SPEED_MPH in 1_cleanDataset.py and the
# flat speed the rest of the RAILWAY graph uses in Step 1.5.
DEFAULT_BRIDGE_SPEED_MPH = {
    "highway": 55,
    "railway": 49,
}

MAX_PASSES_DEFAULT = 10  # matches the spec's own example; hitting this without
                          # convergence prints an explicit warning, not a silent stop.

MILES_PER_KM = 1.60934


# ---------------------------------------------------------------------------
# Import Step 1.2 / Step 1.3's own modules rather than re-implementing them.
# Both files have digit-leading filenames (not valid Python identifiers), so
# they can't be imported with a plain `import`; importlib.util handles that.
# ---------------------------------------------------------------------------

def _import_module_from_path(path: Path, module_label: str):
    """Import a module directly from a file path via importlib.

    Digit-leading filenames (e.g. `2_1_build_naive_graph.py`) are not valid
    Python module identifiers and can't be reached with a normal `import`
    statement -- this loads them straight from disk instead.

    Args:
        path: Path to the .py file to import.
        module_label: Short human-readable label used only in error messages.

    Returns:
        The imported module object.

    Raises:
        ImportError: If the file exists but fails to import/execute.
    """
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # re-raise with context, don't swallow
        raise ImportError(f"Failed to import {module_label} from {path}: {exc}") from exc
    return module


def _find_sibling_module(
    name_patterns: List[str], required_attrs: List[str], module_label: str, explicit_path: Optional[Path] = None
) -> Path:
    """Locate a sibling script by glob pattern, not an exact hardcoded filename.

    This project's Step 1.2/1.3 scripts are not guaranteed to keep the exact
    filenames used during development (e.g. `2_1_build_naive_graph.py`) --
    they may be renamed (`2.1_naive_graph.py`, `naive_graph_diagnostic.py`,
    etc.) as the project evolves. Rather than fail on an exact-filename
    mismatch, this searches this script's own directory for files matching
    any of `name_patterns`, then further narrows candidates to only those
    that actually define every name in `required_attrs` (by importing and
    checking) -- so a same-numbered but unrelated file doesn't get picked by
    accident.

    Args:
        name_patterns: Glob patterns (relative to this script's directory)
            to try, in order, e.g. ["2[._]1*.py", "*naive*graph*.py"].
        required_attrs: Function/attribute names the target module must
            define (e.g. ["build_naive_graph", "load_and_validate"]) --
            used to confirm a glob match is actually the right file.
        module_label: Short human-readable label used only in messages.
        explicit_path: If given, skip searching entirely and use this path.

    Returns:
        Path to the matched module file.

    Raises:
        FileNotFoundError: If no candidate file matching any pattern (and
            defining all `required_attrs`) is found. The error message
            lists every .py file actually present in the directory, so the
            real filename is visible without needing to guess again.
    """
    here = Path(__file__).resolve().parent

    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(f"Explicit path for {module_label} does not exist: {explicit_path}")
        return explicit_path

    seen: List[Path] = []
    for pattern in name_patterns:
        for candidate in sorted(here.glob(pattern)):
            if candidate.resolve() == Path(__file__).resolve():
                continue  # never match this script itself
            if candidate in seen:
                continue
            seen.append(candidate)
            try:
                module = _import_module_from_path(candidate, module_label)
            except ImportError:
                continue  # not importable -- skip, don't crash the search
            if all(hasattr(module, attr) for attr in required_attrs):
                return candidate

    all_py_files = sorted(p.name for p in here.glob("*.py") if p.resolve() != Path(__file__).resolve())
    raise FileNotFoundError(
        f"Could not find {module_label} in {here} -- tried patterns {name_patterns}, "
        f"requiring it to define {required_attrs}.\n"
        f"  .py files actually present in that directory: {all_py_files or '(none)'}\n"
        f"  If it's one of the files above under a name this search didn't catch, "
        f"pass its path explicitly with --step1-2-path / --step1-3-path."
    )


def _load_step1_2_module(explicit_path: Optional[Path] = None):
    """Locate and import Step 1.2's naive-graph diagnostic module.

    Args:
        explicit_path: Optional explicit path (from --step1-2-path), bypassing search.

    Returns:
        The imported Step 1.2 module.
    """
    path = _find_sibling_module(
        name_patterns=["2[._]1*.py", "*naive*graph*.py"],
        required_attrs=["build_naive_graph", "load_and_validate"],
        module_label="Step 1.2 (naive graph)",
        explicit_path=explicit_path,
    )
    return _import_module_from_path(path, "Step 1.2 (naive graph)")


def _load_step1_3_module(explicit_path: Optional[Path] = None):
    """Locate and import Step 1.3's touching/crossing-fix module.

    Args:
        explicit_path: Optional explicit path (from --step1-3-path), bypassing search.

    Returns:
        The imported Step 1.3 module.
    """
    path = _find_sibling_module(
        name_patterns=["2[._]2*.py", "*touch*cross*.py", "*cross*touch*.py"],
        required_attrs=["load_and_validate", "fix_touching_and_crossing"],
        module_label="Step 1.3 (touching/crossing)",
        explicit_path=explicit_path,
    )
    return _import_module_from_path(path, "Step 1.3 (touching/crossing)")


# ---------------------------------------------------------------------------
# Small shared helpers -- copied verbatim from Step 1.3's script per the
# governing task prompt ("copy the helper function verbatim rather than
# reinventing an equivalent").
# ---------------------------------------------------------------------------

def _print_banner(title: str) -> None:
    """Print a section banner, matching the section-header print style used
    throughout 2_1_build_naive_graph.py and 2_2_fix_toucing_and_crossing.py."""
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def _progress_iter(iterable: Iterable, total: int, label: str, every: int = 5000) -> Iterator:
    """Wrap an iterable with periodic progress printing for long-running loops.

    Copied verbatim (aside from docstring wording) from Step 1.3's
    `_progress_iter`: prints a status line every `every` items (and always
    for the final item) so long loops over thousands of dangling nodes or
    tens of thousands of edges are never silently unresponsive.

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


# ---------------------------------------------------------------------------
# Input loading -- reuse Step 1.3's load_and_validate, with one additional
# check layered on top (the noded output should be 100% LineString).
# ---------------------------------------------------------------------------

def load_and_validate_noded(path: Path, step1_3_module) -> gpd.GeoDataFrame:
    """Load and validate Step 1.3's noded GeoJSON output.

    Reuses Step 1.3's own `load_and_validate` for the shared checks (file
    exists, CRS is EPSG:4326, geometry types are LineString/MultiLineString)
    and then layers one additional, stricter check on top: since Step 1.3's
    `unary_union` + explode pipeline should have produced pure LineStrings
    with no remaining MultiLineString rows, this function treats any
    MultiLineString found here as an explicit error rather than silently
    re-exploding it -- if that ever happens, something upstream in Step 1.3
    didn't fully finish its job, and that should be investigated there, not
    quietly patched over here.

    Args:
        path: Path to the noded GeoJSON (Step 1.3's output).
        step1_3_module: The imported Step 1.3 module, used for its
            `load_and_validate` function.

    Returns:
        The loaded, validated GeoDataFrame.

    Raises:
        FileNotFoundError: If `path` does not exist.
        ValueError: If the CRS is wrong, an unexpected geometry type is
            present, or (the additional check) any MultiLineString rows
            remain in what should be fully-noded output.
    """
    gdf = step1_3_module.load_and_validate(path)

    geom_types = set(gdf.geometry.geom_type.unique())
    if "MultiLineString" in geom_types:
        n_multi = int((gdf.geometry.geom_type == "MultiLineString").sum())
        raise ValueError(
            f"{n_multi:,} MultiLineString row(s) found in supposedly-noded "
            f"input {path}. Step 1.3's unary_union + normalize pipeline "
            f"should have produced pure LineStrings only -- this indicates "
            f"Step 1.3 did not fully finish (or this file isn't actually "
            f"its output). Re-run Step 1.3 rather than re-exploding here."
        )

    return gdf


# ---------------------------------------------------------------------------
# 1.4.1 -- Build the preliminary graph
# ---------------------------------------------------------------------------

def build_preliminary_graph(gdf: gpd.GeoDataFrame) -> Tuple[nx.MultiGraph, int]:
    """Build the Step 1.4.1 preliminary graph from noded lines.

    Uses the exact same simple conversion as Step 1.2's naive graph: each
    line's two raw, UNROUNDED endpoint coordinates (as they appear in the
    file) become that edge's two node IDs. Coordinate rounding is explicitly
    deferred to Step 1.5 and must NOT happen here.

    IMPORTANT -- MultiGraph, not Graph:
        A plain `nx.Graph` allows at most one edge between any given pair of
        node IDs; calling `add_edge(u, v, ...)` a second time for the same
        (u, v) does not add a second connection, it silently OVERWRITES the
        first edge's attributes. In this dataset, ~40 distinct (start, end)
        coordinate pairs are shared by two separate source lines each (e.g.
        parallel carriageways, or two lines that happen to terminate at the
        same junction) -- with a plain `Graph` those pairs collapsed to a
        single edge, and roughly 24 of them had meaningfully different
        lengths, meaning one real line's length/travel-time was silently
        discarded with no warning. `nx.MultiGraph` allows multiple distinct
        edges between the same node pair (each gets its own internal key),
        so every source line is preserved as its own edge, whether or not it
        happens to share endpoints with another line. All downstream
        functions in this script (`find_dangling_nodes`, `bridge_gaps_once`,
        `bridge_gaps_iterative`, `sweep_theta`) work unchanged against a
        MultiGraph -- `G.degree()`, `G.has_edge()`, `G.add_edge()`,
        `nx.number_connected_components()`, and `nx.is_connected()` all
        support MultiGraph natively.

        One real behavior change worth knowing: with a plain `Graph`, a
        node where two parallel lines meet the same neighbor was
        (incorrectly) reported as degree 1 ("dangling"), because the
        second line's edge just overwrote the first. With `MultiGraph`,
        that node correctly reports degree 2 (two distinct lines really do
        touch it) and is no longer treated as a dangling loose end. This is
        a correction, not a regression -- it reflects what the source data
        actually says.

    Each edge additionally stores `length_deg` (the geometry's raw length in
    degrees), per the governing spec's own example code -- unused until
    later steps, but kept present now so downstream code can rely on it
    always being there.

    PATCH (Step 1.4.5 prerequisite fix): each edge now ALSO stores
    `length_km` (the geometry's true length, reprojected to EPSG:5070) and
    `source_line_id` (that row's index in `gdf`) at build time. Step 1.4.5's
    component-review stage needs a real per-edge length in km to total up
    component lengths, and it must NOT be recovered later by matching a
    graph edge back to a source line via its (start, end) coordinate pair
    alone -- that lookup is ambiguous for the ~40 pairs described above that
    are shared by two distinct source lines, which is exactly why this
    function builds a `MultiGraph` in the first place. Computing `length_km`
    here, once, while the real `LineString` geometry is still in hand (via a
    single vectorized reproject-and-measure over the whole GeoDataFrame, not
    a per-row reprojection inside the loop), sidesteps that ambiguity
    entirely. Every edge added downstream by `bridge_gaps_once` already
    carries its own `length_km` too, so by the time this graph is pickled as
    `<network>_bridged.pkl`, every edge -- original line or auto-bridge --
    has a trustworthy `length_km` attribute.

    Args:
        gdf: The noded (Step 1.3 output) GeoDataFrame -- every row must
            already be a single LineString (see `load_and_validate_noded`).

    Returns:
        A tuple of (the preliminary networkx.MultiGraph, count of rows
        skipped for missing/invalid/degenerate geometry -- 0 if none).
    """
    total = len(gdf)
    print(f"Building preliminary graph from {total:,} noded line rows...")

    # Step 1.4.5 prerequisite fix: reproject the WHOLE GeoDataFrame to
    # EPSG:5070 once (vectorized), rather than reprojecting one geometry at
    # a time inside the loop below -- gives every row its true length_km
    # up front, cheaply, while row order/index still lines up with `gdf`.
    gdf_m = gdf.to_crs("EPSG:5070")
    lengths_km = (gdf_m.geometry.length / 1000.0).to_numpy()

    G = nx.MultiGraph()
    skipped = 0
    duplicate_pair_edges = 0  # rows sharing a (start, end) pair with an earlier row
    seen_pairs: Dict[Tuple[Tuple[float, float], Tuple[float, float]], int] = {}
    rows = zip(gdf.index, gdf.geometry, lengths_km)
    rows = _progress_iter(rows, total, label="Adding preliminary edges")
    for source_line_id, geom, length_km in rows:
        if geom is None or geom.geom_type != "LineString":
            skipped += 1
            continue
        coords = list(geom.coords)
        if len(coords) < 2:
            skipped += 1
            continue
        start = coords[0]   # raw, unrounded (x, y) -- used as node ID
        end = coords[-1]    # raw, unrounded (x, y) -- used as node ID
        G.add_edge(
            start, end,
            length_deg=geom.length,
            length_km=float(length_km),
            source_line_id=int(source_line_id),
        )

        pair_key = tuple(sorted([start, end]))
        seen_pairs[pair_key] = seen_pairs.get(pair_key, 0) + 1
        if seen_pairs[pair_key] > 1:
            duplicate_pair_edges += 1

    if skipped:
        print(
            f"  NOTE: skipped {skipped:,} row(s) with missing/invalid/non-LineString "
            f"or degenerate (<2 coordinate) geometry while building the preliminary graph."
        )

    if duplicate_pair_edges:
        n_pairs = sum(1 for c in seen_pairs.values() if c > 1)
        print(
            f"  NOTE: {n_pairs:,} distinct (start, end) coordinate pair(s) were shared "
            f"by more than one source line ({duplicate_pair_edges:,} extra edge(s) beyond "
            f"the first per pair). Using nx.MultiGraph here means all of them are kept as "
            f"separate parallel edges, not silently collapsed into one."
        )

    return G, skipped


def report_preliminary_vs_naive(G_prelim: nx.MultiGraph, naive_n_components: Optional[int]) -> int:
    """Print the preliminary graph's stats and compare against the naive baseline.

    This comparison is a required, visible part of Step 1.4.1's output (per
    the governing task prompt) -- not left for the reader to compute by hand.

    Args:
        G_prelim: The preliminary graph built by `build_preliminary_graph`.
        naive_n_components: The Step 1.2 naive graph's component count, or
            None if that baseline could not be obtained (in which case this
            function prints an explicit warning instead of fabricating a
            comparison).

    Returns:
        The preliminary graph's own connected-component count.
    """
    n_components = nx.number_connected_components(G_prelim)
    print(f"Preliminary graph: {G_prelim.number_of_nodes():,} nodes, "
          f"{G_prelim.number_of_edges():,} edges, {n_components:,} components")

    if naive_n_components is None:
        print(
            "  WARNING: no naive-graph (Step 1.2) baseline was available for "
            "comparison -- see the earlier warning about why. Reduction "
            "percentage cannot be reported without it."
        )
    else:
        if naive_n_components > 0:
            reduction = naive_n_components - n_components
            pct = 100 * reduction / naive_n_components
            print(
                f"  Compared to Step 1.2's naive graph ({naive_n_components:,} components): "
                f"the touching/crossing fix (Step 1.3) reduced fragmentation by "
                f"{reduction:,} components ({pct:.1f}% reduction), from "
                f"{naive_n_components:,} down to {n_components:,}."
            )
        else:
            print("  WARNING: naive-graph baseline was 0 components -- cannot compute a reduction percentage.")

    return n_components


# ---------------------------------------------------------------------------
# 1.4.2 -- Dangling (degree-1) nodes
# ---------------------------------------------------------------------------

def find_dangling_nodes(G: nx.Graph) -> List[Tuple[float, float]]:
    """Find every degree-1 ("dangling") node in a graph.

    A dangling node is a loose end -- a point where only one line touches,
    meaning the road just stops. No classification or filtering happens
    here (e.g. no attempt to guess "real dead end" vs. "gap"); that judgment
    belongs to Step 1.4.4's automatic bridging and Step 1.4.5's manual
    review, not to this counting step.

    Works correctly against both `nx.Graph` and `nx.MultiGraph` -- for a
    MultiGraph, `G.degree()` counts edge multiplicity, so a node touched by
    two parallel lines to the same neighbor correctly reports degree 2 (not
    dangling), rather than degree 1 as it would under a plain `Graph` that
    had silently collapsed those two lines into one edge.

    Args:
        G: The graph to scan (typically the Step 1.4.1 preliminary
            MultiGraph).

    Returns:
        A list of node IDs (raw, unrounded (lon, lat) tuples) with degree 1,
        in an order determined by the graph's own node iteration order.
    """
    return [n for n, deg in G.degree() if deg == 1]


# ---------------------------------------------------------------------------
# 1.4.4 -- Iterative gap bridging
# ---------------------------------------------------------------------------

def bridge_gaps_once(
    G: nx.MultiGraph, threshold_m: float, speed_mph: float
) -> Tuple[nx.MultiGraph, int, int]:
    """Run a single bridging pass: connect dangling nodes within `threshold_m`.

    Works unchanged against `nx.MultiGraph`: `G.has_edge(n1, n2)` returns
    True if *any* edge already exists between the pair (regardless of
    multiplicity), which is exactly the "don't bridge an already-connected
    pair again" check this function needs -- no MultiGraph-specific logic
    is required here.

    Reprojects every current dangling node from EPSG:4326 (degrees) to
    EPSG:5070 (meters) purely for distance comparison -- node identity
    itself stays in the original lon/lat coordinate space throughout this
    step; only the transient distance-comparison coordinates are in meters.

    For every dangling node, finds its nearest *other* dangling node with a
    `scipy.spatial.cKDTree` (k=2, taking the second-nearest result since the
    first is the point itself) rather than an O(n^2) all-pairs loop, since
    the graph can have thousands of dangling nodes.

    Args:
        G: The graph to bridge (mutated in place and also returned).
        threshold_m: Maximum distance (meters) for two dangling nodes to be
            considered a data gap worth bridging.
        speed_mph: Default speed used to compute the new bridge edge's
            `travel_time_min` attribute (network-aware -- see
            DEFAULT_BRIDGE_SPEED_MPH).

    Returns:
        A tuple of (the same graph `G`, number of bridge edges added this
        pass, number of dangling nodes skipped because fewer than 2 existed
        to compare -- always 0 unless the graph is nearly edgeless).
    """
    dangling = find_dangling_nodes(G)
    if len(dangling) < 2:
        return G, 0, len(dangling)

    pts = gpd.GeoSeries(
        gpd.points_from_xy([n[0] for n in dangling], [n[1] for n in dangling]),
        crs="EPSG:4326",
    ).to_crs("EPSG:5070")
    coords_m = np.array([(p.x, p.y) for p in pts])

    tree = cKDTree(coords_m)
    dists, idxs = tree.query(coords_m, k=2)  # k=2: nearest OTHER point, not itself

    added = 0
    for i, (dist, j) in enumerate(zip(dists[:, 1], idxs[:, 1])):
        if dist <= threshold_m:
            n1, n2 = dangling[i], dangling[j]
            if n1 == n2 or G.has_edge(n1, n2):
                continue
            length_km = dist / 1000.0
            travel_time_min = (length_km / MILES_PER_KM) / speed_mph * 60
            G.add_edge(
                n1, n2,
                length_km=length_km,
                travel_time_min=travel_time_min,
                is_gap_fill=True,
            )
            added += 1

    return G, added, 0


def bridge_gaps_iterative(
    G: nx.MultiGraph, threshold_m: float, speed_mph: float, max_passes: int = MAX_PASSES_DEFAULT
) -> Tuple[nx.MultiGraph, int, int]:
    """Repeat `bridge_gaps_once` until a full pass adds zero new bridges.

    Closing one gap can turn a 3-fragment chain into a 2-fragment one, which
    still has its own dangling ends that need checking again -- so a
    single-pass implementation is incorrect per the governing spec. Prints
    an explicit per-pass log line (pass number, edges added, running total)
    for every pass, matching the granular progress-reporting style already
    established in Steps 1.2 and 1.3.

    Args:
        G: The graph to bridge (mutated in place and also returned).
        threshold_m: Maximum distance (meters) for a bridge to be added --
            passed straight through to `bridge_gaps_once`.
        speed_mph: Default bridge-edge speed -- passed straight through to
            `bridge_gaps_once`.
        max_passes: Safety cap on the number of passes. Hitting this without
            reaching zero added edges prints an explicit, unmissable
            warning (not a silent truncation), since it likely indicates a
            bug rather than legitimately slow convergence.

    Returns:
        A tuple of (the same graph `G`, total bridge edges added across all
        passes, number of passes actually run).
    """
    total_added = 0
    pass_num = 0
    for pass_num in range(1, max_passes + 1):
        G, added, _skipped = bridge_gaps_once(G, threshold_m, speed_mph)
        total_added += added
        print(f"  pass {pass_num}: added {added:,} bridge edge(s) (running total: {total_added:,})")
        if added == 0:
            break
    else:
        print(
            f"  *** WARNING: reached the max-passes cap ({max_passes}) without a pass "
            f"adding zero new bridges. This likely indicates a bug (e.g. two dangling "
            f"nodes repeatedly bridging back and forth) rather than legitimately slow "
            f"convergence -- investigate before trusting this graph's connectivity. ***"
        )

    return G, total_added, pass_num


# ---------------------------------------------------------------------------
# Optional, opt-in theta sweep (off by default) -- Step 1.4.3's "reproduce
# their sweep yourself" path, not the default fast-path behavior.
# ---------------------------------------------------------------------------

def sweep_theta(
    G_prelim: nx.MultiGraph, speed_mph: float, thetas: Iterable[int], output_dir: Path
) -> None:
    """Run the bridging pass at every theta in `thetas` and plot component count vs. theta.

    This reproduces the paper's own 50m-1000m parameter sweep that justified
    landing on theta=400m. It is opt-in only (`--sweep-theta`), never run by
    default -- the default single-run behavior always uses the fixed 400m
    value (GAP_THRESHOLD_METERS) per the "fast path" the governing spec
    calls for.

    Args:
        G_prelim: The Step 1.4.1 preliminary graph. A fresh copy is bridged
            at each theta so earlier thetas' bridge edges never leak into
            later ones.
        speed_mph: Default bridge-edge speed, passed through to bridging.
        thetas: The theta values (meters) to sweep over.
        output_dir: Where to save the resulting PNG plot.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    print(f"  Running opt-in theta sweep over {list(thetas)} ...")
    results = []
    for theta in thetas:
        G_test, added, passes = bridge_gaps_iterative(G_prelim.copy(), theta, speed_mph)
        n_comp = nx.number_connected_components(G_test)
        results.append((theta, n_comp))
        print(f"    theta={theta}m -> {n_comp:,} components ({added:,} bridges over {passes} passes)")

    df = pd.DataFrame(results, columns=["theta_m", "n_components"])
    fig, ax = plt.subplots(figsize=(8, 6))
    df.plot(x="theta_m", y="n_components", marker="o", ax=ax, legend=False)
    ax.axvline(GAP_THRESHOLD_METERS, color="red", linestyle="--", label=f"chosen theta={GAP_THRESHOLD_METERS}m")
    ax.set_xlabel("theta (m)")
    ax.set_ylabel("connected components")
    ax.set_title("Component count vs. gap threshold theta")
    ax.legend()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "theta_sweep.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Theta sweep plot saved to: {out_path}")


# ---------------------------------------------------------------------------
# Baselines: actually run Step 1.2 and confirm Step 1.3's output, rather
# than fabricating or reusing a stale logged number.
# ---------------------------------------------------------------------------

def get_naive_baseline(network: str, data_dir: Path, step1_2_path: Optional[Path] = None) -> Optional[int]:
    """Actually run Step 1.2's naive-graph diagnostic and return its component count.

    Per the governing task prompt: "Do not fabricate or estimate baseline
    numbers ... Get real ones by actually running Steps 1.2 and 1.3 ...
    rather than inventing a plausible-sounding number or leaving the
    comparison out." This function does exactly that by importing and
    calling Step 1.2's own `load_and_validate` and `build_naive_graph`
    functions directly against the original cleaned input file.

    Args:
        network: "highway" or "railway" -- used to resolve the original
            cleaned input filename via NETWORK_FILENAMES.
        data_dir: Directory containing `<network>_clean.geojson`.
        step1_2_path: Optional explicit path to Step 1.2's script (from
            --step1-2-path), bypassing filename-pattern auto-detection.

    Returns:
        The naive graph's connected-component count, or None if the
        original cleaned input file could not be found or Step 1.2's module
        could not be imported -- in which case a clear warning is printed
        and the caller is responsible for reporting the comparison as
        unavailable rather than fabricating a number.
    """
    clean_path = data_dir / NETWORK_FILENAMES[network]
    if not clean_path.exists():
        print(
            f"  WARNING: original cleaned input {clean_path} not found -- cannot run "
            f"Step 1.2's naive-graph diagnostic for a real baseline. The 1.4.1-vs-1.2 "
            f"comparison below will be reported as unavailable rather than fabricated."
        )
        return None

    try:
        step1_2 = _load_step1_2_module(explicit_path=step1_2_path)
    except (FileNotFoundError, ImportError) as exc:
        print(f"  WARNING: could not import Step 1.2's module: {exc}")
        return None

    print(f"  Running Step 1.2's naive-graph diagnostic against {clean_path} for a real baseline...")
    gdf = step1_2.load_and_validate(str(clean_path))
    G_naive = step1_2.build_naive_graph(gdf)
    n_components = nx.number_connected_components(G_naive)
    print(f"  Naive graph baseline: {G_naive.number_of_nodes():,} nodes, "
          f"{G_naive.number_of_edges():,} edges, {n_components:,} components")
    return n_components


# ---------------------------------------------------------------------------
# Final connected-component report -- printed (and returned) at the very end
# of a network's run, after gap bridging has finished.
# ---------------------------------------------------------------------------

def report_final_components(
    G: nx.MultiGraph, network: str, top_n: int = 5
) -> Dict[str, object]:
    """Print (and return) a final connected-component report for the
    post-bridging graph.

    This is the last thing printed for a given network's run: after Step
    1.4.4's iterative bridging has converged (or hit its pass cap), this
    answers the single most important question left -- "is this graph
    actually connected now, and if not, how fragmented is it left?" -- with
    real numbers rather than leaving the reader to infer it from the
    pass-by-pass bridging log above.

    Reports, in order:
      - total node / edge counts for the whole (possibly still fragmented)
        graph;
      - total connected-component count;
      - whether the graph is a single fully-connected component
        (`nx.is_connected`), printed unambiguously either way;
      - for the largest component: node count, edge count, and what
        percentage of all nodes it holds;
      - a table of the `top_n` largest components by node count (component
        rank, node count, edge count) -- so if the graph is NOT fully
        connected, it's immediately visible whether that's "one dominant
        component plus a handful of tiny stragglers" (usually fine, matches
        real dead-end spurs) or "several large components" (worth
        investigating further, e.g. in Step 1.4.5's manual gap review).

    Args:
        G: The post-bridging graph (`G_bridged` from `bridge_gaps_iterative`).
        network: "highway" or "railway" -- used only in the printed banner.
        top_n: How many of the largest components to list individually
            (default: 5). Harmless if the graph has fewer components than
            this -- it just lists all of them.

    Returns:
        A dict summary of the same numbers printed, in case a caller wants
        to log/serialize this alongside the pickle rather than only reading
        it off stdout:
            {
                "network": str,
                "total_nodes": int,
                "total_edges": int,
                "n_components": int,
                "is_fully_connected": bool,
                "largest_component_nodes": int,
                "largest_component_edges": int,
                "largest_component_pct_of_nodes": float,
                "top_components": [
                    {"rank": int, "nodes": int, "edges": int}, ...
                ],
            }
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

    _print_banner(f"FINAL CONNECTED-COMPONENT REPORT -- {network.upper()}")
    print(f"Total nodes:              {total_nodes:,}")
    print(f"Total edges:              {total_edges:,}")
    print(f"Connected components:     {n_components:,}")
    if is_fully_connected:
        print("Fully connected:          YES -- exactly 1 component.")
    else:
        print(
            f"Fully connected:          NO -- {n_components:,} components remain "
            f"after gap bridging."
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
        print(
            "\nNOTE: remaining components are usually either real dead-end spurs "
            "(single small components, expected) or larger unresolved gaps beyond "
            "theta -- the latter is exactly what Step 1.4.5's manual review is for."
        )

    return {
        "network": network,
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
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments, matching Steps 1.2/1.3's CLI shape.

    Returns:
        The parsed argparse.Namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Steps 1.4.1-1.4.4: build a preliminary graph from noded lines, "
            "find dangling nodes, and iteratively bridge gaps within theta. "
            "Runs BOTH highway and railway automatically unless --network "
            "or a positional input_geojson restricts it to one."
        )
    )
    parser.add_argument(
        "input_geojson",
        nargs="?",
        default=None,
        help=(
            "Path to a specific noded GeoJSON (Step 1.3's output). Optional "
            "-- if omitted, BOTH highway and railway are processed "
            "automatically via NODED_INPUT_DIR (unless --network restricts "
            "this to one). If you pass this, only that one file is "
            "processed."
        ),
    )
    parser.add_argument(
        "--network",
        choices=sorted(NETWORK_FILENAMES.keys()),
        default=None,
        help=(
            "Restrict the run to just this one network. If omitted (and no "
            "positional input_geojson is given), BOTH highway and railway "
            "are processed automatically. Also determines the naive-graph "
            "baseline's input file and the bridging speed default "
            "(HIGHWAY=55mph, RAILWAY=49mph) for whichever network is "
            "actually running."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=(
            f"Directory holding the ORIGINAL <network>_clean.geojson, used only "
            f"for the Step 1.2 naive-graph baseline comparison (default: {DEFAULT_DATA_DIR})."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            f"Directory for diagnostic artifacts, e.g. an optional theta-sweep "
            f"plot (default: {DEFAULT_OUTPUT_DIR}). The primary bridged graph "
            f"pickle always goes to {BRIDGED_OUTPUT_DIR}/, not this directory."
        ),
    )
    parser.add_argument(
        "--theta-m",
        type=float,
        default=GAP_THRESHOLD_METERS,
        help=f"Gap-bridging distance threshold in meters (default: {GAP_THRESHOLD_METERS}, per the paper's own 50m-1000m sweep).",
    )
    parser.add_argument(
        "--max-passes",
        type=int,
        default=MAX_PASSES_DEFAULT,
        help=f"Safety cap on bridging passes (default: {MAX_PASSES_DEFAULT}).",
    )
    parser.add_argument(
        "--sweep-theta",
        action="store_true",
        help=(
            "Opt-in only, OFF by default: also run the full 50m-1000m theta "
            "sweep from the paper and save a component-count-vs-theta plot. "
            "Does not change --theta-m's use for the main bridging run."
        ),
    )
    parser.add_argument(
        "--step1-2-path",
        type=Path,
        default=None,
        help=(
            "Explicit path to the Step 1.2 (naive graph) script, bypassing "
            "auto-detection. Use this if this script can't find it by "
            "filename pattern (e.g. it was renamed)."
        ),
    )
    parser.add_argument(
        "--step1-3-path",
        type=Path,
        default=None,
        help=(
            "Explicit path to the Step 1.3 (touching/crossing) script, "
            "bypassing auto-detection. Use this if this script can't find "
            "it by filename pattern (e.g. it was renamed)."
        ),
    )
    return parser.parse_args()


def resolve_input_path(network: str, args: argparse.Namespace) -> Path:
    """Resolve the noded input GeoJSON path for one network run.

    If a positional `input_geojson` was given, it takes precedence over the
    `NODED_INPUT_DIR` lookup (used only in single-file mode).

    Args:
        network: "highway" or "railway" for this run.
        args: Parsed CLI arguments.

    Returns:
        The resolved path to the noded GeoJSON.
    """
    if args.input_geojson:
        return Path(args.input_geojson)
    return NODED_INPUT_DIR / f"{network}_noded.geojson"


def run_one_network(network: str, args: argparse.Namespace) -> Optional[Dict[str, object]]:
    """Run Steps 1.4.1-1.4.4 end to end for a single network.

    Returns the final connected-component summary dict (see
    `report_final_components`) on success. Returns None (without raising) if
    the noded input file doesn't exist, so a multi-network run can skip this
    network and continue with the other one instead of aborting entirely.
    """
    input_path = resolve_input_path(network, args)
    speed_mph = DEFAULT_BRIDGE_SPEED_MPH[network]

    print()
    print("#" * 70)
    print(f"# NETWORK: {network}")
    print("#" * 70)

    if not input_path.exists():
        print(
            f"  WARNING: noded input file not found, skipping this network:\n"
            f"    {input_path}\n"
            f"  (Run Step 1.3 for this network first, or if this network "
            f"genuinely isn't part of your dataset, this is expected -- "
            f"otherwise pass a path directly.)"
        )
        return None

    _print_banner("STEP 1.4.1-1.4.4 -- Preliminary graph, dangling nodes, gap bridging")
    print(f"Input (noded, Step 1.3 output): {input_path}")
    print(f"Network:                        {network}")
    print(f"Bridging default speed:         {speed_mph} mph")
    print(f"Diagnostics dir:                {args.output_dir}")
    out_preview = BRIDGED_OUTPUT_DIR / f"{network}_bridged.pkl"
    print(f"Will save:                      {out_preview}")

    _print_banner("Loading and validating noded input")
    step1_3 = _load_step1_3_module(explicit_path=args.step1_3_path)
    gdf_noded = load_and_validate_noded(input_path, step1_3)

    _print_banner("Step 1.2 naive-graph baseline (for comparison)")
    naive_n_components = get_naive_baseline(network, args.data_dir, step1_2_path=args.step1_2_path)

    _print_banner("1.4.1 -- Building the preliminary graph")
    G_prelim, skipped = build_preliminary_graph(gdf_noded)
    n_prelim_components = report_preliminary_vs_naive(G_prelim, naive_n_components)

    _print_banner("1.4.2 -- Finding dangling (degree-1) nodes")
    dangling = find_dangling_nodes(G_prelim)
    print(f"{len(dangling):,} dangling node(s) found in the preliminary graph.")

    _print_banner("1.4.3 -- Gap threshold theta")
    print(f"theta = {args.theta_m:g} meters")
    print(
        "  Source: paper's own 50m-1000m parameter sweep -- values below 400m missed "
        "too many true gaps (overly fragmented graph); values above 400m started "
        "wrongly connecting unrelated road segments. 400m was the smallest theta that "
        "did not produce excess fragmentation."
    )
    if args.theta_m != GAP_THRESHOLD_METERS:
        print(f"  NOTE: this run overrides the default GAP_THRESHOLD_METERS={GAP_THRESHOLD_METERS} via --theta-m.")

    if args.sweep_theta:
        _print_banner("1.4.3 (opt-in) -- Full 50m-1000m theta sweep")
        sweep_theta(G_prelim, speed_mph, range(50, 1050, 50), args.output_dir)

    _print_banner("1.4.4 -- Iterative gap bridging")
    G_bridged, total_added, n_passes = bridge_gaps_iterative(
        G_prelim, threshold_m=args.theta_m, speed_mph=speed_mph, max_passes=args.max_passes
    )
    n_bridged_components = nx.number_connected_components(G_bridged)
    reduction = n_prelim_components - n_bridged_components
    pct = (100 * reduction / n_prelim_components) if n_prelim_components else 0.0
    print(
        f"\nPost-bridging: {n_bridged_components:,} components (was {n_prelim_components:,} "
        f"before bridging) -- reduced by {reduction:,} ({pct:.1f}%) across {n_passes} pass(es), "
        f"{total_added:,} total bridge edge(s) added."
    )

    remaining_dangling = find_dangling_nodes(G_bridged)
    print(f"Dangling nodes remaining after bridging: {len(remaining_dangling):,} "
          f"(was {len(dangling):,} before bridging)")

    _print_banner("Saving output")
    out_path = BRIDGED_OUTPUT_DIR / f"{network}_bridged.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(G_bridged, f)
    print(f"Saved post-bridging graph (pickle) to: {out_path}")

    # Final connected-component report -- the last thing printed for this
    # network, run against the actual saved (post-bridging) graph.
    component_summary = report_final_components(G_bridged, network)

    _print_banner("Step 1.4.1-1.4.4 complete")
    print(
        f"Next step: 1.4.5 -- manual review of larger/rarer gaps (suspicious-detour "
        f"flagging, satellite/map cross-referencing, manual connector logging). This "
        f"script deliberately does NOT implement 1.4.5 -- that is intentionally a "
        f"separate task. Start it from '{out_path}' with {len(remaining_dangling):,} "
        f"dangling node(s) still remaining and {component_summary['n_components']:,} "
        f"connected component(s) total."
    )

    return component_summary


def main() -> None:
    """Entry point.

    Runs BOTH highway and railway automatically unless a positional
    `input_geojson` or `--network` restricts the run to a single network
    (see the module docstring's "BOTH NETWORKS BY DEFAULT" section).
    """
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    BRIDGED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.input_geojson:
        # Explicit file given -- single-file mode, labeled by --network if
        # given, else HARDCODED_NETWORK.
        networks_to_run = [args.network or HARDCODED_NETWORK]
    elif args.network:
        # Restricted to one named network.
        networks_to_run = [args.network]
    else:
        # NEW DEFAULT: both networks, in a fixed/deterministic order.
        networks_to_run = sorted(NETWORK_FILENAMES.keys())

    print(f"Running Steps 1.4.1-1.4.4 for: {', '.join(networks_to_run)}")

    results = {network: run_one_network(network, args) for network in networks_to_run}

    ran = [n for n, summary in results.items() if summary is not None]
    skipped = [n for n, summary in results.items() if summary is None]

    print()
    print("=" * 70)
    print("ALL NETWORKS COMPLETE")
    print("=" * 70)
    print(f"Ran successfully: {ran or '(none)'}")
    if skipped:
        print(f"Skipped (noded input file not found): {skipped}")
    if not ran:
        raise SystemExit(
            "\nNo network was actually processed -- every requested noded "
            "input file was missing. Run Step 1.3 first, or pass a path "
            "directly."
        )

    # Final cross-network connected-component summary -- the very last
    # thing this script prints, so "how many connected components does each
    # network end up with" never requires scrolling back through the
    # per-network logs above to answer.
    print()
    print("=" * 70)
    print("CONNECTED-COMPONENT SUMMARY (all networks)")
    print("=" * 70)
    print(f"  {'network':<10}  {'nodes':>10}  {'edges':>10}  {'components':>10}  {'fully connected':>16}")
    for network in ran:
        s = results[network]
        print(
            f"  {network:<10}  {s['total_nodes']:>10,}  {s['total_edges']:>10,}  "
            f"{s['n_components']:>10,}  {('YES' if s['is_fully_connected'] else 'NO'):>16}"
        )


if __name__ == "__main__":
    main()