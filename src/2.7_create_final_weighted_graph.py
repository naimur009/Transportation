#!/usr/bin/env python3
"""
Step 2.7 -- Assemble the final weighted graph from the QGIS-reviewed line
data for HIGHWAY/RAILWAY (Illinois test run).

DATASET NOTE (updated to match the Illinois test run's naming convention):
-----------------------------------------------------------------------
Earlier drafts of this script (and the pipeline generally) used the
STRAHNET/STRACNET naming. The pipeline has since moved to "highway" /
"railway" labels throughout (see 1_cleanDataset.py's OUTPUT_FILENAMES,
2_2_fix_toucing_and_crossing.py's NODED_OUTPUT_DIR, 2_3_gap_repair.py's
BRIDGED_OUTPUT_DIR, 2_4_remove_noise.py's REVIEW_DIR, and
2_5_convert_to_gpkg.py / 2_6_find_connected_components.py's HIGHWAY/RAILWAY
labels). This script has been updated to match:

  * NETWORK_FILENAMES now maps "highway"/"railway" -> their respective
    <network>_clean.geojson filenames, matching 1_cleanDataset.py's
    OUTPUT_FILENAMES exactly.
  * SPEED_COLUMN / FLAT_SPEED_MPH / DEFAULT_FALLBACK_SPEED_MPH keys are now
    "highway"/"railway". HIGHWAY still recovers real per-segment speed via
    spatial join (55 mph default fallback, matching
    HIGHWAY_DEFAULT_SPEED_MPH in 1_cleanDataset.py); RAILWAY still uses a
    flat 49 mph for every edge (matching RAILWAY_SPEED_MPH), no join
    needed.
  * The hardcoded "expect exactly 5 components" check from the earlier
    CONUS-wide STRAHNET draft (Hawaii/Alaska/Puerto Rico as legitimately
    separate pieces) does NOT apply to this Illinois-scoped test run --
    there's no reason a single-state extract should land on 5 components.
    That check is now an optional, explicit `--expected-components` flag
    (default: unset / no check) instead of a hardcoded assumption baked
    into the script. See "Component count" below.
  * The "~71 node pairs shared by more than one line" claim from the
    earlier STRAHNET-scale draft has been removed -- that was a specific
    measurement against the national dataset, not something verified
    against the Illinois data. The script still builds an `nx.MultiGraph`
    regardless (see "Duplicate endpoint pairs" note in 2_3_gap_repair.py
    for why a plain Graph is unsafe here), and now reports the actual
    count of shared-pair edges it finds in THIS run instead of asserting a
    borrowed number.

If 2_5_convert_to_gpkg.py's / 2_6_find_connected_components.py's output
naming changes again, update the paths/dicts below to match -- keep them
in sync by hand, there is no automatic link between the scripts.

What this script actually consumes
-----------------------------------
Earlier drafts of this step assumed a noded-GeoJSON + bridge-pickle +
manual-CSV-log combination. That is NOT what exists on disk once a human
has actually reviewed and hand-fixed the graph in QGIS. What Step 2.5
(2_5_convert_to_gpkg.py) produces and a human then edits/reviews is a
single GeoPackage (`data/graph/qgis_review/<network>_review.gpkg`) with:

  * ONE layer, auto-generated name (varies per export -- do not hardcode
    it, detect it with pyogrio.list_layers()).
  * A pile of junk `cat`, `cat_`, `cat__1`, `cat__1_1`, ... columns --
    accumulated leftovers from repeated GRASS/QGIS `v.clean` attribute
    merges. These carry no information this pipeline needs and are
    dropped on load.
  * `edge_type`: "real" (an actual source line, INCLUDING any lines you
    manually drew in QGIS to fix a gap -- those come through indistinguishable
    from original data) or "auto_bridge" (Step 1.4's automatic gap-bridge
    edges, carried through unchanged).
  * `node_start` / `node_end`: string reprs of raw (lon, lat) tuples.
    IMPORTANT: these do NOT always match the line's own geometry endpoints
    (~0.17% measured drift from QGIS snapping/editing, on top of many
    start/end swaps). This script therefore derives node IDs from each
    line's ACTUAL geometry, never from these attribute columns.
  * `length_km`: populated for every row.
  * `travel_time_min`: populated ONLY for `auto_bridge` rows (computed back
    in Step 1.4 using a default speed). All `real` rows have this as NaN,
    because no speed column survived the QGIS round-trip.

Because of that last point, this script's main job beyond simple assembly
is RECOVERING SPEED for `real` edges by spatially joining each one back to
the original `<network>_clean.geojson` (the file that still has the
authentic per-segment `speed_mph` column, for HIGHWAY). A `real` edge that
can't be matched within `--match-max-dist-m` of any original line is
treated as a manually-drawn connector with no original speed of record,
and falls back to `--fallback-speed-mph` -- logged explicitly, never
silently guessed without a trace.

RAILWAY does not need this join at all -- it uses one flat 49 mph speed
for every edge regardless of source, same as it always has (per 49 CFR
Sec. 236.567(c), matching RAILWAY_SPEED_MPH in 1_cleanDataset.py).

This script builds and saves the final weighted graph. It does NOT drop
down to the largest connected component by default -- component-trimming,
if you want it, is a distinct, separate step (Step 1.6/"keep largest
component" territory) -- run it explicitly afterward, don't fold it in
here. Pass `--expected-components N` if you want this script to flag a
mismatch against a component count you've already confirmed via manual
review; omit it (the default) to just report the count with no
pass/fail judgment -- appropriate for this Illinois test run, where there
is no a-priori "should be exactly N components" expectation the way the
CONUS-wide dataset had (islands/territories, etc.).

Usage
-----
    python 2.7_create_final_weighted_graph.py
    python 2.7_create_final_weighted_graph.py --network railway
    python 2.7_create_final_weighted_graph.py --reviewed-gpkg data/graph/qgis_review/highway_review.gpkg
    python 2.7_create_final_weighted_graph.py --match-max-dist-m 50 --fallback-speed-mph 55
    python 2.7_create_final_weighted_graph.py --expected-components 1   # optional, explicit check

Input
-----
    data/graph/qgis_review/<network>_review.gpkg   (your manually-reviewed file, Step 2.5's output)
    data/cleaned/<network>_clean.geojson            (for the speed_mph join -- HIGHWAY only)

Output
------
    data/graph/final/<network>_final.pkl
    outputs/diagnostics/<network>_speed_match_log.csv   (which real edges fell back, and why)
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import pyogrio

# ---------------------------------------------------------------------------
# Paths / conventions
# ---------------------------------------------------------------------------

CLEAN_DATA_DIR = Path("data/cleaned")                       # original <network>_clean.geojson
REVIEWED_INPUT_DIR = Path("data/graph/qgis_review")          # your manually-reviewed gpkg (Step 2.5's OUTPUT_DIR)
FINAL_OUTPUT_DIR = Path("data/graph/final")                  # this step's output
DEFAULT_OUTPUT_DIR = Path("outputs/diagnostics")             # speed-match log lives here

# Matches 1_cleanDataset.py's OUTPUT_FILENAMES exactly (highway_clean.geojson /
# railway_clean.geojson) -- not independently guessed.
NETWORK_FILENAMES = {
    "highway": "highway_clean.geojson",
    "railway": "railway_clean.geojson",
}

# HARDCODED DEFAULT NETWORK -- same pattern as Steps 1.3-1.6. Only used as
# --network's default; overridable per-run with --network <other>.
HARDCODED_NETWORK = "highway"

ROUND_DECIMALS = 6            # permanent node ID precision (~10cm)
MILES_PER_KM = 1.60934

# HIGHWAY recovers real per-segment speed via spatial join to the original
# cleaned file. RAILWAY never needs the join -- it's a flat speed for
# every edge, full stop, matching how Step 1.4/1.5 always treated it
# (RAILWAY_SPEED_MPH = 49.0 in 1_cleanDataset.py, per 49 CFR Sec. 236.567(c)).
SPEED_COLUMN = {"highway": "speed_mph", "railway": None}
FLAT_SPEED_MPH = {"highway": None, "railway": 49}

# Fallback speed for a `real` edge that cannot be matched to any original
# line within --match-max-dist-m (i.e. a manually-drawn connector with no
# provenance in the source data). Same defaults 1_cleanDataset.py/Step 1.4
# already used (HIGHWAY_DEFAULT_SPEED_MPH=55, RAILWAY_SPEED_MPH=49), so a
# connector's assumed speed is consistent whether a human or the automatic
# bridger added it.
DEFAULT_FALLBACK_SPEED_MPH = {"highway": 55, "railway": 49}

DEFAULT_MATCH_MAX_DIST_M = 30.0   # generous vs. typical QGIS snap/edit drift, tight vs. real distinct roads


# ---------------------------------------------------------------------------
# Small shared helpers -- same conventions as Steps 1.3/1.4
# ---------------------------------------------------------------------------

def _print_banner(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def _progress_iter(iterable: Iterable, total: int, label: str, every: int = 5000) -> Iterator:
    start = time.monotonic()
    for i, item in enumerate(iterable, start=1):
        yield item
        if i % every == 0 or i == total:
            elapsed = time.monotonic() - start
            rate = i / elapsed if elapsed > 0 else 0.0
            print(f"    [{label}] {i:,}/{total:,} ({rate:,.0f}/s, {elapsed:,.1f}s elapsed)")


def _round_point(pt: Tuple[float, float], decimals: int) -> Tuple[float, float]:
    return (round(pt[0], decimals), round(pt[1], decimals))


# ---------------------------------------------------------------------------
# Load the reviewed gpkg (auto-detect layer, drop junk columns)
# ---------------------------------------------------------------------------

def load_reviewed_gpkg(path: Path, layer: Optional[str] = None) -> gpd.GeoDataFrame:
    """Load the QGIS-reviewed gpkg, auto-detecting its layer name.

    QGIS/GRASS exports this with an auto-generated hash layer name that
    changes per export -- never hardcode it. Also strips the accumulated
    `cat`, `cat_`, `cat__1`, ... junk columns left behind by repeated
    `v.clean` attribute-table merges; they carry no information this
    pipeline uses.

    Args:
        path: Path to the reviewed .gpkg file.
        layer: Explicit layer name, or None to auto-detect (only valid if
            exactly one layer exists -- multiple layers require an explicit
            choice since there's no safe way to guess which is authoritative).

    Returns:
        The loaded GeoDataFrame with junk columns dropped.

    Raises:
        FileNotFoundError: If `path` does not exist.
        ValueError: If the gpkg has multiple layers and none was specified,
            or the resolved layer doesn't contain the expected columns.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Reviewed gpkg not found: {path}\n"
            f"Hint: this should be the file you saved out of QGIS after Step 2.6's "
            f"manual review, or pass --reviewed-gpkg explicitly."
        )

    layers = pyogrio.list_layers(path)  # array of [name, geometry_type]
    layer_names = [row[0] for row in layers]
    if layer is None:
        if len(layer_names) != 1:
            raise ValueError(
                f"{path} has {len(layer_names)} layers ({layer_names}) -- pass "
                f"--layer explicitly to pick one; auto-detection only works when "
                f"there's exactly one."
            )
        layer = layer_names[0]
        print(f"Auto-detected layer: {layer!r}")
    elif layer not in layer_names:
        raise ValueError(f"Layer {layer!r} not found in {path}. Available: {layer_names}")

    gdf = gpd.read_file(path, layer=layer)

    junk_cols = [c for c in gdf.columns if c == "cat" or c.startswith("cat_")]
    if junk_cols:
        print(f"Dropping {len(junk_cols)} QGIS/GRASS junk column(s): {junk_cols}")
        gdf = gdf.drop(columns=junk_cols)

    required = {"edge_type", "length_km", "geometry"}
    missing = required - set(gdf.columns)
    if missing:
        raise ValueError(
            f"Reviewed gpkg is missing expected column(s) {sorted(missing)}. "
            f"Columns present: {list(gdf.columns)}. This doesn't look like Step "
            f"2.5/2.6's expected output -- confirm you're pointing at the right file."
        )

    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        raise ValueError(f"Expected CRS EPSG:4326, got {gdf.crs} for {path}.")

    print(f"Loaded {len(gdf):,} rows from layer {layer!r}")
    print(f"  edge_type counts: {gdf['edge_type'].value_counts().to_dict()}")
    return gdf


# ---------------------------------------------------------------------------
# Speed recovery for `real` edges via spatial join to the original cleaned file
# ---------------------------------------------------------------------------

def recover_speed_for_real_edges(
    gdf: gpd.GeoDataFrame,
    clean_path: Path,
    speed_col: str,
    match_max_dist_m: float,
    fallback_speed_mph: float,
    output_dir: Path,
    network: str,
) -> gpd.GeoDataFrame:
    """Recover `speed_mph` for every `real` edge via a nearest-line spatial join.

    Matches each `real` edge's MIDPOINT against the original cleaned lines
    (which still carry the authentic `speed_col`), using `sjoin_nearest`
    capped at `match_max_dist_m`. The midpoint (not an endpoint) is used
    because Step 1.3's noding process can split an original line at a new
    vertex without moving any existing point, so the midpoint of a fully-
    contained sub-segment is still guaranteed to sit on the original line;
    an endpoint could coincide with an unrelated line's endpoint at a
    junction and match the wrong one.

    A `real` edge with no original line within `match_max_dist_m` is treated
    as a connector you drew by hand in QGIS with no corresponding source
    line -- it gets `fallback_speed_mph` instead, and is written to a CSV
    log so you have a record of exactly which edges are running on an
    assumed rather than sourced speed.

    `auto_bridge` edges are left untouched -- they already carry
    `travel_time_min` from Step 1.4 and are not part of this join at all.

    Args:
        gdf: The reviewed GeoDataFrame (post `load_reviewed_gpkg`).
        clean_path: Path to `<network>_clean.geojson`.
        speed_col: Name of the speed column in the clean file (e.g. "speed_mph").
        match_max_dist_m: Maximum join distance in meters.
        fallback_speed_mph: Speed assumed for unmatched `real` edges.
        output_dir: Where to write the speed-match log CSV.
        network: Used only for the log filename.

    Returns:
        A copy of `gdf` with a new `speed_mph` column populated for every
        `real` row (joined or fallback) and left as NaN for `auto_bridge` rows
        (they don't need it -- they already have travel_time_min).
    """
    if not clean_path.exists():
        raise FileNotFoundError(
            f"Original cleaned file not found: {clean_path}\n"
            f"This is required to recover speed_mph for 'real' edges -- pass "
            f"--clean-geojson explicitly if it lives somewhere else."
        )

    gdf_clean = gpd.read_file(clean_path)
    if speed_col not in gdf_clean.columns:
        raise ValueError(
            f"Expected speed column {speed_col!r} not found in {clean_path}. "
            f"Columns present: {list(gdf_clean.columns)}."
        )
    gdf_clean = gdf_clean[[speed_col, "geometry"]].explode(index_parts=False).reset_index(drop=True)
    print(f"Loaded {len(gdf_clean):,} original lines from {clean_path} for speed recovery.")

    is_real = gdf["edge_type"] == "real"
    real_gdf = gdf.loc[is_real].copy()
    print(f"Recovering speed for {len(real_gdf):,} 'real' edges via nearest-line join "
          f"(max distance {match_max_dist_m:.0f}m)...")

    midpoints = real_gdf.geometry.interpolate(0.5, normalized=True)
    mid_gdf = gpd.GeoDataFrame(geometry=midpoints, crs=gdf.crs).to_crs("EPSG:5070")
    clean_m = gdf_clean.to_crs("EPSG:5070")

    joined = gpd.sjoin_nearest(
        mid_gdf, clean_m[[speed_col, "geometry"]],
        max_distance=match_max_dist_m, distance_col="_match_dist_m",
    )
    # sjoin_nearest can return >1 match per point on exact ties; keep the closest.
    joined = joined.sort_values("_match_dist_m").groupby(level=0).first()

    real_gdf["speed_mph"] = np.nan
    real_gdf.loc[joined.index, "speed_mph"] = joined[speed_col].values

    n_matched = real_gdf["speed_mph"].notna().sum()
    n_fallback = len(real_gdf) - n_matched
    print(f"  Matched: {n_matched:,} / {len(real_gdf):,} ({100 * n_matched / len(real_gdf):.1f}%)")
    print(f"  Unmatched (using fallback {fallback_speed_mph} mph): {n_fallback:,}")

    if n_fallback:
        log_path = output_dir / f"{network}_speed_match_log.csv"
        unmatched = real_gdf.loc[real_gdf["speed_mph"].isna()]
        unmatched_log = gpd.GeoDataFrame(
            {"length_km": unmatched["length_km"].values}, geometry=unmatched.geometry.values, crs=gdf.crs,
        )
        unmatched_log["fallback_speed_mph"] = fallback_speed_mph
        unmatched_log.to_csv(log_path, index=False)
        print(f"  Logged {n_fallback:,} fallback edge(s) to: {log_path}")

    real_gdf["speed_mph"] = real_gdf["speed_mph"].fillna(fallback_speed_mph)

    gdf = gdf.copy()
    gdf["speed_mph"] = np.nan
    gdf.loc[real_gdf.index, "speed_mph"] = real_gdf["speed_mph"].values
    return gdf


# ---------------------------------------------------------------------------
# Build the final graph
# ---------------------------------------------------------------------------

def build_final_graph(
    gdf: gpd.GeoDataFrame,
    network: str,
    round_decimals: int = ROUND_DECIMALS,
) -> Tuple[nx.MultiGraph, int]:
    """Build the final weighted MultiGraph from the reviewed + speed-joined lines.

    Node IDs come from each line's own geometry endpoints (rounded to
    `round_decimals`), never from the `node_start`/`node_end` attribute
    columns -- see the module docstring for why those can't be trusted.

    `length_km` is taken from the geometry directly (reprojected to
    EPSG:5070), not the file's own `length_km` column, so the final graph's
    weights reflect the CURRENT (possibly hand-edited) geometry rather than
    a value computed before your QGIS edits. The file's own `length_km` is
    kept only as a cross-check, logged if it disagrees substantially.

    `travel_time_min`:
      - `auto_bridge` rows: kept as-is from Step 1.4 (already correct).
      - `real` rows: computed fresh from the recomputed length_km and the
        `speed_mph` column added by `recover_speed_for_real_edges`
        (HIGHWAY) or `FLAT_SPEED_MPH[network]` (RAILWAY, no join needed).

    Builds an `nx.MultiGraph`, not a plain `Graph`. Step 1.4
    (2_3_gap_repair.py) found ~40 distinct (start, end) coordinate pairs
    shared by more than one source line in this dataset (parallel
    carriageways / shared junctions) -- a plain `Graph` would silently
    drop the extras. This function doesn't assume that exact count still
    holds after QGIS review and rounding; it counts and reports the actual
    number of shared-pair edges it finds in THIS run (see
    `duplicate_pair_edges` in the return summary) rather than asserting a
    borrowed number from an earlier stage or dataset scope.

    Args:
        gdf: The reviewed GeoDataFrame, with `speed_mph` added for HIGHWAY.
        network: "highway" or "railway".
        round_decimals: Decimal places for permanent node IDs.

    Returns:
        A tuple of (the final nx.MultiGraph, count of rows with a length_km
        mismatch > 5% between the file's stored value and the freshly
        recomputed one -- worth spot-checking if non-zero).
    """
    gdf_m = gdf.to_crs("EPSG:5070")
    flat_speed = FLAT_SPEED_MPH[network]

    G = nx.MultiGraph()
    mismatch_count = 0
    total = len(gdf)

    seen_pairs: dict = {}
    duplicate_pair_edges = 0

    stored_travel_times = gdf["travel_time_min"] if "travel_time_min" in gdf else np.full(total, np.nan)
    recovered_speeds = gdf["speed_mph"] if "speed_mph" in gdf else np.full(total, np.nan)

    rows = _progress_iter(
        zip(gdf.geometry, gdf_m.geometry, gdf["edge_type"],
            gdf["length_km"], stored_travel_times, recovered_speeds),
        total, label="Building final graph edges",
    )
    for geom_ll, geom_m, edge_type, stored_length_km, stored_tt_min, speed_mph in rows:
        coords = list(geom_ll.coords)
        u = _round_point(coords[0], round_decimals)
        v = _round_point(coords[-1], round_decimals)

        length_km = geom_m.length / 1000.0
        if stored_length_km and length_km > 0:
            pct_diff = abs(length_km - stored_length_km) / stored_length_km
            if pct_diff > 0.05:
                mismatch_count += 1

        if edge_type == "auto_bridge":
            travel_time_min = stored_tt_min
        else:
            speed = speed_mph if flat_speed is None else flat_speed
            travel_time_min = (length_km / MILES_PER_KM) / speed * 60

        G.add_edge(u, v, length_km=length_km, travel_time_min=travel_time_min,
                   edge_type=edge_type)

        pair_key = tuple(sorted([u, v]))
        seen_pairs[pair_key] = seen_pairs.get(pair_key, 0) + 1
        if seen_pairs[pair_key] > 1:
            duplicate_pair_edges += 1

    if duplicate_pair_edges:
        n_pairs = sum(1 for c in seen_pairs.values() if c > 1)
        print(
            f"  NOTE: {n_pairs:,} distinct (start, end) node pair(s) in this run are shared "
            f"by more than one edge ({duplicate_pair_edges:,} extra edge(s) beyond the first "
            f"per pair) -- kept as separate parallel edges via nx.MultiGraph, not collapsed. "
            f"This is a count observed in THIS run, not a number carried over from an "
            f"earlier pipeline stage or dataset scope."
        )

    if mismatch_count:
        print(
            f"  NOTE: {mismatch_count:,} edge(s) have >5% difference between the file's "
            f"stored length_km and the freshly recomputed one -- likely geometry that was "
            f"hand-edited in QGIS after length_km was last calculated. The recomputed "
            f"value was used."
        )

    return G, mismatch_count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 2.7: assemble the final weighted graph from the "
                     "QGIS-reviewed line data. Runs BOTH highway and railway "
                     "automatically unless --network restricts it to one -- "
                     "matching the 'both by default' convention used by every "
                     "other script in this pipeline (2_1 through 2_6)."
    )
    parser.add_argument(
        "--network", choices=sorted(NETWORK_FILENAMES), default=None,
        help=(
            "Restrict the run to just this one network. If omitted, BOTH "
            "highway and railway are processed automatically, one after "
            "the other. --reviewed-gpkg/--layer/--clean-geojson/"
            "--fallback-speed-mph only make sense for a single network, so "
            "they require an explicit --network too."
        ),
    )
    parser.add_argument("--reviewed-gpkg", type=Path, default=None,
                         help="Path to <network>_review.gpkg. Defaults to "
                              "REVIEWED_INPUT_DIR/<network>_review.gpkg. "
                              "Only valid with a single --network.")
    parser.add_argument("--layer", default=None,
                         help="Explicit layer name in the gpkg. Auto-detected if omitted "
                              "and the file has exactly one layer. Only valid with a "
                              "single --network.")
    parser.add_argument("--clean-geojson", type=Path, default=None,
                         help="Path to the original <network>_clean.geojson, used for the "
                              "speed_mph join (HIGHWAY only). Defaults to "
                              "CLEAN_DATA_DIR/<network>_clean.geojson. Only valid with a "
                              "single --network.")
    parser.add_argument("--match-max-dist-m", type=float, default=DEFAULT_MATCH_MAX_DIST_M,
                         help=f"Max distance (m) for the speed spatial join (default: "
                              f"{DEFAULT_MATCH_MAX_DIST_M}).")
    parser.add_argument("--fallback-speed-mph", type=float, default=None,
                         help="Speed for 'real' edges with no match within --match-max-dist-m. "
                              "Defaults to DEFAULT_FALLBACK_SPEED_MPH[network]. Only valid "
                              "with a single --network.")
    parser.add_argument("--round-decimals", type=int, default=ROUND_DECIMALS)
    parser.add_argument("--output-dir", type=Path, default=FINAL_OUTPUT_DIR)
    parser.add_argument("--diagnostics-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--expected-components", type=int, default=None,
        help="Optional: if set, flag a mismatch if the final graph's component "
             "count differs from this value (e.g. a count you've already "
             "confirmed via manual review). Unset by default -- for this "
             "Illinois test run there's no a-priori expected component count "
             "the way the CONUS-wide dataset had (islands/territories), so "
             "the script just reports the count with no pass/fail judgment "
             "unless you explicitly provide one.",
    )
    args = parser.parse_args()

    single_network_only_flags = {
        "--reviewed-gpkg": args.reviewed_gpkg,
        "--layer": args.layer,
        "--clean-geojson": args.clean_geojson,
        "--fallback-speed-mph": args.fallback_speed_mph,
    }
    if args.network is None:
        used = [flag for flag, val in single_network_only_flags.items() if val is not None]
        if used:
            raise SystemExit(
                f"{', '.join(used)} require a single --network (highway or railway) "
                f"-- they can't be combined with the default 'run both networks' mode."
            )

    return args


def run_one_network(network: str, args: argparse.Namespace) -> bool:
    """Run Step 2.7 end to end for a single network.

    Returns True on success. Returns False (without raising) if the
    reviewed gpkg for this network doesn't exist yet, so a multi-network
    run can skip this network and continue with the other one instead of
    aborting entirely -- matching the "both by default, skip what's
    missing" convention used by 2_1 through 2_6.
    """
    reviewed_path = args.reviewed_gpkg or (REVIEWED_INPUT_DIR / f"{network}_review.gpkg")
    clean_path = args.clean_geojson or (CLEAN_DATA_DIR / NETWORK_FILENAMES[network])
    fallback_speed = args.fallback_speed_mph or DEFAULT_FALLBACK_SPEED_MPH[network]

    print()
    print("#" * 70)
    print(f"# NETWORK: {network}")
    print("#" * 70)

    if not reviewed_path.exists():
        print(
            f"  WARNING: reviewed gpkg not found, skipping this network:\n"
            f"    {reviewed_path}\n"
            f"  (Run 2_5_convert_to_gpkg.py and do the QGIS review for this "
            f"network first -- its OUTPUT_DIR/f'{{network}}_review.gpkg' is "
            f"exactly what this path points at -- or pass --reviewed-gpkg "
            f"directly together with --network {network}.)"
        )
        return False

    _print_banner(f"STEP 2.7 -- Assemble the final weighted graph [{network}]")
    print(f"Network:         {network}")
    print(f"Reviewed gpkg:   {reviewed_path}")
    print(f"Clean geojson:   {clean_path} {'(used for speed join)' if SPEED_COLUMN[network] else '(not needed -- flat speed network)'}")
    print(f"Round decimals:  {args.round_decimals}")

    _print_banner("Loading reviewed gpkg")
    gdf = load_reviewed_gpkg(reviewed_path, layer=args.layer)

    speed_col = SPEED_COLUMN[network]
    if speed_col:
        _print_banner("Recovering speed for 'real' edges via spatial join")
        gdf = recover_speed_for_real_edges(
            gdf, clean_path, speed_col,
            match_max_dist_m=args.match_max_dist_m,
            fallback_speed_mph=fallback_speed,
            output_dir=args.diagnostics_dir,
            network=network,
        )
    else:
        print(f"\n{network} uses a flat {FLAT_SPEED_MPH[network]} mph for every edge -- no join needed.")

    _print_banner("Building final graph")
    G_final, mismatch_count = build_final_graph(gdf, network, args.round_decimals)

    n_components = nx.number_connected_components(G_final)
    total_km = sum(d["length_km"] for _, _, d in G_final.edges(data=True))
    total_miles = total_km / MILES_PER_KM
    print(f"\nFinal graph: {G_final.number_of_nodes():,} nodes, "
          f"{G_final.number_of_edges():,} edges, {n_components:,} component(s)")
    print(f"Total length: {total_miles:,.0f} miles")
    if args.expected_components is not None:
        if n_components != args.expected_components:
            print(
                f"  NOTE: --expected-components was set to {args.expected_components:,}, "
                f"got {n_components:,}. Not necessarily wrong (rounding node IDs here can "
                f"merge points that were distinct pre-rounding), but worth a quick look "
                f"before trusting this."
            )
        else:
            print(f"  Matches the {args.expected_components:,} component(s) you specified.")
    else:
        print(
            "  (No --expected-components was given, so no pass/fail judgment is made on "
            "this count -- for the Illinois test scope there's no a-priori expected "
            "component count the way the CONUS-wide dataset had.)"
        )

    _print_banner("Saving output")
    out_path = args.output_dir / f"{network}_final.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(G_final, f)
    print(f"Saved final weighted graph to: {out_path}")

    _print_banner(f"Step 2.7 complete [{network}]")
    print(
        f"Graph has {n_components:,} component(s). This script does not trim to the "
        f"largest component or otherwise treat remaining components as errors -- that "
        f"judgment (real disconnected piece vs. leftover noise) belongs to the manual "
        f"QGIS review (Step 1.5/2.4-2.6), not here. If you want a single-largest-component "
        f"version for a specific downstream use, that's a separate, explicit step -- "
        f"run it against '{out_path}' rather than folding it into this script."
    )

    return True


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.diagnostics_dir.mkdir(parents=True, exist_ok=True)

    if args.network:
        networks_to_run = [args.network]
    else:
        # NEW DEFAULT: both networks, in a fixed/deterministic order --
        # matches 2_1 through 2_6's "both by default" convention.
        networks_to_run = sorted(NETWORK_FILENAMES.keys())

    print(f"Running Step 2.7 for: {', '.join(networks_to_run)}")

    ran = []
    skipped = []
    for network in networks_to_run:
        ok = run_one_network(network, args)
        (ran if ok else skipped).append(network)

    print()
    print("=" * 70)
    print("ALL NETWORKS COMPLETE")
    print("=" * 70)
    print(f"Ran successfully: {ran or '(none)'}")
    if skipped:
        print(f"Skipped (reviewed gpkg not found): {skipped}")
    if not ran:
        raise SystemExit(
            "\nNo network was actually processed -- every requested reviewed "
            "gpkg was missing. Run 2_5_convert_to_gpkg.py and do the QGIS "
            "review first, or pass --network + --reviewed-gpkg directly."
        )


if __name__ == "__main__":
    main()