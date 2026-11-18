#!/usr/bin/env python3
"""
Step 1.6 -- Keep only the largest connected component.

DATASET NOTE (updated to match the Illinois test run's naming convention):
-----------------------------------------------------------------------
Earlier drafts of this script were written against the old
strahnet/stracnet naming and guessed an input filename of
`<network>_final_graph.pkl`. The pipeline has since moved to
"highway"/"railway" labels throughout (see 1_cleanDataset.py's
OUTPUT_FILENAMES, 2_2_fix_toucing_and_crossing.py's NODED_OUTPUT_DIR,
2_3_gap_repair.py's BRIDGED_OUTPUT_DIR, 2_4_remove_noise.py's REVIEW_DIR,
2_5_convert_to_gpkg.py / 2_6_find_connected_components.py, and
2_7_create_final_weighted_graph.py's FINAL_OUTPUT_DIR). This script has
been updated to match:

  * NETWORK_LABELS / default input filenames now use "highway"/"railway",
    pointing at Step 2.7's ACTUAL own output path/filename --
    `data/graph/final/<network>_final.pkl` (2_7_create_final_weighted_graph.py's
    `FINAL_OUTPUT_DIR / f"{args.network}_final.pkl"`) -- not an
    independently-guessed `<network>_final_graph.pkl`.
  * The graph Step 2.7 hands off is an `nx.MultiGraph` with `length_km` set
    on EVERY edge, always (both `real` and `auto_bridge` edge types --
    see `build_final_graph` in 2_7_create_final_weighted_graph.py). This
    script still checks a short list of length-attribute candidates as a
    defensive fallback (in case it's ever pointed at an older/hand-built
    graph), but `length_km` will always be found first against a genuine
    Step 2.7 output.
  * Following the "both networks by default" convention already
    established in every other script in this pipeline
    (2_1 through 2_7): `--network` is now OPTIONAL. Omitting it runs both
    highway and railway automatically, one after the other, each in its
    own banner section. A missing input pickle for one network prints a
    clear warning and skips just that network rather than aborting the
    whole run. Pass `--network` to restrict the run to just one.
  * `--input` (an explicit override path) now requires `--network` to be
    given too, matching the pattern used by 2_5_convert_to_gpkg.py /
    2_7_create_final_weighted_graph.py -- an explicit single-file override
    only makes sense for a single network at a time.

If 2_7_create_final_weighted_graph.py's output naming changes again,
update FINAL_INPUT_DIR / NETWORK_INPUT_FILENAMES below to match -- keep
them in sync by hand, there is no automatic link between the scripts.

Reads the final weighted graph (a pickled networkx.MultiGraph) out of
data/graph/final/, computes the total length of EVERY connected component,
prints a length-ranked breakdown with percentages, keeps only the largest
component, and pickles that as the graph to hand off to Step 2
(facility snapping).

Usage
-----
    # NEW DEFAULT: runs both highway and railway automatically
    python 2_8_keep_the_largest_element.py

    # restrict to just one network
    python 2_8_keep_the_largest_element.py --network highway
    python 2_8_keep_the_largest_element.py --network railway

    # override the input path (requires --network)
    python 2_8_keep_the_largest_element.py --network highway \\
        --input data/graph/final/some_other_name.pkl

Input
-----
    data/graph/final/<network>_final.pkl
    (Step 2.7's -- 2_7_create_final_weighted_graph.py's -- own output path,
    read directly, not independently guessed)

Output
------
    data/graph/final/<network>_final_largest.pkl  -- final graph, ready for Step 2
    reports/<network>_component_length_breakdown.csv  -- full per-component ledger
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import networkx as nx

# ---------------------------------------------------------------------------
# Paths -- kept in sync by hand with 2_7_create_final_weighted_graph.py's
# own FINAL_OUTPUT_DIR / output filename (`f"{args.network}_final.pkl"`).
# ---------------------------------------------------------------------------

FINAL_INPUT_DIR = Path("data/graph/final")        # Step 2.7's own FINAL_OUTPUT_DIR
DEFAULT_REPORTS_DIR = Path("reports")

NETWORK_LABELS = ["highway", "railway"]

# Matches 2_7_create_final_weighted_graph.py's actual saved filename exactly
# (`out_path = args.output_dir / f"{args.network}_final.pkl"`), not an
# independently-guessed name.
NETWORK_INPUT_FILENAMES = {
    "highway": "highway_final.pkl",
    "railway": "railway_final.pkl",
}

# HARDCODED DEFAULT NETWORK -- same pattern as every other script in this
# pipeline. Only used in single-file mode when --input is given without
# --network (belt-and-suspenders; --input now requires --network, see
# parse_args, but this stays as a safety fallback).
HARDCODED_NETWORK = "highway"

# Attribute names to try, in order, when looking for each edge's length.
# A genuine Step 2.7 output always sets "length_km" on every edge, so that
# candidate will always match first -- the rest are a defensive fallback
# only, in case this is ever pointed at an older/hand-built graph.
LENGTH_ATTR_CANDIDATES = ["length_km", "length", "weight", "length_m"]


def _print_banner(title: str) -> None:
    """Print a section banner, matching the section-header print style used
    throughout the rest of this pipeline (2_2, 2_3, 2_4, 2_7, etc.)."""
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def load_graph(path: Path) -> nx.Graph:
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
            f"Pass the real path explicitly with --network + --input, e.g.:\n"
            f"  python {Path(__file__).name} --network highway --input data/graph/final/<your_file>.pkl"
        )
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise SystemExit(
            f"Loaded object from {path} is a {type(obj)!r}, not a networkx graph. "
            f"This script expects a pickled networkx graph (Step 2.7's output is "
            f"specifically an nx.MultiGraph) -- if your 'final' folder stores "
            f"something else (e.g. a GeoDataFrame or a dict), tell me and I'll "
            f"adjust the loader."
        )
    return obj


def inspect_graph(G: nx.Graph) -> str:
    """Print node ID format and available edge attributes so the user can
    visually confirm this matches what they expect BEFORE trusting the
    length numbers below. Returns the length attribute name that will be used.

    Works unchanged against an `nx.MultiGraph` (Step 2.7's actual output
    type): `G.edges(data=True)` yields one tuple per parallel edge, so
    scanning for attribute keys / printing sample edges still sees every
    distinct source line, not just one per node pair.
    """
    print("=" * 70)
    print("GRAPH INSPECTION (confirm this looks right before trusting output)")
    print("=" * 70)
    print(f"Type:                {type(G).__name__}")
    print(f"Nodes:               {G.number_of_nodes():,}")
    print(f"Edges:               {G.number_of_edges():,}")

    sample_nodes = list(G.nodes())[:3]
    print(f"Sample node IDs:     {sample_nodes}")
    node_id_types = {type(n).__name__ for n in sample_nodes}
    print(f"Node ID Python type(s): {node_id_types}")
    if sample_nodes and isinstance(sample_nodes[0], tuple) and len(sample_nodes[0]) == 2:
        x, y = sample_nodes[0]
        print(
            f"  -> looks like (lon, lat) coordinate tuples, e.g. {sample_nodes[0]} "
            f"(x={x}, y={y}). Step 2.7 rounds these to 6 decimals "
            f"(ROUND_DECIMALS) as the permanent node-ID scheme -- if these "
            f"don't look rounded, confirm you're pointing at its real output."
        )
    else:
        print(
            "  -> node IDs are NOT (lon, lat) tuples. Confirm this is the scheme "
            "you expect; downstream facility-snapping code may assume coordinate "
            "node IDs."
        )

    sample_edges = list(G.edges(data=True))[:3]
    print("Sample edge attribute dict(s):")
    for u, v, data in sample_edges:
        print(f"  {u!r} -- {v!r}: {data}")

    # Decide which length attribute to use
    all_attr_keys = set()
    for _, _, data in G.edges(data=True):
        all_attr_keys.update(data.keys())
    print(f"All edge attribute keys found in graph: {sorted(all_attr_keys)}")

    chosen = None
    for candidate in LENGTH_ATTR_CANDIDATES:
        if candidate in all_attr_keys:
            chosen = candidate
            break

    if chosen is None:
        raise SystemExit(
            f"Could not find any of the expected length attributes "
            f"{LENGTH_ATTR_CANDIDATES} on any edge. Found these attribute "
            f"keys instead: {sorted(all_attr_keys)}. A genuine Step 2.7 "
            f"output always sets 'length_km' on every edge -- if that's "
            f"missing, confirm this file really is 2_7_create_final_weighted_graph.py's "
            f"output rather than an intermediate pickle from an earlier step."
        )
    print(f"Using '{chosen}' as each edge's length attribute.")
    print("=" * 70)
    return chosen


def component_length_breakdown(G: nx.Graph, length_attr: str):
    """Return a list of (component_set, n_nodes, n_edges, total_length),
    sorted by total_length descending.

    Works unchanged against an `nx.MultiGraph`: `sub.edges(data=True)`
    yields one tuple per parallel edge, so a component containing two
    source lines sharing the same (start, end) node pair still has both
    lines' lengths summed, not just one.
    """
    components = list(nx.connected_components(G))
    rows = []
    for comp in components:
        sub = G.subgraph(comp)
        total_length = sum(
            data.get(length_attr, 0.0) for _, _, data in sub.edges(data=True)
        )
        rows.append((comp, len(comp), sub.number_of_edges(), total_length))
    rows.sort(key=lambda r: -r[3])
    return rows


def print_and_save_breakdown(rows, length_attr: str, out_csv: Path) -> float:
    total_length_all = sum(r[3] for r in rows)

    print()
    print("=" * 70)
    print(f"COMPONENT LENGTH BREAKDOWN ({len(rows)} component(s))")
    print("=" * 70)
    print(f"{'rank':<6}{'nodes':<10}{'edges':<10}{'length':<15}{'% of total':<12}")
    for i, (comp, n_nodes, n_edges, length) in enumerate(rows, 1):
        pct = 100 * length / total_length_all if total_length_all else 0.0
        print(f"{i:<6}{n_nodes:<10,}{n_edges:<10,}{length:<15,.2f}{pct:<12.2f}")
    print("=" * 70)
    print(f"Total length across all components ({length_attr}): {total_length_all:,.2f}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "n_nodes", "n_edges", f"total_{length_attr}", "pct_of_total"])
        for i, (comp, n_nodes, n_edges, length) in enumerate(rows, 1):
            pct = 100 * length / total_length_all if total_length_all else 0.0
            writer.writerow([i, n_nodes, n_edges, f"{length:.4f}", f"{pct:.4f}"])
    print(f"Full breakdown saved to: {out_csv}")

    return total_length_all


def keep_largest(G: nx.Graph, rows) -> nx.Graph:
    largest_comp = rows[0][0]
    return G.subgraph(largest_comp).copy()


# ---------------------------------------------------------------------------
# Per-network run
# ---------------------------------------------------------------------------

def run_one_network(
    network: str,
    input_path: Path,
    final_dir: Path,
    reports_dir: Path,
    min_largest_pct: float,
) -> bool:
    """Run Step 1.6 end to end for a single network.

    Returns True on success. Returns False (without raising) if the input
    pickle doesn't exist, so a multi-network run can skip this network and
    continue with the other one instead of aborting entirely -- matching
    the "both by default, skip what's missing" convention used everywhere
    else in this pipeline (2_1 through 2_7).
    """
    print()
    print("#" * 70)
    print(f"# NETWORK: {network}")
    print("#" * 70)

    if not input_path.exists():
        print(
            f"  WARNING: input graph pickle not found, skipping this network:\n"
            f"    {input_path}\n"
            f"  (Run 2_7_create_final_weighted_graph.py for this network first -- "
            f"its FINAL_OUTPUT_DIR/f'{{network}}_final.pkl' is exactly what this "
            f"path points at -- or pass --input directly.)"
        )
        return False

    _print_banner(f"STEP 1.6 -- KEEP LARGEST COMPONENT [{network}]")
    print(f"Loading graph from: {input_path}")
    G = load_graph(input_path)

    length_attr = inspect_graph(G)

    n_components_before = nx.number_connected_components(G)
    print(f"\nConnected components found: {n_components_before}")

    rows = component_length_breakdown(G, length_attr)

    out_csv = reports_dir / f"{network}_component_length_breakdown.csv"
    total_length_all = print_and_save_breakdown(rows, length_attr, out_csv)

    largest_pct = 100 * rows[0][3] / total_length_all if total_length_all else 0.0
    if largest_pct < min_largest_pct:
        print(
            f"\nWARNING: largest component holds only {largest_pct:.2f}% of total "
            f"length, below the {min_largest_pct}% target. This suggests a "
            f"real gap may still be missing (per step 1.4/1.6 guidance) -- worth "
            f"a manual look at the next-largest components in "
            f"{out_csv} before discarding them."
        )
    else:
        print(f"\nLargest component holds {largest_pct:.2f}% of total length (meets {min_largest_pct}% target).")

    G_final = keep_largest(G, rows)

    print(f"\nFinal graph after keeping only the largest component ({network}):")
    print(f"  Nodes: {G_final.number_of_nodes():,}")
    print(f"  Edges: {G_final.number_of_edges():,}")
    print(f"  Is connected: {nx.is_connected(G_final)}")

    out_path = final_dir / f"{network}_final_largest.pkl"
    final_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(G_final, f)
    print(f"\nSaved final graph to: {out_path}")
    print("Ready for Step 2 (facility snapping).")

    _print_banner(f"DONE [{network}]")

    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 1.6: compute per-component lengths, show the percentage "
            "breakdown, and keep only the largest component as the final "
            "graph for facility snapping. Runs BOTH highway and railway "
            "automatically unless --network restricts it to one."
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
            "Path to the pickled networkx graph. If omitted, resolved from "
            f"--network via {FINAL_INPUT_DIR}/<network>_final.pkl (Step 2.7's "
            "own output path). Requires --network to be given explicitly, "
            "since an override path only makes sense for a single network "
            "at a time."
        ),
    )
    parser.add_argument(
        "--final-dir",
        type=Path,
        default=FINAL_INPUT_DIR,
        help=f"Folder holding the Step 2.7 graph(s) and where the output is saved (default: {FINAL_INPUT_DIR}).",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help=f"Folder for the CSV length-breakdown report(s) (default: {DEFAULT_REPORTS_DIR}).",
    )
    parser.add_argument(
        "--min-largest-pct",
        type=float,
        default=95.0,
        help="Warn if the largest component holds less than this %% of total length (default: 95.0).",
    )
    args = parser.parse_args()

    if args.input is not None and args.network is None:
        raise SystemExit(
            "--input requires --network (an explicit input path only makes "
            "sense for a single network at a time)."
        )

    return args


def main() -> None:
    args = parse_args()

    if args.network:
        networks_to_run = [args.network]
    else:
        # NEW DEFAULT: both networks, in a fixed/deterministic order --
        # matches 2_1 through 2_7's "both by default" convention.
        networks_to_run = list(NETWORK_LABELS)

    print(f"Running Step 1.6 for: {', '.join(networks_to_run)}")

    ran = []
    skipped = []

    for network in networks_to_run:
        input_path = args.input if args.input is not None else (
            args.final_dir / NETWORK_INPUT_FILENAMES[network]
        )

        ok = run_one_network(
            network=network,
            input_path=input_path,
            final_dir=args.final_dir,
            reports_dir=args.reports_dir,
            min_largest_pct=args.min_largest_pct,
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
            "graph pickle was missing. Run 2_7_create_final_weighted_graph.py "
            "first, or pass --network + --input directly."
        )


if __name__ == "__main__":
    main()