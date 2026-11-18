#!/usr/bin/env python3
"""
Step 5b -- Targeted disruption (fixed centrality ranking), Illinois test run.

DATASET NOTE (updated to match the Illinois test run's naming convention):
-----------------------------------------------------------------------
The earlier draft of this script ranked and removed edges by fort-to-port
betweenness centrality (disjoint sets O_fort_nodes/D_port_nodes) against
a single hardcoded STRAHNET or STRACNET graph. Neither those two disjoint
facility sets nor those graph paths exist in this pipeline. This script
has been rewritten to match what this project actually produces:

  * Facility set: ONE set of COUNTY nodes, `C`, not two disjoint sets
    `O`/`D`. Traced directly to `4_graph_smoothing.py`'s own output dict
    shape (`{"graph": G_smoothed, "C_county_nodes": sorted(C)}`).
    `nx.edge_betweenness_centrality_subset` is called with BOTH `sources`
    and `targets` set to `C` (matching `6_edge_betweenness.py`'s own
    approach) -- a source/target pair that is the same county
    contributes nothing, so no special-casing is needed.

  * Because the county-to-county travel-time matrix is SYMMETRIC (one
    facility set, undirected graph) with a zero diagonal, every
    delay/functionality statistic is computed over the `N = C(|C|, 2)`
    UNIQUE unordered pairs (the upper triangle), not the full C x C
    matrix -- matching `5_travel_time_matrix.py`'s and
    `3_snap_counties.py`'s own bookkeeping. Naively running the old
    O x D-style calculation over a symmetric C x C matrix would have
    double-counted every route and mixed in C spurious zero
    self-distances.

  * Graph/baseline input: traced directly to `4_graph_smoothing.py`'s
    own output convention (`data/graph/smoothing/<network>_smoothed.pkl`)
    and `5_travel_time_matrix.py`'s own baseline output
    (`data/analysis/<network>_travel_time_matrix.pkl`) -- BOTH "highway"
    and "railway", not "strahnet"/"stracnet". Following the "both
    networks by default" convention already established by every other
    script in this pipeline, `--network` is OPTIONAL: omitting it runs
    the full sweep for BOTH graphs automatically, one after another,
    each with its own outputs. A missing smoothed-graph pickle for one
    network prints a clear warning and skips just that network.

Everything else -- the methodology -- is UNCHANGED:

  1. Validate graph + county nodes.
  2. Compute (or load cached) county-to-county edge betweenness
     centrality on the INTACT graph.
  3. Fix a descending-centrality removal order ONCE -- it is never
     recomputed after an edge is removed (that's adaptive disruption,
     see 7_3_adaptive_disruption.py).
  4. For each intensity 1-50%, follow the fixed ranking until the target
     cumulative length is reached, recompute travel times, classify
     delay, and score functionality.
  5. Save CSV/pickle results, plus a determinism/monotonicity validation
     pass.

Performance notes (unchanged): SciPy sparse Dijkstra evaluates all county
origins in one call per intensity; results are written after every
completed intensity; `--resume` skips already-completed intensities;
MultiGraph edge keys are preserved for centrality/removal.

Input (per network)
--------------------
    data/graph/smoothing/<network>_smoothed.pkl
    data/analysis/<network>_travel_time_matrix.pkl   (rebuilt once if missing)

Output (per network)
----------------------
    reports/<network>_targeted_disruption.csv
        One row per intensity: intensity_pct, n_edges_removed,
        removed_length_km, actual_intensity_pct, n_edges_total,
        n_unaffected, n_delayed, n_disconnected, frac_unaffected,
        frac_delayed, frac_disconnected, functionality, n_pairs,
        n_components, lcc_size, lcc_ratio, elapsed_s.

        n_components / lcc_size / lcc_ratio are STRUCTURAL fragmentation
        metrics (connected components / largest connected component),
        computed via scipy.sparse.csgraph.connected_components on the same
        disrupted sparse adjacency matrix used for the Dijkstra travel-time
        calculation -- no separate graph rebuild. Distinct from
        functionality (a FUNCTIONAL metric: does a route still exist and
        how much slower is it) -- LCC only asks "is the network still
        physically one piece, and how much of it". Because targeted
        disruption removes a strictly growing prefix of one FIXED ranking
        as intensity rises, n_components is guaranteed non-decreasing and
        lcc_ratio is guaranteed non-increasing across a run -- both are
        checked in validate_results().
    data/analysis/<network>_targeted_disruption.pkl
    data/analysis/<network>_targeted_centrality.pkl
    reports/<network>_targeted_disruption_summary.csv

Usage
-----
    python 7_2_targeted_disruption.py                     # BOTH networks
    python 7_2_targeted_disruption.py --network highway
    python 7_2_targeted_disruption.py --network railway --resume
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as sp_connected_components
from scipy.sparse.csgraph import dijkstra as sp_dijkstra

# ---------------------------------------------------------------------------
# PATHS -- matching 4_graph_smoothing.py's / 5_travel_time_matrix.py's own
# output conventions.
# ---------------------------------------------------------------------------

NETWORK_LABELS = ["highway", "railway"]

NETWORK_CONFIG = {
    network: {
        "graph": Path(f"data/graph/smoothing/{network}_smoothed.pkl"),
        "baseline": Path(f"data/analysis/{network}_travel_time_matrix.pkl"),
        "output_csv": Path(f"reports/{network}_targeted_disruption.csv"),
        "output_pickle": Path(f"data/analysis/{network}_targeted_disruption.pkl"),
        "centrality_pickle": Path(f"data/analysis/{network}_targeted_centrality.pkl"),
        "summary_csv": Path(f"reports/{network}_targeted_disruption_summary.csv"),
    }
    for network in NETWORK_LABELS
}

WEIGHT_ATTR = "travel_time_min"
LENGTH_ATTR = "length_km"

DEFAULT_MIN_INTENSITY = 1
DEFAULT_MAX_INTENSITY = 50
DEFAULT_STEP = 1

# HARDCODED DEFAULT NETWORK -- same pattern as every other script in this
# pipeline. Only a safety fallback; --network is optional and defaults to
# "both".
HARDCODED_NETWORK = "highway"

NodeId = Tuple[float, float]
EdgeId = Tuple


def _print_banner(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ---------------------------------------------------------------------------
# Loading / validation
# ---------------------------------------------------------------------------

def load_graph(path: Path):
    if not path.exists():
        raise SystemExit(
            f"Graph not found: {path}\n"
            "Run 4_graph_smoothing.py first."
        )

    with open(path, "rb") as f:
        obj = pickle.load(f)

    if not isinstance(obj, dict) or "graph" not in obj:
        raise SystemExit(
            f"{path} is not the expected {{'graph','C_county_nodes'}} pickle."
        )

    G = obj["graph"]
    C = list(obj.get("C_county_nodes", []))

    if not isinstance(G, (nx.Graph, nx.MultiGraph)):
        raise SystemExit(f"Unsupported graph type: {type(G).__name__}")

    if not C:
        raise SystemExit(f"Empty facility node set: counties={len(C)}")

    if not nx.is_connected(G):
        raise SystemExit(
            "Intact graph is NOT connected. Targeted disruption requires "
            "the intact graph to be connected. Fix Steps 1-4 first."
        )

    missing = [n for n in C if n not in G]
    if missing:
        raise SystemExit(
            f"{len(missing)} county node(s) are missing from the graph. "
            "Re-run facility snapping/smoothing against the current graph."
        )

    bad_weight = 0
    bad_length = 0
    for _, _, d in G.edges(data=True):
        if not np.isfinite(float(d.get(WEIGHT_ATTR, np.nan))) or float(d.get(WEIGHT_ATTR, 0)) <= 0:
            bad_weight += 1
        if not np.isfinite(float(d.get(LENGTH_ATTR, np.nan))) or float(d.get(LENGTH_ATTR, 0)) <= 0:
            bad_length += 1

    if bad_weight or bad_length:
        raise SystemExit(
            f"Invalid edge attributes: {bad_weight} edges have invalid "
            f"{WEIGHT_ATTR}; {bad_length} edges have invalid {LENGTH_ATTR}."
        )

    n_pairs = len(C) * (len(C) - 1) // 2
    print(f"Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    print(f"County nodes (C): {len(C):,}   unique routes N = C(|C|, 2) = {n_pairs:,}")
    print(f"Graph connected: {nx.is_connected(G)}")

    return G, C


def build_travel_time_matrix(G, C, weight=WEIGHT_ATTR) -> pd.DataFrame:
    """Full symmetric C x C matrix, diagonal 0."""
    T = pd.DataFrame(index=C, columns=C, dtype=float)
    for c1 in C:
        lengths = nx.single_source_dijkstra_path_length(G, c1, weight=weight)
        T.loc[c1, :] = [lengths.get(c2, np.inf) if c2 != c1 else 0.0 for c2 in C]
    return T


def load_baseline(path: Path, G, C) -> pd.DataFrame:
    """Reuse Step 5's baseline when available."""
    if not path.exists():
        print(f"Baseline not found at {path}; rebuilding it once...")
        return build_travel_time_matrix(G, C)

    with open(path, "rb") as f:
        obj = pickle.load(f)

    if not isinstance(obj, dict) or "matrix" not in obj:
        raise SystemExit(f"Invalid baseline pickle: {path}")

    T = obj["matrix"]
    saved_C = list(obj.get("C_county_nodes", []))

    if saved_C != C:
        raise SystemExit(
            "Baseline C_county_nodes does not match the current graph's "
            "county list. Re-run 5_travel_time_matrix.py first."
        )

    n = T.shape[0]
    iu = np.triu_indices(n, k=1)
    if np.isinf(T.to_numpy(dtype=float)[iu]).any():
        raise SystemExit("Baseline matrix contains inf values among unique county pairs.")

    print(f"Loaded baseline matrix: {T.shape[0]:,} x {T.shape[1]:,}")
    return T.astype(float)


def upper_triangle_values(df: pd.DataFrame) -> np.ndarray:
    """Extract the unique (i < j) unordered-pair values from a symmetric
    C x C DataFrame -- the N = C(|C|, 2) unique county-to-county routes."""
    n = df.shape[0]
    iu = np.triu_indices(n, k=1)
    return df.to_numpy(dtype=float)[iu]


# ---------------------------------------------------------------------------
# Centrality
# ---------------------------------------------------------------------------

def compute_county_centrality(G, C) -> Dict[EdgeId, float]:
    """
    c_B(e) = sum_{c1,c2 in C} sigma(c1,c2|e) / sigma(c1,c2)

    Restricted to county-to-county pairs, weighted by travel_time_min.
    Passing the same set `C` for sources and targets matches
    6_edge_betweenness.py's own approach -- same-node pairs contribute
    zero automatically.
    """
    print("\nComputing county-to-county edge betweenness centrality...")
    t0 = time.monotonic()

    centrality = nx.edge_betweenness_centrality_subset(
        G, sources=C, targets=C, weight=WEIGHT_ATTR, normalized=False,
    )

    if G.is_multigraph():
        all_edges = list(G.edges(keys=True))
    else:
        all_edges = list(G.edges())

    for e in all_edges:
        centrality.setdefault(e, 0.0)

    nx.set_edge_attributes(G, centrality, "county_centrality")

    elapsed = time.monotonic() - t0
    nonzero = sum(v > 0 for v in centrality.values())
    print(
        f"Centrality complete in {elapsed:,.1f}s | "
        f"edges={len(centrality):,} | nonzero={nonzero:,}"
    )
    return centrality


def load_or_compute_centrality(path: Path, G, C, force=False):
    if path.exists() and not force:
        print(f"Loading cached centrality: {path}")
        with open(path, "rb") as f:
            centrality = pickle.load(f)

        graph_edges = set(G.edges(keys=True)) if G.is_multigraph() else set(G.edges())
        if set(centrality.keys()) != graph_edges:
            print("Cached centrality does not match current graph; recomputing.")
        else:
            nx.set_edge_attributes(G, centrality, "county_centrality")
            print(f"Loaded {len(centrality):,} centrality values.")
            return centrality

    centrality = compute_county_centrality(G, C)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(centrality, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved centrality: {path}")
    return centrality


# ---------------------------------------------------------------------------
# Fixed removal order
# ---------------------------------------------------------------------------

def build_removal_order(centrality, G):
    """Stable deterministic descending-centrality order."""
    def key(item):
        edge, value = item
        return (-float(value), tuple(repr(x) for x in edge))

    ranked = sorted(centrality.items(), key=key)
    return ranked


def total_network_length_km(G) -> float:
    return float(sum(float(d[LENGTH_ATTR]) for _, _, d in G.edges(data=True)))


def edges_to_remove_for_intensity(ranked_edges, G, intensity_pct, total_len_km):
    """Follow the fixed ranking until the target length is reached, without
    overshooting when at least one edge has already been selected."""
    target = total_len_km * intensity_pct / 100.0
    removed = []
    cumulative = 0.0

    for edge, _centrality in ranked_edges:
        if G.is_multigraph():
            u, v, k = edge
            edge_len = float(G[u][v][k][LENGTH_ATTR])
        else:
            u, v = edge
            edge_len = float(G[u][v][LENGTH_ATTR])

        if removed and cumulative + edge_len > target:
            break

        removed.append(edge)
        cumulative += edge_len

        if cumulative >= target:
            break

    return removed, cumulative


# ---------------------------------------------------------------------------
# Fast disrupted shortest paths
# ---------------------------------------------------------------------------

def prepare_sparse_edges(G):
    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    if G.is_multigraph():
        edges = list(G.edges(keys=True, data=True))
        rows = np.fromiter((node_to_idx[u] for u, v, k, d in edges), dtype=np.int64)
        cols = np.fromiter((node_to_idx[v] for u, v, k, d in edges), dtype=np.int64)
        weights = np.fromiter((float(d[WEIGHT_ATTR]) for u, v, k, d in edges), dtype=float)
        edge_ids = [(u, v, k) for u, v, k, d in edges]
    else:
        edges = list(G.edges(data=True))
        rows = np.fromiter((node_to_idx[u] for u, v, d in edges), dtype=np.int64)
        cols = np.fromiter((node_to_idx[v] for u, v, d in edges), dtype=np.int64)
        weights = np.fromiter((float(d[WEIGHT_ATTR]) for u, v, d in edges), dtype=float)
        edge_ids = [(u, v) for u, v, d in edges]

    return nodes, node_to_idx, edge_ids, rows, cols, weights


def disrupted_od_matrix(
    node_to_idx, edge_ids, rows, cols, weights, removed_edges, C, n_nodes,
):
    """One SciPy multi-source Dijkstra call for ALL counties at this
    intensity -> full symmetric C x C matrix, PLUS structural fragmentation
    (connected components / largest connected component) computed on the
    same disrupted sparse adjacency matrix -- no separate graph rebuild.

    Returns:
        A tuple of (T_disrupted, structural) where T_disrupted is the
        C x C travel-time DataFrame (unchanged shape/meaning from before),
        and structural is a dict:
            {"n_components": int, "lcc_size": int, "lcc_ratio": float}
        lcc_ratio is the largest component's node count divided by
        n_nodes -- the graph's TOTAL node count, which never changes in
        this script (only edges are removed, never nodes), so lcc_ratio
        is directly comparable across every intensity.
    """
    removed = set(removed_edges)
    keep = np.fromiter(
        (edge_id not in removed for edge_id in edge_ids),
        dtype=bool,
        count=len(edge_ids),
    )

    r = np.concatenate([rows[keep], cols[keep]])
    c = np.concatenate([cols[keep], rows[keep]])
    w = np.concatenate([weights[keep], weights[keep]])

    A = csr_matrix((w, (r, c)), shape=(n_nodes, n_nodes))

    # Structural fragmentation -- graph structure only (edge weights don't
    # matter for connectivity). An isolated node (e.g. a county that lost
    # every incident edge) counts as its own size-1 component, correctly.
    n_components, labels = sp_connected_components(A, directed=False, connection="weak")
    comp_sizes = np.bincount(labels, minlength=n_components)
    lcc_size = int(comp_sizes.max())
    structural = {
        "n_components": int(n_components),
        "lcc_size": lcc_size,
        "lcc_ratio": lcc_size / n_nodes,
    }

    county_idx = np.asarray([node_to_idx[c1] for c1 in C], dtype=np.int64)

    distances = sp_dijkstra(A, directed=False, indices=county_idx, return_predecessors=False)

    selected = distances[:, county_idx]
    T_disrupted = pd.DataFrame(selected, index=C, columns=C, dtype=float)
    return T_disrupted, structural


# ---------------------------------------------------------------------------
# Classification / functionality
# ---------------------------------------------------------------------------

def classify_and_score(T_baseline, T_disrupted):
    """Computed over the UNIQUE upper-triangle county-to-county pairs
    only, since the matrix is symmetric with a zero diagonal."""
    t = upper_triangle_values(T_baseline)
    tp = upper_triangle_values(T_disrupted)

    with np.errstate(divide="ignore", invalid="ignore"):
        delta = (tp - t) / t

    delta[np.isfinite(delta) & (delta < 0) & (delta > -1e-10)] = 0.0

    unaffected = delta <= 0.1
    disconnected = np.isinf(tp)
    delayed = ~unaffected & ~disconnected

    functionality = 1.0 / (1.0 + delta)
    functionality = np.nan_to_num(functionality, nan=1.0, posinf=0.0, neginf=0.0)

    n = delta.size
    return {
        "n_unaffected": int(unaffected.sum()),
        "n_delayed": int(delayed.sum()),
        "n_disconnected": int(disconnected.sum()),
        "frac_unaffected": float(unaffected.sum() / n),
        "frac_delayed": float(delayed.sum() / n),
        "frac_disconnected": float(disconnected.sum() / n),
        "functionality": float(np.mean(functionality)),
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_results(summary, total_len_km, ranked_edges, G):
    required = [
        "frac_unaffected", "frac_delayed", "frac_disconnected", "functionality",
        "n_components", "lcc_size", "lcc_ratio",
    ]
    for col in required:
        if col not in summary:
            raise AssertionError(f"Missing result column: {col}")

    category_sum = (
        summary["frac_unaffected"] + summary["frac_delayed"] + summary["frac_disconnected"]
    )
    if not np.allclose(category_sum.to_numpy(), 1.0, atol=1e-12):
        raise AssertionError("Path-category fractions do not sum to 1.")

    disconnected_diff = summary["frac_disconnected"].diff().dropna()
    if (disconnected_diff < -1e-10).any():
        raise AssertionError("Disconnected fraction decreased with intensity.")

    functionality_diff = summary["functionality"].diff().dropna()
    if (functionality_diff > 1e-10).any():
        raise AssertionError("Functionality increased with intensity.")

    # Targeted disruption follows a FIXED ranking, and at each intensity
    # takes a growing prefix of it (higher intensity = larger cumulative
    # length target = more edges removed from the same ordered list, a
    # superset of every lower intensity's removed set). So the graph only
    # ever gets more disrupted as intensity rises within one run -- meaning
    # component count must be non-decreasing and LCC ratio must be
    # non-increasing. A violation means edges_to_remove_for_intensity isn't
    # actually producing a growing prefix, which would be a real bug.
    components_diff = summary["n_components"].diff().dropna()
    if (components_diff < 0).any():
        raise AssertionError("Component count decreased with intensity.")

    lcc_diff = summary["lcc_ratio"].diff().dropna()
    if (lcc_diff > 1e-10).any():
        raise AssertionError("LCC ratio increased with intensity.")

    if not ranked_edges:
        raise AssertionError("Removal ranking is empty.")

    for edge, _ in ranked_edges:
        if G.is_multigraph():
            u, v, k = edge
            assert G.has_edge(u, v, k)
        else:
            u, v = edge
            assert G.has_edge(u, v)

    print("Validation passed:")
    print("  category fractions sum to 1")
    print("  disconnected fraction is non-decreasing")
    print("  functionality is non-increasing")
    print("  removal ranking is valid")


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep(
    G, C, baseline, ranked_edges, total_len_km, intensities, output_csv, resume=False,
):
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    nodes, node_to_idx, edge_ids, rows, cols, weights = prepare_sparse_edges(G)
    n_nodes = len(nodes)
    n_pairs = len(C) * (len(C) - 1) // 2

    fields = [
        "intensity_pct", "n_edges_removed", "removed_length_km",
        "actual_intensity_pct", "n_edges_total",
        "n_unaffected", "n_delayed", "n_disconnected",
        "frac_unaffected", "frac_delayed", "frac_disconnected",
        "functionality", "n_pairs",
        "n_components", "lcc_size", "lcc_ratio",
        "elapsed_s",
    ]

    existing = {}
    if resume and output_csv.exists():
        old = pd.read_csv(output_csv)
        if not old.empty:
            existing = {
                int(row["intensity_pct"]): row.to_dict()
                for _, row in old.iterrows()
            }
        print(f"Resume: {len(existing)}/{len(intensities)} intensity levels already done.")

    mode = "a" if resume and output_csv.exists() else "w"

    with open(output_csv, mode, newline="") as f:
        if mode == "w":
            f.write(",".join(fields) + "\n")

        for intensity in intensities:
            if intensity in existing:
                continue

            t0 = time.monotonic()

            removed, removed_km = edges_to_remove_for_intensity(
                ranked_edges, G, intensity, total_len_km
            )

            T_disrupted, structural = disrupted_od_matrix(
                node_to_idx, edge_ids, rows, cols, weights, removed, C, n_nodes,
            )

            metrics = classify_and_score(baseline, T_disrupted)
            elapsed = time.monotonic() - t0
            actual_pct = 100.0 * removed_km / total_len_km

            row = {
                "intensity_pct": intensity,
                "n_edges_removed": len(removed),
                "removed_length_km": removed_km,
                "actual_intensity_pct": actual_pct,
                "n_edges_total": len(edge_ids),
                **metrics,
                "n_pairs": n_pairs,
                **structural,
                "elapsed_s": elapsed,
            }

            pd.DataFrame([row]).to_csv(f, header=False, index=False)
            f.flush()

            print(
                f"  {intensity:2d}% | "
                f"edges={len(removed):,} | "
                f"actual={actual_pct:6.3f}% | "
                f"disc={metrics['frac_disconnected']*100:6.2f}% | "
                f"f(G)={metrics['functionality']:.5f} | "
                f"LCC={structural['lcc_ratio']:.4f} | "
                f"comps={structural['n_components']:,} | "
                f"{elapsed:,.2f}s"
            )

            existing[intensity] = row

    summary = pd.DataFrame(list(existing.values())).sort_values("intensity_pct")
    summary = summary.set_index("intensity_pct")

    return summary


# ---------------------------------------------------------------------------
# Determinism check
# ---------------------------------------------------------------------------

def determinism_check(G, C, baseline, ranked_edges, total_len_km, intensity):
    nodes, node_to_idx, edge_ids, rows, cols, weights = prepare_sparse_edges(G)

    removed_a, km_a = edges_to_remove_for_intensity(ranked_edges, G, intensity, total_len_km)
    removed_b, km_b = edges_to_remove_for_intensity(ranked_edges, G, intensity, total_len_km)

    if removed_a != removed_b or km_a != km_b:
        raise AssertionError("Removal order is not deterministic.")

    Ta, structural_a = disrupted_od_matrix(node_to_idx, edge_ids, rows, cols, weights, removed_a, C, len(nodes))
    Tb, structural_b = disrupted_od_matrix(node_to_idx, edge_ids, rows, cols, weights, removed_b, C, len(nodes))

    if not np.allclose(Ta.to_numpy(), Tb.to_numpy(), equal_nan=True):
        raise AssertionError("Repeated targeted run produced different travel times.")

    if structural_a != structural_b:
        raise AssertionError(
            f"Repeated targeted run produced different structural fragmentation: "
            f"{structural_a} vs {structural_b}."
        )

    print(f"Determinism check passed at {intensity}% (travel times and LCC/component count both match).")


# ---------------------------------------------------------------------------
# Per-network run
# ---------------------------------------------------------------------------

def run_one_network(network: str, args: argparse.Namespace, intensities: List[int]) -> bool:
    cfg = NETWORK_CONFIG[network]

    print()
    print("#" * 70)
    print(f"# NETWORK: {network}")
    print("#" * 70)

    if not cfg["graph"].exists():
        print(
            f"  WARNING: smoothed graph pickle not found, skipping this network:\n"
            f"    {cfg['graph']}\n"
            f"  (Run 4_graph_smoothing.py for this network first.)"
        )
        return False

    _print_banner(f"STEP 5b -- TARGETED DISRUPTION [{network}]")
    print(f"Graph:         {cfg['graph']}")
    print(f"Baseline:      {cfg['baseline']}")
    print(f"Intensities:   {intensities[0]}%..{intensities[-1]}% step {args.step}%")
    print(f"Output CSV:    {cfg['output_csv']}")
    print(f"Output pickle: {cfg['output_pickle']}")
    print(f"Centrality:    {cfg['centrality_pickle']}")

    G, C = load_graph(cfg["graph"])

    baseline = load_baseline(cfg["baseline"], G, C)
    if baseline.shape != (len(C), len(C)):
        raise SystemExit("Baseline matrix shape does not match C x C.")

    n = baseline.shape[0]
    iu = np.triu_indices(n, k=1)
    if np.isinf(baseline.to_numpy()[iu]).any():
        raise SystemExit("Baseline contains unreachable county-to-county pairs.")

    total_len_km = total_network_length_km(G)
    print(f"Total network length: {total_len_km:,.3f} km")

    centrality = load_or_compute_centrality(
        cfg["centrality_pickle"], G, C, force=args.force_centrality,
    )

    ranked_edges = build_removal_order(centrality, G)

    print("\nTop 10 targeted edges:")
    for edge, value in ranked_edges[:10]:
        if G.is_multigraph():
            u, v, k = edge
            length = G[u][v][k][LENGTH_ATTR]
        else:
            u, v = edge
            length = G[u][v][LENGTH_ATTR]
        print(f"  {edge} | cB={value:.3f} | length={length:.3f} km")

    if not args.skip_determinism_check:
        determinism_check(G, C, baseline, ranked_edges, total_len_km, intensities[0])

    print("\nRunning targeted sweep...")
    summary = run_sweep(
        G, C, baseline, ranked_edges, total_len_km, intensities,
        cfg["output_csv"], resume=args.resume,
    )

    validate_results(summary, total_len_km, ranked_edges, G)

    cfg["summary_csv"].parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(cfg["summary_csv"])

    cfg["output_pickle"].parent.mkdir(parents=True, exist_ok=True)
    with open(cfg["output_pickle"], "wb") as f:
        pickle.dump(summary, f, protocol=pickle.HIGHEST_PROTOCOL)

    _print_banner(f"TARGETED DISRUPTION COMPLETE [{network}]")
    print(f"Summary CSV:     {cfg['summary_csv']}")
    print(f"Results pickle:  {cfg['output_pickle']}")

    print("\nEndpoint results:")
    for intensity in [summary.index.min(), summary.index.max()]:
        r = summary.loc[intensity]
        print(
            f"  {intensity}%: actual={r['actual_intensity_pct']:.3f}% | "
            f"disconnected={r['frac_disconnected']:.3%} | "
            f"functionality={r['functionality']:.5f} | "
            f"LCC ratio={r['lcc_ratio']:.4f} | "
            f"components={int(r['n_components']):,}"
        )

    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Deterministic targeted disruption (fixed centrality "
                    "ranking) for the smoothed HIGHWAY and RAILWAY graphs. "
                    "Runs BOTH networks automatically unless --network "
                    "restricts it to one."
    )
    p.add_argument(
        "--network", choices=sorted(NETWORK_LABELS), default=None,
        help=(
            "Restrict the run to just this one network. If omitted, BOTH "
            "highway and railway are processed automatically (matching "
            "every other script in this pipeline)."
        ),
    )
    p.add_argument("--min-intensity", type=int, default=DEFAULT_MIN_INTENSITY)
    p.add_argument("--max-intensity", type=int, default=DEFAULT_MAX_INTENSITY)
    p.add_argument("--step", type=int, default=DEFAULT_STEP)
    p.add_argument(
        "--resume", action="store_true",
        help="Continue an existing CSV instead of overwriting it.",
    )
    p.add_argument(
        "--force-centrality", action="store_true",
        help="Ignore cached centrality and recompute it.",
    )
    p.add_argument("--skip-determinism-check", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.min_intensity < 1 or args.max_intensity > 50:
        raise SystemExit("Intensity must be within 1..50% for the paper reproduction.")
    if args.min_intensity > args.max_intensity:
        raise SystemExit("--min-intensity cannot exceed --max-intensity.")
    if args.step <= 0:
        raise SystemExit("--step must be positive.")

    intensities = list(range(args.min_intensity, args.max_intensity + 1, args.step))

    if args.network:
        networks_to_run = [args.network]
    else:
        networks_to_run = list(NETWORK_LABELS)

    print(f"Running Step 5b (targeted disruption) for: {', '.join(networks_to_run)}")

    ran = []
    skipped = []
    for network in networks_to_run:
        ok = run_one_network(network, args, intensities)
        (ran if ok else skipped).append(network)

    print()
    _print_banner("ALL NETWORKS COMPLETE")
    print(f"Ran successfully: {ran or '(none)'}")
    if skipped:
        print(f"Skipped (smoothed graph pickle not found): {skipped}")
    if not ran:
        raise SystemExit(
            "\nNo network was actually processed -- every requested "
            "smoothed graph pickle was missing. Run 4_graph_smoothing.py "
            "first, or pass --network directly."
        )


if __name__ == "__main__":
    main()