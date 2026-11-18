#!/usr/bin/env python3
"""
Step 6 (shared) -- Repair-analysis common utilities.

Repair analysis is implemented as the reverse of the Step 5 disruption
scripts (7_1_random_disruption.py, 7_2_targeted_disruption.py,
7_3_adaptive_disruption.py): instead of selecting edges to REMOVE, the
repair scripts select removed edges to RESTORE, starting from a fixed
disruption intensity (50% of total network length by default -- "the
50% damaged network") and progressively adding edges back while
re-measuring the same recovery metrics used during disruption.

IMPORTANT ASSUMPTION THIS MODULE MAKES EXPLICIT
-------------------------------------------------
Steps 7_1/7_2/7_3 do NOT persist the actual disrupted graph object or the
specific list of removed edges at any given intensity -- they only write
aggregate per-intensity/per-trial metrics to CSV (that is all the paper's
own figures need). To start repair "from that exact network" at intensity
X%, this module REPLAYS each disruption script's own deterministic
removal procedure up to X% and keeps the resulting graph + removed-edge
list in memory, rather than inventing a different removal method:

  - Random repair replays 7_1's edge-shuffle-and-remove-until-target-km
    procedure for ONE fixed seed -- reproducible, but only one particular
    random 50%-damaged network out of the many trials 7_1 itself explores.
  - Targeted repair replays 7_2's fixed descending-betweenness-centrality
    removal order (loading 7_2's own cached centrality pickle if present,
    else recomputing it), taking the same target-length prefix 7_2's own
    `edges_to_remove_for_intensity` would take.
  - Adaptive repair replays 7_3's recursive recompute-after-every-removal
    procedure up to the target intensity.

If repair analysis becomes a permanent pipeline stage, the cleaner long
-term fix is to have 7_1/7_2/7_3 pickle their disrupted graph + removed-
edge list directly at whatever intensities you care about, so this
replay step can be dropped in favor of just loading a saved object. This
module's replay functions are written so swapping them out later for a
direct load is a small, localized change.

Shared recovery metrics (computed after EVERY restored edge, matching the
disruption scripts' own metrics so degradation and recovery curves are
directly comparable):
  - LCC ratio                         (structural)
  - n_components                       (structural)
  - functionality f(G) = 1/(1+delta_t) (functional)
  - unaffected / delayed / disconnected route counts (functional)

Per the already-diagnosed floating-point bug in the targeted-disruption
trichotomy (Delta_t == 0 failing due to Dijkstra tie-breaking): this
module reuses 7_2/7_3's ALREADY-FIXED `classify_and_score`, which snaps
small negative deltas to exactly 0 before the `np.isclose` check -- so
that bug is not reintroduced here.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as sp_connected_components
from scipy.sparse.csgraph import dijkstra as sp_dijkstra

WEIGHT_ATTR = "travel_time_min"
LENGTH_ATTR = "length_km"

NETWORK_LABELS = ["highway", "railway"]

SMOOTHED_GRAPH_PATH = {n: Path(f"data/graph/smoothing/{n}_smoothed.pkl") for n in NETWORK_LABELS}
BASELINE_MATRIX_PATH = {n: Path(f"data/analysis/{n}_travel_time_matrix.pkl") for n in NETWORK_LABELS}
TARGETED_CENTRALITY_PATH = {n: Path(f"data/analysis/{n}_targeted_centrality.pkl") for n in NETWORK_LABELS}

# "The 50% damaged network" -- the disruption phase's endpoint that
# repair analysis starts from, per the governing task description.
DEFAULT_REPAIR_START_INTENSITY = 50
DEFAULT_SEED = 42

NodeId = Tuple[float, float]
EdgeId = Tuple  # (u, v) for a plain Graph, (u, v, key) for a MultiGraph


def _print_banner(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ---------------------------------------------------------------------------
# Loading -- same contracts as 7_2_targeted_disruption.py's own loaders.
# ---------------------------------------------------------------------------

def load_graph(path: Path) -> Tuple[nx.MultiGraph, List[NodeId]]:
    if not path.exists():
        raise SystemExit(f"Graph not found: {path}\nRun 4_graph_smoothing.py first.")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict) or "graph" not in obj:
        raise SystemExit(f"{path} is not the expected {{'graph','C_county_nodes'}} pickle.")
    G = obj["graph"]
    C = list(obj.get("C_county_nodes", []))
    if not C:
        raise SystemExit(f"Empty facility node set in {path}.")
    if not nx.is_connected(G):
        raise SystemExit(
            "Intact graph is NOT connected. Repair analysis requires the "
            "INTACT (pre-disruption) graph to be connected, since every "
            "removed edge must be restorable to a graph that was "
            "originally whole. Fix Steps 1-4 first."
        )
    return G, C


def build_travel_time_matrix(G, C, weight=WEIGHT_ATTR) -> pd.DataFrame:
    T = pd.DataFrame(index=C, columns=C, dtype=float)
    for c1 in C:
        lengths = nx.single_source_dijkstra_path_length(G, c1, weight=weight)
        T.loc[c1, :] = [lengths.get(c2, np.inf) if c2 != c1 else 0.0 for c2 in C]
    return T


def load_baseline(path: Path, G, C) -> pd.DataFrame:
    """Reuse Step 5's baseline when available (same contract as 7_2/7_3)."""
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
    return T.astype(float)


def total_network_length_km(G) -> float:
    return float(sum(float(d[LENGTH_ATTR]) for _, _, d in G.edges(data=True)))


def upper_triangle_values(df: pd.DataFrame) -> np.ndarray:
    n = df.shape[0]
    iu = np.triu_indices(n, k=1)
    return df.to_numpy(dtype=float)[iu]


# ---------------------------------------------------------------------------
# Classification / functionality -- copy of 7_2/7_3's already-fixed
# version (Dijkstra tie-break epsilon snap included), so disruption and
# repair CSVs stay directly, numerically comparable.
# ---------------------------------------------------------------------------

def classify_and_score(T_baseline: pd.DataFrame, T_disrupted: pd.DataFrame) -> Dict:
    t = upper_triangle_values(T_baseline)
    tp = upper_triangle_values(T_disrupted)

    with np.errstate(divide="ignore", invalid="ignore"):
        delta = (tp - t) / t

    # Snap near-zero deltas from Dijkstra tie-breaking to exactly 0 BEFORE
    # classifying -- fix for the Delta_t==0 floating-point bug in the
    # trichotomy classification.
    delta[np.isfinite(delta) & (delta < 0) & (delta > -1e-10)] = 0.0

    unaffected = np.isclose(delta, 0.0, rtol=0.0, atol=1e-10)
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
# Fast OD matrix + structural fragmentation for whatever G_current
# currently looks like -- generalized from 7_3's compute_od_matrix so
# disruption-replay and repair steps share one implementation.
# ---------------------------------------------------------------------------

def compute_od_matrix_and_structure(G_current: nx.MultiGraph, C: List[NodeId]) -> Tuple[pd.DataFrame, Dict]:
    """Full symmetric C x C travel-time matrix + LCC/component metrics.

    Parallel edges are deduped to the minimum weight per (u, v) pair
    before building the sparse matrix -- csr_matrix otherwise silently
    SUMS duplicate-position weights instead of taking the faster one
    (same fix as 7_1/7_3's own worker code)."""
    nodes = list(G_current.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    n_nodes = len(nodes)

    min_weight_per_pair: Dict[Tuple[int, int], float] = {}
    for u, v, d in G_current.edges(data=True):
        i, j = node_to_idx[u], node_to_idx[v]
        pair = (i, j) if i < j else (j, i)
        w = float(d[WEIGHT_ATTR])
        if pair not in min_weight_per_pair or w < min_weight_per_pair[pair]:
            min_weight_per_pair[pair] = w

    rows, cols, weights = [], [], []
    for (i, j), w in min_weight_per_pair.items():
        rows.extend([i, j])
        cols.extend([j, i])
        weights.extend([w, w])

    if weights:
        A = csr_matrix((weights, (rows, cols)), shape=(n_nodes, n_nodes))
    else:
        A = csr_matrix((n_nodes, n_nodes))

    n_components, labels = sp_connected_components(A, directed=False, connection="weak")
    comp_sizes = np.bincount(labels, minlength=n_components)
    lcc_size = int(comp_sizes.max()) if n_components else 0
    structural = {
        "n_components": int(n_components),
        "lcc_size": lcc_size,
        "lcc_ratio": (lcc_size / n_nodes) if n_nodes else 0.0,
    }

    county_idx = np.array([node_to_idx[c1] for c1 in C], dtype=np.int64)
    distances = sp_dijkstra(A, directed=False, indices=county_idx, return_predecessors=False)
    selected = distances[:, county_idx]
    T = pd.DataFrame(selected, index=C, columns=C, dtype=float)
    return T, structural


def evaluate_state(G_current, C, baseline, total_km) -> Dict:
    """One shared evaluation call used by all three repair scripts after
    every restored edge -- computes the exact same metrics 7_1-7_3 use
    for disruption, so degradation and recovery curves overlay directly."""
    T, structural = compute_od_matrix_and_structure(G_current, C)
    metrics = classify_and_score(baseline, T)
    n_pairs = len(C) * (len(C) - 1) // 2
    return {**metrics, "n_pairs": n_pairs, **structural}


# ---------------------------------------------------------------------------
# Edge helpers -- remove/restore WITH original attributes preserved.
# ---------------------------------------------------------------------------

def edge_data(G, edge_id: EdgeId) -> dict:
    if G.is_multigraph():
        u, v, k = edge_id
        return dict(G[u][v][k])
    u, v = edge_id
    return dict(G[u][v])


def edge_length(G, edge_id: EdgeId) -> float:
    return float(edge_data(G, edge_id)[LENGTH_ATTR])


def has_edge_id(G, edge_id: EdgeId) -> bool:
    if G.is_multigraph():
        u, v, k = edge_id
        return G.has_edge(u, v, k)
    u, v = edge_id
    return G.has_edge(u, v)


def remove_edge_by_id(G, edge_id: EdgeId) -> dict:
    """Remove `edge_id` from G and return its FULL attribute dict, so it
    can be restored later with its exact original attributes."""
    data = edge_data(G, edge_id)
    if G.is_multigraph():
        u, v, k = edge_id
        G.remove_edge(u, v, k)
    else:
        u, v = edge_id
        G.remove_edge(u, v)
    return data


def restore_edge_by_id(G, edge_id: EdgeId, data: dict) -> None:
    """Add `edge_id` back into G with its exact original attributes."""
    if G.is_multigraph():
        u, v, k = edge_id
        G.add_edge(u, v, key=k, **data)
    else:
        u, v = edge_id
        G.add_edge(u, v, **data)


# ---------------------------------------------------------------------------
# Replaying each disruption method up to a target intensity, to reconstruct
# "the exact X% disrupted network" -- see module docstring.
# ---------------------------------------------------------------------------

def build_random_disrupted_state(
    G: nx.MultiGraph, target_pct: float, seed: int = DEFAULT_SEED
) -> Tuple[nx.MultiGraph, List[Tuple[EdgeId, dict]]]:
    """Replay 7_1_random_disruption.py's own removal procedure for a
    single trial at `seed`: shuffle all edges, remove in that order until
    `target_pct`% of total length is removed. Returns (G_disrupted,
    removed_edges), an ORDERED list (removal order) of
    (edge_id, original_attributes)."""
    total_km = total_network_length_km(G)
    target_km = total_km * (target_pct / 100.0)

    all_edges = list(G.edges(keys=True)) if G.is_multigraph() else list(G.edges())
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(all_edges))

    G_disrupted = G.copy()
    removed: List[Tuple[EdgeId, dict]] = []
    removed_km = 0.0

    for idx in order:
        if removed_km >= target_km:
            break
        edge_id = all_edges[idx]
        length_km = edge_length(G_disrupted, edge_id)
        data = remove_edge_by_id(G_disrupted, edge_id)
        removed.append((edge_id, data))
        removed_km += length_km

    print(f"  Random replay (seed={seed}): removed {len(removed):,} edges, "
          f"{removed_km:,.1f} km ({100 * removed_km / total_km:.2f}% of {total_km:,.1f} km).")
    return G_disrupted, removed


def build_targeted_disrupted_state(
    G: nx.MultiGraph, C: List[NodeId], target_pct: float,
    centrality: Optional[Dict[EdgeId, float]] = None,
) -> Tuple[nx.MultiGraph, List[Tuple[EdgeId, dict, float]]]:
    """Replay 7_2_targeted_disruption.py's fixed descending-centrality
    removal order up to `target_pct`% of total length. Returns
    (G_disrupted, removed_edges), an ORDERED list (removal order =
    descending centrality) of (edge_id, original_attributes,
    centrality_at_removal)."""
    if centrality is None:
        print("  No cached centrality supplied -- computing county-to-county "
              "edge betweenness centrality on the intact graph...")
        centrality = nx.edge_betweenness_centrality_subset(
            G, sources=C, targets=C, weight=WEIGHT_ATTR, normalized=False,
        )
        all_edges = list(G.edges(keys=True)) if G.is_multigraph() else list(G.edges())
        for e in all_edges:
            centrality.setdefault(e, 0.0)

    def key(item):
        edge, value = item
        return (-float(value), tuple(repr(x) for x in edge))

    ranked = sorted(centrality.items(), key=key)

    total_km = total_network_length_km(G)
    target_km = total_km * (target_pct / 100.0)

    G_disrupted = G.copy()
    removed: List[Tuple[EdgeId, dict, float]] = []
    removed_km = 0.0

    for edge_id, score in ranked:
        if not has_edge_id(G_disrupted, edge_id):
            continue
        length_km = edge_length(G_disrupted, edge_id)
        if removed and removed_km + length_km > target_km:
            break
        data = remove_edge_by_id(G_disrupted, edge_id)
        removed.append((edge_id, data, float(score)))
        removed_km += length_km
        if removed_km >= target_km:
            break

    print(f"  Targeted replay: removed {len(removed):,} edges, "
          f"{removed_km:,.1f} km ({100 * removed_km / total_km:.2f}% of {total_km:,.1f} km).")
    return G_disrupted, removed


def build_adaptive_disrupted_state(
    G: nx.MultiGraph, C: List[NodeId], target_pct: float,
) -> Tuple[nx.MultiGraph, List[Tuple[EdgeId, dict, float]]]:
    """Replay 7_3_adaptive_disruption.py's batch-adaptive removal procedure
    up to `target_pct`% of total length. Recomputes betweenness once per
    ~1% of network length (matching 7_3's own batch approach), NOT after
    every single edge removal. Returns (G_disrupted, removed_edges), an
    ORDERED list (actual removal order) of (edge_id, original_attributes,
    centrality_at_removal)."""
    total_km = total_network_length_km(G)
    target_km = total_km * (target_pct / 100.0)
    batch_km_target = total_km * 0.01  # ~1% of total per ranking recompute

    # Set up progress bar (tqdm if available, else fallback to prints)
    try:
        from tqdm import tqdm
        bar = tqdm(total=target_km, desc="Adaptive disruption replay",
                   unit="km")
        bar.set_postfix_str("0")
    except ImportError:
        bar = None

    G_current = G.copy()
    removed: List[Tuple[EdgeId, dict, float]] = []
    removed_km = 0.0
    batch_num = 0

    while removed_km < target_km:
        batch_num += 1
        # Recompute ranking once for this batch
        centrality = nx.edge_betweenness_centrality_subset(
            G_current, sources=C, targets=C, weight=WEIGHT_ATTR, normalized=False,
        )
        if not centrality:
            if bar is not None:
                bar.write("  WARNING: no edges left to remove during adaptive replay -- stopping early.")
            else:
                print("  WARNING: no edges left to remove during adaptive replay -- stopping early.")
            break

        if bar is not None:
            bar.set_postfix_str(str(batch_num))

        # Sort all edges by centrality descending
        ranked = sorted(
            centrality.items(),
            key=lambda kv: (-kv[1], tuple(repr(x) for x in kv[0])),
        )

        # Remove edges from this ranking until we've covered ~1% of total km
        # or reached the overall target
        batch_km = 0.0
        for edge_id, score in ranked:
            if removed_km >= target_km:
                break
            if batch_km >= batch_km_target:
                break  # recompute ranking for next batch
            if not has_edge_id(G_current, edge_id):
                continue
            length_km = edge_length(G_current, edge_id)
            data = remove_edge_by_id(G_current, edge_id)
            removed.append((edge_id, data, float(score)))
            removed_km += length_km
            batch_km += length_km
            if bar is not None:
                bar.update(length_km)

        if bar is None and batch_num % 10 == 0:
            print(f"    Adaptive replay batch {batch_num}: {removed_km:,.1f}/{target_km:,.1f} km "
                  f"({len(removed):,} edges removed)")

    if bar is not None:
        bar.close()

    print(f"  Adaptive replay: removed {len(removed):,} edges in {batch_num} batches, "
          f"{removed_km:,.1f} km ({100 * removed_km / total_km:.2f}% of {total_km:,.1f} km).")
    return G_current, removed


def load_cached_targeted_centrality(path: Path, G, expect_edges=True) -> Optional[Dict[EdgeId, float]]:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        centrality = pickle.load(f)
    if expect_edges:
        graph_edges = set(G.edges(keys=True)) if G.is_multigraph() else set(G.edges())
        if set(centrality.keys()) != graph_edges:
            print(f"  Cached centrality at {path} doesn't match the intact graph -- recomputing.")
            return None
    return centrality