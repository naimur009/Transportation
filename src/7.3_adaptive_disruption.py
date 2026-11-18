#!/usr/bin/env python3
"""
Step 5c -- Adaptive (recursive-betweenness) targeted disruption, Illinois
test run.

DATASET NOTE (updated to match the Illinois test run's naming convention):
-----------------------------------------------------------------------
The earlier draft of this script recomputed fort-to-port betweenness
(disjoint sets O_fort_nodes/D_port_nodes) after every single edge removal,
against a single hardcoded STRAHNET or STRACNET graph. Neither those two
disjoint facility sets nor those graph paths exist in this pipeline. This
script has been rewritten to match what this project actually produces:

  * Facility set: ONE set of COUNTY nodes, `C`, not two disjoint sets
    `O`/`D`. Traced directly to `4_graph_smoothing.py`'s own output dict
    shape (`{"graph": G_smoothed, "C_county_nodes": sorted(C)}`).
    `nx.edge_betweenness_centrality_subset` is called each time with BOTH
    `sources` and `targets` set to `C`, matching `6_edge_betweenness.py`'s
    and `7_2_targeted_disruption.py`'s own approach.

  * Because the county-to-county travel-time matrix is SYMMETRIC (one
    facility set, undirected graph) with a zero diagonal, every
    delay/functionality statistic is computed over the `N = C(|C|, 2)`
    UNIQUE unordered pairs (upper triangle), matching
    `7_2_targeted_disruption.py`'s own `classify_and_score` -- this
    script's copy of that function is identical for exactly that reason
    (byte-for-byte comparable CSVs).

  * Graph/baseline input: traced directly to `4_graph_smoothing.py`'s
    own output convention and `5_travel_time_matrix.py`'s own baseline
    output -- BOTH "highway" and "railway", not "strahnet"/"stracnet".
    Following the "both networks by default" convention already
    established by every other script in this pipeline, `--network` is
    OPTIONAL: omitting it runs the full sweep for BOTH graphs
    automatically, one after another. A missing smoothed-graph pickle for
    one network prints a clear warning and skips just that network.

PATCH (LCC / structural fragmentation metrics -- brings this script in
line with 7_1_random_disruption.py and 7_2_targeted_disruption.py):
-----------------------------------------------------------------------
`compute_od_matrix` now ALSO computes `n_components`, `lcc_size`, and
`lcc_ratio` on the SAME disrupted sparse adjacency matrix already built
for the Dijkstra travel-time call -- no separate graph rebuild, no extra
pass over the edge list. These are STRUCTURAL fragmentation metrics ("is
the network still physically one piece, and how much of it") distinct
from `functionality` (a FUNCTIONAL metric: does a route still exist and
how much slower is it) -- the two can diverge, e.g. LCC ratio can stay at
100% while functionality drops because surviving routes got longer, or
LCC can crater while functionality still looks moderate because most
county pairs happen to sit in the surviving largest piece.

`lcc_ratio`'s denominator is the intact graph's TOTAL node count, which
never changes across this script's sweep (only edges are ever removed,
never nodes) -- so it's directly comparable across every intensity, same
guarantee 7_1/7_2 rely on. `n_components`/`lcc_size`/`lcc_ratio` are now
columns in every output CSV/pickle, and `validate_results` checks the
same monotonicity 7_2 already checks: because adaptive disruption only
ever grows its cumulative removed-edge set as intensity rises within one
run (see `run_adaptive_sweep`), `n_components` must be non-decreasing and
`lcc_ratio` must be non-increasing across the sweep -- a violation would
mean the removal loop somehow undid an earlier removal, which would be a
real bug. `8_generate_diagram.py` already reads `lcc_ratio`/`n_components`
generically off any deterministic (targeted/adaptive) CSV that has them,
so Fig. 5 (LCC degradation) picks up the Adaptive row automatically once
this script is re-run -- no changes needed there.

How this differs from 7_2_targeted_disruption.py
--------------------------------------------------
7_2 computes county-to-county edge betweenness centrality ONCE on the
intact graph and then removes edges in that fixed descending order,
regardless of how removing earlier edges reshuffles which roads matter
most.

This script instead:

    1. Compute county-to-county edge betweenness centrality on the
       CURRENT (possibly already-disrupted) graph.
    2. Remove edges in descending centrality order until the next
       intensity threshold is reached.
    3. Recompute centrality on the now-smaller graph.
    4. Repeat for every intensity level.

This is the "batch-adaptive" attack: an attacker who re-surveys the
network at each intensity level and always hits whatever is currently
most critical, rather than committing to a target list computed up front.
The ranking is recomputed once per intensity level (50 times for
1%..50%), NOT after every single edge removal -- this gives a massive
speedup over the per-edge-recompute approach while retaining the
adaptive character (the attacker re-evaluates which edges matter most
as the network degrades).

Checkpointing / resume
-----------------------
The expensive artifact here is the ORDERED SEQUENCE of removed edges (each
one's centrality score recorded from the ranking that was active when it
was removed). The checkpoint pickle stores that full sequence plus each
edge's length and the centrality score it had at the moment it was removed.
On `--resume`, that sequence is replayed against a fresh copy of the intact
graph -- cheap, no centrality recomputation -- to fast-forward back to where
the run left off.

Input (per network)
--------------------
    data/graph/smoothing/<network>_smoothed.pkl
        (dict: {"graph": nx.MultiGraph, "C_county_nodes": [...]})
    data/analysis/<network>_travel_time_matrix.pkl
        (5_travel_time_matrix.py's own output) -- reused as the
        undisrupted baseline. Rebuilt once if missing, same fallback as
        7_2.

Output (per network)
----------------------
    reports/<network>_adaptive_disruption.csv
        One row per intensity: intensity_pct, n_edges_removed_total,
        removed_length_km, actual_intensity_pct, last_removed_edge,
        last_removed_centrality, n_unaffected, n_delayed, n_disconnected,
        frac_unaffected, frac_delayed, frac_disconnected, functionality,
        n_pairs, n_components, lcc_size, lcc_ratio,
        elapsed_s_this_intensity.
    data/analysis/<network>_adaptive_disruption.pkl
    reports/<network>_adaptive_disruption_summary.csv
    data/analysis/<network>_adaptive_disruption_checkpoint.pkl
        (ordered removal sequence + completed rows; deleted automatically
        once the full sweep finishes)

Usage
-----
    python 7_3_adaptive_disruption.py                     # BOTH networks
    python 7_3_adaptive_disruption.py --network highway
    python 7_3_adaptive_disruption.py --network railway --resume
    python 7_3_adaptive_disruption.py --checkpoint-every 10
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as sp_connected_components
from scipy.sparse.csgraph import dijkstra as sp_dijkstra

# ---------------------------------------------------------------------------
# PATHS -- matching 4_graph_smoothing.py's / 5_travel_time_matrix.py's /
# 7_1's / 7_2's own conventions.
# ---------------------------------------------------------------------------

NETWORK_LABELS = ["highway", "railway"]

NETWORK_CONFIG = {
    network: {
        "graph": Path(f"data/graph/smoothing/{network}_smoothed.pkl"),
        "baseline": Path(f"data/analysis/{network}_travel_time_matrix.pkl"),
        "output_csv": Path(f"reports/{network}_adaptive_disruption.csv"),
        "output_pickle": Path(f"data/analysis/{network}_adaptive_disruption.pkl"),
        "checkpoint_pickle": Path(f"data/analysis/{network}_adaptive_disruption_checkpoint.pkl"),
        "summary_csv": Path(f"reports/{network}_adaptive_disruption_summary.csv"),
    }
    for network in NETWORK_LABELS
}

WEIGHT_ATTR = "travel_time_min"
LENGTH_ATTR = "length_km"

DEFAULT_MIN_INTENSITY = 1
DEFAULT_MAX_INTENSITY = 50
DEFAULT_STEP = 1
DEFAULT_CHECKPOINT_EVERY = 25

# HARDCODED DEFAULT NETWORK -- same pattern as every other script in this
# pipeline. Only a safety fallback; --network is optional and defaults to
# "both".
HARDCODED_NETWORK = "highway"

NodeId = Tuple[float, float]
EdgeId = Tuple  # (u, v) for a plain Graph, (u, v, key) for a MultiGraph


def _print_banner(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ---------------------------------------------------------------------------
# Progress bar (PATCH) -- the adaptive sweep can spend a very long time
# between console updates otherwise: the old code only printed once per
# COMPLETED intensity level, and each level can require thousands of
# individual edge removals (each with its own full centrality recompute --
# see the module docstring's "Cost warning"). This gives continuous
# feedback -- a bar tracking cumulative km removed toward the final
# (highest-intensity) target, updated after every single edge removal --
# instead of long silent gaps. Mirrors 7_1_random_disruption.py's own
# tqdm-with-plain-fallback pattern so behavior is identical whether or not
# tqdm is installed.
# ---------------------------------------------------------------------------

def _get_progress_bar(total: float, desc: str):
    try:
        from tqdm import tqdm
        return tqdm(
            total=total, desc=desc, unit="km", unit_scale=False,
            dynamic_ncols=True, bar_format=(
                "{l_bar}{bar}| {n:.1f}/{total_fmt} km "
                "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
            ),
        )
    except ImportError:
        return _PlainProgressBar(total, desc)


def _progress_write(bar, message: str) -> None:
    if hasattr(bar, "write"):
        bar.write(message)
    else:
        print(message)


class _PlainProgressBar:
    """Minimal fallback progress bar (no tqdm dependency) -- same shape as
    the one in 7_1_random_disruption.py, adapted for a float (km) total
    instead of an integer simulation count."""

    def __init__(self, total: float, desc: str, min_interval: float = 0.2):
        self.total = total
        self.desc = desc
        self.n = 0.0
        self.postfix = ""
        self.start = time.monotonic()
        self._last_print = 0.0
        self._min_interval = min_interval

    def update(self, n: float = 1.0) -> None:
        self.n += n
        now = time.monotonic()
        if now - self._last_print >= self._min_interval or self.n >= self.total:
            self._render()
            self._last_print = now

    def set_description(self, desc: str) -> None:
        self.desc = desc

    def set_postfix(self, **kwargs) -> None:
        self.postfix = ", ".join(f"{k}={v}" for k, v in kwargs.items())

    def write(self, message: str) -> None:
        print(message)

    def _render(self) -> None:
        elapsed = time.monotonic() - self.start
        rate = self.n / elapsed if elapsed > 0 else 0.0
        pct = 100 * self.n / self.total if self.total else 100.0
        eta_s = (self.total - self.n) / rate if rate > 0 else float("nan")
        print(
            f"\r{self.desc}: {self.n:,.1f}/{self.total:,.1f} km ({pct:5.1f}%) "
            f"[{elapsed:,.0f}s elapsed, {rate:.4f} km/s, ETA {eta_s:,.0f}s]"
            f"{' -- ' + self.postfix if self.postfix else ''}",
            end="", flush=True,
        )

    def close(self) -> None:
        self._render()
        print()


# ---------------------------------------------------------------------------
# Loading -- same contract as 7_2_targeted_disruption.py's own loaders.
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
            "Intact graph is NOT connected. Adaptive disruption requires "
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
    T = pd.DataFrame(index=C, columns=C, dtype=float)
    for c1 in C:
        lengths = nx.single_source_dijkstra_path_length(G, c1, weight=weight)
        T.loc[c1, :] = [lengths.get(c2, np.inf) if c2 != c1 else 0.0 for c2 in C]
    return T


def load_baseline(path: Path, G, C) -> pd.DataFrame:
    """Reuse Step 5's baseline when available (same contract as 7_2)."""
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


def total_network_length_km(G) -> float:
    return float(sum(float(d[LENGTH_ATTR]) for _, _, d in G.edges(data=True)))


def upper_triangle_values(df: pd.DataFrame) -> np.ndarray:
    """Extract the unique (i < j) unordered-pair values from a symmetric
    C x C DataFrame -- the N = C(|C|, 2) unique county-to-county routes."""
    n = df.shape[0]
    iu = np.triu_indices(n, k=1)
    return df.to_numpy(dtype=float)[iu]


# ---------------------------------------------------------------------------
# Classification / functionality (identical to 7_2's, so the two CSVs are
# byte-for-byte comparable)
# ---------------------------------------------------------------------------

def classify_and_score(T_baseline: pd.DataFrame, T_disrupted: pd.DataFrame) -> Dict:
    t = upper_triangle_values(T_baseline)
    tp = upper_triangle_values(T_disrupted)

    with np.errstate(divide="ignore", invalid="ignore"):
        delta = (tp - t) / t

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
# Adaptive removal core -- BATCH-ADAPTIVE approach
#
# Instead of recomputing betweenness after every single edge removal
# (which is extremely expensive), we recompute ONCE PER INTENSITY LEVEL:
#   1. Compute the full edge betweenness ranking on the current graph.
#   2. Remove edges in descending centrality order until we reach the
#      km target for this intensity level.
#   3. Move to the next intensity level and recompute the ranking.
#
# For 50 intensity levels (1%..50%), this means 50 betweenness
# computations instead of potentially thousands.
# ---------------------------------------------------------------------------

def rank_all_edges(G_current, C: List[NodeId]) -> List[Tuple[EdgeId, float, float]]:
    """
    Compute county-to-county edge betweenness centrality on the CURRENT
    graph and return ALL edges sorted by centrality (descending).

    Returns a list of (edge_id, centrality_score, length_km) tuples,
    sorted from highest to lowest centrality. Ties are broken by a
    stable repr-based key for reproducibility.
    """
    centrality = nx.edge_betweenness_centrality_subset(
        G_current, sources=C, targets=C, weight=WEIGHT_ATTR, normalized=False,
    )
    if not centrality:
        return []

    # Sort descending by centrality, with a stable tiebreaker
    sorted_edges = sorted(
        centrality.items(),
        key=lambda kv: (-kv[1], tuple(repr(x) for x in kv[0])),
    )

    ranked = []
    for edge, score in sorted_edges:
        if G_current.is_multigraph():
            u, v, k = edge
            length_km = float(G_current[u][v][k][LENGTH_ATTR])
        else:
            u, v = edge
            length_km = float(G_current[u][v][LENGTH_ATTR])
        ranked.append((edge, score, length_km))

    return ranked


def remove_edge_by_id(G_current, edge_id: EdgeId) -> None:
    if G_current.is_multigraph():
        u, v, k = edge_id
        G_current.remove_edge(u, v, k)
    else:
        u, v = edge_id
        G_current.remove_edge(u, v)


def compute_od_matrix(G_current, C: List[NodeId]) -> Tuple[pd.DataFrame, Dict]:
    """Full symmetric C x C matrix, rebuilt fresh from whatever G_current
    currently looks like. Only called once per intensity level.

    PATCH: also computes STRUCTURAL fragmentation (connected components /
    largest connected component) on the SAME disrupted sparse adjacency
    matrix used for the Dijkstra travel-time calculation below -- no
    separate graph rebuild, mirroring 7_1_random_disruption.py's and
    7_2_targeted_disruption.py's own LCC/component tracking. This is a
    STRUCTURAL metric ("is the network still physically one piece, and
    how much of it") distinct from the FUNCTIONAL metrics (functionality,
    delay categories) computed from the travel-time matrix returned
    alongside it -- the two can diverge (e.g. the network can stay 100%
    connected while functionality drops because surviving routes got
    longer).

    `lcc_ratio`'s denominator is the graph's TOTAL node count, which never
    changes across this script's sweep (only edges are ever removed,
    never nodes) -- so it's directly comparable across every intensity,
    same guarantee 7_1/7_2 rely on. An isolated node (e.g. a county that
    lost every incident edge) counts as its own size-1 component,
    correctly.

    Returns:
        A tuple of (T_disrupted, structural) where T_disrupted is the
        C x C travel-time DataFrame (unchanged shape/meaning from before),
        and structural is a dict:
            {"n_components": int, "lcc_size": int, "lcc_ratio": float}
    """
    nodes = list(G_current.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    n_nodes = len(nodes)

    missing_C = [c for c in C if c not in node_to_idx]
    if missing_C:
        raise SystemExit(
            "A county node vanished from the graph during adaptive "
            "disruption -- this should never happen since only edges are "
            "ever removed, never nodes. Investigate before trusting results."
        )

    # PATCH (parallel-edge weight bug): scipy's csr_matrix constructor SUMS
    # duplicate (row, col) entries rather than taking the minimum. If two
    # nodes are joined by more than one edge (a real MultiGraph situation
    # in this pipeline -- see 2_3_gap_repair.py's ~40 shared-endpoint-pair
    # note), passing BOTH parallel edges' weights straight into csr_matrix
    # would silently ADD them together (e.g. weights 1.0 and 5.0 -> 6.0),
    # corrupting the shortest-path computation below, which should instead
    # see only the faster parallel edge (weight 1.0) -- exactly the edge
    # that's actually usable for a shortest path. Deduping to the min
    # weight per (u, v) pair BEFORE building the sparse matrix avoids this
    # entirely. This mirrors 7_1_random_disruption.py's own
    # `disrupted_edge_weights` dict, which already does this correctly;
    # this function previously did not.
    if G_current.is_multigraph():
        edges = list(G_current.edges(keys=True, data=True))
        edge_iter = ((u, v, float(d[WEIGHT_ATTR])) for u, v, k, d in edges)
    else:
        edges = list(G_current.edges(data=True))
        edge_iter = ((u, v, float(d[WEIGHT_ATTR])) for u, v, d in edges)

    min_weight_per_pair: Dict[Tuple[int, int], float] = {}
    for u, v, w in edge_iter:
        i, j = node_to_idx[u], node_to_idx[v]
        pair = (i, j) if i < j else (j, i)
        if pair not in min_weight_per_pair or w < min_weight_per_pair[pair]:
            min_weight_per_pair[pair] = w

    rows_dedup, cols_dedup, weights_dedup = [], [], []
    for (i, j), w in min_weight_per_pair.items():
        rows_dedup.extend([i, j])
        cols_dedup.extend([j, i])
        weights_dedup.extend([w, w])

    r = np.array(rows_dedup, dtype=np.int64)
    c = np.array(cols_dedup, dtype=np.int64)
    w = np.array(weights_dedup, dtype=float)
    A = csr_matrix((w, (r, c)), shape=(n_nodes, n_nodes))

    # Structural fragmentation -- graph structure only (edge weights don't
    # matter for connectivity). Computed on the same sparse matrix `A`
    # that feeds the Dijkstra call right below -- no separate rebuild.
    n_components, labels = sp_connected_components(A, directed=False, connection="weak")
    comp_sizes = np.bincount(labels, minlength=n_components)
    lcc_size = int(comp_sizes.max())
    structural = {
        "n_components": int(n_components),
        "lcc_size": lcc_size,
        "lcc_ratio": lcc_size / n_nodes,
    }

    county_idx = np.array([node_to_idx[c1] for c1 in C], dtype=np.int64)

    distances = sp_dijkstra(A, directed=False, indices=county_idx, return_predecessors=False)
    selected = distances[:, county_idx]
    T_disrupted = pd.DataFrame(selected, index=C, columns=C, dtype=float)
    return T_disrupted, structural


# ---------------------------------------------------------------------------
# Checkpointing -- the expensive artifact is the ORDERED removal sequence.
# ---------------------------------------------------------------------------

def load_checkpoint(path: Path):
    if not path.exists():
        return [], 0.0, {}
    with open(path, "rb") as f:
        ck = pickle.load(f)
    return ck["removed_edges_ordered"], ck["cumulative_km"], ck.get("completed_rows", {})


def save_checkpoint(path: Path, removed_edges_ordered, cumulative_km, completed_rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(
            {
                "removed_edges_ordered": removed_edges_ordered,
                "cumulative_km": cumulative_km,
                "completed_rows": completed_rows,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def replay_removals(G, removed_edges_ordered) -> Tuple:
    """Fast-forward a fresh copy of the intact graph `G` through every
    edge in `removed_edges_ordered`, WITHOUT recomputing centrality."""
    G_current = G.copy()
    cumulative_km = 0.0
    for edge_id, length_km, _score in removed_edges_ordered:
        try:
            remove_edge_by_id(G_current, edge_id)
        except nx.NetworkXError as exc:
            raise SystemExit(
                f"Checkpoint replay failed: edge {edge_id} not found in the "
                f"intact graph. The checkpoint likely does not match the "
                f"current input graph -- delete the checkpoint file and "
                f"rerun without --resume."
            ) from exc
        cumulative_km += length_km
    return G_current, cumulative_km


# ---------------------------------------------------------------------------
# Runtime projection
# ---------------------------------------------------------------------------

def estimate_and_confirm(
    G, C, total_len_km: float, max_intensity: int, pause_s: float = 5.0
) -> None:
    """Time a single betweenness computation and project total runtime.

    Under the batch-adaptive approach, the number of betweenness
    computations equals the number of intensity levels (typically 50),
    NOT the number of edges removed. This makes the projection simple
    and accurate.
    """
    n_levels = max_intensity  # 1%..max_intensity% at step=1
    print(
        f"\nTiming 1 edge-betweenness computation to project total runtime "
        f"({n_levels} computations needed for {n_levels} intensity levels)..."
    )
    start = time.monotonic()
    _centrality = nx.edge_betweenness_centrality_subset(
        G, sources=C, targets=C, weight=WEIGHT_ATTR, normalized=False,
    )
    elapsed_one = time.monotonic() - start

    # The graph shrinks at each level, so later computations will be faster.
    # Use the full-graph timing as a conservative upper bound.
    projected_s = elapsed_one * n_levels

    print(
        f"  {elapsed_one:.3f}s for one betweenness computation on the "
        f"intact graph ({G.number_of_edges():,} edges).\n"
        f"  {n_levels} intensity levels x {elapsed_one:.3f}s = "
        f"~{projected_s:,.0f}s (~{projected_s/60:,.1f} min) UPPER BOUND.\n"
        f"  (Actual runtime will be less because the graph shrinks at "
        f"each level, making later computations faster.)\n"
        f"  Starting in {pause_s:.0f}s..."
    )
    time.sleep(pause_s)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "intensity_pct", "n_edges_removed_total", "removed_length_km",
    "actual_intensity_pct", "last_removed_edge", "last_removed_centrality",
    "n_unaffected", "n_delayed", "n_disconnected",
    "frac_unaffected", "frac_delayed", "frac_disconnected",
    "functionality", "n_pairs",
    "n_components", "lcc_size", "lcc_ratio",
    "elapsed_s_this_intensity",
]


def run_adaptive_sweep(
    G, C, baseline: pd.DataFrame, total_len_km: float, intensities: List[int],
    output_csv: Path, checkpoint_path: Path, *, resume: bool = False,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
) -> pd.DataFrame:
    """Batch-adaptive disruption sweep.

    For each intensity level:
      1. Compute edge betweenness centrality ONCE on the current graph.
      2. Remove edges in descending centrality order until cumulative
         removed length reaches the target km for this intensity.
      3. Record metrics (travel time matrix, functionality, LCC, etc.).
      4. Proceed to the next intensity level and recompute rankings.

    This means betweenness is recomputed exactly once per intensity level
    (e.g. 50 times for 1%..50%), NOT after every single edge removal.
    """
    removed_edges_ordered: List[Tuple[EdgeId, float, float]] = []
    cumulative_km = 0.0
    completed_rows: Dict[int, dict] = {}
    n_pairs = len(C) * (len(C) - 1) // 2

    if resume:
        removed_edges_ordered, cumulative_km, completed_rows = load_checkpoint(checkpoint_path)
        if removed_edges_ordered:
            print(
                f"Resuming: replaying {len(removed_edges_ordered):,} previously "
                f"removed edge(s) ({cumulative_km:.1f} km) from checkpoint..."
            )
            G_current, cumulative_km = replay_removals(G, removed_edges_ordered)
        else:
            G_current = G.copy()
    else:
        G_current = G.copy()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if (resume and output_csv.exists() and completed_rows) else "w"
    write_header = mode == "w"

    if resume and completed_rows:
        remaining = [i for i in intensities if i not in completed_rows]
        print(f"Resume: {len(completed_rows)}/{len(intensities)} intensity levels already done, "
              f"{len(remaining)} remaining.")
        if not remaining:
            print("All intensities already completed in checkpoint -- nothing to do.")
            return pd.DataFrame(list(completed_rows.values())).sort_values("intensity_pct").set_index("intensity_pct")

    # One continuous bar spanning the WHOLE sweep -- cumulative_km only ever
    # grows across the entire run.
    final_target_km = total_len_km * max(intensities) / 100.0
    bar = _get_progress_bar(final_target_km, desc="Adaptive sweep")
    if cumulative_km > 0:
        bar.n = min(cumulative_km, final_target_km)
        bar.update(0)

    try:
        with open(output_csv, mode, newline="") as f:
            if write_header:
                f.write(",".join(CSV_FIELDS) + "\n")

            for intensity in intensities:
                if intensity in completed_rows:
                    continue

                target_km = total_len_km * intensity / 100.0
                t0 = time.monotonic()
                last_edge, last_score = None, None
                bar.set_description(f"Adaptive sweep ({intensity}%)")

                # -- BATCH-ADAPTIVE: recompute ranking ONCE for this level --
                if cumulative_km < target_km:
                    _progress_write(
                        bar,
                        f"\n  Recomputing edge betweenness for {intensity}% level "
                        f"({G_current.number_of_edges():,} edges remaining)...",
                    )
                    t_rank = time.monotonic()
                    ranked_edges = rank_all_edges(G_current, C)
                    _progress_write(
                        bar,
                        f"  Ranking done in {time.monotonic() - t_rank:.1f}s -- "
                        f"{len(ranked_edges):,} edges ranked.",
                    )

                    # Walk through the ranking, removing edges until target
                    edge_idx = 0
                    while cumulative_km < target_km and edge_idx < len(ranked_edges):
                        edge_id, score, length_km = ranked_edges[edge_idx]
                        edge_idx += 1

                        # The edge might already have been removed (shouldn't
                        # happen in normal flow, but guard against it)
                        try:
                            remove_edge_by_id(G_current, edge_id)
                        except nx.NetworkXError:
                            continue

                        cumulative_km += length_km
                        removed_edges_ordered.append((edge_id, length_km, score))
                        last_edge, last_score = edge_id, score

                        bar.set_postfix(
                            intensity=f"{intensity}%",
                            edges=f"{len(removed_edges_ordered):,}",
                            last_cB=f"{score:.3f}",
                        )
                        bar.update(length_km)

                        if len(removed_edges_ordered) % checkpoint_every == 0:
                            save_checkpoint(checkpoint_path, removed_edges_ordered, cumulative_km, completed_rows)

                    if cumulative_km < target_km and edge_idx >= len(ranked_edges):
                        _progress_write(
                            bar,
                            f"  WARNING: no edges left to remove at {intensity}% "
                            f"(network fully exhausted) -- stopping early.",
                        )

                T_disrupted, structural = compute_od_matrix(G_current, C)
                metrics = classify_and_score(baseline, T_disrupted)
                elapsed = time.monotonic() - t0
                actual_pct = 100.0 * cumulative_km / total_len_km

                row = {
                    "intensity_pct": intensity,
                    "n_edges_removed_total": len(removed_edges_ordered),
                    "removed_length_km": cumulative_km,
                    "actual_intensity_pct": actual_pct,
                    "last_removed_edge": repr(last_edge),
                    "last_removed_centrality": last_score,
                    **metrics,
                    "n_pairs": n_pairs,
                    **structural,
                    "elapsed_s_this_intensity": elapsed,
                }

                pd.DataFrame([row]).to_csv(f, header=False, index=False)
                f.flush()

                completed_rows[intensity] = row
                save_checkpoint(checkpoint_path, removed_edges_ordered, cumulative_km, completed_rows)

                _progress_write(
                    bar,
                    f"  {intensity:2d}% | edges_removed_total={len(removed_edges_ordered):,} | "
                    f"actual={actual_pct:6.3f}% | disc={metrics['frac_disconnected']*100:6.2f}% | "
                    f"f(G)={metrics['functionality']:.5f} | "
                    f"LCC={structural['lcc_ratio']:.4f} | comps={structural['n_components']:,} | "
                    f"{elapsed:,.1f}s",
                )
    finally:
        bar.close()

    summary = pd.DataFrame(list(completed_rows.values())).sort_values("intensity_pct").set_index("intensity_pct")
    return summary


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_results(summary: pd.DataFrame) -> None:
    required = [
        "frac_unaffected", "frac_delayed", "frac_disconnected", "functionality",
        "n_components", "lcc_size", "lcc_ratio",
    ]
    for col in required:
        if col not in summary:
            raise AssertionError(f"Missing result column: {col}")

    category_sum = summary["frac_unaffected"] + summary["frac_delayed"] + summary["frac_disconnected"]
    if not np.allclose(category_sum.to_numpy(), 1.0, atol=1e-12):
        raise AssertionError("Path-category fractions do not sum to 1.")

    disconnected_diff = summary["frac_disconnected"].diff().dropna()
    if (disconnected_diff < -1e-10).any():
        raise AssertionError("Disconnected fraction decreased with intensity.")

    functionality_diff = summary["functionality"].diff().dropna()
    if (functionality_diff > 1e-10).any():
        raise AssertionError("Functionality increased with intensity.")

    # PATCH: same structural monotonicity check 7_2_targeted_disruption.py
    # already runs. Adaptive disruption's cumulative removed-edge set only
    # ever GROWS as intensity rises within one run (each intensity keeps
    # removing from where the last one left off -- see run_adaptive_sweep's
    # `while cumulative_km < target_km` loop, which never un-removes an
    # edge). So the graph only ever gets more disrupted as intensity rises,
    # meaning component count must be non-decreasing and LCC ratio must be
    # non-increasing. A violation would mean the removal loop somehow undid
    # an earlier removal, which would be a real bug worth investigating.
    components_diff = summary["n_components"].diff().dropna()
    if (components_diff < 0).any():
        raise AssertionError("Component count decreased with intensity.")

    lcc_diff = summary["lcc_ratio"].diff().dropna()
    if (lcc_diff > 1e-10).any():
        raise AssertionError("LCC ratio increased with intensity.")

    print("Validation passed:")
    print("  category fractions sum to 1")
    print("  disconnected fraction is non-decreasing")
    print("  functionality is non-increasing")
    print("  component count is non-decreasing")
    print("  LCC ratio is non-increasing")


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

    _print_banner(f"STEP 5c -- ADAPTIVE (BATCH-RECOMPUTE) DISRUPTION [{network}]")
    print(f"Graph:         {cfg['graph']}")
    print(f"Baseline:      {cfg['baseline']}")
    print(f"Intensities:   {intensities[0]}%..{intensities[-1]}% step {args.step}%")
    print(f"Output CSV:    {cfg['output_csv']}")
    print(f"Output pickle: {cfg['output_pickle']}")
    print(f"Checkpoint:    {cfg['checkpoint_pickle']}")

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

    if not args.skip_runtime_check and not args.resume:
        estimate_and_confirm(G, C, total_len_km, args.max_intensity)

    if args.resume:
        print("\nResuming from checkpoint...")
    else:
        print("\nRunning full adaptive sweep...")

    summary = run_adaptive_sweep(
        G, C, baseline, total_len_km, intensities,
        cfg["output_csv"], cfg["checkpoint_pickle"],
        resume=args.resume, checkpoint_every=args.checkpoint_every,
    )

    validate_results(summary)

    cfg["summary_csv"].parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(cfg["summary_csv"])

    cfg["output_pickle"].parent.mkdir(parents=True, exist_ok=True)
    with open(cfg["output_pickle"], "wb") as f:
        pickle.dump(summary, f, protocol=pickle.HIGHEST_PROTOCOL)

    if cfg["checkpoint_pickle"].exists():
        cfg["checkpoint_pickle"].unlink()
        print(f"Removed checkpoint file: {cfg['checkpoint_pickle']}")

    _print_banner(f"ADAPTIVE DISRUPTION COMPLETE [{network}]")
    print(f"Summary CSV:     {cfg['summary_csv']}")
    print(f"Results pickle:  {cfg['output_pickle']}")

    print("\nEndpoint results:")
    for intensity in [summary.index.min(), summary.index.max()]:
        r = summary.loc[intensity]
        print(
            f"  {intensity}%: actual={r['actual_intensity_pct']:.3f}% | "
            f"edges_removed={int(r['n_edges_removed_total']):,} | "
            f"disconnected={r['frac_disconnected']:.3%} | "
            f"functionality={r['functionality']:.5f} | "
            f"LCC ratio={r['lcc_ratio']:.4f} | "
            f"components={int(r['n_components']):,}"
        )

    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Adaptive (batch-recompute) disruption: recompute "
                    "county-to-county edge betweenness once per intensity "
                    "level, removing edges in descending centrality order "
                    "within each level. Runs BOTH highway and railway "
                    "automatically unless --network restricts it to one."
    )
    p.add_argument(
        "--network", choices=sorted(NETWORK_LABELS), default=None,
        help=(
            "Restrict the run to just this one network. If omitted, BOTH "
            "highway and railway are processed automatically."
        ),
    )
    p.add_argument("--min-intensity", type=int, default=DEFAULT_MIN_INTENSITY)
    p.add_argument("--max-intensity", type=int, default=DEFAULT_MAX_INTENSITY)
    p.add_argument("--step", type=int, default=DEFAULT_STEP)
    p.add_argument(
        "--resume", action="store_true",
        help="Resume from the checkpoint pickle: replay the previously "
             "discovered removal sequence (cheap) and continue the "
             "expensive step-by-step loop from there.",
    )
    p.add_argument(
        "--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY,
        help=f"Save the removal-sequence checkpoint every N edge removals "
             f"(default: {DEFAULT_CHECKPOINT_EVERY}).",
    )
    p.add_argument(
        "--skip-runtime-check", action="store_true",
        help="Skip the probe-run projection and confirmation pause.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.min_intensity < 1 or args.max_intensity > 50:
        raise SystemExit("Intensity must be within 1..50% for the paper reproduction.")
    if args.min_intensity > args.max_intensity:
        raise SystemExit("--min-intensity cannot exceed --max-intensity.")
    if args.step <= 0:
        raise SystemExit("--step must be positive.")
    if args.checkpoint_every <= 0:
        raise SystemExit("--checkpoint-every must be positive.")

    intensities = list(range(args.min_intensity, args.max_intensity + 1, args.step))

    if args.network:
        networks_to_run = [args.network]
    else:
        networks_to_run = list(NETWORK_LABELS)

    print(f"Running Step 5c (adaptive disruption) for: {', '.join(networks_to_run)}")

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

    print(
        "\nNext step: plot alongside random (7_1) and fixed targeted (7_2) "
        "results -- pass this CSV to your diagram-generation script via "
        "--adaptive-csv / --rail-adaptive-csv. 8_generate_diagram.py already "
        "reads lcc_ratio/n_components generically off any deterministic "
        "CSV that has them, so Fig. 5 (LCC degradation) picks up this "
        "script's Adaptive row automatically -- no changes needed there."
    )


if __name__ == "__main__":
    main()