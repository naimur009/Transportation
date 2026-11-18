#!/usr/bin/env python3
"""
Step 5a -- Random disruption simulation (Illinois test run).

DATASET NOTE (updated to match the Illinois test run's naming convention):
-----------------------------------------------------------------------
The earlier draft of this script simulated disruptions to a fort-to-port
travel-time matrix (disjoint sets O_fort_nodes / D_port_nodes) on a single
hardcoded STRAHNET graph. Neither of those two disjoint facility sets nor
that graph path exist in this pipeline. This script has been rewritten to
match what this project actually produces:

  * Facility set: ONE set of COUNTY nodes, `C`, not two disjoint sets
    `O`/`D`. Traced directly to `4_graph_smoothing.py`'s own output dict
    shape (`{"graph": G_smoothed, "C_county_nodes": sorted(C)}`).
    A county-to-county matrix on an UNDIRECTED graph is SYMMETRIC
    (t(c1,c2) == t(c2,c1)) with a zero diagonal, so every statistic below
    (delta_t, unaffected/delayed/disconnected counts, f(G)) is computed
    over only the `N = C(|C|, 2)` UNIQUE unordered pairs (the upper
    triangle of the C x C matrix), matching `3_snap_counties.py`'s and
    `5_travel_time_matrix.py`'s own bookkeeping -- not the old |O| x |D|
    cross-product, which would have double-counted every route and
    included C spurious zero self-distances if applied naively to a
    symmetric matrix.

  * Graph input: BOTH "highway" and "railway", not a single STRAHNET
    graph. Traced directly to `4_graph_smoothing.py`'s own output
    convention (`data/graph/smoothing/<network>_smoothed.pkl`) and
    `5_travel_time_matrix.py`'s own baseline output
    (`data/analysis/<network>_travel_time_matrix.pkl`). Following the
    "both networks by default" convention already established by every
    other script in this pipeline, `--network` is OPTIONAL: omitting it
    runs the full sweep for BOTH graphs automatically, one after another,
    each with its own outputs. A missing smoothed-graph pickle for one
    network prints a clear warning and skips just that network.

Everything else -- the disruption methodology, the performance
optimizations, the checkpoint/resume behavior -- is UNCHANGED from the
original script:

For every disruption intensity from 1% to 50% of total network length (in
1% steps), this script:

  1. Removes a random subset of edges whose combined length_km sums to
     approximately that intensity's percentage of the UNDISRUPTED graph's
     total length. Repeated 100 independent times per intensity.
  2. Recomputes the county-to-county travel-time matrix on the disrupted
     graph (one SciPy sparse Dijkstra call from every county node
     simultaneously).
  3. For every UNIQUE county-to-county pair, computes the relative delay
         delta_t = (t'_c1c2 - t_c1c2) / t_c1c2
     against the UNDISRUPTED baseline loaded from
     data/analysis/<network>_travel_time_matrix.pkl, and classifies the
     pair as unaffected (delta=0), delayed (0<delta<inf), or disconnected
     (delta=inf).
  4. Computes the continuous functionality score per pair,
         f(delta_t) = 1 / (1 + delta_t)
     and averages it across all unique county-to-county pairs into a
     single network functionality score f(G) for that (intensity, trial).

Targeted and adaptive disruption are out of scope for this script -- see
7_2_targeted_disruption.py and 7_3_adaptive_disruption.py, which reuse
this script's per-disruption evaluation logic.

A NOTE ON COST BEFORE YOU RUN THIS
-----------------------------------
The default parameters (50 intensities x 100 trials = 5,000 disrupted
graphs per network) are computationally heavy. This script uses
ProcessPoolExecutor and SciPy's sparse csgraph Dijkstra implementation for
a large speedup over naive NetworkX usage. It times the very first
(intensity, trial) simulation, prints a projected total runtime for the
full sweep BEFORE continuing, and gives you a few seconds to Ctrl-C if
that number is unacceptable. Every parameter that drives that cost
(--n-trials, --min-intensity, --max-intensity, --step) is a CLI flag.

Results are also written incrementally (one CSV row per completed trial,
flushed immediately) -- an interrupted run still leaves a valid partial
results file.

Input (per network)
--------------------
    data/graph/smoothing/<network>_smoothed.pkl
        (dict: {"graph": nx.MultiGraph, "C_county_nodes": [...]})
        -- 4_graph_smoothing.py's own output; same file 5_travel_time_matrix.py
        and 6_edge_betweenness.py load.
    data/analysis/<network>_travel_time_matrix.pkl
        (dict: {"matrix": pandas.DataFrame, "C_county_nodes": [...]})
        -- 5_travel_time_matrix.py's own output; used as the undisrupted
        baseline for every delta_t computed here. NOT recomputed from
        scratch.

Output (per network)
----------------------
    reports/<network>_random_disruption_trials.csv
        One row per (intensity, trial): intensity_pct, trial_idx, seed,
        n_edges_removed, removed_length_km, f_G, n_unaffected, n_delayed,
        n_disconnected, n_pairs (= C(|C|,2)), n_components, lcc_size,
        lcc_ratio, elapsed_s. Written incrementally, one row at a time.

        n_components / lcc_size / lcc_ratio are STRUCTURAL fragmentation
        metrics, computed via scipy.sparse.csgraph.connected_components on
        the same disrupted sparse adjacency matrix used for the Dijkstra
        travel-time calculation -- no separate graph rebuild. This is
        distinct from f_G (a FUNCTIONAL metric: does a route still exist
        and how much slower is it) -- LCC only asks "is the network still
        physically one piece, and how much of it". lcc_ratio is the
        largest component's node count divided by the graph's TOTAL node
        count (constant across the whole sweep, since this script only
        ever removes edges, never nodes) -- so lcc_ratio is directly
        comparable across every intensity and trial.
    data/analysis/<network>_random_disruption.pkl
        The same rows as a pandas.DataFrame, for downstream plotting.

Checkpointing / resume
----------------------
Writes results incrementally, so an interrupted run always leaves a valid
partial CSV. To RESUME from where you left off:

    python 7_1_random_disruption.py --resume

This reads the existing CSV(s) to discover which (network, intensity,
trial) combinations are already done, skips them, and appends new
results. A pickle checkpoint
(data/analysis/<network>_random_disruption_checkpoint.pkl) is also saved
after every completed intensity level per network, and is automatically
cleaned up once that network's full sweep finishes.

Usage
-----
    python 7_1_random_disruption.py                        # BOTH networks, paper defaults
    python 7_1_random_disruption.py --network highway      # just highway
    python 7_1_random_disruption.py --n-trials 5           # fast dev run
    python 7_1_random_disruption.py --workers 4            # use 4 CPU cores
    python 7_1_random_disruption.py --resume               # resume from last checkpoint(s)
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as sp_connected_components
from scipy.sparse.csgraph import dijkstra as sp_dijkstra

# ---------------------------------------------------------------------------
# Paths -- kept in sync by hand with 4_graph_smoothing.py's and
# 5_travel_time_matrix.py's own output conventions.
# ---------------------------------------------------------------------------

SMOOTHED_INPUT_DIR = Path("data/graph/smoothing")   # 4_graph_smoothing.py's own OUTPUT_GRAPH_DIR
BASELINE_INPUT_DIR = Path("data/analysis")          # 5_travel_time_matrix.py's own ANALYSIS_OUTPUT_DIR
ANALYSIS_OUTPUT_DIR = Path("data/analysis")
REPORTS_DIR = Path("reports")

NETWORK_LABELS = ["highway", "railway"]

NETWORK_GRAPH_FILENAMES = {net: f"{net}_smoothed.pkl" for net in NETWORK_LABELS}
NETWORK_BASELINE_FILENAMES = {net: f"{net}_travel_time_matrix.pkl" for net in NETWORK_LABELS}

# HARDCODED DEFAULT NETWORK -- same pattern as every other script in this
# pipeline. Only used as a safety fallback; --network is optional and
# defaults to "both".
HARDCODED_NETWORK = "highway"

WEIGHT_ATTR = "travel_time_min"
LENGTH_ATTR = "length_km"

# Paper-style defaults: 1% to 50% intensity in 1% steps, 100 random trials
# per intensity. CLI-overridable, not fixed constants.
DEFAULT_MIN_INTENSITY = 1
DEFAULT_MAX_INTENSITY = 50
DEFAULT_STEP = 1
DEFAULT_N_TRIALS = 100
DEFAULT_SEED = 42

NodeId = Tuple[float, float]


def _print_banner(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ---------------------------------------------------------------------------
# Worker Initialization & Globals (for memory safety in multiprocessing)
# ---------------------------------------------------------------------------

_WORKER_ALL_EDGES_KEYED = None
_WORKER_BASELINE_TRIU = None       # 1D array: baseline travel time for each UNIQUE county pair
_WORKER_TRIU_INDICES = None        # (row_idx, col_idx) arrays into the full C x C matrix
_WORKER_COUNTY_IDXS = None         # county node -> graph-node-index array, length |C|
_WORKER_N_NODES = None
_WORKER_N_EDGES_TOTAL = None


def worker_initializer(
    all_edges_keyed: List[Tuple[int, int, int, float, float]],
    baseline_triu: np.ndarray,
    triu_indices: Tuple[np.ndarray, np.ndarray],
    county_idxs: np.ndarray,
    n_nodes: int,
    n_edges_total: int,
) -> None:
    """Initialize read-only global data in each worker process once."""
    global _WORKER_ALL_EDGES_KEYED, _WORKER_BASELINE_TRIU, _WORKER_TRIU_INDICES
    global _WORKER_COUNTY_IDXS, _WORKER_N_NODES, _WORKER_N_EDGES_TOTAL
    _WORKER_ALL_EDGES_KEYED = all_edges_keyed
    _WORKER_BASELINE_TRIU = baseline_triu
    _WORKER_TRIU_INDICES = triu_indices
    _WORKER_COUNTY_IDXS = county_idxs
    _WORKER_N_NODES = n_nodes
    _WORKER_N_EDGES_TOTAL = n_edges_total


def run_trial(
    intensity_pct: float,
    trial_idx: int,
    base_seed: int,
    total_length_km: float,
) -> Dict[str, float]:
    """
    Run a single random-disruption trial using the pre-computed worker
    globals -- SciPy sparse Dijkstra, vectorized, no G.copy() overhead.
    """
    t0 = time.monotonic()

    trial_seed = base_seed * 100_000 + int(intensity_pct * 1_000) + trial_idx
    rng = np.random.default_rng(trial_seed)

    target_km = total_length_km * (intensity_pct / 100.0)

    # Random edge removal until the target length is reached.
    order = rng.permutation(len(_WORKER_ALL_EDGES_KEYED))
    removed_set = set()
    removed_km = 0.0
    n_removed = 0

    for idx in order:
        if removed_km >= target_km:
            break
        i, j, k, w, l = _WORKER_ALL_EDGES_KEYED[idx]
        removed_set.add((i, j, k))
        removed_km += l
        n_removed += 1

    # Build disrupted sparse matrix, keeping only the fastest parallel edge
    # per (i, j) pair among survivors.
    disrupted_edge_weights = {}
    for i, j, k, w, l in _WORKER_ALL_EDGES_KEYED:
        if (i, j, k) in removed_set:
            continue
        pair = (i, j) if i < j else (j, i)
        if pair not in disrupted_edge_weights or w < disrupted_edge_weights[pair]:
            disrupted_edge_weights[pair] = w

    d_rows, d_cols, d_weights = [], [], []
    for (i, j), w in disrupted_edge_weights.items():
        d_rows.extend([i, j])
        d_cols.extend([j, i])
        d_weights.extend([w, w])

    disrupted_sparse = csr_matrix((d_weights, (d_rows, d_cols)), shape=(_WORKER_N_NODES, _WORKER_N_NODES))

    # Structural fragmentation: connected components + largest connected
    # component (LCC), computed on the SAME disrupted sparse matrix used
    # for Dijkstra below -- no separate graph rebuild. Uses graph structure
    # only (edge weights are irrelevant to connectivity), and disconnected/
    # isolated nodes (e.g. a county that lost every incident edge) each
    # count as their own size-1 component, exactly as they should.
    n_components, labels = sp_connected_components(
        disrupted_sparse, directed=False, connection="weak"
    )
    comp_sizes = np.bincount(labels, minlength=n_components)
    lcc_size = int(comp_sizes.max())
    # Denominator is the ORIGINAL (undisrupted) node count, not the node
    # count of whatever's left touched by an edge -- node removal never
    # happens in this script (only edges are removed), so _WORKER_N_NODES
    # is constant across every trial and intensity, making lcc_ratio
    # directly comparable across the whole sweep.
    lcc_ratio = lcc_size / _WORKER_N_NODES

    # One Dijkstra call from every county node simultaneously -> full C x C
    # symmetric matrix, then reduce to the unique upper-triangle pairs.
    all_dists = sp_dijkstra(disrupted_sparse, directed=False, indices=_WORKER_COUNTY_IDXS)
    disrupted_full = all_dists[:, _WORKER_COUNTY_IDXS]
    iu = _WORKER_TRIU_INDICES
    disrupted_triu = disrupted_full[iu]

    deltas = np.zeros_like(disrupted_triu, dtype=float)
    positive_baseline = _WORKER_BASELINE_TRIU > 0
    np.divide(
        disrupted_triu - _WORKER_BASELINE_TRIU,
        _WORKER_BASELINE_TRIU,
        out=deltas,
        where=positive_baseline,
    )

    unaffected_mask = (disrupted_triu != np.inf) & (deltas <= 0)
    disconnected_mask = (disrupted_triu == np.inf)
    delayed_mask = ~unaffected_mask & ~disconnected_mask

    n_unaffected = int(unaffected_mask.sum())
    n_delayed = int(delayed_mask.sum())
    n_disconnected = int(disconnected_mask.sum())

    f_vals = np.where(disconnected_mask, 0.0, 1.0 / (1.0 + deltas))
    f_G = float(f_vals.mean())
    n_pairs = len(_WORKER_BASELINE_TRIU)

    elapsed_s = time.monotonic() - t0

    return {
        "intensity_pct": intensity_pct,
        "trial_idx": trial_idx,
        "seed": trial_seed,
        "n_edges_total": _WORKER_N_EDGES_TOTAL,
        "n_edges_removed": n_removed,
        "removed_length_km": removed_km,
        "total_length_km": total_length_km,
        "f_G": f_G,
        "n_unaffected": n_unaffected,
        "n_delayed": n_delayed,
        "n_disconnected": n_disconnected,
        "n_pairs": n_pairs,
        "n_components": int(n_components),
        "lcc_size": lcc_size,
        "lcc_ratio": lcc_ratio,
        "elapsed_s": elapsed_s,
    }


# ---------------------------------------------------------------------------
# Checkpoint / resume helpers
# ---------------------------------------------------------------------------

def load_checkpoint(csv_path: Path) -> Tuple[set, List[Dict]]:
    completed: set = set()
    rows: List[Dict] = []

    if not csv_path.exists():
        print(f"  No existing CSV found at {csv_path} -- starting from scratch.")
        return completed, rows

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ipct = int(row["intensity_pct"])
                tidx = int(row["trial_idx"])
            except (KeyError, ValueError) as e:
                print(f"  WARNING: skipping malformed CSV row: {e}")
                continue
            completed.add((ipct, tidx))
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = int(v) if k in {
                        "intensity_pct", "trial_idx", "seed",
                        "n_edges_total", "n_edges_removed",
                        "n_unaffected", "n_delayed", "n_disconnected", "n_pairs",
                        "n_components", "lcc_size",
                    } else float(v)
                except (ValueError, TypeError):
                    parsed[k] = v
            rows.append(parsed)

    print(f"  Loaded checkpoint: {len(completed):,} completed simulations from {csv_path}")
    return completed, rows


def save_checkpoint_pickle(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    with open(path, "wb") as f:
        pickle.dump(df, f)


def _get_progress_bar(total: int, desc: str):
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc=desc, unit="sim", dynamic_ncols=True)
    except ImportError:
        return _PlainProgressBar(total, desc)


def _progress_write(bar, message: str) -> None:
    if hasattr(bar, "write"):
        bar.write(message)
    else:
        print(message)


class _PlainProgressBar:
    """Minimal fallback progress bar (no tqdm dependency)."""

    def __init__(self, total: int, desc: str, min_interval: float = 0.2):
        self.total = total
        self.desc = desc
        self.n = 0
        self.postfix = ""
        self.start = time.monotonic()
        self._last_print = 0.0
        self._min_interval = min_interval

    def update(self, n: int = 1) -> None:
        self.n += n
        now = time.monotonic()
        if now - self._last_print >= self._min_interval or self.n == self.total:
            self._render()
            self._last_print = now

    def set_postfix(self, **kwargs) -> None:
        self.postfix = ", ".join(f"{k}={v}" for k, v in kwargs.items())

    def _render(self) -> None:
        elapsed = time.monotonic() - self.start
        rate = self.n / elapsed if elapsed > 0 else 0.0
        pct = 100 * self.n / self.total if self.total else 100.0
        eta_s = (self.total - self.n) / rate if rate > 0 else float("nan")
        print(
            f"\r{self.desc}: {self.n:,}/{self.total:,} ({pct:5.1f}%) "
            f"[{elapsed:,.0f}s elapsed, {rate:.2f}/s, ETA {eta_s:,.0f}s]"
            f"{' -- ' + self.postfix if self.postfix else ''}",
            end="", flush=True,
        )

    def close(self) -> None:
        self._render()
        print()


CSV_FIELDNAMES = [
    "intensity_pct", "trial_idx", "seed",
    "n_edges_total", "n_edges_removed", "removed_length_km", "total_length_km",
    "f_G", "n_unaffected", "n_delayed", "n_disconnected", "n_pairs",
    "n_components", "lcc_size", "lcc_ratio",
    "elapsed_s",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_smoothed_graph(path: Path) -> Tuple[nx.MultiGraph, List[NodeId]]:
    """Load the smoothed graph + county node list -- the
    {"graph", "C_county_nodes"} dict shape 4_graph_smoothing.py writes."""
    if not path.exists():
        raise SystemExit(f"Input not found: {path}")

    with open(path, "rb") as f:
        obj = pickle.load(f)

    if not (isinstance(obj, dict) and "graph" in obj):
        raise SystemExit(
            f"{path} is not the expected {{'graph', 'C_county_nodes'}} dict "
            f"produced by 4_graph_smoothing.py -- refusing to guess."
        )

    G = obj["graph"]
    C = list(obj.get("C_county_nodes", []))

    if not C:
        raise SystemExit(
            f"Loaded graph but C_county_nodes ({len(C)}) is empty -- nothing "
            f"to disrupt a travel-time matrix over."
        )

    n_pairs = len(C) * (len(C) - 1) // 2
    print(f"Loaded graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    print(f"County nodes (C): {len(C):,}   unique routes N = C(|C|, 2) = {n_pairs:,}")
    return G, C


def load_baseline_matrix(path: Path, C: List[NodeId]) -> pd.DataFrame:
    """Load 5_travel_time_matrix.py's own output as the undisrupted
    baseline. Does NOT recompute it -- if it doesn't match the smoothed
    graph's county list, something is inconsistent upstream."""
    if not path.exists():
        raise SystemExit(
            f"Baseline travel-time matrix not found: {path}\n"
            f"Run 5_travel_time_matrix.py first -- this script reuses its "
            f"output as the undisrupted baseline rather than recomputing "
            f"it from scratch."
        )

    with open(path, "rb") as f:
        obj = pickle.load(f)

    if not (isinstance(obj, dict) and "matrix" in obj):
        raise SystemExit(f"{path} is not the expected {{'matrix', ...}} dict from 5_travel_time_matrix.py.")

    matrix = obj["matrix"]
    saved_C = list(obj.get("C_county_nodes", []))

    if saved_C != C:
        raise SystemExit(
            f"C_county_nodes in {path} does not match the smoothed graph's "
            f"C_county_nodes. This baseline matrix was computed against a "
            f"different (or since-changed) graph -- re-run "
            f"5_travel_time_matrix.py against the current smoothed graph "
            f"before trusting delta_t values from this script."
        )

    n = matrix.shape[0]
    iu = np.triu_indices(n, k=1)
    off_diag = matrix.to_numpy(dtype=float)[iu]
    if (off_diag == float("inf")).any():
        raise SystemExit(
            f"Baseline matrix in {path} contains infinite (disconnected) "
            f"off-diagonal entries. The UNDISRUPTED network must be "
            f"connected so every unique county pair has a finite travel "
            f"time -- fix Step 3/4 (smoothing/connectivity) before running "
            f"disruption simulations."
        )

    print(f"Loaded baseline travel-time matrix: {n:,} x {n:,} counties, "
          f"all unique pairs finite (undisrupted network is connected).")
    return matrix


def total_network_length_km(G: nx.MultiGraph) -> float:
    total = sum(d.get(LENGTH_ATTR, 0.0) for _, _, d in G.edges(data=True))
    if total <= 0:
        raise SystemExit(
            f"Total network length computed as {total} -- every edge is "
            f"missing '{LENGTH_ATTR}' or the graph is empty."
        )
    return total


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def _prepare_worker_data(G: nx.MultiGraph, C: List[NodeId], baseline: pd.DataFrame) -> Tuple:
    """Pre-compute the integer mappings and arrays needed for the fast
    SciPy path, including the baseline reduced to its unique-pair vector."""
    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    n_nodes = len(nodes)

    county_idxs = np.array([node_to_idx[c] for c in C])

    n_c = len(C)
    triu_indices = np.triu_indices(n_c, k=1)
    baseline_triu = baseline.to_numpy(dtype=float)[triu_indices]

    all_edges_keyed = []
    for u, v, k, d in G.edges(keys=True, data=True):
        i = node_to_idx[u]
        j = node_to_idx[v]
        w = d.get(WEIGHT_ATTR, float("inf"))
        l = d.get(LENGTH_ATTR, 0.0)
        all_edges_keyed.append((i, j, k, w, l))

    n_edges_total = G.number_of_edges()

    return all_edges_keyed, baseline_triu, triu_indices, county_idxs, n_nodes, n_edges_total


def estimate_and_confirm(
    G: nx.MultiGraph, C: List[NodeId], baseline: pd.DataFrame,
    total_length_km: float, intensities: List[int], n_trials: int, seed: int,
    workers: int, pause_s: float = 8.0,
) -> None:
    if not intensities:
        raise SystemExit("No intensities to run -- check --min-intensity/--max-intensity/--step.")

    print("\nPreparing fast simulation data structures...")
    worker_args = _prepare_worker_data(G, C, baseline)
    worker_initializer(*worker_args)

    probe_intensity = intensities[len(intensities) // 2]

    print(f"Timing one optimized probe simulation at intensity={probe_intensity}% ...")
    start = time.monotonic()
    result = run_trial(probe_intensity, 0, seed, total_length_km)
    probe_elapsed = time.monotonic() - start

    total_sims = len(intensities) * n_trials
    effective_workers = max(1, workers * 0.85)
    projected_s = (probe_elapsed * total_sims) / effective_workers

    print(
        f"  Probe took {probe_elapsed:.4f}s for one (intensity, trial) simulation "
        f"({result['n_edges_removed']:,} edges removed, {result['removed_length_km']:,.1f} km).\n"
        f"  Full sweep: {len(intensities)} intensit{'y' if len(intensities)==1 else 'ies'} "
        f"x {n_trials} trial(s) = {total_sims:,} total simulations.\n"
        f"  Using {workers} worker(s).\n"
        f"  PROJECTED TOTAL RUNTIME: ~{projected_s:,.0f}s "
        f"(~{projected_s/60:,.1f} min, ~{projected_s/3600:,.2f} hr).\n"
        f"  If this is too slow, Ctrl-C now."
    )
    print(f"  Starting full sweep in {pause_s:.0f}s...")
    time.sleep(pause_s)


def run_sweep(
    G: nx.MultiGraph, C: List[NodeId], baseline: pd.DataFrame,
    total_length_km: float, intensities: List[int], n_trials: int, seed: int,
    csv_path: Path, checkpoint_path: Path, workers: int, *, resume: bool = False,
) -> pd.DataFrame:
    """Run every (intensity, trial) combination in parallel using
    ProcessPoolExecutor. Writes results incrementally."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if resume:
        completed, rows = load_checkpoint(csv_path)
    else:
        completed, rows = set(), []

    total_sims = len(intensities) * n_trials
    already_done = len(completed)
    remaining_sims = total_sims - already_done

    if remaining_sims <= 0:
        print(f"\nAll {total_sims:,} simulations already completed -- nothing to do.")
        return pd.DataFrame(rows)

    if resume and already_done > 0:
        print(f"  Resuming: {already_done:,}/{total_sims:,} already done, "
              f"{remaining_sims:,} remaining.")

    print("Preparing graph data structures for parallel workers...")
    worker_args = _prepare_worker_data(G, C, baseline)

    sweep_start = time.monotonic()
    bar = _get_progress_bar(total_sims, desc="Disruption sweep")
    if already_done > 0:
        bar.n = already_done
        bar.update(0)

    csv_mode = "a" if (resume and csv_path.exists() and already_done > 0) else "w"
    write_header = (csv_mode == "w")

    tasks_by_intensity = {}
    for intensity_pct in intensities:
        trials = [t for t in range(n_trials) if (intensity_pct, t) not in completed]
        if trials:
            tasks_by_intensity[intensity_pct] = trials

    print(f"Starting parallel execution with {workers} worker process(es)...")

    try:
        with open(csv_path, csv_mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            if write_header:
                writer.writeheader()
                f.flush()

            for intensity_pct in intensities:
                trials = tasks_by_intensity.get(intensity_pct, [])
                if not trials:
                    continue

                intensity_start = time.monotonic()
                intensity_results = []

                if workers == 1:
                    worker_initializer(*worker_args)
                    for trial_idx in trials:
                        res = run_trial(intensity_pct, trial_idx, seed, total_length_km)
                        writer.writerow(res)
                        f.flush()
                        rows.append(res)
                        intensity_results.append(res)
                        bar.set_postfix(
                            intensity=f"{res['intensity_pct']}%",
                            trial=f"{res['trial_idx'] + 1}/{n_trials}",
                            sim_time=f"{res['elapsed_s']:.3f}s",
                            f_G=f"{res['f_G']:.4f}",
                            lcc=f"{res['lcc_ratio']:.4f}",
                        )
                        bar.update(1)
                else:
                    with ProcessPoolExecutor(
                        max_workers=workers,
                        initializer=worker_initializer,
                        initargs=worker_args,
                    ) as executor:
                        futures = {
                            executor.submit(
                                run_trial, intensity_pct, trial_idx, seed, total_length_km
                            ): trial_idx for trial_idx in trials
                        }
                        for future in futures:
                            try:
                                res = future.result()
                                writer.writerow(res)
                                f.flush()
                                rows.append(res)
                                intensity_results.append(res)
                                bar.set_postfix(
                                    intensity=f"{res['intensity_pct']}%",
                                    trial=f"{res['trial_idx'] + 1}/{n_trials}",
                                    sim_time=f"{res['elapsed_s']:.3f}s",
                                    f_G=f"{res['f_G']:.4f}",
                                    lcc=f"{res['lcc_ratio']:.4f}",
                                )
                                bar.update(1)
                            except Exception as e:
                                _progress_write(bar, f"Worker error on intensity {intensity_pct}: {e}")
                                raise

                f_vals = [r["f_G"] for r in intensity_results]
                lcc_vals = [r["lcc_ratio"] for r in intensity_results]
                intensity_elapsed = time.monotonic() - intensity_start
                _progress_write(
                    bar,
                    f"Intensity {intensity_pct}% complete in {intensity_elapsed:,.1f}s: "
                    f"f(G) mean={np.mean(f_vals):.4f} std={np.std(f_vals):.4f}  |  "
                    f"LCC ratio mean={np.mean(lcc_vals):.4f} std={np.std(lcc_vals):.4f} "
                    f"over {len(f_vals)} trial(s)",
                )

                save_checkpoint_pickle(rows, checkpoint_path)

    finally:
        bar.close()

    sweep_elapsed = time.monotonic() - sweep_start
    print(f"\nFull sweep finished in {sweep_elapsed:,.1f}s "
          f"({sweep_elapsed/60:,.1f} min). Incremental CSV written throughout to: {csv_path}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-network run
# ---------------------------------------------------------------------------

def run_one_network(network: str, args: argparse.Namespace, intensities: List[int]) -> bool:
    """Run Step 5a end to end for a single network.

    Returns True on success. Returns False (without raising) if the
    smoothed graph pickle doesn't exist, so a multi-network run can skip
    this network and continue with the other one instead of aborting
    entirely -- matching the "both by default, skip what's missing"
    convention used everywhere else in this pipeline.
    """
    graph_path = SMOOTHED_INPUT_DIR / NETWORK_GRAPH_FILENAMES[network]
    baseline_path = BASELINE_INPUT_DIR / NETWORK_BASELINE_FILENAMES[network]
    output_csv_path = REPORTS_DIR / f"{network}_random_disruption_trials.csv"
    output_pickle_path = ANALYSIS_OUTPUT_DIR / f"{network}_random_disruption.pkl"
    checkpoint_pickle_path = ANALYSIS_OUTPUT_DIR / f"{network}_random_disruption_checkpoint.pkl"

    print()
    print("#" * 70)
    print(f"# NETWORK: {network}")
    print("#" * 70)

    if not graph_path.exists():
        print(
            f"  WARNING: smoothed graph pickle not found, skipping this network:\n"
            f"    {graph_path}\n"
            f"  (Run 4_graph_smoothing.py for this network first.)"
        )
        return False

    _print_banner(f"STEP 5a -- RANDOM DISRUPTION SIMULATION [{network}]")
    print(f"Smoothed graph:    {graph_path}")
    print(f"Baseline matrix:   {baseline_path}")
    print(f"Intensities:       {intensities[0]}% to {intensities[-1]}% step {args.step}% "
          f"({len(intensities)} levels)")
    print(f"Trials/intensity:  {args.n_trials}")
    print(f"Workers:           {args.workers}")
    print(f"Base seed:         {args.seed}")
    print(f"Resume mode:       {'ON' if args.resume else 'OFF'}")
    print(f"Output CSV:        {output_csv_path}")
    print(f"Output pickle:     {output_pickle_path}")
    print(f"Checkpoint pickle: {checkpoint_pickle_path}")

    G, C = load_smoothed_graph(graph_path)
    baseline = load_baseline_matrix(baseline_path, C)
    total_km = total_network_length_km(G)
    print(f"Total undisrupted network length: {total_km:,.1f} km")

    if not args.skip_runtime_check and not args.resume:
        estimate_and_confirm(
            G, C, baseline, total_km, intensities, args.n_trials, args.seed, args.workers
        )

    if args.resume:
        print("\nResuming from last checkpoint...")
    else:
        print("\nRunning full sweep...")

    results = run_sweep(
        G, C, baseline, total_km, intensities, args.n_trials, args.seed,
        output_csv_path, checkpoint_pickle_path, args.workers, resume=args.resume,
    )

    output_pickle_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pickle_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved full results DataFrame to: {output_pickle_path} ({len(results):,} rows)")

    if checkpoint_pickle_path.exists():
        checkpoint_pickle_path.unlink()
        print(f"Removed checkpoint file: {checkpoint_pickle_path}")

    _print_banner(f"RANDOM DISRUPTION SWEEP COMPLETE [{network}]")
    summary = results.groupby("intensity_pct")[["f_G", "lcc_ratio"]].agg(["mean", "std"])
    print(summary.to_string())

    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 5a: random disruption simulation for the smoothed "
            "HIGHWAY and RAILWAY graphs. Runs BOTH networks automatically "
            "unless --network restricts it to one."
        )
    )
    parser.add_argument(
        "--network", choices=sorted(NETWORK_LABELS), default=None,
        help=(
            "Restrict the run to just this one network. If omitted, BOTH "
            "highway and railway are processed automatically (matching "
            "every other script in this pipeline)."
        ),
    )
    parser.add_argument(
        "--min-intensity", type=int, default=DEFAULT_MIN_INTENSITY,
        help=f"Lowest disruption intensity, in %% of total network length (default: {DEFAULT_MIN_INTENSITY}).",
    )
    parser.add_argument(
        "--max-intensity", type=int, default=DEFAULT_MAX_INTENSITY,
        help=f"Highest disruption intensity, inclusive (default: {DEFAULT_MAX_INTENSITY}).",
    )
    parser.add_argument(
        "--step", type=int, default=DEFAULT_STEP,
        help=f"Intensity step size (default: {DEFAULT_STEP}).",
    )
    parser.add_argument(
        "--n-trials", type=int, default=DEFAULT_N_TRIALS,
        help=f"Random trials per intensity (default: {DEFAULT_N_TRIALS}).",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Base RNG seed for reproducible trials (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--workers", type=int, default=multiprocessing.cpu_count(),
        help="Number of parallel worker processes to use (default: logical CPU count).",
    )
    parser.add_argument(
        "--skip-runtime-check", action="store_true",
        help="Skip the probe-run runtime projection and confirmation pause.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from the last checkpoint(s) per network.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    intensities = list(range(args.min_intensity, args.max_intensity + 1, args.step))

    if args.network:
        networks_to_run = [args.network]
    else:
        networks_to_run = list(NETWORK_LABELS)

    print(f"Running Step 5a (random disruption) for: {', '.join(networks_to_run)}")

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
        "\nNext step: targeted disruption (deterministic, driven by "
        "6_edge_betweenness.py's fixed centrality ranking, no trials/RNG) -- "
        "see 7_2_targeted_disruption.py, which reuses this script's "
        "evaluation logic. Once both results files exist, plot f(G) vs. "
        "intensity for targeted (single curve) against random (mean +/- "
        "std band from this file's output)."
    )


if __name__ == "__main__":
    main()