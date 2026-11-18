#!/usr/bin/env python3
"""
Step 6a -- Random repair (reverse of Step 5a random disruption).
FAST REWRITE: SciPy sparse evaluation core + (trial x checkpoint) parallelism.

See 8.0_common_repair_utilities.py's module docstring for the key
assumption this script makes: since 7_1_random_disruption.py never
persists the actual disrupted graph object, this script first REPLAYS
7_1's own random-edge-removal procedure (same algorithm, one fixed seed)
to reconstruct "the exact --intensity% damaged network" (default: 50%,
the disruption phase's endpoint), then reverses it.

What is randomized here is NOT which edges got removed (that is fixed
once per --disruption-seed, via the replay above) -- it is the REPAIR
ORDER. For each of --n-trials trials, this script draws a different
random permutation of the SAME removed-edge list and restores them in
that order, evaluating the same recovery metrics after every
restoration. Trials are then averaged (mean/std) per number of edges
repaired, mirroring how 7_1 averages its disruption trials per
intensity.

WHY THIS VERSION IS FAST (and what did NOT change)
---------------------------------------------------
The output schema, the metric definitions, and the repair-order RNG are
unchanged -- `--seed`/`--n-trials`/`--log-every` reproduce the same
`n_repaired` checkpoints and the same permutations as before, so results
are directly comparable to earlier runs.

8.0's `evaluate_state` is already sensible about the expensive part: one
`scipy.sparse.csgraph.dijkstra` call from every county at once, with
parallel edges deduped to their minimum weight. Dijkstra is ~99% of an
evaluation's cost and that call is not improved on here. What this
version removes is the per-checkpoint SETUP around it, and the ceiling
on how many cores the sweep can actually use:

  1. The graph -> sparse rebuild happens ONCE, not once per checkpoint.
     8.0's `compute_od_matrix_and_structure` rebuilds `list(G.nodes())`,
     a node->index dict, a Python-level min-per-pair dict over every
     edge, three Python lists, and a tuple-indexed DataFrame on EVERY
     call. This script instead builds one immutable integer/float edge
     table up front and represents any repair state as a boolean mask
     over it, so each checkpoint does array work plus the Dijkstra and
     nothing else. That also removes the per-trial `G.copy()` and the
     per-edge `add_edge`.

  2. Parallelism at (trial, checkpoint) granularity, not per trial.
     The old code submitted one task per trial AND defaulted to
     `min(2, cpu_count)` workers -- so on a 16-core box it used two
     cores, and even with --workers raised it could never exceed
     --n-trials of them, finishing as slowly as the unluckiest worker.
     Work is now chunked as (trial, slice-of-checkpoints); every chunk
     recomputes its own prefix from the trial's seed, so chunks are
     independent, evenly sized, and scale past --n-trials cores. Default
     is now every core. No shared mutable state, no progress-queue
     monitor thread.

  3. `--eval-points K` evaluates at K evenly spaced checkpoints instead
     of all of them. Since cost is exactly linear in
     (n_trials x n_checkpoints) and Dijkstra is irreducible, this is by
     far the biggest remaining lever: at 50% intensity there are
     thousands of removed edges, and the recovery curve is smooth enough
     that K=50-100 usually looks identical to `--log-every 1`.

Parallel edges are still collapsed to the fastest ACTIVE edge per node
pair before each Dijkstra -- required for correctness, not speed, since
`csr_matrix` SUMS duplicate (i, j) entries and would otherwise invent a
slow combined edge on a MultiGraph. Which parallel edge is active changes
as repair proceeds, so this is recomputed per state exactly as 8.0 does.

Correctness is not assumed: `--validate N` (on by default, N=2)
re-evaluates a few sampled repair states through 8.0's own
`evaluate_state` and asserts every metric agrees, so a divergence
surfaces immediately rather than silently contaminating a long run.

Output (unchanged)
------------------
    reports/<network>_random_repair_trials.csv
        One row per (trial, n_repaired): trial_idx, n_repaired,
        n_remaining, repaired_length_km, cumulative_pct_of_total,
        functionality, n_unaffected/delayed/disconnected, n_pairs,
        n_components, lcc_size, lcc_ratio.
    reports/<network>_random_repair_summary.csv
        Same rows aggregated (mean/std) by n_repaired -- the recovery
        curve to actually plot.
    data/analysis/<network>_random_repair.pkl

Usage
-----
    python 8.1_random_repair.py                        # BOTH networks
    python 8.1_random_repair.py --network highway
    python 8.1_random_repair.py --n-trials 100 --eval-points 50
    python 8.1_random_repair.py --workers 16 --no-validate
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import multiprocessing as mp
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as sp_connected_components
from scipy.sparse.csgraph import dijkstra as sp_dijkstra


# ---------------------------------------------------------------------------
# Shared repair utilities (8.0) -- loaded by path, since a digit-leading
# filename isn't a valid Python module identifier. Unchanged from the
# original script.
# ---------------------------------------------------------------------------

_COMMON_UTILITIES_NAMES = (
    "8.0_common_repair_utilities.py",   # the name 8_2/8_3 also reference
    "8_0_common_repair_utilities.py",   # underscore variant, in case of a rename
)
_COMMON_UTILITIES_PATH = next(
    (p for p in (Path(__file__).with_name(n) for n in _COMMON_UTILITIES_NAMES) if p.exists()),
    None,
)
if _COMMON_UTILITIES_PATH is None:
    raise ImportError(
        "Could not find the shared repair utilities module next to this script. "
        f"Looked for {list(_COMMON_UTILITIES_NAMES)} in "
        f"{Path(__file__).resolve().parent}."
    )
_COMMON_UTILITIES_SPEC = importlib.util.spec_from_file_location(
    "common_repair_utilities", _COMMON_UTILITIES_PATH
)
if _COMMON_UTILITIES_SPEC is None or _COMMON_UTILITIES_SPEC.loader is None:
    raise ImportError(f"Unable to load repair utilities from {_COMMON_UTILITIES_PATH}")
rc = importlib.util.module_from_spec(_COMMON_UTILITIES_SPEC)
_COMMON_UTILITIES_SPEC.loader.exec_module(rc)

# Attribute names are read off 8.0 when it defines them, so this script
# never hardcodes a name that module might have changed.
WEIGHT_ATTR = getattr(rc, "WEIGHT_ATTR", "travel_time_min")
LENGTH_ATTR = getattr(rc, "LENGTH_ATTR", "length_km")

# Metric keys the fast path produces. Used by the validation cross-check
# to report anything 8.0's evaluate_state returns that this script does
# not reproduce (rather than silently dropping a column).
FAST_METRIC_KEYS = (
    "n_unaffected", "n_delayed", "n_disconnected",
    "frac_unaffected", "frac_delayed", "frac_disconnected",
    "functionality",
    "n_pairs", "n_components", "lcc_size", "lcc_ratio",
)

# Aggregated in the summary CSV -- unchanged from the original script.
AGG_COLS = [
    "functionality", "lcc_ratio", "n_components",
    "frac_unaffected", "frac_delayed", "frac_disconnected",
    "cumulative_pct_of_total",
]


# ---------------------------------------------------------------------------
# Sparse state model
# ---------------------------------------------------------------------------
#
# Every repair state is just a boolean mask over ONE fixed edge table:
#
#   edge indices [0, n_surv)                 -> edges that survived the
#                                               disruption (always active)
#   edge indices [n_surv, n_surv + n_removed) -> the removed edges, in the
#                                               exact order of
#                                               `removed_edges`, so repair
#                                               step k activates
#                                               n_surv + order[:k]
#
# Nothing below ever mutates a graph object, so there is no per-trial
# `G.copy()` and no per-edge `add_edge`.


def _pair_structures(rows: np.ndarray, cols: np.ndarray, n_nodes: int) -> dict:
    """Group edges by undirected node pair, so parallel edges can be
    collapsed to their fastest ACTIVE member at evaluation time.

    Why this matters: `csr_matrix((w, (r, c)))` SUMS duplicate (i, j)
    entries. Handing a MultiGraph's parallel edges straight to SciPy would
    therefore fabricate a slower combined edge instead of keeping the real
    fastest one. 7_1_random_disruption.py collapses to the minimum for the
    same reason; this does it per state, since which parallel edge is
    active changes as repair proceeds.

    Self-loops are dropped: they can never shorten a path and never affect
    connectivity, so they are simply never activated in the matrix (their
    length still counts in the km bookkeeping, which is handled separately).
    """
    lo = np.minimum(rows, cols).astype(np.int64)
    hi = np.maximum(rows, cols).astype(np.int64)
    not_self = lo != hi

    key = lo * n_nodes + hi
    uniq, inv_kept = np.unique(key[not_self], return_inverse=True)

    inv = np.full(rows.shape[0], -1, dtype=np.int64)
    inv[not_self] = inv_kept

    counts = np.bincount(inv_kept, minlength=uniq.shape[0])

    # Edges whose pair has exactly one member: a plain scatter-assign.
    single_pair_mask = counts[inv] == 1
    single_pair_mask &= not_self
    single_edge_idx = np.flatnonzero(single_pair_mask)
    single_pair_id = inv[single_edge_idx]

    # The handful of genuinely parallel pairs get a tiny per-group min.
    dup_groups: List[Tuple[int, np.ndarray]] = []
    dup_pair_ids = np.flatnonzero(counts > 1)
    if dup_pair_ids.size:
        dup_edge_idx = np.flatnonzero((counts[inv] > 1) & not_self)
        order = np.argsort(inv[dup_edge_idx], kind="stable")
        dup_edge_idx = dup_edge_idx[order]
        group_ids = inv[dup_edge_idx]
        boundaries = np.flatnonzero(np.diff(group_ids)) + 1
        for chunk in np.split(dup_edge_idx, boundaries):
            dup_groups.append((int(inv[chunk[0]]), chunk))

    return {
        "n_upairs": int(uniq.shape[0]),
        "pair_r": (uniq // n_nodes).astype(np.int32),
        "pair_c": (uniq % n_nodes).astype(np.int32),
        "single_edge_idx": single_edge_idx.astype(np.int64),
        "single_pair_id": single_pair_id.astype(np.int64),
        "dup_groups": dup_groups,
        "n_parallel_pairs": int(dup_pair_ids.size),
        "n_self_loops": int((~not_self).sum()),
    }


def _edge_attr(d: dict, key: str, where: str) -> float:
    """Read one required numeric edge attribute, with a useful error."""
    try:
        return float(d[key])
    except KeyError as exc:
        raise SystemExit(
            f"A {where} edge carries no '{key}' attribute (attrs: "
            f"{sorted(d)}). Every edge in a genuine Step 4 output has both "
            f"'{WEIGHT_ATTR}' and '{LENGTH_ATTR}' -- the same precondition "
            f"7_2_targeted_disruption.py validates. Fix Steps 2.7/3/4 first."
        ) from exc


def _baseline_in_C_order(baseline: pd.DataFrame, C: Sequence) -> np.ndarray:
    """Return the baseline matrix as a plain array in C's own order.

    8.0's `upper_triangle_values` reads the baseline POSITIONALLY
    (`df.to_numpy()[iu]`), which is correct only while the matrix's
    index/column order already equals `C_county_nodes`. It normally does,
    since 5_travel_time_matrix.py reindexes to the same list this graph
    pickle carries -- so the common path here is a no-op that keeps this
    script bit-identical to the rest of the pipeline. If the orders ever
    diverge, this reorders and says so loudly, because in that case the
    positional read elsewhere in the pipeline is silently comparing the
    wrong county pairs.
    """
    if baseline.shape != (len(C), len(C)):
        raise SystemExit(
            f"Baseline matrix is {baseline.shape}, expected "
            f"({len(C)}, {len(C)}) to match the graph's C_county_nodes. "
            f"Re-run 5_travel_time_matrix.py against the current smoothed graph."
        )

    target = [tuple(c) for c in C]
    idx = [tuple(k) if isinstance(k, tuple) else k for k in baseline.index]
    cols = [tuple(k) if isinstance(k, tuple) else k for k in baseline.columns]
    M = baseline.to_numpy(dtype=float)

    if idx == target and cols == target:
        return M

    pos_r = {k: i for i, k in enumerate(idx)}
    pos_c = {k: i for i, k in enumerate(cols)}
    try:
        ri = [pos_r[c] for c in target]
        ci = [pos_c[c] for c in target]
    except KeyError as exc:
        raise SystemExit(
            f"County node {exc.args[0]!r} is in the graph's C_county_nodes but "
            f"not in the baseline travel-time matrix. Re-run "
            f"5_travel_time_matrix.py against the current smoothed graph."
        ) from exc

    print("  WARNING: the baseline matrix's row/column order differs from the "
          "graph's C_county_nodes. Reordering it here -- but note 8.0/7_2 read "
          "that matrix positionally, so they would have compared the wrong "
          "county pairs. Re-run 5_travel_time_matrix.py.")
    return M[np.ix_(ri, ci)]


def prepare_sparse_state(
    G_intact, G_disrupted, removed_edges, C, baseline, total_km: float
) -> dict:
    """Build the immutable arrays every evaluation reads from.

    Node indexing comes from the INTACT graph, so component counts and
    lcc_ratio are taken over the same node universe the slow path sees
    (removing an edge in NetworkX leaves both endpoints in place).
    """
    nodes = list(G_intact.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    n_nodes = len(nodes)

    missing = [n for n in G_disrupted.nodes() if n not in node_to_idx]
    if missing:
        raise SystemExit(
            f"{len(missing)} node(s) in the disrupted graph are absent from the "
            f"intact graph -- the replay in build_random_disrupted_state() is "
            f"not a pure edge removal, so node-based metrics (n_components, "
            f"lcc_ratio) can't be compared across states. Investigate 8.0."
        )

    # --- surviving edges ---
    surv_it = (
        G_disrupted.edges(keys=True, data=True) if G_disrupted.is_multigraph()
        else ((u, v, None, d) for u, v, d in G_disrupted.edges(data=True))
    )
    s_rows, s_cols, s_w = [], [], []
    for u, v, _k, d in surv_it:
        s_rows.append(node_to_idx[u])
        s_cols.append(node_to_idx[v])
        s_w.append(_edge_attr(d, WEIGHT_ATTR, "surviving"))
    n_surv = len(s_rows)

    # --- removed edges, in `removed_edges` order ---
    r_rows, r_cols, r_w, r_len = [], [], [], []
    for edge_id, d in removed_edges:
        u, v = edge_id[0], edge_id[1]
        r_rows.append(node_to_idx[u])
        r_cols.append(node_to_idx[v])
        r_w.append(_edge_attr(d, WEIGHT_ATTR, "removed"))
        r_len.append(_edge_attr(d, LENGTH_ATTR, "removed"))

    rows = np.asarray(s_rows + r_rows, dtype=np.int64)
    cols = np.asarray(s_cols + r_cols, dtype=np.int64)
    w = np.asarray(s_w + r_w, dtype=np.float64)

    if not np.all(np.isfinite(w)) or np.any(w <= 0):
        n_bad = int((~np.isfinite(w)).sum() + (w <= 0).sum())
        raise SystemExit(
            f"{n_bad} edge(s) have a non-finite or non-positive "
            f"'{WEIGHT_ATTR}'. SciPy's sparse representation treats a zero "
            f"weight as 'no edge', so these would silently disappear. Fix "
            f"Steps 2.7/3/4 (same validation 7_2_targeted_disruption.py does)."
        )

    base_active = np.zeros(rows.shape[0], dtype=bool)
    base_active[:n_surv] = True

    state = {
        "n_nodes": n_nodes,
        "n_surv": n_surv,
        "n_removed": len(r_rows),
        "w": w,
        "rem_len": np.asarray(r_len, dtype=np.float64),
        "base_active": base_active,
        "county_idx": np.asarray([node_to_idx[c] for c in C], dtype=np.int64),
        "total_km": float(total_km),
    }
    state.update(_pair_structures(rows, cols, n_nodes))

    n_c = len(C)
    iu = np.triu_indices(n_c, k=1)
    state["iu"] = iu
    state["baseline_triu"] = _baseline_in_C_order(baseline, C)[iu]

    if np.isinf(state["baseline_triu"]).any():
        raise SystemExit(
            "Baseline matrix contains infinite entries among the unique "
            "county-to-county pairs -- the intact network must be connected "
            "before repair curves mean anything. Fix Steps 3/4 first."
        )

    print(f"Sparse model: {n_nodes:,} nodes, {rows.shape[0]:,} edge records "
          f"({n_surv:,} surviving + {state['n_removed']:,} removed), "
          f"{state['n_upairs']:,} distinct node pairs")
    if state["n_parallel_pairs"]:
        print(f"  {state['n_parallel_pairs']:,} node pair(s) carry parallel edges -- "
              f"collapsed to the fastest ACTIVE edge per evaluation.")
    if state["n_self_loops"]:
        print(f"  {state['n_self_loops']:,} self-loop(s) excluded from the shortest-path "
              f"matrix (they can't shorten a route or change connectivity).")
    return state


def _build_matrix(S: dict, active: np.ndarray) -> csr_matrix:
    """Collapse the active edge set to one fastest edge per node pair and
    return it as a symmetric CSR matrix."""
    w_min = np.full(S["n_upairs"], np.inf)

    single_idx = S["single_edge_idx"]
    m = active[single_idx]
    w_min[S["single_pair_id"][m]] = S["w"][single_idx[m]]

    for pair_id, edge_idx in S["dup_groups"]:
        a = active[edge_idx]
        if a.any():
            w_min[pair_id] = S["w"][edge_idx][a].min()

    keep = np.isfinite(w_min)
    n = S["n_nodes"]
    if not keep.any():
        # No active edges at all (only reachable if the disruption removed
        # everything). 8.0 handles the same case with an empty matrix.
        return csr_matrix((n, n))

    wk = w_min[keep]
    r = S["pair_r"][keep]
    c = S["pair_c"][keep]

    return csr_matrix(
        (np.concatenate([wk, wk]), (np.concatenate([r, c]), np.concatenate([c, r]))),
        shape=(n, n),
    )


def evaluate_active(S: dict, active: np.ndarray) -> Dict[str, float]:
    """Fast equivalent of 8.0's `evaluate_state`, for one repair state.

    Metric definitions deliberately match 7_2_targeted_disruption.py's
    `classify_and_score`: delay relative to the undisrupted baseline over
    the N = C(|C|, 2) UNIQUE unordered county pairs (upper triangle of the
    symmetric matrix), with f = mean(1 / (1 + delta)) and disconnected
    pairs scoring 0.
    """
    A = _build_matrix(S, active)
    county_idx = S["county_idx"]

    distances = sp_dijkstra(A, directed=False, indices=county_idx, return_predecessors=False)
    tp = distances[:, county_idx][S["iu"]]
    t = S["baseline_triu"]

    with np.errstate(divide="ignore", invalid="ignore"):
        delta = (tp - t) / t
    # Kill floating-point negative noise, exactly as 7_2 does.
    delta[np.isfinite(delta) & (delta < 0) & (delta > -1e-10)] = 0.0

    unaffected = np.isclose(delta, 0.0, rtol=0.0, atol=1e-10)
    disconnected = np.isinf(tp)
    delayed = ~unaffected & ~disconnected

    functionality = 1.0 / (1.0 + delta)
    functionality = np.nan_to_num(functionality, nan=1.0, posinf=0.0, neginf=0.0)

    n_comp, labels = sp_connected_components(A, directed=False)
    sizes = np.bincount(labels)
    lcc_size = int(sizes.max()) if sizes.size else 0

    n_pairs = int(delta.size)
    n_unaffected = int(unaffected.sum())
    n_delayed = int(delayed.sum())
    n_disconnected = int(disconnected.sum())

    # Key order matches 8.0's evaluate_state() exactly -- these keys become
    # CSV columns via `**metrics`, so a different order would silently
    # reshuffle the output file's columns.
    return {
        "n_unaffected": n_unaffected,
        "n_delayed": n_delayed,
        "n_disconnected": n_disconnected,
        "frac_unaffected": n_unaffected / n_pairs,
        "frac_delayed": n_delayed / n_pairs,
        "frac_disconnected": n_disconnected / n_pairs,
        "functionality": float(functionality.mean()),
        "n_pairs": n_pairs,
        "n_components": int(n_comp),
        "lcc_size": lcc_size,
        "lcc_ratio": lcc_size / S["n_nodes"] if S["n_nodes"] else 0.0,
    }


# ---------------------------------------------------------------------------
# Checkpoints and trial work
# ---------------------------------------------------------------------------

def checkpoint_steps(n_removed: int, log_every: int, eval_points: int | None) -> List[int]:
    """The `n_repaired` values each trial reports a row for.

    With `--log-every` this reproduces the original script's checkpoints
    exactly: 0 (the disrupted state itself), then every `log_every`-th
    repaired edge, and always the fully-repaired state.
    """
    if eval_points is not None:
        if eval_points < 2:
            raise SystemExit("--eval-points must be at least 2.")
        steps = np.unique(
            np.round(np.linspace(0, n_removed, min(eval_points, n_removed + 1))).astype(int)
        )
        return [int(s) for s in steps]

    steps = [0]
    steps += [s for s in range(1, n_removed + 1) if s % log_every == 0 or s == n_removed]
    return steps


def repair_order(n_removed: int, base_seed: int, trial_idx: int) -> np.ndarray:
    """Per-trial random repair order.

    Identical RNG to the original script (`seed * 100_000 + trial_idx`,
    `default_rng(...).permutation(n)`), so a given --seed reproduces the
    same permutations and therefore the same numbers as before. Recomputed
    per work chunk rather than shared, which is what makes the chunks
    independent.
    """
    return np.random.default_rng(base_seed * 100_000 + trial_idx).permutation(n_removed)


def trial_chunk_rows(
    S: dict, base_seed: int, trial_idx: int, steps: Sequence[int]
) -> List[dict]:
    """Evaluate one trial at an ascending slice of its checkpoints."""
    order = repair_order(S["n_removed"], base_seed, trial_idx)
    cum_km = np.concatenate([[0.0], np.cumsum(S["rem_len"][order])])

    active = S["base_active"].copy()
    n_surv = S["n_surv"]
    total_km = S["total_km"]
    n_removed = S["n_removed"]

    rows: List[dict] = []
    filled = 0
    for step in steps:
        if step > filled:
            active[n_surv + order[filled:step]] = True
            filled = step
        metrics = evaluate_active(S, active)
        row = {
            "n_repaired": int(step),
            "n_remaining": int(n_removed - step),
            "repaired_length_km": float(cum_km[step]),
            "cumulative_pct_of_total": float(100.0 * cum_km[step] / total_km),
            **metrics,
        }
        # Appended last so the CSV column order matches the original script.
        row["trial_idx"] = trial_idx
        rows.append(row)
    return rows


_WORKER_STATE: dict | None = None
_WORKER_SEED: int = 0


def _init_worker(state: dict, base_seed: int) -> None:
    global _WORKER_STATE, _WORKER_SEED
    _WORKER_STATE = state
    _WORKER_SEED = base_seed


def _run_chunk(task: Tuple[int, Tuple[int, ...]]) -> List[dict]:
    trial_idx, steps = task
    return trial_chunk_rows(_WORKER_STATE, _WORKER_SEED, trial_idx, steps)


def build_tasks(
    n_trials: int, steps: Sequence[int], workers: int
) -> List[Tuple[int, Tuple[int, ...]]]:
    """Split the sweep into (trial, checkpoint-slice) chunks.

    Aims for roughly 16 chunks per worker: enough for even load balancing
    and usable progress granularity, few enough that per-chunk setup
    (one permutation + one prefix fill) stays negligible. This is what
    lets the run scale past --n-trials cores, which per-trial tasking
    could not.
    """
    per_trial_chunks = max(1, math.ceil(workers * 16 / max(1, n_trials)))
    chunk_size = max(1, math.ceil(len(steps) / per_trial_chunks))

    tasks = []
    for trial_idx in range(n_trials):
        for start in range(0, len(steps), chunk_size):
            tasks.append((trial_idx, tuple(steps[start:start + chunk_size])))
    return tasks


# ---------------------------------------------------------------------------
# Validation against the original slow path
# ---------------------------------------------------------------------------

def validate_fast_path(
    S: dict, G_disrupted, removed_edges, C, baseline, total_km: float,
    base_seed: int, n_checks: int,
) -> None:
    """Cross-check the sparse core against 8.0's own `evaluate_state`.

    Picks a few repair states from trial 0, rebuilds each as a real
    NetworkX graph via 8.0's own `restore_edge_by_id`, and compares every
    metric. A mismatch
    means the fast path and the rest of the pipeline disagree about a
    definition -- far better to fail here than after a multi-hour sweep.
    """
    n_removed = S["n_removed"]
    order = repair_order(n_removed, base_seed, 0)

    candidates = [0, n_removed // 2, n_removed]
    checks = sorted(set(candidates[:max(1, n_checks)]))

    print(f"\nValidating the fast evaluator against 8.0's evaluate_state at "
          f"n_repaired = {checks} ...")

    G_val = G_disrupted.copy()
    active = S["base_active"].copy()
    filled = 0
    worst = 0.0

    for step in checks:
        for idx in order[filled:step]:
            edge_id, data = removed_edges[idx]
            rc.restore_edge_by_id(G_val, edge_id, data)
        active[S["n_surv"] + order[filled:step]] = True
        filled = step

        slow = rc.evaluate_state(G_val, C, baseline, total_km)
        fast = evaluate_active(S, active)

        missing = [k for k in slow if k not in fast]
        if missing:
            print(f"  WARNING: 8.0's evaluate_state also returns {missing}, which "
                  f"this fast path does not compute. Those columns will be absent "
                  f"from the output -- tell me and I'll add them.")

        for key in sorted(set(slow) & set(fast)):
            a, b = slow[key], fast[key]
            if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
                if int(a) != int(b):
                    raise SystemExit(
                        f"VALIDATION FAILED at n_repaired={step}: {key} "
                        f"slow={a} vs fast={b}. Refusing to run the sweep."
                    )
            else:
                diff = abs(float(a) - float(b))
                scale = max(1.0, abs(float(a)))
                if diff / scale > 1e-9:
                    raise SystemExit(
                        f"VALIDATION FAILED at n_repaired={step}: {key} "
                        f"slow={a!r} vs fast={b!r} (abs diff {diff:.3e}). "
                        f"Refusing to run the sweep."
                    )
                worst = max(worst, diff)

    print(f"  Validation passed at every checked state (largest float "
          f"discrepancy {worst:.3e}).")


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------

class _PlainProgressBar:
    """Minimal fallback progress bar (no tqdm dependency) -- same shape as
    the one in 7_1_random_disruption.py."""

    def __init__(self, total: int, desc: str, min_interval: float = 0.3):
        self.total = total
        self.desc = desc
        self.n = 0
        self.start = time.monotonic()
        self._last = 0.0
        self._min_interval = min_interval

    def update(self, n: int = 1) -> None:
        self.n += n
        now = time.monotonic()
        if now - self._last >= self._min_interval or self.n >= self.total:
            self._render()
            self._last = now

    def _render(self) -> None:
        elapsed = time.monotonic() - self.start
        rate = self.n / elapsed if elapsed > 0 else 0.0
        pct = 100 * self.n / self.total if self.total else 100.0
        eta = (self.total - self.n) / rate if rate > 0 else float("nan")
        print(f"\r{self.desc}: {self.n:,}/{self.total:,} ({pct:5.1f}%) "
              f"[{elapsed:,.0f}s elapsed, {rate:,.1f}/s, ETA {eta:,.0f}s]",
              end="", flush=True)

    def close(self) -> None:
        self._render()
        print()


def _get_progress_bar(total: int, desc: str):
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc=desc, unit="eval", dynamic_ncols=True)
    except ImportError:
        return _PlainProgressBar(total, desc)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep(S: dict, args: argparse.Namespace, steps: Sequence[int]) -> pd.DataFrame:
    total_evals = args.n_trials * len(steps)
    workers = min(args.workers, max(1, total_evals))

    if not args.skip_runtime_check:
        mid = steps[len(steps) // 2]
        order = repair_order(S["n_removed"], args.seed, 0)
        active = S["base_active"].copy()
        active[S["n_surv"] + order[:mid]] = True
        t0 = time.monotonic()
        evaluate_active(S, active)
        probe = time.monotonic() - t0
        projected = probe * total_evals / max(1.0, workers * 0.85)
        print(f"\nProbe: one state evaluated in {probe:.4f}s.")
        print(f"  Sweep: {args.n_trials:,} trial(s) x {len(steps):,} checkpoint(s) "
              f"= {total_evals:,} evaluations across {workers} worker(s).")
        print(f"  PROJECTED RUNTIME: ~{projected:,.0f}s "
              f"(~{projected / 60:,.1f} min, ~{projected / 3600:,.2f} hr).")
        if projected > 900 and args.eval_points is None:
            print("  If that's too slow, the cheapest lever is --eval-points 50 "
                  "(evenly spaced checkpoints); the recovery curve is smooth, so "
                  "it usually looks identical. Ctrl-C now to change it.")

    tasks = build_tasks(args.n_trials, steps, workers)
    bar = _get_progress_bar(total_evals, "Random repair")
    all_rows: List[dict] = []
    t0 = time.monotonic()

    try:
        if workers == 1:
            _init_worker(S, args.seed)
            for task in tasks:
                rows = _run_chunk(task)
                all_rows.extend(rows)
                bar.update(len(rows))
        else:
            try:
                ctx = mp.get_context("fork")
            except ValueError:
                ctx = mp.get_context()
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=ctx,
                initializer=_init_worker,
                initargs=(S, args.seed),
            ) as executor:
                futures = [executor.submit(_run_chunk, task) for task in tasks]
                for future in as_completed(futures):
                    rows = future.result()
                    all_rows.extend(rows)
                    bar.update(len(rows))
    finally:
        bar.close()

    elapsed = time.monotonic() - t0
    print(f"{args.n_trials} trial(s) / {total_evals:,} evaluation(s) completed in "
          f"{elapsed:,.1f}s ({total_evals / elapsed:,.1f} eval/s).")

    all_rows.sort(key=lambda row: (row["trial_idx"], row["n_repaired"]))
    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Per-network run
# ---------------------------------------------------------------------------

def run_one_network(network: str, args: argparse.Namespace) -> bool:
    graph_path = rc.SMOOTHED_GRAPH_PATH[network]
    baseline_path = rc.BASELINE_MATRIX_PATH[network]

    print()
    print("#" * 70)
    print(f"# NETWORK: {network}")
    print("#" * 70)

    if not graph_path.exists():
        print(f"  WARNING: smoothed graph not found, skipping this network:\n    {graph_path}")
        return False

    rc._print_banner(f"STEP 6a -- RANDOM REPAIR [{network}]")
    G, C = rc.load_graph(graph_path)
    baseline = rc.load_baseline(baseline_path, G, C)
    total_km = rc.total_network_length_km(G)
    print(f"Total intact network length: {total_km:,.1f} km")

    print(f"\nReplaying random disruption to {args.intensity}% "
          f"(seed={args.disruption_seed}) to reconstruct the damaged network...")
    G_disrupted, removed_edges = rc.build_random_disrupted_state(
        G, args.intensity, seed=args.disruption_seed
    )
    n_removed = len(removed_edges)
    if n_removed == 0:
        print("  Nothing was removed at this intensity -- nothing to repair.")
        return False

    print("\nBuilding the sparse repair model (done once, reused by every trial)...")
    S = prepare_sparse_state(G, G_disrupted, removed_edges, C, baseline, total_km)

    if args.validate > 0:
        validate_fast_path(
            S, G_disrupted, removed_edges, C, baseline, total_km,
            args.seed, args.validate,
        )

    steps = checkpoint_steps(n_removed, args.log_every, args.eval_points)
    print(f"\nCheckpoints per trial: {len(steps):,} "
          f"(n_repaired from {steps[0]} to {steps[-1]})")

    df = run_sweep(S, args, steps)

    output_csv = Path(f"reports/{network}_random_repair_trials.csv")
    summary_csv = Path(f"reports/{network}_random_repair_summary.csv")
    output_pickle = Path(f"data/analysis/{network}_random_repair.pkl")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_pickle.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_csv, index=False)
    print(f"Saved per-trial results to: {output_csv} ({len(df):,} rows)")

    summary = df.groupby("n_repaired")[AGG_COLS].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(summary_csv, index=False)
    print(f"Saved aggregated recovery curve to: {summary_csv} ({len(summary):,} rows)")

    with open(output_pickle, "wb") as f:
        pickle.dump({"trials": df, "summary": summary, "n_removed": n_removed}, f)
    print(f"Saved full results to: {output_pickle}")

    print("\nRecovery curve preview (mean LCC ratio / functionality by n_repaired):")
    preview = summary[["n_repaired", "lcc_ratio_mean", "functionality_mean"]]
    stride = max(1, len(preview) // 10)
    print(preview.iloc[::stride].to_string(index=False))

    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Random repair: reverse of random disruption (fast "
                    "SciPy-sparse core, parallel over trials x checkpoints)."
    )
    p.add_argument("--network", choices=sorted(rc.NETWORK_LABELS), default=None,
                   help="Restrict to one network. If omitted, BOTH highway and railway run.")
    p.add_argument("--intensity", type=float, default=rc.DEFAULT_REPAIR_START_INTENSITY,
                   help="Disruption intensity (%% of length) to start repair from (default: 50).")
    p.add_argument("--disruption-seed", type=int, default=rc.DEFAULT_SEED,
                   help="Seed used to reconstruct the damaged network via replay (default: 42).")
    p.add_argument("--seed", type=int, default=rc.DEFAULT_SEED,
                   help="Base seed for randomizing REPAIR order across trials.")
    p.add_argument("--n-trials", type=int, default=20,
                   help="Number of random repair-order trials to average (default: 20; "
                        "paper-style reproduction uses 100).")
    p.add_argument("--log-every", type=int, default=1,
                   help="Evaluate metrics every N repaired edges (default: 1 -- every edge). "
                        "Ignored if --eval-points is given.")
    p.add_argument("--eval-points", type=int, default=None,
                   help="Instead of --log-every, evaluate at this many evenly spaced "
                        "checkpoints (e.g. 50). The single biggest speed lever: cost is "
                        "linear in the number of checkpoints, and the recovery curve is "
                        "smooth enough that 50-100 points usually matches --log-every 1.")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                   help="Parallel worker processes (default: all logical cores). Use 1 for "
                        "sequential execution.")
    p.add_argument("--validate", type=int, default=2, metavar="N",
                   help="Cross-check N sampled repair states against 8.0's own "
                        "evaluate_state before sweeping (default: 2). 0 disables.")
    p.add_argument("--no-validate", dest="validate", action="store_const", const=0,
                   help="Shorthand for --validate 0.")
    p.add_argument("--skip-runtime-check", action="store_true",
                   help="Skip the probe timing and runtime projection.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive.")
    if args.log_every <= 0:
        raise SystemExit("--log-every must be positive.")
    if args.n_trials <= 0:
        raise SystemExit("--n-trials must be positive.")

    networks = [args.network] if args.network else list(rc.NETWORK_LABELS)
    print(f"Running Step 6a (random repair) for: {', '.join(networks)}")

    ran, skipped = [], []
    for network in networks:
        ok = run_one_network(network, args)
        (ran if ok else skipped).append(network)

    print()
    rc._print_banner("ALL NETWORKS COMPLETE")
    print(f"Ran successfully: {ran or '(none)'}")
    if skipped:
        print(f"Skipped: {skipped}")
    if not ran:
        raise SystemExit(
            "\nNo network was actually processed -- every requested "
            "smoothed graph pickle was missing. Run 4_graph_smoothing.py "
            "first."
        )


if __name__ == "__main__":
    main()