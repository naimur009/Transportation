#!/usr/bin/env python3
"""
Step 10 -- Origin-destination specification and single-point disruption tables.

WHAT THIS SCRIPT PRODUCES
-------------------------
Three tables, each written BOTH as a machine-readable CSV (full precision)
and as a PNG typeset in the style of the base paper (Chung et al.,
"Contested logistics: Resilience of strategic highways and railways",
Transportation Research Interdisciplinary Perspectives 32 (2025) 101507):

  Table 1  reports/table1_od_baseline.{csv,png}
      The UNDISRUPTED specification of every origin-destination pair:
          Origin | Destination | Highway Length (km) | Travel Time (min)
                               | Railway Length (km) | Travel Time (min)
      Mirrors the paper's Table 1 ("Specifications of the US Power
      Projection Platforms"), reduced to the six columns that apply to a
      county-to-county study.

  Table 2  reports/table2_highway_disruption.{csv,png}
  Table 3  reports/table3_railway_disruption.{csv,png}
      The impact of single-point disruptions on each route, mirroring the
      paper's Tables 2 and 3:
          Origin | Destination | travel time (min)
                 | 25th percentile (time increase, % time increase)
                 | 50th percentile (time increase, % time increase)
                 | 75th percentile (time increase, % time increase)
                 | maximum disruption (time increase, % time increase)
                 | length causing maximum (km) | % length causing maximum

  Audit trail (one row per removed edge, the raw evidence behind every
  percentile in Tables 2 and 3):
      reports/table2_highway_edge_detail.csv
      reports/table3_railway_edge_detail.csv


METHODOLOGY -- traced to the base paper, NOT to the Monte Carlo sweep
--------------------------------------------------------------------
This is the single most important thing to understand about this script,
because it is easy to reach for the wrong existing artifact.

The 1%-50% / 100-trial random disruption sweep in `7.1_random_disruption.py`
is a DIFFERENT experiment (paper Sections 3.4.2 / 3.5.2, which produce the
f(G)-vs-intensity figures). It must NOT be used for Tables 2 and 3.

Tables 2 and 3 come from the paper's power-projection-platform model
(Sections 3.4.1 / 3.5.1), which is exhaustive and DETERMINISTIC -- there is
no RNG, no intensity parameter, and no trial count:

    "we focus on disrupting individual paths rather than the disruptions in
     the whole transportation network ... For each fort-to-port PPP path, we
     measured the impact of removal of each edge along the path. Here, the
     road length of the removed edge is considered as the cost of the
     disruption and the travel time delay caused by the removal of an edge
     measures the importance of the edge."   -- Section 3.4.1

    "Disruptions in the PPP model were single point disruptions of disabling
     one edge e on the shortest path between origin and destination pair od.
     The N edges on the shortest path from origin o to destination d were
     ordered by their respective delay values dt_1, dt_2, ..., dt_N, each
     associated with an edge length l_1, l_2, ..., l_N representing the cost
     of disruption. The 50th percentile delay, calculated as a
     length-weighted median, was defined as the kth delay where k satisfies

         sum_{i=1}^{k-1} l_i / L <= 1/2   and   sum_{i=k+1}^{n} l_i / L <= 1/2

     where L = sum_{i=1}^{N} l_i is the total length of the route ... In
     addition to the 50th percentile delay, the 25th and 75th percentiles and
     the maximum disruption for each installation and port pair were also
     calculated (Table 2 and 3)."   -- Section 3.5.1

So, per route:

  1. Find the shortest path by Dijkstra MINIMIZING TOTAL TRAVEL TIME
     (Section 3.3.1: "the shortest path can be determined by the Dijkstra's
     algorithm minimizing the total sum of edge travel times"). The Table 1
     "Length (km)" column is the geometric length of THAT SAME path -- the
     paper never computes a second, distance-minimizing route, so length and
     time always describe one and the same path.

  2. Remove each of that path's N edges ONE AT A TIME (restoring it before
     the next removal) and recompute the origin-destination travel time on
     the otherwise intact graph. The relative delay is

         dt_i := (t'_od - t_od) / t_od

     following Section 3.4.3, with dt_i = inf when removing edge i
     disconnects the origin from the destination entirely (t'_od = inf).
     An edge whose removal disconnects the route is a BRIDGE (cut edge) of
     the graph -- this is why `inf` is common in the railway table (sparse,
     tree-like, little redundancy) and rare outside the maximum column in
     the highway table (meshed, many detours).

  3. Sort the N delays ASCENDING, weight each by its own edge length l_i, and
     read off the delay at which the cumulative length first reaches 25%,
     50% and 75% of the route length L. This is a LENGTH-WEIGHTED quantile,
     not an ordinary one: it answers "if a failure lands uniformly at random
     along this route in proportion to length, how bad is the delay at each
     quartile", which is exactly why a long benign stretch can pin all three
     percentiles to the same value.

  4. The "maximum disruption" is the worst single-edge removal. "Length
     causing maximum (km)" is the SUMMED LENGTH OF EVERY EDGE tied at that
     maximum -- which is how a km figure stays finite even when the delay
     itself is inf. It reads as "how much infrastructure you would have to
     harden to eliminate the worst case", per Section 5: "it might be more
     difficult to strengthen railways since it would require hardening and
     building detours of length greater than 100 km in many cases, whereas
     less than 30 km is required for highways." The accompanying
     "% length causing maximum" is that sum divided by the route length L.

  5. The Average row is an UNWEIGHTED mean across routes that SILENTLY SKIPS
     inf entries, matching the paper (verified numerically against its own
     printed Table 3 seaport averages).

Because the percentiles are length-weighted over a sorted list, the columns
are monotone by construction: 25th <= 50th <= 75th <= maximum. The script
asserts this, so a violation fails loudly rather than shipping a wrong table.


INPUTS
------
    data/origin_destination/origin_des.json
        [{"origin": "Cook", "destination": "St. Clair"}, ...]
        County names WITHOUT the " County" suffix; the suffix is appended to
        match the `facility_names` node attribute written by
        `3_snap_counties.py`.

    data/graph/smoothing/<network>_smoothed.pkl
        `4_graph_smoothing.py`'s own output: a
        {"graph": nx.MultiGraph, "C_county_nodes": [...]} dict whose edges
        carry `length_km` and `travel_time_min`. Same file that
        `5_travel_time_matrix.py`, `6_edge_betweenness.py` and
        `7.1_random_disruption.py` load, so Table 1's travel times are
        directly comparable to the existing travel-time matrix.

USAGE
-----
    python src/10_od_pair_tables.py                  # all three tables
    python src/10_od_pair_tables.py --network highway    # Tables 1 + 2 only
    python src/10_od_pair_tables.py --no-png         # CSVs only, no rendering
    python src/10_od_pair_tables.py --dpi 600        # higher-resolution PNGs

Run from the project root (paths are relative to it).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless: never try to open a window

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402

# ---------------------------------------------------------------------------
# Paths -- kept in sync by hand with 4_graph_smoothing.py's own output
# convention (output_graph_dir / f"{network}_smoothed.pkl").
# ---------------------------------------------------------------------------

SMOOTHED_INPUT_DIR = Path("data/graph/smoothing")
OD_PAIRS_PATH = Path("data/origin_destination/origin_des.json")
REPORTS_DIR = Path("reports/od_pair_tables")

NETWORK_LABELS = ["highway", "railway"]
NETWORK_GRAPH_FILENAMES = {net: f"{net}_smoothed.pkl" for net in NETWORK_LABELS}

WEIGHT_ATTR = "travel_time_min"
LENGTH_ATTR = "length_km"

# `3_snap_counties.py` records counties as "Cook County"; origin_des.json
# stores the bare name "Cook". Bridging the two is a single suffix.
COUNTY_SUFFIX = " County"

# The three quantiles the paper reports, as fractions of cumulative route
# length. Section 3.5.1 defines only the 50th explicitly; the 25th and 75th
# use the identical construction with the threshold changed.
PERCENTILES: Tuple[float, float, float] = (0.25, 0.50, 0.75)

# Tolerance for deciding that two delays are "the same" when summing the
# length of every edge tied at the maximum.
TIE_REL_TOL = 1e-9
TIE_ABS_TOL = 1e-12

NodeId = Tuple[float, float]
INF = float("inf")


def _print_banner(title: str) -> None:
    """Section-banner style matching the rest of this pipeline."""
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ===========================================================================
# Data model
# ===========================================================================


@dataclass
class RouteEdge:
    """One edge lying on a shortest path, with the two attributes the paper
    needs: its length (the 'cost of disruption') and its travel time."""

    u: NodeId
    v: NodeId
    key: int
    length_km: float
    travel_time_min: float


@dataclass
class Route:
    """A time-minimizing shortest path between two county nodes."""

    origin: str
    destination: str
    network: str
    nodes: List[NodeId]
    edges: List[RouteEdge]

    @property
    def length_km(self) -> float:
        """Geometric length of the TIME-minimizing path -- the paper's
        Table 1 'Length (km)' column. Not a separate distance-minimizing
        route."""
        return sum(e.length_km for e in self.edges)

    @property
    def travel_time_min(self) -> float:
        return sum(e.travel_time_min for e in self.edges)


@dataclass
class EdgeImpact:
    """The result of removing exactly one edge from the route."""

    edge: RouteEdge
    disrupted_time_min: float  # inf when the removal disconnects o from d
    delta: float  # (t' - t) / t, inf on disconnection

    @property
    def disconnects(self) -> bool:
        return math.isinf(self.delta)


@dataclass
class RouteDisruption:
    """Every statistic Tables 2 and 3 report for a single route."""

    route: Route
    impacts: List[EdgeImpact]
    percentile_deltas: Dict[float, float] = field(default_factory=dict)
    max_delta: float = 0.0
    length_causing_max_km: float = 0.0
    n_disconnecting_edges: int = 0

    @property
    def baseline_min(self) -> float:
        return self.route.travel_time_min

    @property
    def route_length_km(self) -> float:
        return self.route.length_km

    @property
    def pct_length_causing_max(self) -> float:
        L = self.route_length_km
        return self.length_causing_max_km / L if L > 0 else 0.0

    def delta_to_minutes(self, delta: float) -> float:
        """Convert a relative delay back into a time increase in minutes.
        inf stays inf -- a disconnected route has no finite delay."""
        if math.isinf(delta):
            return INF
        return delta * self.baseline_min


# ===========================================================================
# Loading
# ===========================================================================


def load_od_pairs(path: Path) -> List[Tuple[str, str]]:
    """Read the reference origin-destination list."""
    if not path.exists():
        raise SystemExit(f"Origin-destination file not found: {path}")

    with open(path, "r") as f:
        raw = json.load(f)

    if not isinstance(raw, list) or not raw:
        raise SystemExit(f"{path} is not a non-empty JSON list of OD pairs.")

    pairs: List[Tuple[str, str]] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict) or "origin" not in entry or "destination" not in entry:
            raise SystemExit(
                f"{path} entry {i} is missing 'origin' and/or 'destination': {entry!r}"
            )
        pairs.append((str(entry["origin"]).strip(), str(entry["destination"]).strip()))

    print(f"Loaded {len(pairs)} origin-destination pair(s) from {path}")
    return pairs


def load_smoothed_graph(path: Path) -> Tuple[nx.MultiGraph, List[NodeId]]:
    """Load the {"graph", "C_county_nodes"} dict 4_graph_smoothing.py writes."""
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
        raise SystemExit(f"Loaded graph but C_county_nodes is empty in {path}.")

    print(
        f"Loaded graph: {G.number_of_nodes():,} nodes, "
        f"{G.number_of_edges():,} edges, {len(C):,} county nodes"
    )
    return G, C


def build_county_lookup(G: nx.MultiGraph, C: Sequence[NodeId]) -> Dict[str, NodeId]:
    """Map county display name -> graph node, from the `facility_names`
    attribute set by 3_snap_counties.py.

    Fails loudly if one name maps to several nodes: silently picking one
    would make every downstream number unreproducible.
    """
    lookup: Dict[str, List[NodeId]] = {}
    for node in C:
        for name in G.nodes[node].get("facility_names", []):
            lookup.setdefault(name, []).append(node)

    duplicates = {name: nodes for name, nodes in lookup.items() if len(nodes) > 1}
    if duplicates:
        listed = ", ".join(f"{n} ({len(v)} nodes)" for n, v in list(duplicates.items())[:5])
        raise SystemExit(
            f"{len(duplicates)} county name(s) map to more than one graph node "
            f"({listed}). Re-run 3_snap_counties.py -- a county must snap to "
            f"exactly one node for these tables to be reproducible."
        )

    return {name: nodes[0] for name, nodes in lookup.items()}


def resolve_county_node(
    county: str, lookup: Dict[str, NodeId], network: str
) -> NodeId:
    """Resolve a bare county name from origin_des.json to a graph node.

    Tries the bare name first, then the " County"-suffixed form, then a
    case-insensitive match, before giving up with a helpful message.
    """
    for candidate in (county, county + COUNTY_SUFFIX):
        if candidate in lookup:
            return lookup[candidate]

    lowered = {name.lower(): node for name, node in lookup.items()}
    for candidate in (county.lower(), (county + COUNTY_SUFFIX).lower()):
        if candidate in lowered:
            return lowered[candidate]

    near = sorted(n for n in lookup if county.lower()[:4] in n.lower())[:5]
    raise SystemExit(
        f"County '{county}' not found among the {network} graph's county "
        f"nodes (tried '{county}' and '{county + COUNTY_SUFFIX}'). "
        f"Closest names: {near or 'none'}"
    )


# ===========================================================================
# Shortest paths (paper Section 3.3.1 -- minimize total travel time)
# ===========================================================================


def cheapest_parallel_edge(G: nx.MultiGraph, u: NodeId, v: NodeId) -> Tuple[int, dict]:
    """On a MultiGraph two nodes may be joined by several parallel edges.
    Dijkstra traverses the fastest one, so that is the edge that actually
    lies on the path and the one whose removal we must simulate."""
    best_key, best_data, best_w = None, None, INF
    for key, data in G[u][v].items():
        w = data.get(WEIGHT_ATTR, INF)
        if w < best_w:
            best_key, best_data, best_w = key, data, w
    if best_key is None:
        raise SystemExit(f"No edge data between {u} and {v} -- graph is inconsistent.")
    return best_key, best_data


def compute_route(
    G: nx.MultiGraph, origin: str, destination: str, o_node: NodeId, d_node: NodeId,
    network: str,
) -> Route:
    """Dijkstra minimizing `travel_time_min`, then read the traversed edges
    back off the node path so we can report the path's own length."""
    if o_node == d_node:
        raise SystemExit(
            f"Origin and destination resolve to the same node for "
            f"{origin} -> {destination} on the {network} network."
        )

    try:
        _time, nodes = nx.bidirectional_dijkstra(G, o_node, d_node, weight=WEIGHT_ATTR)
    except nx.NetworkXNoPath:
        raise SystemExit(
            f"No {network} path between {origin} and {destination} on the "
            f"UNDISRUPTED graph. Every county pair must be reachable before "
            f"disruption analysis is meaningful -- check the connectivity "
            f"validation in 4_graph_smoothing.py."
        )

    edges: List[RouteEdge] = []
    for u, v in zip(nodes[:-1], nodes[1:]):
        key, data = cheapest_parallel_edge(G, u, v)
        edges.append(
            RouteEdge(
                u=u,
                v=v,
                key=key,
                length_km=float(data.get(LENGTH_ATTR, 0.0)),
                travel_time_min=float(data.get(WEIGHT_ATTR, 0.0)),
            )
        )

    return Route(
        origin=origin, destination=destination, network=network, nodes=nodes, edges=edges
    )


# ===========================================================================
# Single-point disruption (paper Sections 3.4.1 / 3.4.3 / 3.5.1)
# ===========================================================================


def length_weighted_quantile(
    deltas: Sequence[float], lengths: Sequence[float], q: float
) -> float:
    """The paper's length-weighted quantile (Section 3.5.1).

    Sort the delays ascending, walk them accumulating each one's edge length,
    and return the delay at which the cumulative length first reaches q * L.

    That index k satisfies the paper's own two conditions, because the walk
    stops at the FIRST crossing:
        sum_{i<k} l_i / L <= q          (we had not yet reached q*L at k-1)
        sum_{i>k} l_i / L <= 1 - q      (since cum_k >= q*L)

    `inf` sorts last, so a route only reports an infinite percentile when
    disconnecting edges make up more than (1-q) of its total length.
    """
    if not deltas:
        raise SystemExit("Cannot compute a quantile over an empty route.")

    order = sorted(range(len(deltas)), key=lambda i: deltas[i])
    L = sum(lengths)

    if L <= 0:
        # Degenerate: no length to weight by. Fall back to the plain
        # order statistic rather than silently returning the minimum.
        idx = min(int(q * (len(order) - 1)), len(order) - 1)
        return deltas[order[idx]]

    target = q * L
    cumulative = 0.0
    for i in order:
        cumulative += lengths[i]
        if cumulative >= target - 1e-12:
            return deltas[i]

    return deltas[order[-1]]  # floating-point safety net


def analyse_route(G: nx.MultiGraph, route: Route, progress_prefix: str = "") -> RouteDisruption:
    """Remove each edge of `route` in turn and measure the resulting delay.

    The graph is mutated and restored in place rather than copied: a copy per
    edge would dominate the runtime for no benefit, and the restore is exact
    because we keep the edge's key and attribute dict.
    """
    baseline = route.travel_time_min
    if baseline <= 0:
        raise SystemExit(
            f"Baseline travel time for {route.origin} -> {route.destination} "
            f"({route.network}) is {baseline}; a relative delay is undefined."
        )

    o_node, d_node = route.nodes[0], route.nodes[-1]
    impacts: List[EdgeImpact] = []

    for edge in route.edges:
        data = dict(G[edge.u][edge.v][edge.key])  # snapshot before removal
        G.remove_edge(edge.u, edge.v, key=edge.key)
        try:
            disrupted, _path = nx.bidirectional_dijkstra(
                G, o_node, d_node, weight=WEIGHT_ATTR
            )
        except nx.NetworkXNoPath:
            disrupted = INF
        finally:
            G.add_edge(edge.u, edge.v, key=edge.key, **data)  # always restore

        if math.isinf(disrupted):
            delta = INF
        else:
            # Removing an edge can never make a shortest path faster; clamp
            # away floating-point noise around zero.
            delta = max(0.0, (disrupted - baseline) / baseline)

        impacts.append(
            EdgeImpact(edge=edge, disrupted_time_min=disrupted, delta=delta)
        )

    deltas = [im.delta for im in impacts]
    lengths = [im.edge.length_km for im in impacts]

    result = RouteDisruption(route=route, impacts=impacts)
    result.percentile_deltas = {
        q: length_weighted_quantile(deltas, lengths, q) for q in PERCENTILES
    }

    max_delta = max(deltas)
    result.max_delta = max_delta

    if math.isinf(max_delta):
        tied = [i for i, d in enumerate(deltas) if math.isinf(d)]
    else:
        tied = [
            i
            for i, d in enumerate(deltas)
            if math.isclose(d, max_delta, rel_tol=TIE_REL_TOL, abs_tol=TIE_ABS_TOL)
        ]
    result.length_causing_max_km = sum(lengths[i] for i in tied)
    result.n_disconnecting_edges = sum(1 for d in deltas if math.isinf(d))

    # Monotonicity is guaranteed by construction (a quantile over an ascending
    # sort). If it ever fails, something is wrong with the sort or the
    # accumulation, and a silently wrong table is worse than a crash.
    p25, p50, p75 = (result.percentile_deltas[q] for q in PERCENTILES)
    if not (p25 <= p50 <= p75 <= max_delta):
        raise SystemExit(
            f"Percentiles are not monotone for {route.origin} -> "
            f"{route.destination} ({route.network}): "
            f"25th={p25}, 50th={p50}, 75th={p75}, max={max_delta}."
        )

    if progress_prefix:
        print(
            f"{progress_prefix} {route.origin} -> {route.destination}: "
            f"{len(route.edges):,} edges, baseline {baseline:,.1f} min, "
            f"{result.n_disconnecting_edges:,} disconnecting"
        )

    return result


# ===========================================================================
# Aggregation helpers
# ===========================================================================


def finite_mean(values: Sequence[float]) -> float:
    """Unweighted mean over FINITE entries only.

    The paper's Average rows silently drop `inf` (verified numerically
    against its printed Table 3: the seaport 25th-percentile average of
    342.9 is the mean of exactly the 13 finite values). If every entry is
    infinite there is nothing finite to average, so the mean is inf.
    """
    finite = [v for v in values if not math.isinf(v)]
    if not finite:
        return INF
    return sum(finite) / len(finite)


# ===========================================================================
# Number formatting
# ===========================================================================


def fmt_fixed(value: Optional[float], decimals: int = 1, suffix: str = "") -> str:
    """Format for the PNG. `inf` prints as the paper prints it: 'inf'."""
    if value is None:
        return ""
    if math.isinf(value):
        return "inf"
    if math.isnan(value):
        return ""
    return f"{value:.{decimals}f}{suffix}"


def fmt_pct(value: Optional[float], decimals: int = 1) -> str:
    """A relative delay (0.112) as the paper prints it ('11.2 %')."""
    if value is None:
        return ""
    if math.isinf(value):
        return "inf"
    if math.isnan(value):
        return ""
    return f"{value * 100:.{decimals}f} %"


def csv_value(value: Optional[float]) -> str:
    """Full precision for the CSV. `inf` round-trips through float()."""
    if value is None:
        return ""
    if math.isinf(value):
        return "inf"
    return repr(float(value))


def csv_pct(value: Optional[float]) -> str:
    """Full precision for a CSV column whose NAME says percent.

    Internally a relative delay is carried as a fraction (0.112). Writing that
    fraction into a column called `..._pct_...` would silently mislead anyone
    who opens the CSV, and would disagree with the PNG, which prints '11.2 %'.
    So percent-named columns are scaled here, at the single point of write.
    """
    if value is None:
        return ""
    if math.isinf(value):
        return "inf"
    return repr(float(value) * 100.0)


# ===========================================================================
# Paper-style table renderer
# ===========================================================================
#
# The base paper's tables are set in the Elsevier/booktabs idiom: a serif
# face, HORIZONTAL RULES ONLY (no vertical lines, no grid, no shading), a
# top rule, a rule under the header block, and a bottom rule. Column headers
# stack over several lines and group headers span a set of columns without
# their own rule. Body text columns are left-aligned; every numeric column is
# right-aligned so decimal points line up.
#
# matplotlib's built-in `table` cannot do stacked multi-line headers, spanning
# group labels, or per-column alignment, so the layout is done by hand: text
# is measured with the real renderer, column widths are derived from the
# widest cell, and everything is positioned in INCHES (the axes are set up so
# one data unit equals one inch, which makes the arithmetic below literal).

# Preference order: the paper's own face first, then progressively more
# common Times/Palatino clones, then matplotlib's guaranteed fallback. The
# first font actually installed wins, so this renders sensibly anywhere.
SERIF_STACK = [
    "Charter",
    "Bitstream Charter",
    "Utopia",
    "P052",  # URW Palladio / Palatino clone
    "TeX Gyre Pagella",
    "Nimbus Roman",
    "TeX Gyre Termes",
    "Times New Roman",
    "Liberation Serif",
    "DejaVu Serif",
]

FS_TITLE = 10.0  # "Table 1"
FS_CAPTION = 9.5  # the caption sentence
FS_HEADER = 9.0
FS_BODY = 9.0

LINE_SPACING = 1.38  # multiple of font size, for stacked header/body lines
COL_GUTTER_IN = 0.20  # horizontal space between adjacent columns
MARGIN_X_IN = 0.14
MARGIN_TOP_IN = 0.10
MARGIN_BOTTOM_IN = 0.12

RULE_LW_OUTER = 1.0  # top and bottom rules
RULE_LW_INNER = 0.7  # the rule under the header block
RULE_GAP_IN = 0.07  # breathing room either side of a rule
CAPTION_TO_RULE_IN = 0.13  # caption baseline down to the top rule

# Reserved row key: maps a column key to an alignment that overrides the
# column's own alignment for that one row. Never treated as a cell.
ROW_ALIGN_KEY = "__align__"


@dataclass
class Column:
    """One column of a rendered table."""

    key: str
    header: Tuple[str, ...]  # stacked header lines, rendered bottom-aligned
    align: str = "right"  # body alignment: 'left' | 'right' | 'center'
    header_align: Optional[str] = None  # defaults to `align`
    header_bold: bool = True

    def effective_header_align(self) -> str:
        return self.header_align if self.header_align is not None else self.align


@dataclass
class Group:
    """A header label spanning columns `first` through `last` inclusive."""

    label: str
    first: int
    last: int


def _pick_font() -> str:
    """Return the first font from SERIF_STACK that is actually installed."""
    import matplotlib.font_manager as fm

    installed = {f.name for f in fm.fontManager.ttflist}
    for name in SERIF_STACK:
        if name in installed:
            return name
    return "serif"


def _measure(fig, text: str, fontsize: float, bold: bool, font: str) -> float:
    """Width of `text` in INCHES, measured with the real renderer rather than
    estimated from character counts (the face is proportional, so counting
    characters would misalign every column)."""
    if not text:
        return 0.0
    artist = fig.text(
        0.0, 0.0, text,
        fontsize=fontsize,
        fontweight="bold" if bold else "normal",
        fontfamily=font,
    )
    extent = artist.get_window_extent(renderer=fig.canvas.get_renderer())
    artist.remove()
    return extent.width / fig.dpi


def _x_for(x_left: float, width: float, align: str) -> Tuple[float, str]:
    """Anchor point and matplotlib horizontal-alignment for a cell."""
    if align == "left":
        return x_left, "left"
    if align == "center":
        return x_left + width / 2.0, "center"
    return x_left + width, "right"


def render_table_png(
    out_path: Path,
    title: str,
    caption: str,
    columns: Sequence[Column],
    rows: Sequence[Dict[str, str]],
    groups: Optional[Sequence[Group]] = None,
    gap_before_last_row: bool = False,
    dpi: int = 300,
) -> None:
    """Render `rows` as a PNG typeset in the base paper's table style.

    `gap_before_last_row` inserts a little vertical space before the final
    row, which is how the paper visually separates its Average row from the
    data without drawing another rule.
    """
    groups = list(groups or [])
    font = _pick_font()

    # --- Pass 1: measure everything on a throwaway figure -----------------
    scratch = plt.figure(figsize=(6, 6), dpi=100)
    scratch.canvas.draw()

    n_header_lines = max(len(c.header) for c in columns)

    widths: List[float] = []
    for col in columns:
        candidates = [
            _measure(scratch, line, FS_HEADER, col.header_bold, font)
            for line in col.header
        ]
        candidates += [
            _measure(scratch, row.get(col.key, ""), FS_BODY, False, font)
            for row in rows
        ]
        widths.append(max(candidates) if candidates else 0.0)

    # A group label must fit across the columns it spans. If it does not,
    # widen those columns evenly rather than letting the label overflow into
    # its neighbour.
    for group in groups:
        span = sum(widths[group.first : group.last + 1])
        span += COL_GUTTER_IN * (group.last - group.first)
        label_w = _measure(scratch, group.label, FS_HEADER, True, font)
        if label_w > span:
            n_cols = group.last - group.first + 1
            extra = (label_w - span) / n_cols
            for i in range(group.first, group.last + 1):
                widths[i] += extra

    title_w = _measure(scratch, title, FS_TITLE, True, font)
    caption_w = _measure(scratch, caption, FS_CAPTION, False, font)
    plt.close(scratch)

    # --- Geometry ---------------------------------------------------------
    table_w = sum(widths) + COL_GUTTER_IN * (len(columns) - 1)
    rule_w = max(table_w, title_w, caption_w)
    fig_w = rule_w + 2 * MARGIN_X_IN

    line_h = FS_BODY * LINE_SPACING / 72.0
    header_line_h = FS_HEADER * LINE_SPACING / 72.0
    caption_h = FS_CAPTION * LINE_SPACING / 72.0

    n_group_lines = 1 if groups else 0
    header_block_h = (n_group_lines + n_header_lines) * header_line_h
    body_h = len(rows) * line_h + (line_h * 0.45 if gap_before_last_row else 0.0)

    # Every term below corresponds one-for-one to a step in the drawing pass,
    # so the figure is exactly as tall as its contents -- no bbox_inches
    # trimming, which would otherwise crop the rules.
    fig_h = (
        MARGIN_TOP_IN
        + FS_TITLE / 72.0       # top of the title down to its baseline
        + caption_h             # title baseline down to caption baseline
        + CAPTION_TO_RULE_IN    # caption baseline down to the top rule
        + RULE_GAP_IN           # top rule down to the header block
        + header_block_h
        + RULE_GAP_IN           # header block down to the mid rule
        + RULE_GAP_IN           # mid rule down to the body
        + body_h
        + RULE_GAP_IN           # body down to the bottom rule
        + MARGIN_BOTTOM_IN
    )

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w)   # one data unit == one inch, so the maths below
    ax.set_ylim(0, fig_h)   # can be written directly in inches
    ax.axis("off")
    ax.set_facecolor("white")

    def draw(x, y, s, *, size, bold=False, align="left"):
        if not s:
            return
        ax.text(
            x, y, s,
            fontsize=size,
            fontweight="bold" if bold else "normal",
            fontfamily=font,
            ha=align,
            va="baseline",
            color="black",
        )

    def rule(y, lw):
        # Rules span the full content width (which may be set by a long
        # caption rather than by the columns), as they do in the paper.
        ax.plot(
            [MARGIN_X_IN, MARGIN_X_IN + rule_w], [y, y],
            lw=lw, color="black", solid_capstyle="butt",
        )

    # Column left edges. When the caption (or title) is wider than the
    # columns, the leftover width is shared equally between the gutters so the
    # table fills the rules instead of huddling on the left. This is exactly
    # what Elsevier's `tabular*` + `\extracolsep{\fill}` does in the paper.
    gutter = COL_GUTTER_IN
    if len(columns) > 1 and rule_w > table_w:
        gutter += (rule_w - table_w) / (len(columns) - 1)

    x_left: List[float] = []
    x = MARGIN_X_IN
    for w in widths:
        x_left.append(x)
        x += w + gutter

    # --- Draw, top to bottom ---------------------------------------------
    y = fig_h - MARGIN_TOP_IN - FS_TITLE / 72.0
    draw(MARGIN_X_IN, y, title, size=FS_TITLE, bold=True)

    y -= caption_h
    draw(MARGIN_X_IN, y, caption, size=FS_CAPTION)

    y -= CAPTION_TO_RULE_IN
    rule(y, RULE_LW_OUTER)
    y -= RULE_GAP_IN

    # From here `y` tracks the TOP EDGE of the next line. Each line drops to
    # its baseline, draws, then drops the remainder of the line height, so
    # the block consumes exactly `header_block_h`.
    if groups:
        y -= FS_HEADER / 72.0
        for group in groups:
            span = sum(widths[group.first : group.last + 1])
            span += gutter * (group.last - group.first)
            anchor, ha = _x_for(
                x_left[group.first], span,
                columns[group.first].effective_header_align(),
            )
            draw(anchor, y, group.label, size=FS_HEADER, bold=True, align=ha)
        y -= header_line_h - FS_HEADER / 72.0

    # Header lines are BOTTOM-aligned: a one-line header sits on the same
    # baseline as the last line of a three-line header, so "(km)" and "(min)"
    # line up across the whole table.
    for line_idx in range(n_header_lines):
        y -= FS_HEADER / 72.0
        for col_idx, col in enumerate(columns):
            offset = n_header_lines - len(col.header)
            if line_idx < offset:
                continue
            text = col.header[line_idx - offset]
            anchor, ha = _x_for(
                x_left[col_idx], widths[col_idx], col.effective_header_align()
            )
            draw(anchor, y, text, size=FS_HEADER, bold=col.header_bold, align=ha)
        y -= header_line_h - FS_HEADER / 72.0

    y -= RULE_GAP_IN
    rule(y, RULE_LW_INNER)
    y -= RULE_GAP_IN

    for row_idx, row in enumerate(rows):
        if gap_before_last_row and row_idx == len(rows) - 1:
            y -= line_h * 0.45
        y -= FS_BODY / 72.0
        # A row may override the alignment of individual cells via the
        # reserved "__align__" key -- used so the "Average" label hugs the
        # numeric block it summarises, as it does in the paper.
        overrides = row.get(ROW_ALIGN_KEY) or {}
        for col_idx, col in enumerate(columns):
            text = row.get(col.key, "")
            align = overrides.get(col.key, col.align)
            anchor, ha = _x_for(x_left[col_idx], widths[col_idx], align)
            draw(anchor, y, text, size=FS_BODY, align=ha)
        y -= line_h - FS_BODY / 72.0

    y -= RULE_GAP_IN
    rule(y, RULE_LW_OUTER)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="white", bbox_inches=None)
    plt.close(fig)
    print(f"  PNG  -> {out_path}  ({fig_w:.2f} x {fig_h:.2f} in @ {dpi} dpi, font '{font}')")


# ===========================================================================
# Table 1 -- undisrupted specification of every origin-destination pair
# ===========================================================================

TABLE1_CAPTION = (
    "Specifications of the origin–destination county pairs on the Illinois "
    "highway and railway networks."
)


def write_table1(
    routes_by_network: Dict[str, List[Route]],
    od_pairs: Sequence[Tuple[str, str]],
    reports_dir: Path,
    make_png: bool,
    dpi: int,
) -> None:
    """Emit Table 1 as CSV (full precision) and PNG (paper rounding).

    Networks that were not run are written as blank cells rather than zeros,
    so a `--network highway` run produces an honest half-filled table instead
    of one that silently claims the railway distance is zero.
    """
    _print_banner("TABLE 1 -- ORIGIN-DESTINATION SPECIFICATIONS")

    hw = {(r.origin, r.destination): r for r in routes_by_network.get("highway", [])}
    rw = {(r.origin, r.destination): r for r in routes_by_network.get("railway", [])}

    csv_path = reports_dir / "table1_od_baseline.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    png_rows: List[Dict[str, str]] = []
    hw_len, hw_time, rw_len, rw_time = [], [], [], []

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "origin", "destination",
            "highway_length_km", "highway_travel_time_min",
            "railway_length_km", "railway_travel_time_min",
            "highway_n_edges", "railway_n_edges",
        ])

        for origin, destination in od_pairs:
            h = hw.get((origin, destination))
            r = rw.get((origin, destination))

            if h:
                hw_len.append(h.length_km)
                hw_time.append(h.travel_time_min)
            if r:
                rw_len.append(r.length_km)
                rw_time.append(r.travel_time_min)

            writer.writerow([
                origin, destination,
                csv_value(h.length_km) if h else "",
                csv_value(h.travel_time_min) if h else "",
                csv_value(r.length_km) if r else "",
                csv_value(r.travel_time_min) if r else "",
                len(h.edges) if h else "",
                len(r.edges) if r else "",
            ])

            png_rows.append({
                "origin": origin,
                "destination": destination,
                "hw_len": fmt_fixed(h.length_km, 0) if h else "",
                "hw_time": fmt_fixed(h.travel_time_min, 0) if h else "",
                "rw_len": fmt_fixed(r.length_km, 0) if r else "",
                "rw_time": fmt_fixed(r.travel_time_min, 0) if r else "",
            })

        avg = [
            "Average", "",
            csv_value(finite_mean(hw_len)) if hw_len else "",
            csv_value(finite_mean(hw_time)) if hw_time else "",
            csv_value(finite_mean(rw_len)) if rw_len else "",
            csv_value(finite_mean(rw_time)) if rw_time else "",
            "", "",
        ]
        writer.writerow(avg)

    png_rows.append({
        "origin": "",
        "destination": "Average",
        "hw_len": fmt_fixed(finite_mean(hw_len), 0) if hw_len else "",
        "hw_time": fmt_fixed(finite_mean(hw_time), 0) if hw_time else "",
        "rw_len": fmt_fixed(finite_mean(rw_len), 0) if rw_len else "",
        "rw_time": fmt_fixed(finite_mean(rw_time), 0) if rw_time else "",
        ROW_ALIGN_KEY: {"destination": "right"},
    })

    print(f"  CSV  -> {csv_path}  ({len(od_pairs)} route rows + Average)")

    if not make_png:
        return

    columns = [
        Column("origin", ("Origin",), align="left", header_align="left"),
        Column("destination", ("Destination",), align="left", header_align="left"),
        Column("hw_len", ("Length", "(km)"), align="right", header_align="center"),
        Column("hw_time", ("Travel", "Time", "(min)"), align="right", header_align="center"),
        Column("rw_len", ("Length", "(km)"), align="right", header_align="center"),
        Column("rw_time", ("Travel", "Time", "(min)"), align="right", header_align="center"),
    ]
    groups = [Group("Highway", 2, 3), Group("Railway", 4, 5)]

    render_table_png(
        reports_dir / "table1_od_baseline.png",
        "Table 1",
        TABLE1_CAPTION,
        columns,
        png_rows,
        groups=groups,
        gap_before_last_row=True,
        dpi=dpi,
    )


# ===========================================================================
# Tables 2 and 3 -- single-point disruption impact
# ===========================================================================

TABLE_NUMBER_BY_NETWORK = {"highway": "Table 2", "railway": "Table 3"}
TABLE_FILENAME_BY_NETWORK = {
    "highway": "table2_highway_disruption",
    "railway": "table3_railway_disruption",
}


def disruption_caption(network: str) -> str:
    return (
        f"Impact of {network.capitalize()} Disruptions on Shortest Paths "
        f"between Origin and Destination Counties."
    )


def write_edge_detail(results: Sequence[RouteDisruption], path: Path) -> None:
    """One row per removed edge -- the raw evidence behind every percentile.

    Without this, the percentile columns are unfalsifiable: this file lets
    anyone re-derive them independently.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "origin", "destination", "network",
            "edge_index_on_path", "edge_u_lon", "edge_u_lat",
            "edge_v_lon", "edge_v_lat", "edge_key",
            "edge_length_km", "edge_travel_time_min",
            "baseline_time_min", "disrupted_time_min",
            "delta_relative", "time_increase_min", "disconnects",
        ])
        n = 0
        for res in results:
            for i, im in enumerate(res.impacts):
                writer.writerow([
                    res.route.origin, res.route.destination, res.route.network,
                    i, im.edge.u[0], im.edge.u[1], im.edge.v[0], im.edge.v[1],
                    im.edge.key,
                    csv_value(im.edge.length_km), csv_value(im.edge.travel_time_min),
                    csv_value(res.baseline_min), csv_value(im.disrupted_time_min),
                    csv_value(im.delta), csv_value(res.delta_to_minutes(im.delta)),
                    "yes" if im.disconnects else "no",
                ])
                n += 1
    print(f"  CSV  -> {path}  ({n:,} removed-edge rows)")


def write_disruption_table(
    network: str,
    results: Sequence[RouteDisruption],
    reports_dir: Path,
    make_png: bool,
    dpi: int,
) -> None:
    """Emit Table 2 (highway) or Table 3 (railway) as CSV and PNG."""
    label = TABLE_NUMBER_BY_NETWORK[network]
    _print_banner(f"{label.upper()} -- {network.upper()} SINGLE-POINT DISRUPTIONS")

    stem = TABLE_FILENAME_BY_NETWORK[network]
    csv_path = reports_dir / f"{stem}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    q25, q50, q75 = PERCENTILES
    png_rows: List[Dict[str, str]] = []

    # Column-wise accumulators for the Average row.
    acc: Dict[str, List[float]] = {k: [] for k in (
        "baseline", "d25", "p25", "d50", "p50", "d75", "p75",
        "dmax", "pmax", "lmax", "plmax",
    )}

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "origin", "destination", "network",
            "travel_time_min", "route_length_km", "n_edges_on_path",
            "n_disconnecting_edges",
            "p25_time_increase_min", "p25_pct_time_increase",
            "p50_time_increase_min", "p50_pct_time_increase",
            "p75_time_increase_min", "p75_pct_time_increase",
            "max_time_increase_min", "max_pct_time_increase",
            "length_causing_maximum_km", "pct_length_causing_maximum",
        ])

        for res in results:
            d25 = res.delta_to_minutes(res.percentile_deltas[q25])
            d50 = res.delta_to_minutes(res.percentile_deltas[q50])
            d75 = res.delta_to_minutes(res.percentile_deltas[q75])
            dmax = res.delta_to_minutes(res.max_delta)

            acc["baseline"].append(res.baseline_min)
            acc["d25"].append(d25)
            acc["p25"].append(res.percentile_deltas[q25])
            acc["d50"].append(d50)
            acc["p50"].append(res.percentile_deltas[q50])
            acc["d75"].append(d75)
            acc["p75"].append(res.percentile_deltas[q75])
            acc["dmax"].append(dmax)
            acc["pmax"].append(res.max_delta)
            acc["lmax"].append(res.length_causing_max_km)
            acc["plmax"].append(res.pct_length_causing_max)

            writer.writerow([
                res.route.origin, res.route.destination, network,
                csv_value(res.baseline_min), csv_value(res.route_length_km),
                len(res.route.edges), res.n_disconnecting_edges,
                csv_value(d25), csv_pct(res.percentile_deltas[q25]),
                csv_value(d50), csv_pct(res.percentile_deltas[q50]),
                csv_value(d75), csv_pct(res.percentile_deltas[q75]),
                csv_value(dmax), csv_pct(res.max_delta),
                csv_value(res.length_causing_max_km),
                csv_pct(res.pct_length_causing_max),
            ])

            png_rows.append({
                "origin": res.route.origin,
                "destination": res.route.destination,
                "baseline": fmt_fixed(res.baseline_min, 0),
                "d25": fmt_fixed(d25), "p25": fmt_pct(res.percentile_deltas[q25]),
                "d50": fmt_fixed(d50), "p50": fmt_pct(res.percentile_deltas[q50]),
                "d75": fmt_fixed(d75), "p75": fmt_pct(res.percentile_deltas[q75]),
                "dmax": fmt_fixed(dmax), "pmax": fmt_pct(res.max_delta),
                "lmax": fmt_fixed(res.length_causing_max_km),
                "plmax": fmt_pct(res.pct_length_causing_max),
            })

        means = {k: finite_mean(v) for k, v in acc.items()}
        writer.writerow([
            "Average", "", network,
            csv_value(means["baseline"]), "", "", "",
            csv_value(means["d25"]), csv_pct(means["p25"]),
            csv_value(means["d50"]), csv_pct(means["p50"]),
            csv_value(means["d75"]), csv_pct(means["p75"]),
            csv_value(means["dmax"]), csv_pct(means["pmax"]),
            csv_value(means["lmax"]), csv_pct(means["plmax"]),
        ])

    png_rows.append({
        "origin": "",
        "destination": "Average",
        "baseline": fmt_fixed(means["baseline"], 0),
        "d25": fmt_fixed(means["d25"]), "p25": fmt_pct(means["p25"]),
        "d50": fmt_fixed(means["d50"]), "p50": fmt_pct(means["p50"]),
        "d75": fmt_fixed(means["d75"]), "p75": fmt_pct(means["p75"]),
        "dmax": fmt_fixed(means["dmax"]), "pmax": fmt_pct(means["pmax"]),
        "lmax": fmt_fixed(means["lmax"]), "plmax": fmt_pct(means["plmax"]),
        ROW_ALIGN_KEY: {"destination": "right"},
    })

    print(f"  CSV  -> {csv_path}  ({len(results)} route rows + Average)")

    write_edge_detail(results, reports_dir / f"{stem.replace('_disruption', '')}_edge_detail.csv")

    if not make_png:
        return

    # Sub-column headers are set in regular weight while the group labels
    # above them are bold -- exactly as the paper sets its Tables 2 and 3.
    def inc_pair(prefix: str) -> List[Column]:
        return [
            Column(f"d{prefix}", ("time", "increase", "(min)"),
                   align="right", header_align="left", header_bold=False),
            Column(f"p{prefix}", ("% time", "increase"),
                   align="right", header_align="left", header_bold=False),
        ]

    columns = [
        Column("origin", ("Origin",), align="left", header_align="left"),
        Column("destination", ("Destination",), align="left", header_align="left"),
        Column("baseline", ("travel", "time", "(min)"), align="right", header_align="left"),
        *inc_pair("25"), *inc_pair("50"), *inc_pair("75"), *inc_pair("max"),
        Column("lmax", ("length", "causing", "maximum", "(km)"),
               align="right", header_align="left", header_bold=False),
        Column("plmax", ("% length", "causing", "maximum"),
               align="right", header_align="left", header_bold=False),
    ]

    groups = [
        Group(network.capitalize(), 2, 2),
        Group("25th percentile", 3, 4),
        Group("50th percentile", 5, 6),
        Group("75th percentile", 7, 8),
        Group("maximum disruption", 9, 10),
    ]

    render_table_png(
        reports_dir / f"{stem}.png",
        label,
        disruption_caption(network),
        columns,
        png_rows,
        groups=groups,
        gap_before_last_row=True,
        dpi=dpi,
    )


# ===========================================================================
# Per-network driver
# ===========================================================================


def run_network(
    network: str, od_pairs: Sequence[Tuple[str, str]], graph_dir: Path
) -> Tuple[List[Route], List[RouteDisruption]]:
    """Compute every route and its single-edge disruption profile."""
    graph_path = graph_dir / NETWORK_GRAPH_FILENAMES[network]

    print()
    print("#" * 70)
    print(f"# NETWORK: {network}")
    print("#" * 70)

    G, C = load_smoothed_graph(graph_path)
    lookup = build_county_lookup(G, C)

    routes: List[Route] = []
    for origin, destination in od_pairs:
        o_node = resolve_county_node(origin, lookup, network)
        d_node = resolve_county_node(destination, lookup, network)
        routes.append(compute_route(G, origin, destination, o_node, d_node, network))

    total_edges = sum(len(r.edges) for r in routes)
    print(
        f"\nResolved {len(routes)} shortest path(s), {total_edges:,} edges in total.\n"
        f"Removing each edge one at a time and recomputing the "
        f"origin-destination travel time ({total_edges:,} Dijkstra runs)..."
    )

    started = time.monotonic()
    results = [
        analyse_route(G, route, progress_prefix=f"  [{i:>2}/{len(routes)}]")
        for i, route in enumerate(routes, start=1)
    ]
    elapsed = time.monotonic() - started
    print(f"Done in {elapsed:,.1f}s ({total_edges / max(elapsed, 1e-9):,.0f} removals/s)")

    return routes, results


# ===========================================================================
# CLI
# ===========================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 10: build the origin-destination specification table "
            "(Table 1) and the single-point disruption impact tables "
            "(Table 2 highway, Table 3 railway), each as CSV and as a PNG "
            "typeset in the base paper's style."
        )
    )
    parser.add_argument(
        "--network", choices=sorted(NETWORK_LABELS), default=None,
        help=(
            "Restrict the run to one network. If omitted, BOTH highway and "
            "railway are processed (matching every other script in this "
            "pipeline)."
        ),
    )
    parser.add_argument(
        "--od-pairs", type=Path, default=OD_PAIRS_PATH,
        help=f"Origin-destination JSON file (default: {OD_PAIRS_PATH}).",
    )
    parser.add_argument(
        "--graph-dir", type=Path, default=SMOOTHED_INPUT_DIR,
        help=f"Folder holding <network>_smoothed.pkl (default: {SMOOTHED_INPUT_DIR}).",
    )
    parser.add_argument(
        "--reports-dir", type=Path, default=REPORTS_DIR,
        help=f"Folder for the CSV and PNG outputs (default: {REPORTS_DIR}).",
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="Resolution of the rendered PNG tables (default: 300).",
    )
    parser.add_argument(
        "--no-png", action="store_true",
        help="Write only the CSVs, skipping PNG rendering.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_png = not args.no_png

    networks = [args.network] if args.network else list(NETWORK_LABELS)
    print(f"Running Step 10 (origin-destination tables) for: {', '.join(networks)}")

    od_pairs = load_od_pairs(args.od_pairs)

    routes_by_network: Dict[str, List[Route]] = {}
    results_by_network: Dict[str, List[RouteDisruption]] = {}

    for network in networks:
        graph_path = args.graph_dir / NETWORK_GRAPH_FILENAMES[network]
        if not graph_path.exists():
            print(
                f"\n  WARNING: smoothed graph pickle not found, skipping "
                f"{network}:\n    {graph_path}\n"
                f"  (Run 4_graph_smoothing.py for this network first.)"
            )
            continue
        routes, results = run_network(network, od_pairs, args.graph_dir)
        routes_by_network[network] = routes
        results_by_network[network] = results

    if not routes_by_network:
        raise SystemExit(
            "\nNo network was processed -- every requested smoothed graph "
            "pickle was missing. Run 4_graph_smoothing.py first."
        )

    write_table1(routes_by_network, od_pairs, args.reports_dir, make_png, args.dpi)

    for network in NETWORK_LABELS:
        if network in results_by_network:
            write_disruption_table(
                network, results_by_network[network], args.reports_dir, make_png, args.dpi
            )

    _print_banner("ALL TABLES COMPLETE")
    print(f"Networks processed: {list(routes_by_network)}")
    print(f"Output folder:      {args.reports_dir}")


if __name__ == "__main__":
    main()
