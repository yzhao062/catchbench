"""Gold v2: named-value substrate and injection over tau-bench trajectories.

The substrate keeps the values a trajectory actually carries, so dependency edges are derived
from current values rather than stored. An injection mutates one argument value and rebuilds the
graph, removing the file-level process artifact where both faults broke an invariant that clean
construction never breaks.

Two signal classes, and only one carries the admissibility bar:
  process artifact  : a property produced by the act of editing (format, schema, position skew).
                      Must sit at the matched floor, or the substrate is inadmissible.
  fault-definitional: the semantic content of the fault (consuming a superseded value; consuming
                      an ungrounded value). Reported as an oracle ceiling, expected to be high.

Labels are injection sites fixed at construction and independent of any detector.
"""
from __future__ import annotations

import copy
import json
import random
import re
import statistics as st
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from catchbench import _reuse  # noqa: F401  side effect: sys.path for grade + auditable

import agent_graph_tau_bench as tau  # noqa: E402

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@#-]{3,}$")
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._@#-]{3,}")

DERIVED, GIVEN, UNGROUNDED = "derived", "given", "ungrounded"


def _norm(v: Any) -> str:
    """Match key for a scalar: integral floats render as integers, whitespace trimmed."""
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return str(v).strip()


def _is_scalar(v: Any) -> bool:
    return isinstance(v, (str, int, float)) and not isinstance(v, bool)


def identifier_shaped(v: Any) -> bool:
    """Identifier-shaped: length >= 4, token pattern, and both a letter and a digit."""
    s = _norm(v)
    if len(s) < 4 or not _ID_RE.match(s):
        return False
    return any(c.isdigit() for c in s) and any(c.isalpha() for c in s)


def _leaves(obj: Any, out: List[str]) -> None:
    if isinstance(obj, dict):
        for v in obj.values():
            _leaves(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _leaves(v, out)
    elif _is_scalar(obj):
        out.append(_norm(obj))


def _arg_leaves(obj: Any, prefix: tuple = ()):
    """Every scalar leaf of a call-argument structure as (path_tuple, norm_value). The path holds
    real dict keys and list indices, so an injection can address the exact leaf; nested arguments
    count as values, not only top-level scalars."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _arg_leaves(v, prefix + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _arg_leaves(v, prefix + (i,))
    elif _is_scalar(obj):
        yield prefix, _norm(obj)


def _field_key(path: tuple) -> str:
    """Field identity for grouping and donor pools: the path with list indices collapsed, so
    repeated records share one field (``items[].id``) while a top-level arg stays its own name."""
    return ".".join("[]" if isinstance(p, int) else str(p) for p in path)


def _set_by_path(container: Any, path: tuple, value: Any) -> None:
    """Set the scalar leaf at ``path`` inside a (deep-copied) argument structure."""
    for p in path[:-1]:
        container = container[p]
    container[path[-1]] = value


def _leaf_map(obj: Any, out: Dict[str, set], path: str = "") -> None:
    """Scalar leaves keyed by JSON path, so a value keeps its field identity. List indices
    collapse to ``[]`` so repeated records share one path."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _leaf_map(v, out, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for v in obj:
            _leaf_map(v, out, f"{path}[]")
    elif _is_scalar(obj):
        out.setdefault(path or "", set()).add(_norm(obj))


def _parse(content: Any) -> Any:
    if not isinstance(content, str):
        return content
    try:
        return json.loads(content)
    except Exception:
        return content


@dataclass
class Event:
    """One tool event. ``args`` is mutable: an injection rewrites a value here, then the graph
    is rebuilt so the derived edges follow the values."""

    idx: int
    kind: str                      # "call" | "result"
    tool: str
    args: Dict[str, Any] = field(default_factory=dict)   # call only
    payload: Any = None                                  # result only


@dataclass
class Consumption:
    """One named argument value carried by a call, with its provenance in this run.

    ``key`` is the field identity (list indices collapsed) used for stats and donor pools;
    ``path`` is the exact addressable path (real indices) an injection mutates through.
    """

    event_idx: int
    tool: str
    key: str
    value: str
    provenance: str
    producers: Tuple[int, ...] = ()
    path: tuple = ()


@dataclass
class Injection:
    """Injection-site ground truth, fixed at construction, detector-independent."""

    kind: str            # "stale" | "dropped"
    event_idx: int
    key: str
    original: str
    injected: str
    detail: dict = field(default_factory=dict)
    path: tuple = ()     # exact leaf path; site identity is (event_idx, path)


@dataclass
class ValueGraph:
    run_id: str
    events: List[Event]
    given: set
    consumptions: List[Consumption]
    entities: Dict[Tuple[str, str], List[Tuple[int, dict]]]
    res_idx: Dict[int, set] = field(default_factory=dict)
    fld_idx: Dict[int, Dict[str, set]] = field(default_factory=dict)
    _stale_cache: Optional[List[dict]] = None

    def eligible_dropped(self) -> List[Consumption]:
        """Derived identifier-shaped consumptions: the dropped-grounding eligible pool."""
        return [c for c in self.consumptions
                if c.provenance == DERIVED and identifier_shaped(c.value)]

    def eligible_stale(self) -> List[dict]:
        """Stale triples: an entity observation superseded by a later one, then a consumer after
        it carrying a value of that entity. Memoized: a built graph is not mutated in place."""
        if self._stale_cache is not None:
            return self._stale_cache
        out = []
        for key, obs in self.entities.items():
            for (i_a, map_a), (i_b, map_b) in zip(obs, obs[1:]):
                for path, old_vals in map_a.items():
                    new_vals = map_b.get(path)
                    if not new_vals:
                        continue
                    superseded = old_vals - new_vals   # values of THIS field that are now gone
                    if not superseded:
                        continue
                    for c in self.consumptions:
                        # the consumer must carry the CURRENT value of that same field, so the
                        # rewrite stays inside the field's own value domain
                        if c.event_idx > i_b and c.value in new_vals:
                            out.append({"entity": key, "field": path, "obs_old": i_a,
                                        "obs_new": i_b, "superseded": tuple(sorted(superseded)),
                                        "consumption": c})
                            break
        self._stale_cache = out
        return out


def extract_events(rec: dict) -> Tuple[List[Event], set]:
    """Ordered tool events plus the token set supplied by the user or system before the end of
    the run. ``given`` marks values the user handed the agent, which are legitimately producer-free."""
    events: List[Event] = []
    given: set = set()
    for m in rec.get("messages") or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role in ("user", "system"):
            text = m.get("content")
            if isinstance(text, str):
                given |= {_norm(t) for t in _TOKEN_RE.findall(text)}
        elif role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function") or {}
                args = _parse(fn.get("arguments"))
                events.append(Event(len(events), "call", fn.get("name") or "?",
                                    args=dict(args) if isinstance(args, dict) else {}))
        elif role == "tool":
            events.append(Event(len(events), "result", m.get("name") or "?",
                                payload=_parse(m.get("content"))))
    return events, given


def result_index(events: List[Event]) -> Dict[int, set]:
    """Scalar leaves per result event. Injection never touches results, so this is invariant
    across rebuilds and is computed once per run rather than on every mutation."""
    idx: Dict[int, set] = {}
    for i, ev in enumerate(events):
        if ev.kind == "result":
            vals: List[str] = []
            _leaves(ev.payload, vals)
            idx[i] = set(vals)
    return idx


def field_index(events: List[Event]) -> Dict[int, Dict[str, set]]:
    """Per result event, a map from JSON path to the set of values at that path. Field identity
    is what keeps a stale injection inside the value domain of the field it rewrites."""
    idx: Dict[int, Dict[str, set]] = {}
    for i, ev in enumerate(events):
        if ev.kind == "result":
            m: Dict[str, set] = {}
            _leaf_map(ev.payload, m)
            idx[i] = m
    return idx


def build_graph(run_id: str, events: List[Event], given: set,
                res_idx: Optional[Dict[int, set]] = None,
                fld_idx: Optional[Dict[int, Dict[str, set]]] = None) -> ValueGraph:
    """Derive consumptions, provenance, and entity histories from the values present now."""
    if res_idx is None:
        res_idx = result_index(events)
    if fld_idx is None:
        fld_idx = field_index(events)
    produced: Dict[str, List[int]] = defaultdict(list)
    consumptions: List[Consumption] = []
    entities: Dict[Tuple[str, str], List[Tuple[int, dict]]] = defaultdict(list)

    for i, ev in enumerate(events):
        if ev.kind == "result":
            for v in res_idx.get(i, ()):
                produced[v].append(i)
            continue
        # call: entity identity is (tool, first identifier-shaped argument)
        ids = [_norm(v) for v in ev.args.values() if identifier_shaped(v)]
        nxt = events[i + 1] if i + 1 < len(events) else None
        if ids and nxt is not None and nxt.kind == "result":
            entities[(ev.tool, ids[0])].append((i, fld_idx.get(i + 1, {})))
        for path, nv in _arg_leaves(ev.args):   # recurse into nested argument leaves, not only top level
            prior = [p for p in produced.get(nv, ()) if p < i]
            prods = (prior[-1],) if prior else ()  # edge to the latest prior producer
            if prods:
                prov = DERIVED
            elif nv in given:
                prov = GIVEN
            else:
                prov = UNGROUNDED
            consumptions.append(Consumption(i, ev.tool, _field_key(path), nv, prov, prods, path))

    for k in entities:
        entities[k].sort()
    return ValueGraph(run_id, events, given, consumptions, dict(entities), res_idx, fld_idx)


def load_graphs(limit: Optional[int] = None) -> List[ValueGraph]:
    """Build one graph per tau-bench run from the pinned corpus."""
    out = []
    for n, rec in enumerate(tau.load_runs(tau._ensure_files())):
        if limit is not None and n >= limit:
            break
        events, given = extract_events(rec)
        if not events:
            continue
        out.append(build_graph(f"tau-{n}", events, given,
                               result_index(events), field_index(events)))
    return out


def _rebuild(graph: ValueGraph) -> ValueGraph:
    return build_graph(graph.run_id, graph.events, graph.given, graph.res_idx, graph.fld_idx)


def _copy(graph: ValueGraph) -> ValueGraph:
    """Deep-copy the mutable half (call arguments); results and their scalar index are shared,
    because injection never touches a result."""
    events = [Event(e.idx, e.kind, e.tool, copy.deepcopy(e.args), e.payload) for e in graph.events]
    return build_graph(graph.run_id, events, graph.given, graph.res_idx, graph.fld_idx)


def donor_pool(graphs: Sequence[ValueGraph]) -> Dict[str, List[str]]:
    """Identifier-shaped values seen per argument key across the corpus, for dropped grounding.
    Donors are real values from other runs, never synthesized, so format stays in distribution."""
    pool: Dict[str, set] = defaultdict(set)
    for g in graphs:
        for c in g.consumptions:
            if identifier_shaped(c.value):
                pool[c.key].add(c.value)
    return {k: sorted(v) for k, v in pool.items()}


def inject_stale(graph: ValueGraph, rng: random.Random) -> Optional[Tuple[ValueGraph, Injection]]:
    """Rewrite one consumed value to a version a later observation of its entity superseded."""
    sites = graph.eligible_stale()
    if not sites:
        return None
    site = sites[rng.randrange(len(sites))]
    c: Consumption = site["consumption"]
    candidates = [v for v in site["superseded"] if v != c.value]
    if not candidates:
        return None
    v_old = candidates[rng.randrange(len(candidates))]
    out = _copy(graph)
    _set_by_path(out.events[c.event_idx].args, c.path, v_old)
    out = _rebuild(out)
    return out, Injection("stale", c.event_idx, c.key, c.value, v_old,
                          {"entity": list(site["entity"]), "field": site["field"],
                           "obs_old": site["obs_old"], "obs_new": site["obs_new"]}, c.path)


def _lev(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _neighbour_distance(value: str, neighbours: Sequence[str]) -> int:
    """Minimum edit distance from ``value`` to the run's other values of the same field."""
    others = [o for o in neighbours if o != value]
    return min((_lev(value, o) for o in others), default=0)


def inject_dropped(graph: ValueGraph, rng: random.Random,
                   pool: Dict[str, List[str]],
                   match_neighbourhood: bool = True) -> Optional[Tuple[ValueGraph, Injection]]:
    """Rewrite one grounded consumption to a real value from another run that this run never
    produced and the user never supplied, so the consumption becomes ungrounded.

    Donor selection is matched on the field's local string-distance profile. A donor drawn purely
    at random is an unfamiliar string in this run, so it sits farther from the run's other values
    of the same field than the value it replaces, and an edit-distance control reads that gap as
    signal. Choosing the donor whose neighbour distance is closest to the original's equalizes the
    statistic by construction, which is an injector fix rather than a relaxed criterion.
    """
    sites = graph.eligible_dropped()
    rng.shuffle(sites)
    # A donor must have no source in this run at all: not consumed, not user-given, and not even
    # produced-and-unconsumed. Omitting produced-unconsumed values (High 3) let the prefilter admit
    # a donor this run produced, which the final ungrounded check then rejected by abandoning the
    # site, biasing site selection.
    produced = set().union(*graph.res_idx.values()) if graph.res_idx else set()
    local = {c.value for c in graph.consumptions} | set(graph.given) | produced
    for c in sites:
        donors = [d for d in pool.get(c.key, ()) if d not in local]
        if not donors:
            continue
        if match_neighbourhood:
            # The neighbourhood a control sees is the POST-injection one, where this site's own
            # value has been replaced. Matching against the pre-injection neighbourhood (which
            # still contains the original) targets the wrong quantity and leaves a residual.
            # Exclude only the exact site (event, path), not the whole event: after list indices
            # collapse into one field key, a same-event sibling leaf (flights[0].id vs flights[1].id)
            # shares the key but is a different value the edit-distance control does see, so it must
            # stay in the matching neighbourhood or the matched statistic diverges from the control's.
            neigh = [x.value for x in graph.consumptions
                     if x.key == c.key and (x.event_idx, x.path) != (c.event_idx, c.path)]
            target = _neighbour_distance(c.value, neigh)
            sample = donors if len(donors) <= 64 else rng.sample(donors, 64)
            best = min(abs(_neighbour_distance(d, neigh) - target) for d in sample)
            tied = [d for d in sample if abs(_neighbour_distance(d, neigh) - target) == best]
            v_ung = tied[rng.randrange(len(tied))]
        else:
            v_ung = donors[rng.randrange(len(donors))]
        out = _copy(graph)
        _set_by_path(out.events[c.event_idx].args, c.path, v_ung)
        out = _rebuild(out)
        # the mutation must actually produce the intended shape at the same leaf path
        hit = [x for x in out.consumptions
               if x.event_idx == c.event_idx and x.path == c.path]
        if hit and hit[0].provenance == UNGROUNDED:
            return out, Injection("dropped", c.event_idx, c.key, c.value, v_ung, {}, c.path)
    return None


@dataclass
class Pair:
    """One clean graph and its injected copy, with the injection-site label."""

    clean: ValueGraph
    injected: ValueGraph
    label: Injection


def build_corpus(graphs: Sequence[ValueGraph], seed: int = 0) -> List[Pair]:
    """Paired corpus: alternate fault kind across eligible runs so the label is the fault, not
    the run. Deterministic under ``seed``.

    Use this for a corpus whose label is the fault kind. It is the wrong builder for a
    single-kind board: alternating the try-order and stopping at the first success means a run
    eligible for both kinds is assigned one of them, so filtering the result down to one kind
    silently drops the runs that went the other way. ``build_single_kind_corpus`` is the builder
    for that case.
    """
    rng = random.Random(seed)
    pool = donor_pool(graphs)
    pairs: List[Pair] = []
    for n, g in enumerate(graphs):
        order = ("stale", "dropped") if n % 2 == 0 else ("dropped", "stale")
        for kind in order:
            res = (inject_stale(g, rng) if kind == "stale"
                   else inject_dropped(g, rng, pool))
            if res is not None:
                inj_graph, label = res
                pairs.append(Pair(g, inj_graph, label))
                break
    return pairs


def build_single_kind_corpus(
    graphs: Sequence[ValueGraph], kind: str, seed: int = 0
) -> List[Pair]:
    """Paired corpus of one fault kind, over every run that kind is eligible for.

    A board that scores one kind declares its population by that kind's eligibility and by
    nothing else. Building the mixed corpus and filtering it does not produce that population:
    it lost two of the 614 dropped-eligible runs to stale injections chosen by graph-enumeration
    parity, a rule with no bearing on what dropped grounding is. Deterministic under ``seed``.
    """
    if kind not in ("stale", "dropped"):
        raise ValueError(f"unknown fault kind: {kind}")
    rng = random.Random(seed)
    pool = donor_pool(graphs) if kind == "dropped" else None
    pairs: List[Pair] = []
    for g in graphs:
        res = (inject_stale(g, rng) if kind == "stale"
               else inject_dropped(g, rng, pool))
        if res is not None:
            inj_graph, label = res
            pairs.append(Pair(g, inj_graph, label))
    return pairs


# --- Gold v2 scored task ----------------------------------------------------------------------

GOLD_V2_SEEDS = (0, 1, 2, 3, 4)
GOLD_V2_TOP1_MARGIN = 0.05
GOLD_V2_AUC_BAND = (0.45, 0.55)
_T975_DF4 = 2.776
_GRAPH_CACHE: Optional[List[ValueGraph]] = None


def corpus_stats(graphs: Sequence[ValueGraph]) -> dict:
    """Corpus summaries used by process controls. They are computed on the clean substrate."""
    field_len: Dict[str, list] = defaultdict(list)
    field_charclass: Dict[str, Counter] = defaultdict(Counter)
    tool_keys: Dict[str, Counter] = defaultdict(Counter)
    key_count = Counter()
    for graph in graphs:
        for consumption in graph.consumptions:
            field_len[consumption.key].append(len(consumption.value))
            field_charclass[consumption.key][_charclass(consumption.value)] += 1
            key_count[consumption.key] += 1
        for event in graph.events:
            if event.kind == "call":
                tool_keys[event.tool][tuple(sorted(event.args))] += 1
    return {
        "field_len": {
            key: (st.mean(values), st.pstdev(values) or 1.0)
            for key, values in field_len.items()
        },
        "field_charclass": field_charclass,
        "tool_keys": tool_keys,
        "key_count": key_count,
    }


def _charclass(value: str) -> tuple:
    return (
        any(char.isdigit() for char in value),
        any(char.isalpha() for char in value),
        any(not char.isalnum() for char in value),
    )


def _ctl_format_outlier(graph: ValueGraph, consumption: Consumption, stats: dict) -> float:
    mu, sd = stats["field_len"].get(consumption.key, (len(consumption.value), 1.0))
    z_score = abs(len(consumption.value) - mu) / (sd or 1.0)
    classes = stats["field_charclass"].get(consumption.key, Counter())
    total = sum(classes.values()) or 1
    rarity = 1.0 - classes.get(_charclass(consumption.value), 0) / total
    return z_score + rarity


def _ctl_schema_shape(graph: ValueGraph, consumption: Consumption, stats: dict) -> float:
    event = graph.events[consumption.event_idx]
    observed = stats["tool_keys"].get(event.tool, Counter())
    total = sum(observed.values()) or 1
    return 1.0 - observed.get(tuple(sorted(event.args)), 0) / total


def _ctl_position_prior(graph: ValueGraph, consumption: Consumption, stats: dict) -> float:
    return consumption.event_idx / max(len(graph.events) - 1, 1)


def _ctl_field_prior(graph: ValueGraph, consumption: Consumption, stats: dict) -> float:
    return -stats["key_count"].get(consumption.key, 0)


def _ctl_tool_prior(graph: ValueGraph, consumption: Consumption, stats: dict) -> float:
    event = graph.events[consumption.event_idx]
    return -sum(stats["tool_keys"].get(event.tool, Counter()).values())


def _ctl_edit_distance(graph: ValueGraph, consumption: Consumption, stats: dict) -> float:
    neighbours = [
        other.value
        for other in graph.consumptions
        if other.key == consumption.key and other.value != consumption.value
    ]
    return min((_lev(consumption.value, other) for other in neighbours), default=0)


def _orc_superseded(graph: ValueGraph, consumption: Consumption, stats: dict) -> float:
    """Return one when the consumption carries a field value superseded earlier in the run."""
    for observations in graph.entities.values():
        for (_old_idx, old_map), (new_idx, new_map) in zip(observations, observations[1:]):
            if consumption.event_idx <= new_idx:
                continue
            for path, old_values in old_map.items():
                new_values = new_map.get(path)
                if new_values and consumption.value in (old_values - new_values):
                    return 1.0
    return 0.0


def _orc_provenance(graph: ValueGraph, consumption: Consumption, stats: dict) -> float:
    return 1.0 if consumption.provenance == UNGROUNDED else 0.0


GOLD_V2_CONTROLS = (
    ("format-outlier", _ctl_format_outlier),
    ("schema-shape", _ctl_schema_shape),
    ("position-prior", _ctl_position_prior),
    ("field-prior", _ctl_field_prior),
    ("tool-prior", _ctl_tool_prior),
    ("edit-distance", _ctl_edit_distance),
)
GOLD_V2_ORACLES = (
    ("superseded-value", _orc_superseded),
    ("provenance", _orc_provenance),
)


def _eligible_pool(graph: ValueGraph, kind: str) -> List[Consumption]:
    """Exact clean-run candidate pool used by the injector for one fault kind."""
    if kind == "dropped":
        return graph.eligible_dropped()
    seen = set()
    pool = []
    for site in graph.eligible_stale():
        consumption = site["consumption"]
        identity = (consumption.event_idx, consumption.path)
        if identity not in seen:
            seen.add(identity)
            pool.append(consumption)
    return pool


def _pools_for(pairs: Sequence[Pair]) -> List[List[tuple]]:
    """Candidate site identities from each clean graph, never from its injected copy."""
    pools = []
    for pair in pairs:
        pool = [(c.event_idx, c.path) for c in _eligible_pool(pair.clean, pair.label.kind)]
        target = (pair.label.event_idx, pair.label.path)
        if target not in pool:
            pool.append(target)
        pools.append(pool)
    return pools


def _tie_aware_top1(pool_scores: Sequence[float], target_idx: int) -> float:
    """Expected Top-1 under uniform tie breaking."""
    target_score = pool_scores[target_idx]
    greater = sum(score > target_score for score in pool_scores)
    tied = sum(score == target_score for score in pool_scores)
    return (1.0 / tied) if greater == 0 else 0.0


def _top1_and_floor(pairs: Sequence[Pair], score_fn, stats: dict,
                    pools: Sequence[Sequence[tuple]]) -> tuple[float, float]:
    """Tie-aware site Top-1 and its per-run eligibility-matched random floor."""
    hits, floors = [], []
    for pair, pool in zip(pairs, pools):
        if not pool:
            continue
        by_site = {(c.event_idx, c.path): c for c in pair.injected.consumptions}
        scored = [
            (site, score_fn(pair.injected, by_site[site], stats))
            for site in pool
            if site in by_site
        ]
        if not scored:
            continue
        target = (pair.label.event_idx, pair.label.path)
        target_idx = next((i for i, (site, _score) in enumerate(scored) if site == target), None)
        hits.append(
            _tie_aware_top1([score for _site, score in scored], target_idx)
            if target_idx is not None
            else 0.0
        )
        floors.append(1.0 / len(scored))
    return (st.mean(hits) if hits else 0.0), (st.mean(floors) if floors else 0.0)


def _run_auc(pairs: Sequence[Pair], score_fn, stats: dict) -> float:
    """Clean-versus-injected ROC-AUC from each run's maximum consumption score."""
    injected_scores, clean_scores = [], []
    for pair in pairs:
        injected_scores.append(max(
            (score_fn(pair.injected, c, stats) for c in pair.injected.consumptions),
            default=0.0,
        ))
        clean_scores.append(max(
            (score_fn(pair.clean, c, stats) for c in pair.clean.consumptions),
            default=0.0,
        ))
    if not injected_scores or not clean_scores:
        return 0.5
    wins = ties = 0
    for injected_score in injected_scores:
        for clean_score in clean_scores:
            if injected_score > clean_score:
                wins += 1
            elif injected_score == clean_score:
                ties += 1
    return (wins + 0.5 * ties) / (len(injected_scores) * len(clean_scores))


def _cached_graphs() -> List[ValueGraph]:
    global _GRAPH_CACHE
    if _GRAPH_CACHE is None:
        _GRAPH_CACHE = load_graphs()
    return _GRAPH_CACHE


class GoldNamedValue:
    """POST / Gold v2 task over explicit named-value flow in tau-bench trajectories.

    Tau-bench does not support a scored stale-state arm: it has 16 sites in only 6 runs. The board
    therefore scores dropped grounding only. Top-1 localizes the changed argument leaf within the
    clean eligible pool, while run AUC tests clean-versus-injected separation by the same scorer.
    """

    task_id = "gold_v2_namedvalue"
    pillar = "POST"
    granularity = "value"
    dataset = "tau-bench-gold-v2"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self._loaded = False
        self._score_cache: Dict[str, Mapping[str, float]] = {}

    def setup(self) -> None:
        if self._loaded:
            return
        self.graphs = _cached_graphs()
        self.stats = corpus_stats(self.graphs)
        self.stale_sites = sum(len(graph.eligible_stale()) for graph in self.graphs)
        self.stale_runs = sum(bool(graph.eligible_stale()) for graph in self.graphs)
        self.dropped_sites = sum(len(graph.eligible_dropped()) for graph in self.graphs)
        self.dropped_runs = sum(bool(graph.eligible_dropped()) for graph in self.graphs)
        self.pairs = build_single_kind_corpus(self.graphs, "dropped", seed=self.seed)
        self.pools = _pools_for(self.pairs)
        self._loaded = True

    def score(self, method_id: str, score_fn) -> Mapping[str, float]:
        self.setup()
        if method_id not in self._score_cache:
            top1, _floor = _top1_and_floor(self.pairs, score_fn, self.stats, self.pools)
            self._score_cache[method_id] = {
                "top1": float(top1),
                "run_auc": float(_run_auc(self.pairs, score_fn, self.stats)),
            }
        return self._score_cache[method_id]

    def matched_floor(self) -> float:
        self.setup()
        return float(st.mean(1.0 / len(pool) for pool in self.pools if pool))

    def corpus_line(self) -> str:
        self.setup()
        return (
            f"{self.dataset}: {len(self.graphs)} tau-bench named-value runs; "
            f"{len(self.pairs)} dropped-grounding clean/injected pairs at seed {self.seed}, "
            f"from {self.dropped_sites} eligible sites in {self.dropped_runs} runs. "
            f"Stale-state has {self.stale_sites} sites in {self.stale_runs} runs and is not scored."
        )


class _GoldV2Floor:
    method_id = "random (matched floor)"
    supports = {"gold_v2_namedvalue"}
    category = "floor"

    def evaluate(self, task: GoldNamedValue) -> Mapping[str, float]:
        return {"top1": task.matched_floor(), "run_auc": 0.5}


class _GoldV2ScoreMethod:
    def __init__(self, method_id: str, score_fn, category: str) -> None:
        self.method_id = method_id
        self._score_fn = score_fn
        self.category = category
        self.supports = {"gold_v2_namedvalue"}

    def evaluate(self, task: GoldNamedValue) -> Mapping[str, float]:
        return task.score(self.method_id, self._score_fn)


def gold_v2_methods() -> list:
    """Matched floor, construction-only process controls, and fault-definitional oracles."""
    methods = [_GoldV2Floor()]
    methods.extend(
        _GoldV2ScoreMethod(name, score_fn, "process control")
        for name, score_fn in GOLD_V2_CONTROLS
    )
    methods.extend(
        _GoldV2ScoreMethod(name, score_fn, "fault oracle")
        for name, score_fn in GOLD_V2_ORACLES
    )
    return methods


def _seed_interval(values: Sequence[float]) -> tuple[float, float]:
    """Mean and 95% t half-width over the fixed five injection seeds."""
    if len(values) != len(GOLD_V2_SEEDS):
        raise ValueError("Gold v2 seed intervals require the fixed five-seed panel")
    return st.mean(values), _T975_DF4 * st.stdev(values) / (len(values) ** 0.5)


def gold_v2_breakdown(task: GoldNamedValue, methods: Sequence) -> str:
    """Five-seed fixed-margin diagnostic, with process controls separate from semantic oracles.

    Meeting the margins is not a full no-artifact-leakage PASS. Positive-control power currently
    covers three of six controls on Top-1 and none on run AUC, so the bar item remains undetermined.
    """
    task.setup()
    scored = [method for method in methods if hasattr(method, "_score_fn")]
    per_method = {method.method_id: {"gap": [], "auc": []} for method in scored}
    pair_counts = []
    for seed in GOLD_V2_SEEDS:
        if seed == task.seed:
            pairs, pools = task.pairs, task.pools
        else:
            pairs = build_single_kind_corpus(task.graphs, "dropped", seed=seed)
            pools = _pools_for(pairs)
        pair_counts.append(len(pairs))
        for method in scored:
            if seed == task.seed:
                metrics = method.evaluate(task)
                top1, auc = metrics["top1"], metrics["run_auc"]
                floor = task.matched_floor()
            else:
                top1, floor = _top1_and_floor(
                    pairs, method._score_fn, task.stats, pools
                )
                auc = _run_auc(pairs, method._score_fn, task.stats)
            per_method[method.method_id]["gap"].append(float(top1 - floor))
            per_method[method.method_id]["auc"].append(float(auc))

    lines = [
        "\nGold v2 named-value fixed-margin diagnostic "
        f"({len(GOLD_V2_SEEDS)} injection seeds, dropped grounding, pair counts {pair_counts}):",
        f"  {'process control':26s}{'Top-1 - floor':>22s}{'run AUC':>20s}  fixed margin",
    ]
    fixed_margin_pass = True
    for method in (item for item in scored if item.category == "process control"):
        gap_mean, gap_half = _seed_interval(per_method[method.method_id]["gap"])
        auc_mean, auc_half = _seed_interval(per_method[method.method_id]["auc"])
        passes = (
            abs(gap_mean) + gap_half <= GOLD_V2_TOP1_MARGIN
            and GOLD_V2_AUC_BAND[0] <= auc_mean - auc_half
            and auc_mean + auc_half <= GOLD_V2_AUC_BAND[1]
        )
        fixed_margin_pass &= passes
        lines.append(
            f"  {method.method_id:26s}"
            f"{f'{gap_mean:+.3f}+/-{gap_half:.3f}':>22s}"
            f"{f'{auc_mean:.3f}+/-{auc_half:.3f}':>20s}  "
            f"{'PASS' if passes else 'FAIL'}"
        )

    lines.extend([
        "",
        f"  {'fault oracle':26s}{'Top-1 - floor':>22s}{'run AUC':>20s}  role",
    ])
    for method in (item for item in scored if item.category == "fault oracle"):
        gap_mean, gap_half = _seed_interval(per_method[method.method_id]["gap"])
        auc_mean, auc_half = _seed_interval(per_method[method.method_id]["auc"])
        role = "matching oracle" if method.method_id == "provenance" else "other-fault oracle"
        lines.append(
            f"  {method.method_id:26s}"
            f"{f'{gap_mean:+.3f}+/-{gap_half:.3f}':>22s}"
            f"{f'{auc_mean:.3f}+/-{auc_half:.3f}':>20s}  {role}"
        )

    lines.extend([
        "",
        "  Fixed-margin panel: "
        + ("PASS" if fixed_margin_pass else "FAIL")
        + f" under Top-1 gap [{-GOLD_V2_TOP1_MARGIN:+.2f}, {GOLD_V2_TOP1_MARGIN:+.2f}] "
        f"and run-AUC CI [{GOLD_V2_AUC_BAND[0]:.2f}, {GOLD_V2_AUC_BAND[1]:.2f}].",
        "  No-artifact-leakage bar: UNDETERMINED. Positive-control power is demonstrated for "
        "format-outlier, schema-shape, and position-prior on Top-1 only. The other three controls "
        "and every run-AUC path still lack a planted-artifact power test.",
        "  Registry status: every Gold v2 value above is a displayed diagnostic cell. "
        "tools/statistical_tests_results.json registers no Gold v2 contrast.",
        f"  Stale-state status: NOT SCORED. The full corpus has {task.stale_sites} eligible sites "
        f"in {task.stale_runs} runs, which is inadequate for a board.",
    ])
    return "\n".join(lines)
