"""Gold v2: named-value substrate and injection over tau-bench trajectories.

Implements `research/auditablebench-namedvalue-injection.md`. The substrate keeps the values a
trajectory actually carries, so a dependency edge is DERIVED from values rather than stored. An
injection mutates one value and the edge set recomputes, which is what removes the v1 file-level
process artifact (there, both faults broke an invariant clean construction never breaks).

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
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from auditablebench import _reuse  # noqa: F401  side effect: sys.path for grade + auditable

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
    are values too (spec section 3), not only top-level scalars."""
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
            prods = (prior[-1],) if prior else ()  # spec section 3: edge to the LATEST prior producer
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
    the run. Deterministic under ``seed``."""
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
