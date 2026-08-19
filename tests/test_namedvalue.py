"""Contract tests for the Gold v2 named-value substrate and injectors.

These lock the properties the admissibility argument rests on, so a later refactor cannot quietly
reintroduce the v1 failure mode:

  - an injection mutates a VALUE; edges are derived, never stored and edited;
  - the source corpus is never mutated (deep-copy semantics);
  - labels are injection sites fixed at construction, independent of any detector;
  - a corpus is reproducible from (seed, parameters);
  - each fault actually produces its intended semantic shape.
"""
import pytest

# Not pytest.importorskip: the ImportError comes from catchbench._reuse, a module deeper than
# the one requested, and pytest changed how it treats that. 9.0 skipped; 9.1 lets the error escape
# and breaks collection. An explicit guard behaves the same on every version. CI installs GRADE, so
# this path is for a contributor working without a GRADE checkout.
try:
    from catchbench import namedvalue as nv
except ImportError as exc:  # pragma: no cover - exercised only without a GRADE checkout
    pytest.skip(f"needs a GRADE checkout: {exc}", allow_module_level=True)


def _mk(events, given=()):
    evs = [nv.Event(i, k, t, dict(a) if a else {}, p)
           for i, (k, t, a, p) in enumerate(events)]
    return nv.build_graph("t", evs, set(given))


CALL, RES = "call", "result"


def test_provenance_classes():
    """derived / given / ungrounded partition every consumption."""
    g = _mk([
        (CALL, "search", {"q": "AB12"}, None),          # given by the user
        (RES, "search", None, {"order_id": "W9999"}),   # produces W9999
        (CALL, "fetch", {"order_id": "W9999"}, None),   # derived
        (RES, "fetch", None, {"ok": 1}),
        (CALL, "act", {"order_id": "ZZ77"}, None),      # ungrounded
    ], given={"AB12"})
    prov = {(c.event_idx, c.key): c.provenance for c in g.consumptions}
    assert prov[(0, "q")] == nv.GIVEN
    assert prov[(2, "order_id")] == nv.DERIVED
    assert prov[(4, "order_id")] == nv.UNGROUNDED


def test_edges_are_derived_not_stored():
    """Rewriting a consumed value re-points the dependency without touching any stored edge."""
    g = _mk([
        (RES, "read", None, {"v": "AA11"}),
        (CALL, "x", {"k": "AA11"}, None),
        (RES, "x", None, {"v": "BB22"}),
        (CALL, "y", {"k": "BB22"}, None),
    ])
    before = [c for c in g.consumptions if c.event_idx == 3][0]
    assert before.producers == (2,)
    g.events[3].args["k"] = "AA11"          # mutate the VALUE only
    rebuilt = nv.build_graph(g.run_id, g.events, g.given, g.res_idx)
    after = [c for c in rebuilt.consumptions if c.event_idx == 3][0]
    assert after.producers == (0,), "edge must follow the value it was derived from"


def test_supersession_and_stale_eligibility():
    g = _mk([
        (CALL, "get", {"id": "R100"}, None),
        (RES, "get", None, {"id": "R100", "seat": "S1"}),      # observation 1
        (CALL, "get", {"id": "R100"}, None),
        (RES, "get", None, {"id": "R100", "seat": "S2"}),      # S1 superseded
        (CALL, "use", {"id": "R100", "seat": "S2"}, None),     # consumer after
    ])
    sites = g.eligible_stale()
    assert sites, "a superseded value plus a later consumer is an eligible triple"
    assert "S1" in sites[0]["superseded"]


def test_inject_stale_writes_a_superseded_value_and_leaves_source_clean():
    import random
    g = _mk([
        (CALL, "get", {"id": "R100"}, None),
        (RES, "get", None, {"id": "R100", "seat": "S1"}),
        (CALL, "get", {"id": "R100"}, None),
        (RES, "get", None, {"id": "R100", "seat": "S2"}),
        (CALL, "use", {"id": "R100", "seat": "S2"}, None),
    ])
    out = nv.inject_stale(g, random.Random(0))
    assert out is not None
    injected, label = out
    assert label.kind == "stale"
    assert label.injected == "S1" and label.original == "S2"
    assert injected.events[label.event_idx].args[label.key] == "S1"
    # the source graph is untouched
    assert g.events[4].args["seat"] == "S2"


def test_inject_dropped_produces_ungrounded_provenance():
    import random
    g = _mk([
        (RES, "read", None, {"order_id": "W1111"}),
        (CALL, "act", {"order_id": "W1111"}, None),
    ])
    pool = {"order_id": ["W1111", "W2222"]}
    out = nv.inject_dropped(g, random.Random(0), pool)
    assert out is not None
    injected, label = out
    assert label.kind == "dropped"
    hit = [c for c in injected.consumptions
           if c.event_idx == label.event_idx and c.key == label.key][0]
    assert hit.provenance == nv.UNGROUNDED
    assert label.injected != label.original
    assert g.events[1].args["order_id"] == "W1111"   # source untouched


def test_donor_values_are_real_corpus_values():
    """Donors come from the corpus, never synthesized, so format stays in distribution."""
    a = _mk([(RES, "r", None, {"order_id": "W1111"}), (CALL, "c", {"order_id": "W1111"}, None)])
    b = _mk([(RES, "r", None, {"order_id": "W2222"}), (CALL, "c", {"order_id": "W2222"}, None)])
    pool = nv.donor_pool([a, b])
    assert set(pool["order_id"]) == {"W1111", "W2222"}


def test_corpus_is_deterministic_under_seed():
    graphs = [
        _mk([(RES, "r", None, {"order_id": f"W{i}000"}),
             (CALL, "c", {"order_id": f"W{i}000"}, None)])
        for i in range(4)
    ]
    a = nv.build_corpus(graphs, seed=7)
    b = nv.build_corpus(graphs, seed=7)
    assert [(p.label.kind, p.label.event_idx, p.label.injected) for p in a] == \
           [(p.label.kind, p.label.event_idx, p.label.injected) for p in b]


def test_nested_argument_leaves_become_consumptions():
    """Spec section 3: values nested in argument dicts/lists are consumptions too, addressed by
    path, not only top-level scalars."""
    g = _mk([
        (RES, "search", None, {"order_id": "W1234"}),
        (CALL, "act", {"payload": {"order_id": "W1234"}, "items": ["A1B2"]}, None),
    ])
    keys = {c.key for c in g.consumptions}
    assert "payload.order_id" in keys and "items.[]" in keys
    nested = [c for c in g.consumptions if c.key == "payload.order_id"][0]
    assert nested.path == ("payload", "order_id") and nested.provenance == nv.DERIVED


def test_inject_dropped_mutates_the_exact_nested_leaf():
    """Injection targets the addressed leaf and leaves the rest of the structure intact."""
    import random
    g = _mk([
        (RES, "r", None, {"order_id": "W1111"}),
        (CALL, "act", {"payload": {"order_id": "W1111", "keep": "S1"}}, None),
    ])
    out = nv.inject_dropped(g, random.Random(0), {"payload.order_id": ["W1111", "W2222"]})
    assert out is not None
    injected, label = out
    assert label.path == ("payload", "order_id")
    assert injected.events[1].args["payload"]["order_id"] == "W2222"
    assert injected.events[1].args["payload"]["keep"] == "S1"       # sibling untouched
    assert g.events[1].args["payload"]["order_id"] == "W1111"       # source untouched


def test_list_siblings_share_key_but_have_distinct_sites():
    """Prerequisite for the exact-site donor rule: same-collapsed-key list siblings (flights[0] vs
    flights[1]) are distinct (event, path) sites, so the injector can exclude one while keeping the
    other. Donor-matching behaviour itself is locked by the next test."""
    g = _mk([
        (RES, "s", None, {"a": "F0001", "b": "F0002"}),
        (CALL, "book", {"flights": ["F0001", "F0002"]}, None),
    ])
    sibs = [c for c in g.consumptions if c.key == "flights.[]"]
    assert len(sibs) == 2
    assert {s.path for s in sibs} == {("flights", 0), ("flights", 1)}
    assert sibs[0].key == sibs[1].key and sibs[0].path != sibs[1].path


def test_donor_matching_excludes_only_the_exact_site_not_the_event():
    """Locks the High-1 fix by exercising donor selection, not just site identity. The target
    flights[0]=W1111 (derived) has a same-event sibling flights[1]=W9999 (ungrounded). Under
    exact-site exclusion the sibling stays in the neighbourhood, so neigh=[W9999], the target's
    neighbour distance is 4, and W0000 is the unique gap-0 donor. Reverting to whole-event exclusion
    empties the neighbourhood, ties both donors at gap 0, and Random(0) then selects Z9999. Asserting
    the exact-site choice therefore fails on that regression."""
    import random
    g = _mk([
        (RES, "search", None, {"order_id": "W1111"}),
        (CALL, "book", {"flights": ["W1111", "W9999"]}, None),
    ])
    out = nv.inject_dropped(g, random.Random(0), {"flights.[]": ["W0000", "Z9999"]})
    assert out is not None
    _, label = out
    assert label.path == ("flights", 0), "the derived list leaf is the injected site"
    assert label.injected == "W0000", "exact-site matching picks the uniquely distance-matched donor"


def test_edge_records_only_latest_producer():
    """Spec section 3: a consumption's dependency is the LATEST prior producer, not all of them."""
    g = _mk([
        (RES, "r", None, {"v": "W1000"}),      # producer 0
        (RES, "r", None, {"v": "W1000"}),      # producer 1 (later, same value)
        (CALL, "c", {"v": "W1000"}, None),     # consumer
    ])
    c = [x for x in g.consumptions if x.event_idx == 2][0]
    assert c.producers == (1,), "only the latest prior producer is recorded"


def test_deep_copy_isolates_nested_arguments():
    """_copy must not alias nested argument structures back to the source graph."""
    import random
    g = _mk([
        (RES, "r", None, {"order_id": "W1111"}),
        (CALL, "act", {"order_id": "W1111", "opts": {"nested": "keep"}}, None),
    ])
    out = nv.inject_dropped(g, random.Random(0), {"order_id": ["W1111", "W2222"]})
    assert out is not None
    injected, _ = out
    injected.events[1].args["opts"]["nested"] = "mutated"
    assert g.events[1].args["opts"]["nested"] == "keep", "source nested arg must stay clean"


def test_dropped_donor_excludes_produced_but_unconsumed_values():
    """A donor must have no source in the run at all, including a value produced but never
    consumed. Otherwise the injected 'ungrounded' value was in fact produced here (High 3)."""
    import random
    g = _mk([
        (RES, "r", None, {"order_id": "W1111"}),          # W1111 consumed below
        (CALL, "act", {"order_id": "W1111"}, None),
        (RES, "look", None, {"order_id": "W9999"}),       # W9999 produced, never consumed
    ])
    # donor pool offers W9999 (produced-unconsumed here) and W2222 (truly foreign)
    out = nv.inject_dropped(g, random.Random(0), {"order_id": ["W9999", "W2222"]})
    assert out is not None
    _, label = out
    assert label.injected == "W2222", "must not pick a value this run produced"


def test_identifier_shape_rule():
    assert nv.identifier_shaped("HAT069") and nv.identifier_shaped("W1234")
    assert not nv.identifier_shaped("available")   # no digit
    assert not nv.identifier_shaped("12345")       # no letter
    assert not nv.identifier_shaped("AB1")         # too short


def test_constant_score_matches_analytic_floors():
    """Uniform ties must give the target its expected share, independent of pool order."""
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    # `tools` is a plain directory beside the package, not an installed module, so it is importable
    # only when the repository root is on sys.path. That holds when pytest runs from the root, as CI
    # does, and not when it is pointed at an absolute test path from elsewhere. Resolve it here so
    # the test reports a real result rather than a ModuleNotFoundError that looks like a code fault.
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from tools import namedvalue_admissibility as adm

    sites = [(0, ("a",)), (1, ("b",)), (2, ("target",))]
    consumptions = [SimpleNamespace(event_idx=event_idx, path=path)
                    for event_idx, path in sites]
    graph = SimpleNamespace(consumptions=consumptions)
    label = SimpleNamespace(event_idx=2, path=("target",))
    pair = SimpleNamespace(clean=graph, injected=graph, label=label)
    constant_score = lambda _graph, _consumption, _stats: 0.0

    top1, floor = adm.top1_and_floor([pair], constant_score, {}, [sites])

    assert top1 == pytest.approx(floor)
    assert floor == pytest.approx(1.0 / len(sites))
    assert adm.run_auc([pair], constant_score, {}) == pytest.approx(0.5)
