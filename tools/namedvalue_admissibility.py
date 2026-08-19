"""Gold v2 admissibility check on the named-value substrate.

Compares each injection site with the injector's matched eligible pool and tests the
FIXED-THRESHOLD criterion:

  process-artifact controls : injection-site Top-1 minus matched floor must have its seed-block
                              95% CI inside [-0.05, +0.05], AND run-level ROC-AUC CI inside
                              [0.45, 0.55]. Five seeds (0..4).
  fault-definitional oracles: reported, expected high; they are the ceiling a method must beat,
                              not evidence of leakage.

The decision bands and the seed count are held fixed across the reported runs. The injector was
revised after control feedback, so this is a fixed decision threshold applied to an adaptive,
in-sample diagnostic, not a confirmatory pre-registered test.

Run:  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src python tools/namedvalue_admissibility.py
"""
import os
import statistics as st
import sys
from collections import Counter, defaultdict

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from catchbench import namedvalue as nv  # noqa: E402

T975_4 = 2.776  # t(0.975, df=4): seed-block CI half-width over 5 seeds
SEEDS = (0, 1, 2, 3, 4)
TOP1_MARGIN = 0.05
AUC_BAND = (0.45, 0.55)
KIND = os.environ.get("NV_KIND", "dropped")  # tau affords no usable stale population


# --- corpus-level statistics the controls compare against -------------------------------------

def corpus_stats(graphs):
    field_len, field_charclass, tool_keys, key_count = defaultdict(list), defaultdict(Counter), \
        defaultdict(Counter), Counter()
    for g in graphs:
        for c in g.consumptions:
            field_len[c.key].append(len(c.value))
            field_charclass[c.key][_charclass(c.value)] += 1
            key_count[c.key] += 1
        for e in g.events:
            if e.kind == "call":
                tool_keys[e.tool][tuple(sorted(e.args))] += 1
    return {"field_len": {k: (st.mean(v), st.pstdev(v) or 1.0) for k, v in field_len.items()},
            "field_charclass": field_charclass, "tool_keys": tool_keys, "key_count": key_count}


def _charclass(s):
    return (any(c.isdigit() for c in s), any(c.isalpha() for c in s),
            any(not c.isalnum() for c in s))


# --- controls: process artifacts only (no fault semantics) ------------------------------------

def ctl_format_outlier(g, c, stats):
    mu, sd = stats["field_len"].get(c.key, (len(c.value), 1.0))
    z = abs(len(c.value) - mu) / (sd or 1.0)
    cc = stats["field_charclass"].get(c.key, Counter())
    tot = sum(cc.values()) or 1
    rarity = 1.0 - cc.get(_charclass(c.value), 0) / tot
    return z + rarity


def ctl_schema_shape(g, c, stats):
    ev = g.events[c.event_idx]
    seen = stats["tool_keys"].get(ev.tool, Counter())
    tot = sum(seen.values()) or 1
    return 1.0 - seen.get(tuple(sorted(ev.args)), 0) / tot


def ctl_position_prior(g, c, stats):
    n = max(len(g.events) - 1, 1)
    return c.event_idx / n


def ctl_field_prior(g, c, stats):
    return -stats["key_count"].get(c.key, 0)


def ctl_tool_prior(g, c, stats):
    ev = g.events[c.event_idx]
    return -sum(stats["tool_keys"].get(ev.tool, Counter()).values())


def ctl_edit_distance(g, c, stats):
    others = [x.value for x in g.consumptions if x.key == c.key and x.value != c.value]
    if not others:
        return 0.0
    return min(_lev(c.value, o) for o in others)


def _lev(a, b):
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# --- oracles: fault-definitional (expected high, reported separately) -------------------------

def orc_superseded(g, c, stats):
    """Does this consumption carry a value a later observation of its entity superseded, at the
    level of the field that held it? Field-level, matching the injector's eligibility rule."""
    for _key, obs in g.entities.items():
        for (i_a, map_a), (i_b, map_b) in zip(obs, obs[1:]):
            if c.event_idx <= i_b:
                continue
            for path, old_vals in map_a.items():
                new_vals = map_b.get(path)
                if new_vals and c.value in (old_vals - new_vals):
                    return 1.0
    return 0.0


def orc_provenance(g, c, stats):
    return 1.0 if c.provenance == nv.UNGROUNDED else 0.0


CONTROLS = [("format-outlier", ctl_format_outlier), ("schema-shape", ctl_schema_shape),
            ("position-prior", ctl_position_prior), ("field-prior", ctl_field_prior),
            ("tool-prior", ctl_tool_prior), ("edit-distance", ctl_edit_distance)]
ORACLES = [("superseded-value", orc_superseded), ("provenance", orc_provenance)]


# --- evaluation --------------------------------------------------------------------------------

def eligible_pool(g, kind):
    """Rank only inside the injector's eligible pool, so target selection is not read as signal."""
    if kind == "dropped":
        return g.eligible_dropped()
    seen, pool = set(), []
    for site in g.eligible_stale():
        c = site["consumption"]
        if (c.event_idx, c.path) not in seen:
            seen.add((c.event_idx, c.path))
            pool.append(c)
    return pool


def pools_for(pairs):
    """Eligible pool per pair, as (event_idx, path) sites taken from the CLEAN graph.

    Sites are keyed by the exact leaf path, not the field key, because nested argument leaves can
    share a field key while addressing different values. The pool must come from the clean run,
    because it is the set of sites the injector could have chosen. Reading it off the injected graph
    is wrong and silently self-defeating: injection moves the chosen site out of its own eligibility
    class (a dropped-grounding target stops being ``derived``), so the true site would not appear
    among the candidates being ranked.
    """
    out = []
    for p in pairs:
        pool = [(c.event_idx, c.path) for c in eligible_pool(p.clean, p.label.kind)]
        if (p.label.event_idx, p.label.path) not in pool:
            pool.append((p.label.event_idx, p.label.path))
        out.append(pool)
    return out


def _tie_aware_top1(pool_scores, inj_local):
    """Expected Top-1 for the injected site under uniform random tie breaking."""
    injected_score = pool_scores[inj_local]
    greater = sum(score > injected_score for score in pool_scores)
    tied = sum(score == injected_score for score in pool_scores)
    ranks = range(greater + 1, greater + tied + 1)
    return sum(rank == 1 for rank in ranks) / tied


def top1_and_floor(pairs, fn, stats, pools):
    """Tie-aware injection-site Top-1 inside the eligible pool, and the matched random floor.

    Sites come from the clean run; scores are read off the injected run at those same sites.
    """
    hits, floors = [], []
    for p, pool in zip(pairs, pools):
        if not pool:
            continue
        by_site = {(c.event_idx, c.path): c for c in p.injected.consumptions}
        scored = [(site, fn(p.injected, by_site[site], stats))
                  for site in pool if site in by_site]
        if not scored:
            continue
        target = (p.label.event_idx, p.label.path)
        target_idx = next((i for i, (site, _) in enumerate(scored) if site == target), None)
        hits.append(_tie_aware_top1([score for _, score in scored], target_idx)
                    if target_idx is not None else 0.0)
        floors.append(1.0 / len(scored))
    return (st.mean(hits) if hits else 0.0), (st.mean(floors) if floors else 0.0)


def run_auc(pairs, fn, stats):
    """Run-level clean vs injected separation, scoring a run by its maximum consumption score."""
    pos, neg = [], []
    for p in pairs:
        pos.append(max((fn(p.injected, c, stats) for c in p.injected.consumptions), default=0.0))
        neg.append(max((fn(p.clean, c, stats) for c in p.clean.consumptions), default=0.0))
    return _auc(pos, neg)


def _auc(pos, neg):
    if not pos or not neg:
        return 0.5
    wins = ties = 0
    for a in pos:
        for b in neg:
            if a > b:
                wins += 1
            elif a == b:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def ci(vals):
    if len(vals) < 2:
        return (vals[0], 0.0) if vals else (0.0, 0.0)
    return st.mean(vals), T975_4 * st.stdev(vals) / (len(vals) ** 0.5)


def main():
    print("loading tau-bench and building named-value graphs...", flush=True)
    graphs = nv.load_graphs()
    stats = corpus_stats(graphs)
    print(f"named-value substrate: {len(graphs)} runs\n", flush=True)

    per_seed = defaultdict(lambda: {"gap": [], "auc": []})
    kinds = Counter()
    sizes = []
    for seed in SEEDS:
        pairs = [p for p in nv.build_corpus(graphs, seed=seed) if p.label.kind == KIND]
        pools = pools_for(pairs)
        sizes.append(len(pairs))
        print(f"  seed {seed}: {len(pairs)} pairs built", flush=True)
        if seed == SEEDS[0]:
            kinds.update(p.label.kind for p in pairs)
        for name, fn in CONTROLS + ORACLES:
            t1, floor = top1_and_floor(pairs, fn, stats, pools)
            print(f"    {name} scored", flush=True)
            per_seed[name]["gap"].append(t1 - floor)
            per_seed[name]["auc"].append(run_auc(pairs, fn, stats))
            if seed == SEEDS[0]:
                per_seed[name]["t1_first"] = t1
                per_seed[name]["floor_first"] = floor

    print(f"paired corpus: {sizes[0]} pairs (seed {SEEDS[0]}); "
          f"kinds {dict(kinds)}; sizes across seeds {sizes}\n")
    print(f"FIXED-THRESHOLD criterion: Top-1 gap CI inside [{-TOP1_MARGIN:+.2f}, {TOP1_MARGIN:+.2f}] "
          f"AND run-level AUC CI inside [{AUC_BAND[0]:.2f}, {AUC_BAND[1]:.2f}], {len(SEEDS)} seeds\n")

    print(f"{'control (process artifact)':28s}{'Top-1 - floor':>22s}{'run AUC':>20s}  verdict")
    all_pass = True
    for name, _ in CONTROLS:
        gm, gh = ci(per_seed[name]["gap"])
        am, ah = ci(per_seed[name]["auc"])
        ok = (abs(gm) + gh <= TOP1_MARGIN) and (AUC_BAND[0] <= am - ah and am + ah <= AUC_BAND[1])
        all_pass &= ok
        print(f"  {name:26s}{f'{gm:+.3f}+/-{gh:.3f}':>22s}{f'{am:.3f}+/-{ah:.3f}':>20s}  "
              f"{'PASS' if ok else 'FAIL'}")
    print()
    print(f"{'oracle (fault-definitional)':28s}{'Top-1 - floor':>22s}{'run AUC':>20s}  (expected high)")
    for name, _ in ORACLES:
        gm, gh = ci(per_seed[name]["gap"])
        am, ah = ci(per_seed[name]["auc"])
        print(f"  {name:26s}{f'{gm:+.3f}+/-{gh:.3f}':>22s}{f'{am:.3f}+/-{ah:.3f}':>20s}")
    print()
    print("ADMISSIBLE" if all_pass else "NOT ADMISSIBLE",
          "- every process-artifact control sits at the matched floor within the declared margin."
          if all_pass else "- at least one process-artifact control separates the classes; "
                           "the injector or the substrate needs work before this board is evidence.")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
