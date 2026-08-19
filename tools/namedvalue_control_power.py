"""Positive control: do the admissibility controls actually have detection power?

A PASS from `namedvalue_admissibility.py` is only meaningful if the controls can catch a process
artifact when one is present. Otherwise "every control sits at the floor" would report a broken
control rather than a clean substrate, and a false PASS is far worse than a false FAIL.

So plant artifacts deliberately and require the matching control to fire:

  synthesized-format : rewrite the value to something no tool emits (wrong length and character
                       class for the field). `format-outlier` must catch it.
  schema-mutation    : add an argument key the tool never carries. `schema-shape` must catch it.
  position-locked    : always inject at the last eligible consumption. `position-prior` must catch it.

Each is exactly the kind of mistake a careless injector makes, and each is what the real injector
avoids by drawing donors from real corpus values of the same field and leaving argument structure
alone.

Run:  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src python tools/namedvalue_control_power.py
Exit 0 iff every planted artifact is caught by its control.
"""
import os
import random
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from catchbench import namedvalue as nv  # noqa: E402


def _load_checker():
    """Import the checker module by path so this stays a sibling tool without a package."""
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "nv_adm", os.path.join(here, "namedvalue_admissibility.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def plant(graphs, mode, seed=0, limit=200):
    """Build a paired corpus whose injection deliberately carries a process artifact."""
    rng = random.Random(seed)
    pairs = []
    for g in graphs[:limit]:
        sites = g.eligible_dropped()
        if not sites:
            continue
        if mode == "position-locked":
            c = sites[-1]                      # always the last eligible site
        else:
            c = sites[rng.randrange(len(sites))]
        out = nv._copy(g)
        if mode == "synthesized-format":
            nv._set_by_path(out.events[c.event_idx].args, c.path, "ZZZZZZZZZZZZZZZZZZZZZZZZ_synth_0000")
        elif mode == "schema-mutation":
            out.events[c.event_idx].args["__injected_flag__"] = "1"
            nv._set_by_path(out.events[c.event_idx].args, c.path, "QQ99")
        else:                                   # position-locked keeps the value plausible
            nv._set_by_path(out.events[c.event_idx].args, c.path, "W0000")
        out = nv._rebuild(out)
        pairs.append(nv.Pair(g, out,
                             nv.Injection("dropped", c.event_idx, c.key, c.value, "x", {}, c.path)))
    return pairs


EXPECT = {"synthesized-format": "format-outlier",
          "schema-mutation": "schema-shape",
          "position-locked": "position-prior"}


def main():
    adm = _load_checker()
    print("loading substrate...", flush=True)
    graphs = nv.load_graphs()
    stats = adm.corpus_stats(graphs)
    ctl = dict(adm.CONTROLS)
    ok = True
    print(f"substrate: {len(graphs)} runs; planting artifacts into the first 200\n", flush=True)
    for mode, expected in EXPECT.items():
        pairs = plant(graphs, mode)
        pools = adm.pools_for(pairs)
        t1, floor = adm.top1_and_floor(pairs, ctl[expected], stats, pools)
        gap = t1 - floor
        fired = gap > adm.TOP1_MARGIN
        ok &= fired
        print(f"  planted {mode:20s} -> {expected:16s} Top-1 {t1:.3f} vs floor {floor:.3f} "
              f"(gap {gap:+.3f})  {'CAUGHT' if fired else 'MISSED'}", flush=True)
    print()
    if ok:
        print("PARTIAL CONTROL POWER - planted artifacts were caught for format-outlier, "
              "schema-shape, and position-prior on Top-1. This covers three of the six controls on "
              "one output; a future PASS is not supported until all six controls are power-tested on "
              "Top-1 and run-level AUC.")
    else:
        print("CONTROL FAILURE - a planted artifact went undetected. Any PASS from the "
              "admissibility check is meaningless until the missed control is fixed.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
