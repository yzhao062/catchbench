r"""Score the two PRE label makers as methods against the held-out judge, on a common population.

Appendix~\ref{app:artifact} of the paper used to say that no committed command reproduces this
comparison, and that reproducing it would need model access and a prose-retaining harvest. Both
halves were wrong. The comparison rescores committed votes; it generates nothing. The three inputs
are data/pre/gpt_judge_votes/*.json, data/pre/claude_judge_votes/*.json and
data/pre/llm_judge_method/llama-3.3-70b.json, all released, and this script reads only those plus the
de-identified records in data/pre/*.json. Generating fresh judgments is the operation that needs a
model and the prose; rescoring stored ones is not.

The comparison also had a second defect, which is the reason this script reports two populations
instead of one. The two label makers voted on all 661 judge-labeled configurations. The held-out
model answered all of them too, but five of its replies named a capability whose spelling did not
match the declared roster, and the strict parser behind data/pre/llm_judge_method discarded those
five whole judgments, so the held-out row rests on 656. Scoring a method on a population another
method did not face is the comparison pre_score's own docstring rules out: "Head-to-head claims
should be made on a common evaluable set, not on these cells." The five are not interchangeable with
any other five, either. One of them is a 622-capability MCP server carrying 337 excess labels, which
is 11.7 percent of this corpus's positive class.

So this prints both. The per-method column is each method's own coverage, which is the number a
single-method claim may use. The common column restricts all three to the 656 configurations every
method judged, and that column is the one a head-to-head sentence may quote.

What the comparison does NOT establish is a skill gap. A label maker's votes went into the merge
that built the key, and the merge marks a capability excess only when both judges call it unneeded,
so each label maker scores recall 1.000 by construction and its F1 is decided by precision alone.
That circularity is the finding. The numbers only measure how large it is.

Exits 0 when the six F1 values match what the paper prints, and 1 when they have drifted::

    python tools/pre_label_maker_diagnostic.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from catchbench.pre import PreOverPrivilege, pre_score  # noqa: E402
from pre_merge_judges import CLAUDE_DIR, GPT_DIR, PRE, SOURCES, load_claude, load_gpt  # noqa: E402

HELD_OUT_CACHE = PRE / "llm_judge_method" / "llama-3.3-70b.json"

# What Appendix D prints. A relabel or a reparse that moves these should fail loudly rather than
# leave the paper quoting a number no artifact produces.
EXPECTED_OWN = {"gpt-5.5": 0.854, "claude-opus-4-8": 0.960, "llama-3.3-70b": 0.703}
EXPECTED_COMMON = {"gpt-5.5": 0.824, "claude-opus-4-8": 0.955, "llama-3.3-70b": 0.703}


def _held_out_votes() -> dict[str, set]:
    """The held-out cache, in the id -> needed-set shape load_gpt and load_claude return."""
    raw = json.loads(HELD_OUT_CACHE.read_text(encoding="utf-8"))
    return {iid: set(names) for iid, names in raw.items() if isinstance(names, list)}


def _label_maker_votes(load) -> dict[str, set]:
    out: dict[str, set] = {}
    for source in SOURCES:
        out.update(load(source))
    return out


def _flagged(votes: dict[str, set], instances: list) -> dict[str, set]:
    """Declared minus needed, for the configurations this method actually voted on."""
    out = {}
    for inst in instances:
        if inst.instance_id not in votes:
            continue
        declared = {c["name"] for c in inst.declared_capabilities}
        out[inst.instance_id] = declared - votes[inst.instance_id]
    return out


def main() -> int:
    if not (GPT_DIR.exists() and CLAUDE_DIR.exists() and HELD_OUT_CACHE.exists()):
        print("missing committed vote records; nothing to score")
        return 1

    task = PreOverPrivilege()
    task.setup()
    judged = [o for o in task.instances if o.labels["label_source"] == "llm_judge"]

    methods = {
        "gpt-5.5": _label_maker_votes(load_gpt),
        "claude-opus-4-8": _label_maker_votes(load_claude),
        "llama-3.3-70b": _held_out_votes(),
    }
    flagged = {name: _flagged(votes, judged) for name, votes in methods.items()}
    common = set.intersection(*(set(f) for f in flagged.values()))

    print("PRE label makers scored as methods, against the held-out judge")
    print("judge-labeled configurations: %d" % len(judged))
    print()
    print("%-18s %-10s %8s %8s %8s   %8s %8s %8s"
          % ("method", "role", "own n", "own P", "own F1", "common n", "common P", "common F1"))
    roles = {"gpt-5.5": "label maker", "claude-opus-4-8": "label maker",
             "llama-3.3-70b": "held out"}
    drifted = []
    for name, f in flagged.items():
        own = pre_score(f, judged, evaluable=set(f))
        both = pre_score(f, judged, evaluable=common)
        print("%-18s %-10s %8d %8.3f %8.3f   %8d %8.3f %8.3f"
              % (name, roles[name], len(f), own["precision"], own["f1"],
                 len(common), both["precision"], both["f1"]))
        if own["f1"] != EXPECTED_OWN[name]:
            drifted.append("%s own F1 %.3f, paper prints %.3f"
                           % (name, own["f1"], EXPECTED_OWN[name]))
        if both["f1"] != EXPECTED_COMMON[name]:
            drifted.append("%s common F1 %.3f, paper prints %.3f"
                           % (name, both["f1"], EXPECTED_COMMON[name]))

    missing = sorted(set.union(*(set(f) for f in flagged.values())) - common)
    print()
    print("configurations outside the common population (%d): %s"
          % (len(missing), ", ".join(missing) if missing else "none"))
    print("recall is 1.000 by construction for a label maker: its votes built the key it is scored")
    print("against, so read precision, and read the common columns for any head-to-head claim.")

    if drifted:
        print()
        for line in drifted:
            print("DRIFT: %s" % line)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
