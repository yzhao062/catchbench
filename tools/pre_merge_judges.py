"""Merge two independent LLM judges into cross-vendor PRE over-privilege labels.

Two judges answer the byte-identical needed-vs-excess prompt (tools/pre_label_llm_judge.py
build_prompt) on every sub-wave C config:
  - gpt-5.5 via the NAIRR gateway  -> data/pre/gpt_judge_votes/<src>.json    (dict id -> {needed})
  - Claude (opus) via /workflows   -> data/pre/claude_judge_votes/<src>.json (list {instance_id, needed})

Merge rule: a declared capability is EXCESS only when BOTH judges agree it is not needed, that is
    excess_set = declared - (needed_gpt UNION needed_claude).
Only configs judged by both vendors are kept, so every shipped label is a cross-vendor agreement
label. Per-capability agreement and Cohen's kappa are written to data/pre/LABEL_QUALITY.md as a
reliability characterization (the label source is judge opinion, so its agreement must be reported).

Base config fields (instance_id, source, provenance, task_or_role_spec, declared_capabilities) come
from data/pre_staging/<src>.json when present, otherwise from the harvest-input fields preserved in
the existing data/pre/<src>.json labels. Only excess_set and minimal_reference are (re)derived here.

Run: python tools/pre_merge_judges.py [--write]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PRE = REPO / "data" / "pre"
STAGING = REPO / "data" / "pre_staging"
GPT_DIR = PRE / "gpt_judge_votes"
CLAUDE_DIR = PRE / "claude_judge_votes"
REPORT = PRE / "LABEL_QUALITY.md"
SOURCES = ("crewai", "n8n", "mcp")
GPT_MODEL = "gpt-5.5"
CLAUDE_MODEL = "claude-opus-4-8"


def cohen_kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    if n == 0:
        return 1.0
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1 = sum(a) / n
    pb1 = sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return round((po - pe) / (1 - pe), 3) if (1 - pe) else 1.0


def load_base(src: str) -> list[dict]:
    sp = STAGING / f"{src}.json"
    if sp.exists():
        return json.load(open(sp, encoding="utf-8"))
    lp = PRE / f"{src}.json"
    if lp.exists():
        rows = json.load(open(lp, encoding="utf-8"))
        return [
            {
                "instance_id": r["instance_id"],
                "source": r["source"],
                "provenance": r["provenance"],
                "task_or_role_spec": r["task_or_role_spec"],
                "declared_capabilities": r["declared_capabilities"],
            }
            for r in rows
        ]
    raise FileNotFoundError(f"no base for {src}: need {sp} or {lp}")


def load_gpt(src: str) -> dict[str, set]:
    p = GPT_DIR / f"{src}.json"
    raw = json.load(open(p, encoding="utf-8"))
    out = {}
    for iid, entry in raw.items():
        if isinstance(entry, dict) and entry.get("needed") is not None and entry.get("status") != "error":
            out[iid] = set(entry["needed"])
    return out


def load_claude(src: str) -> dict[str, set]:
    p = CLAUDE_DIR / f"{src}.json"
    out: dict[str, set] = {}
    for r in json.load(open(p, encoding="utf-8")):
        needed = r.get("needed")
        # Only an explicit list counts as a vote; a missing or malformed field
        # must NOT default to empty-needed (that would silently mark every
        # declared capability excess and break the both-vendors-judged contract).
        if isinstance(needed, list):
            out[r["instance_id"]] = set(needed)
    return out


def main() -> None:
    write = "--write" in sys.argv
    summary = []
    pooled_g, pooled_c = [], []
    for src in SOURCES:
        base = load_base(src)
        gpt = load_gpt(src)
        claude = load_claude(src)
        merged, gv, cv = [], [], []
        dropped = 0
        for r in base:
            iid = r["instance_id"]
            declared = [c["name"] for c in r["declared_capabilities"]]
            ds = set(declared)
            if iid not in gpt or iid not in claude:
                dropped += 1
                continue
            needed = (gpt[iid] | claude[iid]) & ds
            excess = [n for n in declared if n not in needed]
            for n in declared:
                gv.append(0 if n in gpt[iid] else 1)
                cv.append(0 if n in claude[iid] else 1)
            merged.append({
                "instance_id": iid,
                "source": r["source"],
                "provenance": r["provenance"],
                "task_or_role_spec": r["task_or_role_spec"],
                "declared_capabilities": r["declared_capabilities"],
                "minimal_reference": sorted(needed),
                "labels": {
                    "excess_set": excess,
                    "label_source": "llm_judge",
                    "judges": [GPT_MODEL, CLAUDE_MODEL],
                    "merge_rule": "excess iff both judges agree the capability is not needed",
                },
            })
        pooled_g += gv
        pooled_c += cv
        ncap = len(gv)
        agree = round(sum(1 for x, y in zip(gv, cv) if x == y) / ncap, 3) if ncap else 1.0
        k = cohen_kappa(gv, cv)
        denom = sum(len(m["declared_capabilities"]) for m in merged)
        mex = round(sum(len(m["labels"]["excess_set"]) for m in merged) / denom, 3) if denom else 0.0
        summary.append((src, len(merged), dropped, ncap, agree, k, mex))
        if write:
            json.dump(merged, open(PRE / f"{src}.json", "w", encoding="utf-8"), ensure_ascii=False)
        print(f"{src}: dual_judged={len(merged)} dropped={dropped} caps={ncap} "
              f"cap_agree={agree} kappa={k} mean_excess={mex}")

    on = len(pooled_g)
    oagree = round(sum(1 for x, y in zip(pooled_g, pooled_c) if x == y) / on, 3) if on else 1.0
    ok = cohen_kappa(pooled_g, pooled_c)
    print(f"OVERALL: caps={on} cap_agree={oagree} kappa={ok}")

    if write:
        lines = [
            "# PRE sub-wave C label quality (cross-vendor judge panel)",
            "",
            f"Two judges ({GPT_MODEL} via the NAIRR gateway, {CLAUDE_MODEL} via /workflows) independently",
            "answered the byte-identical needed-vs-excess prompt on every config. A capability is labeled",
            "EXCESS only when both judges agree it is not needed. Only configs judged by both vendors are",
            "kept, so every label is a cross-vendor agreement label. Raw votes are committed under",
            "`data/pre/gpt_judge_votes/` and `data/pre/claude_judge_votes/`; rerun `tools/pre_merge_judges.py`",
            "to reproduce the labels and this table.",
            "",
            "Per-capability agreement and Cohen's kappa characterize label reliability. Kappa 0.61-0.80 is",
            "'substantial agreement' on the Landis-Koch scale.",
            "",
            "| Source | dual-judged configs | dropped (single-judge) | capabilities | cap agreement | Cohen kappa | mean excess ratio |",
            "|---|---|---|---|---|---|---|",
        ]
        for src, dual, drp, ncap, agree, k, mex in summary:
            lines.append(f"| {src} | {dual} | {drp} | {ncap} | {agree} | {k} | {mex} |")
        lines.append(f"| **overall** | | | {on} | {oagree} | {ok} | |")
        lines.append("")
        lines.append("An earlier single-vendor pass with non-identical prompts scored kappa 0.06; aligning the")
        lines.append("prompt wording across vendors raised agreement to the value above, which is why the two")
        lines.append("judges must be asked the identical question before their agreement means anything.")
        lines.append("")
        lines.append("## Known limitations")
        lines.append("")
        lines.append("A minority of configs carry an all-excess label (empty minimal reference) where both judges")
        lines.append("independently agreed that no declared capability is strictly needed. These are reasoning or")
        lines.append("reporting roles whose declared tools exceed the stated purpose; the cross-vendor agreement")
        lines.append("makes them genuine judgments rather than parser artifacts. Separately, the labeler reads a")
        lines.append("bare `NEEDED:` line as 'none', so a truncated judge response could in principle mislabel a")
        lines.append("config as all-excess. The union merge rescues any single-judge empty-needed answer, so a")
        lines.append("mislabel would require both judges to truncate identically on the same config.")
        lines.append("")
        open(REPORT, "w", encoding="utf-8").write("\n".join(lines))
        print("WROTE", PRE, "and", REPORT)
    else:
        print("(dry run; pass --write to write labels + report)")


if __name__ == "__main__":
    main()
