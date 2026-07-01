# PRE sub-wave C label quality (cross-vendor judge panel)

Two judges (gpt-5.5 via the NAIRR gateway, claude-opus-4-8 via /workflows) independently
answered the byte-identical needed-vs-excess prompt on every config. A capability is labeled
EXCESS only when both judges agree it is not needed. Only configs judged by both vendors are
kept, so every label is a cross-vendor agreement label. Raw votes are committed under
`data/pre/gpt_judge_votes/` and `data/pre/claude_judge_votes/`; rerun `tools/pre_merge_judges.py`
to reproduce the labels and this table.

Per-capability agreement and Cohen's kappa characterize label reliability. Kappa 0.61-0.80 is
'substantial agreement' on the Landis-Koch scale.

| Source | dual-judged configs | dropped (single-judge) | capabilities | cap agreement | Cohen kappa | mean excess ratio |
|---|---|---|---|---|---|---|
| crewai | 298 | 0 | 598 | 0.753 | 0.492 | 0.241 |
| n8n | 219 | 0 | 550 | 0.88 | 0.515 | 0.084 |
| mcp | 144 | 0 | 3002 | 0.838 | 0.674 | 0.485 |
| **overall** | | | 4150 | 0.832 | 0.666 | |

An earlier single-vendor pass with non-identical prompts scored kappa 0.06; aligning the
prompt wording across vendors raised agreement to the value above, which is why the two
judges must be asked the identical question before their agreement means anything.
