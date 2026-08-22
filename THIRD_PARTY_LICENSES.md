# Third-party sources and licences

CatchBench's MIT licence covers CatchBench-authored code and data. It does not replace the terms of
upstream material. The generated [`ASSET_MANIFEST.json`](ASSET_MANIFEST.json) is the authoritative
file-level inventory: it gives every upstream repository or dataset URL, source identifier, path,
record count, and declared licence distribution. Each artifact row counts the declarations of the
records that artifact actually holds, so a file covering part of a source set does not report the
whole set's distribution. Source identifiers are typed: `immutable_revision` is a git commit or a
Hugging Face dataset revision, and `mutable_identifier` is a name whose content can change under
it, which is the case for n8n template IDs and the SWE-bench submission label.

Redistributed licence text travels in this repository rather than by link. Every value that appears
as a declared licence on the committed PRE records falls into one of three classes:

- **One shared text covers every record.** Apache-2.0, GPL-3.0, and CC-BY-4.0 read the same for
  every project that declares them, so one copy is enough.
  [`third_party/licenses/CANONICAL-TEXTS.md`](third_party/licenses/CANONICAL-TEXTS.md) records,
  for each, where the copy was retrieved, its SHA-256, and every pinned upstream project whose
  records declare it.
- **Each project needs its own text.** MIT names the copyright holder, so one project's notice
  cannot stand in for another's. The CrewAI and MCP notices are in
  [`MIT-crewai.md`](third_party/licenses/MIT-crewai.md) and
  [`MIT-mcp.md`](third_party/licenses/MIT-mcp.md), the single InjecAgent notice is reproduced inline
  below, and the 56 synthetic records are first-party and covered by [`LICENSE`](LICENSE).
- **There is no text, because there is no licence.** `NOASSERTION`, `Other`, and `unverified` record
  the absence of a declaration. The records carrying them stay unfinished; a source section marked
  **UNFINISHED** is not cleared for redistribution merely because other records in that source have
  permissive declarations.

Reproduced upstream `NOTICE` text is in [`NOTICE`](NOTICE). Regenerate every generated file with
`python tools/emit_third_party_notices.py --fetch`. Offline, `--check` enumerates the licence values
the committed PRE records actually declare, rather than a list written into the tool, and fails when
any of them has no local text. A source that begins declaring a licence nobody anticipated therefore
fails the check by name instead of passing it silently. The enumeration reads `data/pre/`; the
Who&When artifacts under `data/llm_judge/` are covered by their own section below.

## Why the GPL-3.0 and CC-BY-4.0 records are in the release

Nine of the 298 CrewAI records declare GPL-3.0, and one of the 144 MCP records declares CC-BY-4.0.
All three upstream declarations were verified at their pinned revisions.
`opahopa/crewai-factory-crew` and `tom333/cv` each ship the GNU General Public License version 3,
byte-for-byte identical to the text the Free Software Foundation serves. `MicrosoftDocs/mcp` ships
the Creative Commons Attribution 4.0 International text, differing from the Creative Commons copy
only in whitespace. Digests for all of these are in
[`CANONICAL-TEXTS.md`](third_party/licenses/CANONICAL-TEXTS.md).

These records stay in the release. The obligation is acknowledged, the licence text ships here, and
the reasoning is written out below so that a reader can disagree with it on the evidence.

### What a GPL-3.0 record carries

Each of the nine records holds four fields. Open `data/pre/crewai.json` and search for `"GPL-3.0"` to
read all nine.

`declared_capabilities` lists the agent's tools, each reduced to a name, a type, and a permission
level. Across the nine records that is 28 entries drawing on 11 distinct names. The names are
identifiers copied from the upstream project: all 11 appear in that project's `crew.py`, because
neither project's YAML declares a `tools:` key and the harvester then falls back to a tool map parsed
out of the project's Python. Seven of the names are classes from the third-party `crewai_tools`
library, among them `SerperDevTool` and `FileReadTool`. Four are names the project wrote itself:
`CodeDocsSearchTool`, `ShellCommandTool`, `read_resume`, and `semantic_search_resume`. The `type` and
`permission_level` fields are CatchBench's own classification, assigned from a lookup table in
`tools/pre_harvest_crewai.py`, and neither exists in the upstream source.

`spec_tokens` holds single words taken from the agent's role, goal, and backstory prose. A word
survives only where a static scanner rule can consult it for that capability roster, and only after
identifiers are scrubbed. Across all nine records this is 20 tokens drawn from seven distinct words:
`code`, `creating`, `posting`, `resume`, `resumes`, `role`, and `roles`. `tools/pre_deidentify.py`
removes the prose from the record. For scale, the upstream `config/coding_crew/agents.yaml` is 39
lines of role descriptions and invented biographies, and none of those lines is in the record.

`minimal_reference` and `labels` are CatchBench's annotation: the smallest sufficient tool set, and
the excess set produced by the cross-vendor judge panel described in
[`data/pre/LABEL_QUALITY.md`](data/pre/LABEL_QUALITY.md). Neither field carries an upstream
statement.

`provenance` records the repository, the commit, the path, and the declared licence. One caveat about
that path: it names the YAML file the agent definition came from, while the tool names came from
`crew.py` in the same checkout at the same commit.

### The cached judge reply, and what was removed from it

Those four fields are not the only committed artifact that touches these records. The held-out judge
cached in `data/pre/llm_judge_method_votes/llama-3.3-70b.json` was prompted with each config's prose
before that prose was removed, and its reply restated part of it. An earlier draft of this file said
that no sentence of upstream prose was here. That was measured against the records and missed this
second file, and it was wrong.

Across the nine records the cached replies reproduced 15 runs of four or more consecutive words from
the upstream YAML. The longest ran to 19 words, and three replies carried 13 words or more, in two
cases most of an upstream `goal:` field. The label-making judges in `data/pre/gpt_judge_votes/` were
prompted with the same prose and are checked by the same pass; their replies for these nine records
paraphrase rather than quote, and none carries a run of four words. Naming the two cache directories
rather than one file is deliberate, because the first version of this pass looked at the held-out
judge alone and would have missed a label-making reply that did quote.
`tools/pre_deidentify.py --upstream-dir` removes those runs,
replacing each with `[upstream text removed: see THIRD_PARTY_LICENSES.md]`, and records what it
removed in [`UPSTREAM_REDACTIONS.json`](third_party/UPSTREAM_REDACTIONS.json). The pass selects
records by their declared licence rather than by a list of identifiers, so a copyleft source added
later is covered without a code change. `tools/pre_deidentify.py --check-upstream` re-checks the
result offline. The upstream sources the pass reads are fetched at their pinned commits and are not
committed here, because carrying them would redistribute in full the text the pass exists to remove.

The judge's own reasoning and its `needed` verdict come through the pass untouched, and no scored
number moves. The board reads the published `data/pre/llm_judge_method/` cache, which holds only the
`needed` lists, and the freshness predicate in `tools/pre_judge_method.py` reads `status`, `model`,
`model_id`, `needed`, and the prompt digest, never the reply text.

No source file is redistributed. What survives of the upstream prose is seven common English words
in `spec_tokens`, alongside 11 tool identifiers.

### Why this is a position and not a clearance

GPL-3.0 attaches to a work and to works derived from it. Whether nine rows of tool identifiers,
CatchBench permission labels, and seven scanner words form a derived work of a configuration is a
question this repository takes a position on rather than one it can settle. The position is that a
list of names read out of a project, stripped of the file structure and the prose that surrounded
them, describes that project rather than copying it, and that the labels attached to those names are
CatchBench's own work. That is a reading of the licence, offered as such, and it is not a legal
opinion.

Because it is a reading, the obligation is carried rather than argued away. The full licence text
ships in [`GPL-3.0.txt`](third_party/licenses/GPL-3.0.txt), both projects are named with their pinned
commits, and every artifact holding these records reports them as GPL-3.0 in its declared
distribution. A reader who finds the derivation too thin to trigger copyleft loses nothing by our
carrying the text. A reader who finds the opposite has the licence, the attribution, and the exact
records to work from.

Three things are enough to judge this independently: the nine records in `data/pre/crewai.json`, the
two upstream projects at the commits listed in
[`CANONICAL-TEXTS.md`](third_party/licenses/CANONICAL-TEXTS.md), and `derive_spec_features` in
`src/catchbench/pre_static_scanner.py`, which is the function deciding which words survive. If you
reach a different conclusion, open an issue. These are 9 records out of 1,187, each identified by
instance id, so removing them is a small change.

### The CC-BY-4.0 record

One MCP record, `mcp-0146-com-microsoft-microsoft-learn-mcp`, comes from the Microsoft Learn MCP
server, whose repository declares CC-BY-4.0. That licence is permissive and its section 3(a)
obligation is attribution, which this file and
[`CANONICAL-TEXTS.md`](third_party/licenses/CANONICAL-TEXTS.md) supply: the project, the pinned
commit, and the full licence text.

Two details complicate the simple reading, and neither is hidden here. The repository is dual
licensed in the usual Microsoft documentation pattern, CC-BY-4.0 in `LICENSE` for content and MIT in
`LICENSE-CODE` for code; CatchBench records the CC-BY-4.0 declaration because that is what `LICENSE`
says. Separately, this record's `declared_capabilities` were read from the running server's
`tools/list` response rather than from the repository, so the repository licence is recorded as
where the record came from and is not asserted to license that response. The second point applies to
all 144 MCP records and is repeated in the MCP section below.

## InjecAgent

**Status:** complete for the pinned source.

- Derived material: normalized declared tool capabilities, the designated user-tool reference, the
  attacker-tool excess label, attack type, and deidentified task tokens.
- CatchBench files: `data/pre/injecagent.json` and the InjecAgent rows in
  `data/pre/llm_judge_method/llama-3.3-70b.json` and
  `data/pre/llm_judge_method_votes/llama-3.3-70b.json`.
- Upstream project: [uiuc-kang-lab/InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent).
- Upstream revision: `f19c9f2c79a41046eb13c03c51a24c567a8ffa07`.
- Upstream inputs: `data/test_cases_dh_base.json` and `data/test_cases_ds_base.json`.
- Licence: MIT. The exact pinned source is
  [`LICENCE`](https://github.com/uiuc-kang-lab/InjecAgent/blob/f19c9f2c79a41046eb13c03c51a24c567a8ffa07/LICENCE).

Full licence text:

```text
MIT License

Copyright (c) 2023 Qiusi Zhan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Who&When

**Status: UNFINISHED - the dataset terms are still being established.**

- Derived material: 31 caches of LLM localization predictions, including their `raw` model output,
  and one legacy-to-content-address key map. The predictions were generated from 126 failed
  Who&When traces and some raw outputs quote or restate trace content.
- CatchBench files: `data/llm_judge/whoandwhen__*.json` and
  `data/llm_judge/legacy_run_keys.json`.
- Upstream dataset: [Kevin355/Who_and_When](https://huggingface.co/datasets/Kevin355/Who_and_When),
  revision `59b9fcba1aaed7bbf206b5f4d3c68b8face2f49c`.
- Associated code project:
  [mingyin1/Agents_Failure_Attribution](https://github.com/mingyin1/Agents_Failure_Attribution).
- Licence: the code project is MIT. The pinned dataset card does not declare a dataset licence, so
  the code licence is not treated here as a licence grant for the separately hosted traces.

The associated code project's MIT notice is reproduced for attribution. This does not resolve the
dataset terms.

```text
MIT License

Copyright (c) 2025 Ming Yin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Public CrewAI projects

**Status: UNFINISHED - 230 records are `NOASSERTION` and 2 declared-MIT records have no retrievable
licence file, so 232 of the 298 have no established terms. The 9 GPL-3.0 records have established
terms and a stated position; see [Why the GPL-3.0 and CC-BY-4.0 records are in the
release](#why-the-gpl-30-and-cc-by-40-records-are-in-the-release).**

- Derived material: normalized agent tool rosters, capability types, deidentified role tokens, and
  cross-vendor model-judge labels from 105 public repositories.
- CatchBench files: `data/pre/crewai.json`, `data/pre/gpt_judge_votes/crewai.json`,
  `data/pre/claude_judge_votes/crewai.json`, CrewAI rows in both `llm_judge_method` files, and
  `data/pre/LABEL_QUALITY.md`.
- Upstream projects and identifiers: the `crewai.upstream_identifiers` array in
  [`ASSET_MANIFEST.json`](ASSET_MANIFEST.json). Every entry carries its GitHub project URL, exact
  commit, source path, record count, and declared licence.
- Declared distribution: 53 MIT, 6 Apache-2.0, 9 GPL-3.0, and 230 `NOASSERTION` records.

The licence files for 24 of the 25 repositories declared MIT, Apache-2.0, or GPL-3.0 were retrieved
at their pinned revisions. For entries whose manifest field is one of those three, the exact licence
pointer is `{upstream_url}/blob/{revision}/LICENSE`, except:

- `Subrahmanyam2305/claude_computer_use` and `zcaceres/agentstack-receipts-manager` use
  `LICENSE.md` at the recorded revision.
- The declared-MIT `jasonssdev/dl-crewai` revision is no longer retrievable and remains unfinished.

The retrieved Apache-2.0 repositories had no root `NOTICE`, based on checks of `NOTICE`,
`NOTICE.md`, and `NOTICE.txt`.

The pinned links above are references to where each record came from. Redistributed terms travel
locally in [`third_party/licenses/Apache-2.0.txt`](third_party/licenses/Apache-2.0.txt),
[`third_party/licenses/GPL-3.0.txt`](third_party/licenses/GPL-3.0.txt), and the source-specific MIT
copyright and permission notices in
[`third_party/licenses/MIT-crewai.md`](third_party/licenses/MIT-crewai.md). Applicable upstream
`NOTICE` text is reproduced in [`NOTICE`](NOTICE). That appendix carries the notice of 17 of the 18
MIT-declared projects; `jasonssdev/dl-crewai` is marked unresolved there for the same reason it is
marked unresolved above. The 9 GPL-3.0 records carry the licence text and the position stated
[above](#why-the-gpl-30-and-cc-by-40-records-are-in-the-release). No conclusion is made here for the
230 `NOASSERTION` records.

## n8n workflow templates

**Status: UNFINISHED - terms for the selected workflow artifacts are unverified.**

- Derived material: normalized AI-agent node tool rosters, capability types, deidentified workflow
  tokens, and cross-vendor model-judge labels from 219 public workflow templates.
- CatchBench files: `data/pre/n8n.json`, `data/pre/gpt_judge_votes/n8n.json`,
  `data/pre/claude_judge_votes/n8n.json`, n8n rows in both `llm_judge_method` files, and
  `data/pre/LABEL_QUALITY.md`.
- Upstream project: [n8n workflow templates](https://n8n.io/workflows/). Each template ID and source
  path is recorded under `n8n.upstream_identifiers` in
  [`ASSET_MANIFEST.json`](ASSET_MANIFEST.json). A template ID is a mutable identifier, not a pinned
  revision: it names a gallery entry whose workflow can be edited or withdrawn, so it records where
  a record came from and not which bytes were read. The harvester read each template from
  `https://api.n8n.io/api/templates/workflows/<id>`, recorded in the manifest as
  `n8n.artifact_base_url`.
- Licence: unverified for all 219 released records. No licence is invented here.

## Model Context Protocol Registry servers

**Status: UNFINISHED - 29 records are `NOASSERTION`, 12 are `Other`, and 4 records declare a licence
whose pinned file could not be retrieved, so 45 of the 144 have no established terms. The 1
CC-BY-4.0 record has established terms; see [Why the GPL-3.0 and CC-BY-4.0 records are in the
release](#why-the-gpl-30-and-cc-by-40-records-are-in-the-release).**

- Derived material: normalized live `tools/list` capability names and types, deidentified server
  description tokens, the source repository and commit, and cross-vendor model-judge labels from 121
  projects represented by 144 server records.
- CatchBench files: `data/pre/mcp.json`, `data/pre/gpt_judge_votes/mcp.json`,
  `data/pre/claude_judge_votes/mcp.json`, MCP rows in both `llm_judge_method` files, and
  `data/pre/LABEL_QUALITY.md`.
- Upstream project: [Model Context Protocol Registry](https://registry.modelcontextprotocol.io/).
  Every server repository URL and exact commit is recorded under `mcp.upstream_identifiers` in
  [`ASSET_MANIFEST.json`](ASSET_MANIFEST.json).
- Declared distribution: 87 MIT, 15 Apache-2.0, 1 CC-BY-4.0, 29 `NOASSERTION`, and 12 `Other`
  records.

Each declaration is the pinned repository's licence. The capability roster was read from the server's
live endpoint, so a repository licence is not asserted here to license that endpoint's output. The
[SWE-agent section](#swe-agent-trajectories) states the same limit for the same reason.

The licence files for 77 of the 81 repositories declared MIT, Apache-2.0, or CC-BY-4.0 were retrieved
at their pinned revisions. Their exact licence pointer is
`{upstream_url}/blob/{revision}/LICENSE`. These four recorded declarations could not be checked
against a retrievable pinned licence file and remain unfinished:

- `Darko893/mcp-server` (MIT)
- `Loop-XXI/loop-mcp` (MIT)
- `mustafasalimerek-bit/launchtrust-mcp` (MIT)
- `TunnelMind/sigil-mcp` (Apache-2.0)

Two Apache-2.0 projects and one MIT project carry a root `NOTICE`. Their exact pinned pointers are:

- [Helixar-AI/helixar-mcp NOTICE](https://github.com/Helixar-AI/helixar-mcp/blob/c0b73c99d2fe8eaab7f1db5a450bee7f4934e801/NOTICE)
- [HemmaBo-se/hemmabo-mcp-server NOTICE](https://github.com/HemmaBo-se/hemmabo-mcp-server/blob/fe57b7c04b39a9db774bc248d85c4aada1acaf09/NOTICE)
- [Evlek/evlek-mcp NOTICE](https://github.com/Evlek/evlek-mcp/blob/3639e1ec983afb88e0f258511a99f5bcb137f388/NOTICE)

The pinned links above are references to where each record came from. Redistributed terms travel
locally in [`third_party/licenses/Apache-2.0.txt`](third_party/licenses/Apache-2.0.txt),
[`third_party/licenses/CC-BY-4.0.txt`](third_party/licenses/CC-BY-4.0.txt), and the source-specific
MIT copyright and permission notices in
[`third_party/licenses/MIT-mcp.md`](third_party/licenses/MIT-mcp.md). Applicable upstream `NOTICE`
text is reproduced in [`NOTICE`](NOTICE). That appendix carries the notice of 65 of the 68
MIT-declared projects; the three unretrievable MIT projects listed above are marked unresolved
there as well. The 1 CC-BY-4.0 record carries the licence text and the attribution stated
[above](#the-cc-by-40-record). The `NOASSERTION` and `Other` subsets remain unfinished.

### The BUSL-1.1 record

One MCP Registry record declares the Business Source License 1.1:
[`MadaBurns/bv-mcp`](https://github.com/MadaBurns/bv-mcp/tree/5d4c08fefc349fa6ee56bf29a70a5c161ce45a36).
Its text is reproduced here rather than in the shared canonical set, because BUSL-1.1 is published
as a template whose Parameters block the licensor fills in. Two projects under it do not grant the
same thing, so one shared copy would state somebody else's terms.

The parameters at the pinned commit:

- Licensor: BLACKVEIL Security
- Licensed Work: Blackveil DNS, (c) 2025-2026 BLACKVEIL Security
- Additional Use Grant: non-commercial use. The licence states that providing the work as a hosted
  service for a fee, or embedding it in a commercial product, is commercial use.
- Change Date: 2030-03-17
- Change License: MIT License

The licence carries its own notice on what it is: "The Business Source License (this document, or
the 'License') is not an Open Source license. However, the Licensed Work will eventually be made
available under an Open Source License, as stated in this License." CatchBench is a research
benchmark distributed without charge, which is within the non-commercial grant above.

## SWE-agent trajectories

**Status: UNFINISHED - the trajectory artifact terms are unverified.**

- Derived material: normalized declared shell/editor command rosters, commands observed as used,
  declared-minus-used excess labels, and deidentified issue tokens from 130 trajectories.
- CatchBench files: `data/pre/sweagent.json` and SWE-agent rows in both `llm_judge_method` files.
- Upstream project: [SWE-bench experiments](https://github.com/SWE-bench/experiments), submission
  `20240728_sweagent_gpt4o`, with individual trajectory paths recorded under
  `sweagent.upstream_identifiers` in [`ASSET_MANIFEST.json`](ASSET_MANIFEST.json).
- Source identifier: the submission label is a mutable identifier, not a pinned revision. It names a
  submission rather than an object version, and the trajectories were downloaded from
  `https://swe-bench-submissions.s3.amazonaws.com/lite/20240728_sweagent_gpt4o/trajs/`, recorded in
  the manifest as `sweagent.artifact_base_url`. That object store is served without a version ID or
  checksum here, so these records carry no proof of which bytes were read.
- Licence: unverified for all 130 released records. A code repository licence is not asserted to
  license the generated trajectory artifacts.

## First-party synthetic records

`data/pre/synthetic.json` contains 56 CatchBench-authored synthetic PRE records. They are not a
third-party source and are covered by CatchBench's [`LICENSE`](LICENSE). Their downstream parsed and
raw model-judge artifacts are listed in [`ASSET_MANIFEST.json`](ASSET_MANIFEST.json).
