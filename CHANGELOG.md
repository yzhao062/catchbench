# Changelog

Notable changes to CatchBench. Board numbers are called out explicitly whenever they move, because
a benchmark that changes a score without saying so is not usable as a reference point.

## 0.1.2 (2026-09-13)

No board number changes. Every score reported in the paper and printed by `catchbench --task pre`
is identical to 0.1.1.

### Added

- **Judge addendum records.** Twelve prediction caches and twelve sidecars under
  `data/llm_judge_addendum/`, covering later releases scored on the same 126 Who&When runs and the
  same all-at-once protocol as the published board. These sit outside the frozen arena, add no
  entrant to it, and declare no contrast. The 0.1.1 wheel carried none of them.
- **`examples/add_a_method.py`.** A runnable end-to-end method, scored on the offline PRE board in
  under a second with no key, no corpus download, no GRADE checkout and no torch. It uses the same
  `pre_score` the board calls, and lands below the `flag_all` floor, which is the honest result for
  a keyword rule and the point the example is making.
- **A named-value Gold board**, `gold_v2_namedvalue` on `tau-bench-gold-v2`, with its five-seed
  fixed-margin diagnostic.
- **Project URLs** in the package metadata, so PyPI links to the repository, the paper, the issue
  tracker and this file.

### Fixed

- **A missing GRADE checkout is reported instead of raised.** Running plain `catchbench` after
  installing the 0.1.1 wheel ended in an unhandled `ImportError` traceback and exit 1. It now
  prints one paragraph naming the setup that works, links the README's Full Board section, and
  exits 2. `--task gold-v2` is new in this release rather than repaired: 0.1.1's parser accepted
  only `all` and `pre`, so that argument was an argparse invalid-choice error there. It reports a
  missing checkout the same way, at exit 2.
- **The GRADE checkout is found in the working directory.** Resolution looked only at `GRADE_DIR`
  and the parent of the installed package. The old message asked the reader to clone GRADE to
  `../grade`, beside a repository, which is a real location from a checkout and no location at all
  from a wheel install. Discovery in the working directory is new, and the message now asks for a
  checkout the resolver will actually find.
- **The wheel carries the licence documents its own `NOTICE` cites.** `THIRD_PARTY_LICENSES.md`,
  `ASSET_MANIFEST.json` and `third_party/licenses/` were absent from the 0.1.0 and 0.1.1 wheels
  while the bundled `NOTICE` referred readers to them for the terms on the bundled records.
- **The PRE footer points somewhere useful**, at the new example and at Full Board setup, rather
  than at a command that fails without GRADE.

### Changed

- Two release-inventory tests were added. The existing distribution tests asserted that the PRE
  records are present and the corpora absent, and both passed on a wheel missing every licence
  document and every addendum cache.

## 0.1.1 (2026-08-22)

First wheel to include the PRE records and the `catchbench` console command. The 0.1.0 wheel omitted
both, so `catchbench --task pre` works from a clean install starting here.

## 0.1.0 (2026-08-22)

Initial release.
