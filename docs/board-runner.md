# Running the board check on a self-hosted runner

`verify.yml` finishes in about ninety seconds because it never runs the scoring board. `board.yml`
runs the whole board and compares it against `tests/golden/board.txt`, which takes roughly nine
minutes and fetches about 320 MB of corpora. That is too slow for a hosted runner on every push and
cheap on a lab machine, where the Hugging Face cache persists between jobs.

## Read this before registering a runner on a public repository

A self-hosted runner executes whatever a workflow tells it to, on hardware you own and inside your
network. On a public repository, anyone can open a pull request that changes both the workflow and
the code it runs. GitHub's own guidance is not to pair self-hosted runners with public repositories.

`board.yml` is written around that risk rather than ignoring it:

- it has no `pull_request` trigger, and a job-level `if` refuses to run for one even if a trigger is
  added later;
- it runs on pushes to `main`, which only a maintainer can make, plus a weekly schedule and manual
  dispatch;
- it lives in its own file, so a repository with no runner registered sees no queued job.

Before this repository goes public, confirm in **Settings, Actions, General** that
*Fork pull request workflows* is set so that workflows from outside collaborators require approval.
If you would rather not accept the residual risk at all, delete `board.yml` and run
`python tools/check_board.py` by hand before each release; the check itself does not need CI.

## Machine prerequisites

Python 3.10 or newer, git, and a working `pip`. A CUDA driver is optional: the PyGOD rows run on CPU
without one, more slowly. On an ARM64 machine such as a DGX Spark, the compiled PyTorch Geometric
backend is available (`pyg_lib` publishes `manylinux_2_27_aarch64` wheels for the CUDA builds), so
the graph-AD path works there.

## Register the runner

From the repository's **Settings, Actions, Runners, New self-hosted runner**, follow the commands
GitHub shows for Linux ARM64. When it asks for labels, accept the defaults so the runner carries
`self-hosted`, `linux`, and `ARM64`, which is what `board.yml` targets. Then install it as a service
so it survives a reboot:

```bash
./config.sh --url https://github.com/<owner>/<repo> --token <token> --labels self-hosted,linux,ARM64
sudo ./svc.sh install
sudo ./svc.sh start
```

## The golden is platform-specific, and that is not yet settled

`tests/golden/board.txt` is currently generated on the maintainer's Windows workstation with a CUDA
build of PyTorch. Whether a Linux ARM64 runner reproduces it exactly at three decimals is an open
question: the PyGOD rows go through torch and a compiled sampler, and those can differ across BLAS
implementations and builds.

The first runner job answers it. Two outcomes:

- **The board matches.** Nothing to do; the golden is portable and a local `check_board.py` run is
  as authoritative as CI.
- **Only the PyGOD rows differ.** Regenerate the golden on the runner with
  `python tools/check_board.py --update`, commit it with the diff in the message, and record here
  that the runner is the authority. A local run will then show that same difference, which is
  expected rather than drift.

Do not respond by loosening `check_board.py` into a tolerance comparison. Exact comparison is what
makes the check worth having; a tolerance hides the small real movements it exists to catch.

## Regenerating the golden

Whenever a board number changes on purpose:

```bash
python tools/check_board.py --update
```

Put the resulting diff in the commit message. A golden update with no explanation is indistinguishable
from a result quietly moving.
