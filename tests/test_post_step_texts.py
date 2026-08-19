"""Contract tests for ``PostLocalization.step_texts``.

The attribute exists so an outside text-reading localizer can enter the localization board, which
the LLM judges currently lead. Its whole risk is misalignment: if the text loader and the scoring
loader ever select or order records differently, every step's text lands on the wrong step and a
method is scored against words it never saw. The guard that prevents that is the thing worth
testing, so these cases drive it rather than the happy path alone.

They run offline. ``setup`` and ``load_judge_runs`` are replaced with small fixtures, because the
real pair downloads the Who&When corpus and this suite is the ninety-second one.
"""
import pytest

from auditablebench.post import PostLocalization
import auditablebench.post as post_module


def _steps(*pairs):
    return [{"idx": idx, "agent": agent, "kind": "message"} for idx, agent in pairs]


def _task(monkeypatch, scored, judged):
    """Build a task whose scoring runs are ``scored`` and whose text records are ``judged``."""
    task = PostLocalization()

    def fake_setup(self):
        self.runs = scored
        self._loaded = True

    monkeypatch.setattr(PostLocalization, "setup", fake_setup)
    monkeypatch.setattr(post_module, "load_judge_runs", lambda: judged)
    return task


def test_aligned_records_yield_text_per_step(monkeypatch):
    scored = [{"steps": _steps((0, "a"), (1, "b")), "mistake": 0},
              {"steps": _steps((0, "c")), "mistake": 0}]
    judged = [{"steps": _steps((0, "a"), (1, "b")), "texts": ["first", "second"], "mistake": 0},
              {"steps": _steps((0, "c")), "texts": ["third"], "mistake": 0}]
    task = _task(monkeypatch, scored, judged)
    assert task.step_texts == (("first", "second"), ("third",))


def test_shape_matches_runs_exactly(monkeypatch):
    """One text per step, in run order: this is what lets a method concatenate scores and score."""
    scored = [{"steps": _steps((0, "a"), (1, "b"), (2, "c")), "mistake": 1}]
    judged = [{"steps": _steps((0, "a"), (1, "b"), (2, "c")), "texts": ["x", "y", "z"], "mistake": 1}]
    task = _task(monkeypatch, scored, judged)
    assert [len(run) for run in task.step_texts] == [len(r["steps"]) for r in scored]


def test_returns_text_only_and_no_annotation(monkeypatch):
    """A label reaching the text view would hand a text-reading method the answer."""
    scored = [{"steps": _steps((0, "a")), "mistake": 0}]
    judged = [{"steps": _steps((0, "a")), "texts": ["body"], "mistake": 0, "mistake_step": 0,
               "mistake_agent": "a", "mistake_reason": "wrong file", "ground_truth": "x"}]
    task = _task(monkeypatch, scored, judged)
    texts = task.step_texts
    assert texts == (("body",),)
    assert all(isinstance(cell, str) for run in texts for cell in run)


@pytest.mark.parametrize("judged, why", [
    ([], "run count differs"),
    ([{"steps": _steps((0, "a")), "texts": ["only one step"], "mistake": 0}], "step count differs"),
    ([{"steps": _steps((0, "a"), (9, "b")), "texts": ["p", "q"], "mistake": 0}], "step index differs"),
    ([{"steps": _steps((0, "a"), (1, "z")), "texts": ["p", "q"], "mistake": 0}], "step agent differs"),
])
def test_misalignment_refuses_rather_than_mislabels(monkeypatch, judged, why):
    """Each of these would silently attribute text to the wrong step, so each must raise."""
    scored = [{"steps": _steps((0, "a"), (1, "b")), "mistake": 0}]
    task = _task(monkeypatch, scored, judged)
    with pytest.raises(RuntimeError, match="do not line up"):
        _ = task.step_texts


def test_built_once_and_cached(monkeypatch):
    """It is a cached_property so a structural method never pays for the text load."""
    scored = [{"steps": _steps((0, "a")), "mistake": 0}]
    judged = [{"steps": _steps((0, "a")), "texts": ["body"], "mistake": 0}]
    calls = []

    def counting_loader():
        calls.append(1)
        return judged

    task = _task(monkeypatch, scored, judged)
    monkeypatch.setattr(post_module, "load_judge_runs", counting_loader)
    first, second = task.step_texts, task.step_texts
    assert first is second
    assert len(calls) == 1


def test_same_shaped_runs_cannot_be_swapped(monkeypatch):
    """Two runs with identical step signatures must not be interchangeable.

    The signature check compares run count, step count, and each (idx, agent). Two runs built the
    same way satisfy all three, so swapping them passes and hands each run the other's words. The
    decisive-step label is the only field both loaders carry, so it is what separates them.
    """
    sig = _steps((0, "a"), (1, "b"))
    scored = [{"steps": [dict(s) for s in sig], "mistake": 0},
              {"steps": [dict(s) for s in sig], "mistake": 1}]
    swapped = [{"steps": [dict(s) for s in sig], "texts": ["B0", "B1"], "mistake": 1},
               {"steps": [dict(s) for s in sig], "texts": ["A0", "A1"], "mistake": 0}]
    task = _task(monkeypatch, scored, swapped)
    with pytest.raises(RuntimeError, match="do not line up"):
        _ = task.step_texts


def test_same_shaped_runs_in_the_right_order_are_accepted(monkeypatch):
    """The identity check must not reject the correct pairing it is meant to protect."""
    sig = _steps((0, "a"), (1, "b"))
    scored = [{"steps": [dict(s) for s in sig], "mistake": 0},
              {"steps": [dict(s) for s in sig], "mistake": 1}]
    right = [{"steps": [dict(s) for s in sig], "texts": ["A0", "A1"], "mistake": 0},
             {"steps": [dict(s) for s in sig], "texts": ["B0", "B1"], "mistake": 1}]
    task = _task(monkeypatch, scored, right)
    assert task.step_texts == (("A0", "A1"), ("B0", "B1"))


@pytest.mark.parametrize("texts", [["only one"], ["a", "b", "c"], []])
def test_text_count_must_equal_step_count(monkeypatch, texts):
    """A short or long text list shifts every later text onto the wrong step."""
    scored = [{"steps": _steps((0, "a"), (1, "b")), "mistake": 0}]
    judged = [{"steps": _steps((0, "a"), (1, "b")), "texts": texts, "mistake": 0}]
    task = _task(monkeypatch, scored, judged)
    with pytest.raises(RuntimeError, match="do not line up"):
        _ = task.step_texts


def test_every_run_is_checked_not_only_the_first(monkeypatch):
    """A guard that returns after run 0 passes every single-run case, so drive the defect later.

    This is the mutation the earlier cases could not see: they put the misalignment in the first
    run, or used one run, so short-circuiting after the first still satisfied them.
    """
    sig = _steps((0, "a"))
    scored = [{"steps": [dict(s) for s in sig], "mistake": 0},
              {"steps": [dict(s) for s in sig], "mistake": 0},
              {"steps": _steps((0, "a"), (1, "b")), "mistake": 1}]
    judged = [{"steps": [dict(s) for s in sig], "texts": ["ok0"], "mistake": 0},
              {"steps": [dict(s) for s in sig], "texts": ["ok1"], "mistake": 0},
              # Third run disagrees: the agent at step 1 is wrong.
              {"steps": _steps((0, "a"), (1, "WRONG")), "texts": ["p", "q"], "mistake": 1}]
    task = _task(monkeypatch, scored, judged)
    with pytest.raises(RuntimeError, match="do not line up"):
        _ = task.step_texts


@pytest.mark.parametrize("extra_judged", [
    [{"steps": _steps((0, "a")), "texts": ["surplus"], "mistake": 0}],
    [],
])
def test_run_counts_must_match_exactly_in_both_directions(monkeypatch, extra_judged):
    """Not "at least as many". A surplus text record means the two loaders disagree on selection."""
    scored = [{"steps": _steps((0, "a")), "mistake": 0}]
    judged = [{"steps": _steps((0, "a")), "texts": ["body"], "mistake": 0}] + extra_judged
    task = _task(monkeypatch, scored, judged)
    if extra_judged:
        with pytest.raises(RuntimeError, match="do not line up"):
            _ = task.step_texts
    else:
        assert task.step_texts == (("body",),)


def test_no_label_value_appears_among_the_returned_texts(monkeypatch):
    """The returned strings must be step content, never the decisive-step answer.

    A method reading `step_texts` is meant to work from what the agent said. If the label leaked in
    as a string, a text-reading entrant could score by reading it rather than by localizing.
    """
    scored = [{"steps": _steps((0, "a"), (1, "b")), "mistake": 1}]
    judged = [{"steps": _steps((0, "a"), (1, "b")), "texts": ["first", "second"],
               "mistake": 1, "mistake_agent": "b", "mistake_reason": "used a stale read"}]
    task = _task(monkeypatch, scored, judged)
    texts = task.step_texts
    assert texts == (("first", "second"),)
    flat = [cell for run in texts for cell in run]
    assert len(flat) == 2, "one string per step and nothing appended"
    for forbidden in ("1", "b", "used a stale read"):
        assert not any(cell == forbidden for cell in flat)
