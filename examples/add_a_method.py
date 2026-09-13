r"""Add a method to the PRE over-privilege board and score it, offline, in about a second.

This is the whole extension contract, running end to end. A method declares which ``task_id``s it
handles and implements ``evaluate``; the task hands it the same records every other entrant sees,
and the same ``pre_score`` the board calls turns its answer into precision, recall and F1. Nothing
here is special-cased for the example.

The method below is deliberately simple, so that the plumbing rather than the model is what shows:
it flags a declared capability when the capability name looks like it can write, execute, or reach
the network. That is a real rule of the kind the board already scores, and it lands between the two
floors rather than beating them, which is the honest outcome for four lines of matching.

No model key, no corpus download, no GRADE checkout, and no torch. The records ship with the
package, so this runs the same way from a clone or from ``pip install catchbench``.

CatchBench must be importable first, so install it before running this from a clone:

    python -m pip install -e .        # from a checkout, or: python -m pip install catchbench

Run it:

    python examples/add_a_method.py

Then read the comparison it prints against ``flag_all``, the floor that flags every capability and
therefore has recall 1.000 by construction. A method earns its false alarms only by clearing that
floor.
"""
from catchbench.pre import FlagAllMethod, FlagNoneMethod, PreOverPrivilege, pre_score

# Substrings that suggest a capability can change the world rather than only read it. A real entrant
# would learn or curate this; the point here is that the method sees the declaration and nothing
# else, which is what makes the PRE state PRE.
RISKY = ("write", "delete", "remove", "exec", "run", "shell", "command",
         "http", "request", "fetch", "browse", "network", "upload", "send")


class KeywordOverPrivilege:
    """Flag a declared capability when its name contains a write, execute, or network verb."""

    method_id = "keyword-rule (example)"
    supports = {"pre_over_privilege"}

    def evaluate(self, task):
        task.setup()
        flagged = {}
        for config in task.method_view():
            hits = {
                capability["name"]
                for capability in config["declared_capabilities"]
                if any(word in capability["name"].lower() for word in RISKY)
            }
            flagged[config["instance_id"]] = hits
        # Same scorer the board uses. The optional `evaluable` argument names the configurations
        # a method actually judged; leaving it out, as here, scores every configuration, so a
        # missing entry counts as "flagged nothing" rather than as an abstention. A method that
        # genuinely cannot answer some configuration should pass the ones it did answer, which is
        # how the held-out judge's five abstentions stay out of its denominator.
        return pre_score(flagged, task.instances)


def main() -> int:
    task = PreOverPrivilege()
    task.setup()
    print(task.corpus_line())
    print()

    entrants = [KeywordOverPrivilege(), FlagAllMethod(), FlagNoneMethod()]
    width = max(len(m.method_id) for m in entrants)
    print(f"  {'method':{width}s}  {'precision':>9s}  {'recall':>6s}  {'F1':>6s}")
    scores = {}
    for method in entrants:
        s = method.evaluate(task)
        scores[method.method_id] = s
        print(f"  {method.method_id:{width}s}  {s['precision']:9.3f}  {s['recall']:6.3f}  "
              f"{s['f1']:6.3f}")

    mine = scores["keyword-rule (example)"]["f1"]
    floor = scores["flag_all"]["f1"]
    print()
    print(f"  keyword rule F1 {mine:.3f} against the flag_all floor {floor:.3f}: "
          f"{'above' if mine > floor else 'below'} the floor.")
    print("  Flagging everything has recall 1.000 by construction, so the floor is what a method")
    print("  has to clear before its false alarms are worth anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
