"""AuditableBench: the benchmark for agent auditing across the PRE / LIVE / POST lifecycle.

The public arena where methods compete at finding and attributing agent failures over real
agent traces. Organized along the agent lifecycle (PRE deploy gate, LIVE real-time, POST
forensics); each pillar holds scenarios, each scenario is one ``Task`` with its own labels,
metric, and baseline set. See the design in the agent-startup-thesis planning repo
(``research/auditablebench-design.md``).
"""
from auditablebench.core import Method, ResultRow, RunPipeline, Task

__all__ = ["Method", "ResultRow", "RunPipeline", "Task"]
__version__ = "0.0.1"
