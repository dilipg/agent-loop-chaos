"""The demo agent, with all twelve weaknesses closed.

Same graph, same tools, same happy-path answer. It is the demo suite's negative
control: it must pass every scenario, and a probe that fires on it has a false
positive.

`git diff --no-index examples/trip_planner examples/trip_planner_fixed` is meant to
read as documentation -- the difference between the two trees is exactly the
difference between an agent that looks fine and one that is.
"""

from __future__ import annotations
