# Refinement loop

- rounds: 3
- final round: 1 passed, 2 failed
- fixed: 1
- regressions: 1
- judge disagreement: 0%
- wall time: 300 ms

| scenario | r1 | r2 | r3 | outcome |
|---|---|---|---|---|
| tool.drop_required_key | FAIL | PASS | PASS | fixed in r2 |
| loop.pinned_tool_output | FAIL | FAIL | FAIL | still failing (max_iterations_exhausted) |
| state.drop_location | PASS | FAIL | FAIL | **REGRESSION in r2** |

## Regressions

- state.drop_location: regressed in round 2 (state_corruption_propagated); it passed in round 1

## Work orders written

- `/out/loop.pinned_tool_output/run-loop.pin/AGENT_TASK.md`

Hand a work order to a coding agent as-is. Do not edit the scenario, the `must_not` list, or the probes to make a run pass: the loop hashes all three between rounds and reports a flip that follows a change as a regression.
