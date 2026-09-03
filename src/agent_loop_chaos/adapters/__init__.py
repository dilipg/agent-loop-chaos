"""Framework adapters.

An adapter's only job is to turn its framework's hooks into `Crossing` objects, so
the core never learns which framework is in play. Each implements `instrument()`,
`run()` and `state_of()`.

`vanilla.py` arrives in M1 and `langgraph.py` in M5. A framework is always an
optional extra, imported locally, raising `MissingExtraError` when absent — so
importing `agent_loop_chaos` never pulls in `langgraph`.
"""

from __future__ import annotations

__all__: list[str] = []
