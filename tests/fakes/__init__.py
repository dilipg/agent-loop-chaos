"""Fake agents: deliberate, minimal, and buggy in known ways.

`docs/07-TESTING.md` §2. These are the fixtures every end-to-end test runs against.
`good_agent` is the negative control and must pass every scenario in the fake suite.

**The fakes are calibrated against the probe rules, not the reverse.** If a new probe
makes `good_agent` fail, the probe has a false positive and the probe is wrong until
proven otherwise. Do not loosen a rule to make a fixture pass.
"""

from __future__ import annotations

from . import (
    good_agent,
    loopy_agent,
    naive_tool_agent,
    no_retry_agent,
    retry_storm_agent,
    stateless_agent,
    strict_json_agent,
    trusting_agent,
)
from .fake_llm import FakeLLM

__all__ = [
    "FakeLLM",
    "good_agent",
    "loopy_agent",
    "naive_tool_agent",
    "no_retry_agent",
    "retry_storm_agent",
    "stateless_agent",
    "strict_json_agent",
    "trusting_agent",
]
