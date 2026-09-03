"""A scripted model, so end-to-end tests are deterministic without a real one."""

from __future__ import annotations

from typing import Any


class FakeLLM:
    """Returns scripted responses keyed by call index, and records what it received.

    The recording is what lets every `pre`-phase LLM fault assert on **the messages
    the model actually got**, rather than on the fault's own return value.
    """

    def __init__(self, script: list[Any] | None = None) -> None:
        """Initialise with a response script.

        Args:
            script: Responses to return in order. The last one repeats once the
                script runs out, so a looping agent does not crash the fixture
                before the probe has seen the loop.
        """
        self.script: list[Any] = list(script or ["ok"])
        self.received: list[Any] = []
        self.calls = 0

    def __call__(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        """Record the call and return the next scripted response.

        Args:
            messages: Whatever the agent sent.
            *args: Ignored.
            **kwargs: Ignored.

        Returns:
            The scripted response for this call index.
        """
        self.received.append(messages)
        index = min(self.calls, len(self.script) - 1)
        self.calls += 1
        return self.script[index]
