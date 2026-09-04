"""The two integration snippets from `docs/02-API.md` §11, run verbatim.

"Both snippets must work verbatim by the end of phase 05." A quickstart that does
not run is worse than none: it costs a reader their first half hour and their trust.

Kept in sync by construction -- if either snippet changes, this file must change with
it, and the test names say which one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


def test_the_five_line_langgraph_promise(tmp_path: Path, monkeypatch: Any) -> None:
    """`docs/02-API.md` §11, the LangGraph snippet.

    The promise is five lines: construct, register, instrument, run, print. Anything
    more and the "≤5 lines" success criterion in `docs/00-VISION.md` is not met.
    """
    pytest.importorskip("langgraph", reason="the langgraph extra is not installed")
    pytest.importorskip("langchain_core", reason="the langgraph extra is not installed")
    monkeypatch.chdir(tmp_path)

    from tests.fakes.lg_agent import build

    app = build(lambda prompt: "Warm; pack light layers.").compile()

    # --- the snippet ------------------------------------------------------------
    from agent_loop_chaos import ChaosEngine
    from agent_loop_chaos.adapters.langgraph import instrument_graph
    from agent_loop_chaos.faults import ToolCorruptionFault

    engine = ChaosEngine(seed=1337)
    engine.register_fault(
        ToolCorruptionFault(mutation_type="drop_key", keys=["temp_c"]),
        target_tool="get_weather_data",
    )
    result = engine.run(instrument_graph(app, engine), inputs={"query": "Pack list for Paris"})
    print(result.to_json())
    # --- end of the snippet -----------------------------------------------------

    assert result.run_id
    assert result.validate() == []


def test_the_plain_python_blueprint_shape(tmp_path: Path, monkeypatch: Any) -> None:
    """`docs/02-API.md` §11, the vanilla snippet.

    This is the original blueprint's call shape, kept working on purpose: the API
    doc froze it, and `run_with_state` exists precisely so it keeps parsing.
    """
    monkeypatch.chdir(tmp_path)

    # --- the snippet ------------------------------------------------------------
    from agent_loop_chaos import ChaosEngine
    from agent_loop_chaos.faults import ToolCorruptionFault

    chaos = ChaosEngine()
    chaos.register_fault(
        ToolCorruptionFault(mutation_type="empty_json"), target_tool="get_weather_data"
    )

    @chaos.tool
    def get_weather_data(location: str) -> list[dict[str, Any]]:
        return [{"temp_c": 21, "city": location}]

    def llm_generate(user_query: str, data: Any) -> str:
        return f"{user_query}: {data}"

    @chaos.intercept_tools()
    def weather_agent(user_query: str, state: dict[str, Any]) -> str:
        data = get_weather_data(state["location"])
        return llm_generate(user_query, data)

    result = chaos.run_with_state(
        weather_agent, query="Pack list for Paris", initial_state={"location": "Paris"}
    )
    print(result.to_json())
    # --- end of the snippet -----------------------------------------------------

    assert result.run_id
    assert result.validate() == []
    assert result.injected_faults[0]["fired"] is True


def test_the_snippets_are_still_the_ones_in_the_doc() -> None:
    """The doc and these tests must not drift apart.

    A snippet that works here and not in the README is the same failure as one that
    works nowhere -- the reader only ever sees the README.
    """
    doc = (Path(__file__).resolve().parents[1] / "docs" / "02-API.md").read_text(encoding="utf-8")
    section = doc.split("## 11. The 5-line integration promise")[1]
    for line in (
        "engine = ChaosEngine(seed=1337)",
        "instrument_graph(app, engine)",
        "chaos = ChaosEngine()",
        "@chaos.intercept_tools()",
        "chaos.run_with_state(",
    ):
        assert line in section, f"the doc no longer contains {line!r}"
