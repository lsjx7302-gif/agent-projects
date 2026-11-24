# -*- coding: utf-8 -*-
"""Validate the power graph topology + constraints with a mock LLM (no API)."""
from power_graph import (
    PMDecision, SegmentDecision, SegmentView, build_power_graph, check_constraints,
    initial_state,
)


class _Resp:
    def __init__(self, content):
        self.content = content


class MockLLM:
    def invoke(self, messages):
        return _Resp("mock analyst/debate/risk opinion: peak=up, valley=down, flat=flat")

    def with_structured_output(self, schema):
        class _Wrapped:
            def invoke(self, messages):
                if schema is SegmentView:
                    return SegmentView(
                        peak=SegmentDecision(direction="up", confidence=0.8, reason="mock"),
                        valley=SegmentDecision(direction="down", confidence=0.7, reason="mock"),
                        flat=SegmentDecision(direction="flat", confidence=0.2, reason="mock"),
                    )
                return PMDecision(risk_level="medium", explanation="mock PM")
        return _Wrapped()


import power_graph as pg
pg.make_llm = lambda model, base_url, api_key: MockLLM()

graph = build_power_graph("mock", "http://mock", "mock", max_debate_rounds=1)
state = graph.invoke(initial_state("2026-11-23"))
print("llm_calls:", state.get("llm_calls"))
print("report keys:", sorted(state.get("reports", {})))
print("debate roles:", [d["role"] for d in state.get("debate", [])])
print("risk keys:", sorted(state.get("risk_opinions", {})))
print("risk_level:", state.get("risk_level"))
print("price points:", len(state.get("price_view", [])))
print("constraints:", check_constraints(state["price_view"], state["baseline"]))
assert state["llm_calls"] == 11, state["llm_calls"]
assert sorted(state["reports"]) == ["load", "price", "renewable", "supply"]
assert [d["role"] for d in state["debate"]] == ["bull", "bear"]
assert sorted(state["risk_opinions"]) == ["aggressive", "conservative", "neutral"]
assert len(state["price_view"]) == 96
assert max(check_constraints(state["price_view"], state["baseline"]).values()) <= 1e-6
print("MOCK GRAPH OK")
