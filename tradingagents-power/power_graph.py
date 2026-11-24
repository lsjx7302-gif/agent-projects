# -*- coding: utf-8 -*-
"""Power-scenario secondary development of TradingAgents' role structure.

analysts (load / renewable / supply / price) -> bull vs bear debate
-> research manager -> trader (96-point view) -> risk committee
-> portfolio manager -> hard constraints.
"""
from __future__ import annotations

import json
import operator
import os
from typing import Annotated, Dict, List, Literal, Optional, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from power_data import baseline_prices, features_text, get_series, segment_avg

PEAK = list(range(68, 84))
VALLEY = list(range(44, 60))
FLAT = [i for i in range(96) if i not in PEAK and i not in VALLEY]
SEG_INDEX = {"peak": PEAK, "valley": VALLEY, "flat": FLAT}


class SegmentDecision(BaseModel):
    direction: Literal["up", "down", "flat"] = Field(description="相对昨日的方向")
    confidence: float = Field(ge=0.0, le=1.0, description="置信度 0-1")
    reason: str = Field(description="一句话理由")


class SegmentView(BaseModel):
    peak: SegmentDecision
    valley: SegmentDecision
    flat: SegmentDecision


class PMDecision(BaseModel):
    risk_level: Literal["low", "medium", "high"]
    explanation: str


def merge_dict(a: Optional[Dict], b: Optional[Dict]) -> Dict:
    return {**(a or {}), **(b or {})}


def add_int(a: Optional[int], b: Optional[int]) -> int:
    return (a or 0) + (b or 0)


class PowerState(TypedDict, total=False):
    day: str
    features: str
    reports: Annotated[Dict[str, str], merge_dict]
    debate: Annotated[List[Dict[str, str]], operator.add]
    round: int
    segment_view: Dict[str, Dict]
    baseline: List[float]
    price_view: List[float]
    risk_opinions: Annotated[Dict[str, str], merge_dict]
    risk_level: str
    final_decision: str
    llm_calls: Annotated[int, add_int]


ANALYST_PROMPTS = {
    "load": (
        "你是电力现货市场的负荷分析师。只根据给定的负荷预测和昨日电价，判断明天峰段/谷段/平段的负荷压力方向。"
        "不要给具体价格，不要编造数据。输出 3-4 句，必须明确写出：峰段=up/down/flat，谷段=up/down/flat，平段=up/down/flat。"
    ),
    "renewable": (
        "你是新能源分析师。根据风电/光伏预测和昨日电价，判断新能源出力对明天峰段/谷段/平段价格的方向影响（新能源越多，价格越承压）。"
        "不要给具体价格。输出 3-4 句，必须明确写出：峰段=up/down/flat，谷段=up/down/flat，平段=up/down/flat。"
    ),
    "supply": (
        "你是供给与检修分析师。根据检修容量、负荷和新能源，判断供给紧张程度对明天峰段/谷段/平段价格的方向影响。"
        "不要给具体价格。输出 3-4 句，必须明确写出：峰段=up/down/flat，谷段=up/down/flat，平段=up/down/flat。"
    ),
    "price": (
        "你是价格与技术面分析师。根据昨日 96 点价格的高低价、分段均价和峰谷形态，判断明天峰段/谷段/平段的方向。"
        "不要给具体价格。输出 3-4 句，必须明确写出：峰段=up/down/flat，谷段=up/down/flat，平段=up/down/flat。"
    ),
}

BULL_PROMPT = (
    "你是多头研究员。基于四位分析师的报告，论证明天哪些时段价格会强于昨日、哪些会弱于昨日；"
    "重点找支撑「峰段更强、谷段抬升」的证据。不要给具体数字，3-5 句。"
)
BEAR_PROMPT = (
    "你是空头研究员。基于四位分析师的报告，反驳多头的乐观判断，论证明天哪些时段价格会弱于昨日；"
    "重点找「新能源大发压价、检修减少、负荷回落」的证据。不要给具体数字，3-5 句。"
)
RM_PROMPT = (
    "你是研究经理。综合四位分析师报告和多空辩论，给出明天峰段/谷段/平段相对昨日的方向判断。"
    "只输出 JSON，格式：{\"peak\":{\"direction\":\"up|down|flat\",\"confidence\":0.0,\"reason\":\"...\"},"
    "\"valley\":{...},\"flat\":{...}}。不要输出 markdown。"
)
RISK_PROMPTS = {
    "aggressive": "你是激进风控。关注峰谷价差机会，指出当前方案是否过于保守、哪些时段可以更进取。3 句以内。",
    "conservative": "你是保守风控。关注偏差考核和硬约束，指出当前方案的风险点，建议降低偏离。3 句以内。",
    "neutral": "你是中性风控。在收益与风险之间平衡，给出风险等级建议（low/medium/high）。3 句以内。",
}
PM_PROMPT = (
    "你是投资组合经理。根据三位风控意见，决定最终风险等级（low/medium/high）并解释。"
    "只输出 JSON：{\"risk_level\":\"low|medium|high\",\"explanation\":\"...\"}。不要输出 markdown。"
)


def make_llm(model: str, base_url: str, api_key: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=0.2,
        timeout=120,
        max_retries=2,
    )


def _invoke(llm, messages) -> str:
    return (llm.invoke(messages).content or "").strip()


def _parse_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON object in: {text[:200]}")
    return json.loads(text[start:end + 1])


def _structured(llm, messages, schema, fallback_prompt: str):
    try:
        return llm.with_structured_output(schema).invoke(messages)
    except Exception:
        text = _invoke(llm, messages + [("human", fallback_prompt)])
        return schema(**_parse_json(text))


# ---------------------------------------------------------------- nodes
def analyst_node(llm, key: str):
    def node(state: PowerState):
        text = _invoke(llm, [("system", ANALYST_PROMPTS[key]), ("human", state["features"])])
        return {"reports": {key: text}, "llm_calls": 1}
    return node


def bull_node(llm):
    def node(state: PowerState):
        reports = "\n\n".join(f"[{k}]\n{v}" for k, v in state["reports"].items())
        text = _invoke(llm, [("system", BULL_PROMPT), ("human", reports)])
        return {"debate": [{"role": "bull", "content": text}],
                "round": state.get("round", 0) + 1, "llm_calls": 1}
    return node


def bear_node(llm):
    def node(state: PowerState):
        reports = "\n\n".join(f"[{k}]\n{v}" for k, v in state["reports"].items())
        debate = "\n".join(f"[{d['role']}] {d['content']}" for d in state.get("debate", []))
        text = _invoke(llm, [("system", BEAR_PROMPT), ("human", reports + "\n\n多头观点：\n" + debate)])
        return {"debate": [{"role": "bear", "content": text}], "llm_calls": 1}
    return node


def should_continue_debate(state: PowerState, max_rounds: int) -> str:
    if state.get("round", 0) < max_rounds:
        return "Bull Researcher"
    return "Research Manager"


def research_manager_node(llm):
    def node(state: PowerState):
        reports = "\n\n".join(f"[{k}]\n{v}" for k, v in state["reports"].items())
        debate = "\n".join(f"[{d['role']}] {d['content']}" for d in state.get("debate", []))
        messages = [("system", RM_PROMPT), ("human", reports + "\n\n辩论：\n" + debate)]
        view = _structured(llm, messages, SegmentView, "只输出符合格式的 JSON。")
        data = view.model_dump() if hasattr(view, "model_dump") else view
        return {"segment_view": data, "llm_calls": 1}
    return node


def adjust_factor(direction: str, confidence: float) -> float:
    base = {"up": 0.08, "down": -0.08, "flat": 0.0}[direction]
    return 1.0 + base * max(0.0, min(1.0, confidence))


def build_price_view(baseline: List[float], view: Dict[str, Dict]) -> List[float]:
    prices = list(baseline)
    for seg, idx in SEG_INDEX.items():
        decision = view.get(seg, {"direction": "flat", "confidence": 0.0})
        factor = adjust_factor(decision.get("direction", "flat"), float(decision.get("confidence", 0.0)))
        for i in idx:
            prices[i] = baseline[i] * factor
    return prices


def risk_node(llm, key: str):
    def node(state: PowerState):
        summary = (
            f"交易日：{state['day']}\n"
            f"分段方向：{json.dumps(state.get('segment_view', {}), ensure_ascii=False)}\n"
            f"96 点方案均价：{sum(state['price_view']) / 96:.2f}；"
            f"昨日均价：{sum(state['baseline']) / 96:.2f}"
        )
        text = _invoke(llm, [("system", RISK_PROMPTS[key]), ("human", summary)])
        return {"risk_opinions": {key: text}, "llm_calls": 1}
    return node


def pm_node(llm):
    def node(state: PowerState):
        summary = "\n".join(f"[{k}] {v}" for k, v in state["risk_opinions"].items())
        messages = [("system", PM_PROMPT), ("human", summary)]
        decision = _structured(llm, messages, PMDecision, "只输出符合格式的 JSON。")
        data = decision.model_dump() if hasattr(decision, "model_dump") else decision
        risk_level = data.get("risk_level", "medium")
        factor = {"high": 1.0, "medium": 0.8, "low": 0.6}[risk_level]
        base = state["baseline"]
        scaled = [base[i] + (state["price_view"][i] - base[i]) * factor for i in range(96)]
        final = apply_constraints(scaled, base)
        return {
            "risk_level": risk_level,
            "final_decision": data.get("explanation", ""),
            "price_view": final,
            "llm_calls": 1,
        }
    return node


# ---------------------------------------------------------------- constraints
def apply_constraints(prices: List[float], baseline: List[float]) -> List[float]:
    """逐点 ±30% / 分段 ±20% / 全天 ±15% / 价格 [40,650] / 跳变 ≤20."""
    def clamp_point(p, b):
        return max(40.0, min(650.0, max(b * 0.7, min(b * 1.3, p))))

    def scale_segment(prices, seg, lo, hi):
        idx = SEG_INDEX[seg]
        cur = sum(prices[i] for i in idx) / len(idx)
        if cur <= 0:
            return prices
        target = max(lo, min(hi, cur))
        k = target / cur
        return [p * k if i in idx else p for i, p in enumerate(prices)]

    out = [clamp_point(p, b) for p, b in zip(prices, baseline)]
    # Projection loop: segment/daily scaling, then point bounds + adjacent jumps.
    for _ in range(6):
        for seg in ("peak", "valley", "flat"):
            b_avg = sum(baseline[i] for i in SEG_INDEX[seg]) / len(SEG_INDEX[seg])
            out = scale_segment(out, seg, b_avg * 0.8, b_avg * 1.2)
        daily_b = sum(baseline) / 96
        daily = sum(out) / 96
        target = max(daily_b * 0.85, min(daily_b * 1.15, daily))
        if daily > 0:
            out = [p * target / daily for p in out]
        for _ in range(8):
            out = [clamp_point(p, b) for p, b in zip(out, baseline)]
            for i in range(1, 96):
                out[i] = min(out[i - 1] + 20, max(out[i - 1] - 20, out[i]))
            for i in range(94, -1, -1):
                out[i] = min(out[i + 1] + 20, max(out[i + 1] - 20, out[i]))
            out = [clamp_point(p, b) for p, b in zip(out, baseline)]
        violations = check_constraints(out, baseline)
        if max(violations.values()) <= 1e-6:
            break
    return [round(p, 2) for p in out]


def check_constraints(prices: List[float], baseline: List[float]) -> Dict[str, float]:
    point_viol = max(max(abs(p - b) / b - 0.3, 0) for p, b in zip(prices, baseline))
    seg_viol = 0.0
    for seg in ("peak", "valley", "flat"):
        idx = SEG_INDEX[seg]
        b_avg = sum(baseline[i] for i in idx) / len(idx)
        p_avg = sum(prices[i] for i in idx) / len(idx)
        seg_viol = max(seg_viol, abs(p_avg - b_avg) / b_avg - 0.2)
    daily_b, daily_p = sum(baseline) / 96, sum(prices) / 96
    daily_viol = abs(daily_p - daily_b) / daily_b - 0.15
    jump_viol = max((abs(prices[i] - prices[i - 1]) - 20 for i in range(1, 96)), default=0.0)
    range_viol = max(max(40 - p, p - 650, 0) for p in prices)
    return {
        "point_violation": round(max(0.0, point_viol), 6),
        "segment_violation": round(max(0.0, seg_viol), 6),
        "daily_violation": round(max(0.0, daily_viol), 6),
        "jump_violation": round(max(0.0, jump_viol), 6),
        "range_violation": round(max(0.0, range_viol), 6),
    }


# ---------------------------------------------------------------- graph
def build_power_graph(model: str, base_url: str, api_key: str, max_debate_rounds: int = 1):
    llm = make_llm(model, base_url, api_key)
    g = StateGraph(PowerState)

    analysts = ["Load Analyst", "Renewable Analyst", "Supply Analyst", "Price Analyst"]
    g.add_node("Load Analyst", analyst_node(llm, "load"))
    g.add_node("Renewable Analyst", analyst_node(llm, "renewable"))
    g.add_node("Supply Analyst", analyst_node(llm, "supply"))
    g.add_node("Price Analyst", analyst_node(llm, "price"))
    g.add_node("Join", lambda state: {})
    g.add_node("Bull Researcher", bull_node(llm))
    g.add_node("Bear Researcher", bear_node(llm))
    g.add_node("Research Manager", research_manager_node(llm))
    g.add_node("Trader", lambda state: {"price_view": build_price_view(state["baseline"], state["segment_view"])})
    g.add_node("Aggressive Risk", risk_node(llm, "aggressive"))
    g.add_node("Conservative Risk", risk_node(llm, "conservative"))
    g.add_node("Neutral Risk", risk_node(llm, "neutral"))
    g.add_node("Portfolio Manager", pm_node(llm))

    for name in analysts:
        g.add_edge(START, name)
        g.add_edge(name, "Join")
    g.add_edge("Join", "Bull Researcher")
    g.add_edge("Bull Researcher", "Bear Researcher")
    g.add_conditional_edges(
        "Bear Researcher",
        lambda state: should_continue_debate(state, max_debate_rounds),
        {"Bull Researcher": "Bull Researcher", "Research Manager": "Research Manager"},
    )
    g.add_edge("Research Manager", "Trader")
    g.add_edge("Trader", "Aggressive Risk")
    g.add_edge("Aggressive Risk", "Conservative Risk")
    g.add_edge("Conservative Risk", "Neutral Risk")
    g.add_edge("Neutral Risk", "Portfolio Manager")
    g.add_edge("Portfolio Manager", END)
    return g.compile()


def initial_state(day: str) -> PowerState:
    return {
        "day": day,
        "features": features_text(day),
        "baseline": baseline_prices(day),
        "reports": {},
        "debate": [],
        "round": 0,
        "risk_opinions": {},
        "llm_calls": 0,
    }
