# -*- coding: utf-8 -*-
"""Tool implementations + OpenAI function schemas for the evaluation Agent."""
from __future__ import annotations

import ast
import json
import operator
import re
from typing import Any, Dict, List

from data import PRICE_DATES, RULES, WEATHER, price_summary

TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市今天的天气（天气现象、最高温、最低温、风力）。",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市名，例如 北京"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_rules",
            "description": "在电力现货市场规则库中检索相关条款，返回最相关的 1-3 条。",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "检索问题或关键词"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算四则运算表达式，支持 + - * / 和括号。",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "例如 (650-40)*0.9"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_price_summary",
            "description": "查询某天 96 点日前电价的最高价、最低价、均价及出现时间。",
            "parameters": {
                "type": "object",
                "properties": {"day": {"type": "string", "description": "日期，格式 YYYY-MM-DD"}},
                "required": ["day"],
            },
        },
    },
]

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_calc(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_calc(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_safe_calc(node.left), _safe_calc(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_safe_calc(node.operand))
    raise ValueError("只支持数字和 + - * / 括号")


def get_weather(city: str) -> str:
    info = WEATHER.get(city.strip())
    if info is None:
        return f'ERROR: 未找到城市 "{city}" 的天气数据，可用城市：{"、".join(WEATHER)}。'
    return f'{city}：{info["condition"]}，最高 {info["high"]}°C，最低 {info["low"]}°C，{info["wind"]}。'


def search_rules(query: str) -> str:
    q = query.lower()
    scored = []
    for rule in RULES:
        score = 0
        for kw in rule["keywords"].split():
            if kw.lower() in q:
                score += 2
        for token in re.findall(r"[一-鿿]{2,}", q):
            if token in rule["text"]:
                score += 1
        if score:
            scored.append((score, rule))
    if not scored:
        return f'ERROR: 规则库中没有找到与 "{query}" 相关的内容。'
    scored.sort(key=lambda x: (-x[0], x[1]["id"]))
    lines = [f'[{r["id"]}] {r["text"]}' for _, r in scored[:3]]
    return "\n".join(lines)


def calculate(expression: str) -> str:
    try:
        value = _safe_calc(ast.parse(expression.strip(), mode="eval"))
    except ZeroDivisionError:
        return "ERROR: 不能除以 0。"
    except Exception as exc:
        return f"ERROR: 表达式无效（{exc}）。"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def get_price_summary(day: str) -> str:
    summary = price_summary(day.strip())
    if summary is None:
        return f'ERROR: 日期 "{day}" 无电价数据，可用日期：{"、".join(PRICE_DATES)}。'
    return json.dumps(summary, ensure_ascii=False)


def execute_tool(name: str, args: Dict[str, Any]) -> str:
    """Never raises: tool errors are returned as observations for the model."""
    try:
        if name == "get_weather":
            return get_weather(str(args.get("city", "")))
        if name == "search_rules":
            return search_rules(str(args.get("query", "")))
        if name == "calculate":
            return calculate(str(args.get("expression", "")))
        if name == "get_price_summary":
            return get_price_summary(str(args.get("day", "")))
        return f"ERROR: 未知工具 {name}。"
    except Exception as exc:  # defensive: a tool bug must not kill the run
        return f"ERROR: 工具 {name} 执行失败：{exc}"
