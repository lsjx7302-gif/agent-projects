# -*- coding: utf-8 -*-
"""Minimal function-calling Agent (OpenAI-compatible endpoint)."""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

from tools import TOOLS, execute_tool

SYSTEM_PROMPT = (
    "你是一个工具调用 Agent。优先调用工具获取事实，不要凭记忆编造。"
    "如果问题与可用工具的能力无关（例如订票、股市预测、公司内部数据），不要调用工具，直接说明无法回答。"
    "工具返回 ERROR 时，要如实告诉用户无法获取，并给出可用的替代信息；不要编造答案。"
    "回答尽量简短，包含关键数字；完成任务后直接给出最终答案，不再调用工具。"
)


def _dump_tool_calls(tool_calls) -> List[Dict[str, Any]]:
    out = []
    for tc in tool_calls:
        out.append({
            "id": tc.id,
            "type": "function",
            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
        })
    return out


def run_agent(
    question: str,
    *,
    api_key: str,
    base_url: str,
    model: str,
    max_steps: int = 6,
    temperature: float = 0.0,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """Run one task and return answer + full tool-call trace."""
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=90)
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    trace: List[Dict[str, Any]] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    started = time.time()

    for step in range(1, max_steps + 1):
        response = None
        last_error: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=temperature,
                )
                break
            except Exception as exc:  # rate limits / transient network errors
                last_error = exc
                if attempt < max_retries:
                    time.sleep(3 * (attempt + 1))
        if response is None:
            return {
                "answer": f"ERROR: 模型调用失败：{last_error}",
                "tool_calls": trace,
                "llm_turns": step,
                "latency_s": round(time.time() - started, 2),
                "usage": usage,
            }

        if response.usage:
            usage["prompt_tokens"] += response.usage.prompt_tokens or 0
            usage["completion_tokens"] += response.usage.completion_tokens or 0
            usage["total_tokens"] += response.usage.total_tokens or 0

        message = response.choices[0].message
        if message.tool_calls:
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": _dump_tool_calls(message.tool_calls),
            })
            for tc in message.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = execute_tool(name, args)
                trace.append({
                    "step": step,
                    "name": name,
                    "args": args,
                    "result": result,
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": result,
                })
            continue

        return {
            "answer": (message.content or "").strip(),
            "tool_calls": trace,
            "llm_turns": step,
            "latency_s": round(time.time() - started, 2),
            "usage": usage,
        }

    return {
        "answer": f"达到最大工具调用轮数（{max_steps}），未能完成任务。",
        "tool_calls": trace,
        "llm_turns": max_steps,
        "latency_s": round(time.time() - started, 2),
        "usage": usage,
    }
