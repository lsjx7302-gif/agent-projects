# -*- coding: utf-8 -*-
"""Validate tasks + tools without an LLM: execute reference tool calls and check answers."""
from __future__ import annotations

import json
from pathlib import Path

from run_eval import check_answer
from tools import execute_tool

HERE = Path(__file__).resolve().parent
tasks = json.loads((HERE / "tasks.json").read_text(encoding="utf-8"))

passed = 0
checked = 0
for task in tasks:
    calls = task["reference_tool_calls"]
    if not calls:
        print(f"{task['id']}: no tool expected (skip)")
        continue
    outputs = []
    for call in calls:
        outputs.append(execute_tool(call["name"], call.get("args", {})))
    combined = "\n".join(outputs)
    ok, checks = check_answer(combined, task)
    checked += 1
    passed += int(ok)
    print(f"{task['id']}: {'OK ' if ok else 'FAIL'}  {checks}")
    if not ok:
        print("  outputs:", combined[:300])

print(f"\nchecked {checked}, passed {passed}")
