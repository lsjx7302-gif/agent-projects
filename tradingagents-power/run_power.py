# -*- coding: utf-8 -*-
"""Run the power-scenario graph on 3 target days and compare with a single agent.

Metrics: segment direction accuracy (9 segment-day pairs), MAE vs actual prices,
constraint violations, LLM calls, latency.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from power_data import TARGET_DATES, actual_directions, actual_prices, baseline_prices, features_text, segment_avg
from power_graph import (
    SegmentView, apply_constraints, build_price_view, build_power_graph,
    check_constraints, initial_state, make_llm, _structured,
)

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

MODEL = os.getenv("POWER_MODEL", "qwen3.8-flash")
BASE_URL = os.getenv("POWER_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
API_KEY = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise SystemExit("DASHSCOPE_API_KEY not set")


def single_agent(day: str):
    llm = make_llm(MODEL, BASE_URL, API_KEY)
    prompt = (
        "你是电力现货价格分析师。根据以下特征，直接判断明天峰段/谷段/平段相对昨日的方向。"
        "只输出 JSON，格式：{\"peak\":{\"direction\":\"up|down|flat\",\"confidence\":0.0,\"reason\":\"...\"},"
        "\"valley\":{...},\"flat\":{...}}。不要输出 markdown。"
    )
    view = _structured(llm, [("system", prompt), ("human", features_text(day))],
                       SegmentView, "只输出符合格式的 JSON。")
    data = view.model_dump() if hasattr(view, "model_dump") else view
    base = baseline_prices(day)
    raw = build_price_view(base, data)
    scaled = [base[i] + (raw[i] - base[i]) * 0.8 for i in range(96)]  # medium risk
    final = apply_constraints(scaled, base)
    return data, final, 1


def evaluate(day: str, view: dict, prices: list) -> dict:
    actual = actual_prices(day)
    actual_dir = actual_directions(day)
    pred_dir = {seg: view[seg]["direction"] for seg in ("peak", "valley", "flat")}
    correct = sum(1 for s in pred_dir if pred_dir[s] == actual_dir[s])
    mae = sum(abs(p - a) for p, a in zip(prices, actual)) / 96
    base = baseline_prices(day)
    base_mae = sum(abs(p - a) for p, a in zip(base, actual)) / 96
    return {
        "day": day,
        "pred_directions": pred_dir,
        "actual_directions": actual_dir,
        "direction_correct": correct,
        "direction_total": 3,
        "mae": round(mae, 2),
        "baseline_mae": round(base_mae, 2),
        "segment_avgs": {s: {"pred": round(segment_avg(prices, s), 2),
                             "actual": round(segment_avg(actual, s), 2),
                             "baseline": round(segment_avg(base, s), 2)}
                         for s in ("peak", "valley", "flat")},
        "constraints": check_constraints(prices, base),
        "first4": [round(x, 2) for x in prices[:4]],
        "last4": [round(x, 2) for x in prices[-4:]],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", default="", help="comma-separated dates; default: all target days")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    days = args.days.split(",") if args.days else list(TARGET_DATES)
    if args.limit:
        days = days[:args.limit]

    graph = build_power_graph(MODEL, BASE_URL, API_KEY, max_debate_rounds=1)
    rows = []
    for day in days:
        print(f"=== {day} multi-agent ===")
        t0 = time.time()
        state = graph.invoke(initial_state(day))
        elapsed = round(time.time() - t0, 1)
        multi = evaluate(day, state["segment_view"], state["price_view"])
        multi.update({"latency_s": elapsed, "llm_calls": state.get("llm_calls", 0),
                      "risk_level": state.get("risk_level"), "risk_opinions": state.get("risk_opinions", {}),
                      "debate": state.get("debate", []), "reports": state.get("reports", {})})
        print(f"  directions={multi['pred_directions']} actual={multi['actual_directions']} "
              f"correct={multi['direction_correct']}/3 MAE={multi['mae']} calls={multi['llm_calls']}")

        print(f"=== {day} single-agent ===")
        t1 = time.time()
        view, prices, calls = single_agent(day)
        single = evaluate(day, view, prices)
        single.update({"latency_s": round(time.time() - t1, 1), "llm_calls": calls})
        print(f"  directions={single['pred_directions']} actual={single['actual_directions']} "
              f"correct={single['direction_correct']}/3 MAE={single['mae']} calls={single['llm_calls']}")
        rows.append({"day": day, "multi": multi, "single": single})

    total = len(rows) * 3
    summary = {
        "model": MODEL,
        "days": TARGET_DATES,
        "multi": {
            "direction_correct": sum(r["multi"]["direction_correct"] for r in rows),
            "direction_total": total,
            "avg_mae": round(sum(r["multi"]["mae"] for r in rows) / len(rows), 2),
            "avg_calls": round(sum(r["multi"]["llm_calls"] for r in rows) / len(rows), 1),
            "avg_latency_s": round(sum(r["multi"]["latency_s"] for r in rows) / len(rows), 1),
        },
        "single": {
            "direction_correct": sum(r["single"]["direction_correct"] for r in rows),
            "direction_total": total,
            "avg_mae": round(sum(r["single"]["mae"] for r in rows) / len(rows), 2),
            "avg_calls": round(sum(r["single"]["llm_calls"] for r in rows) / len(rows), 1),
            "avg_latency_s": round(sum(r["single"]["latency_s"] for r in rows) / len(rows), 1),
        },
        "baseline_mae": round(sum(r["multi"]["baseline_mae"] for r in rows) / len(rows), 2),
    }
    payload = {"summary": summary, "rows": rows}
    (RESULTS / "power_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# 电力现货多智能体复盘（二次开发）结果",
        "",
        f"- 模型：`{MODEL}`",
        f"- 目标日：{', '.join(TARGET_DATES)}",
        f"- 数据：合成 96 点负荷/风光/检修/电价",
        "",
        "## 总体",
        "",
        "| 指标 | 多 Agent（分析师 + 辩论 + 风控） | 单 Agent | 不变化基线 |",
        "|---|---|---|---|",
        f"| 分段方向正确 | {summary['multi']['direction_correct']}/{total} | {summary['single']['direction_correct']}/{total} | - |",
        f"| MAE（元/MWh） | {summary['multi']['avg_mae']} | {summary['single']['avg_mae']} | {summary['baseline_mae']} |",
        f"| 平均 LLM 调用 | {summary['multi']['avg_calls']} | {summary['single']['avg_calls']} | - |",
        f"| 平均时延（s） | {summary['multi']['avg_latency_s']} | {summary['single']['avg_latency_s']} | - |",
        "",
        "## 每日明细",
        "",
    ]
    for r in rows:
        lines += [
            f"### {r['day']}",
            f"- 多 Agent：方向 {r['multi']['pred_directions']} / 实际 {r['multi']['actual_directions']}；"
            f"正确 {r['multi']['direction_correct']}/3；MAE {r['multi']['mae']}；风险等级 {r['multi']['risk_level']}",
            f"- 单 Agent：方向 {r['single']['pred_directions']} / 实际 {r['single']['actual_directions']}；"
            f"正确 {r['single']['direction_correct']}/3；MAE {r['single']['mae']}",
            f"- 约束违规：{r['multi']['constraints']}",
            "",
        ]
    (RESULTS / "power_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n=== summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("saved:", RESULTS / "power_results.json")
    print("saved:", RESULTS / "power_summary.md")


if __name__ == "__main__":
    main()
