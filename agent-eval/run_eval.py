# -*- coding: utf-8 -*-
"""Run the 24-task Agent evaluation and write results.json + summary.md."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from agent import run_agent
from tools import _safe_calc

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-flash"


# ---------------------------------------------------------------- answer checks
def _numbers(text: str) -> List[float]:
    text = text.replace(",", "").replace("，", "")
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]


def check_answer(answer: str, task: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
    checks: List[Dict[str, Any]] = []
    nums = _numbers(answer)
    for expected in task.get("expected_numbers", []):
        tol = max(0.02, abs(expected) * 0.01) if abs(expected) < 100 else max(1.0, abs(expected) * 0.005)
        ok = any(abs(n - expected) <= tol for n in nums)
        checks.append({"type": "number", "expected": expected, "tolerance": round(tol, 3), "ok": ok})
    for kw in task.get("expected_keywords_all", []):
        checks.append({"type": "keyword_all", "expected": kw, "ok": kw in answer})
    any_kws = task.get("expected_keywords_any", [])
    if any_kws:
        hit = [kw for kw in any_kws if kw in answer]
        checks.append({"type": "keyword_any", "expected": any_kws, "hit": hit, "ok": bool(hit)})
    return all(c["ok"] for c in checks), checks


# ---------------------------------------------------------------- tool checks
def _norm(text: Any) -> str:
    return str(text).replace(" ", "").strip().lower()


def _match_args(name: str, expected: Dict[str, Any], actual: Dict[str, Any]) -> float:
    if name == "get_weather":
        return 1.0 if _norm(expected.get("city")) == _norm(actual.get("city")) else 0.0
    if name == "get_price_summary":
        return 1.0 if _norm(expected.get("day")) == _norm(actual.get("day")) else 0.0
    if name == "calculate":
        try:
            a = _safe_calc(__import__("ast").parse(str(expected.get("expression", "")), mode="eval"))
            b = _safe_calc(__import__("ast").parse(str(actual.get("expression", "")), mode="eval"))
            return 1.0 if abs(a - b) <= 1e-6 * max(1.0, abs(a)) else 0.0
        except Exception:
            return 1.0 if _norm(expected.get("expression")) == _norm(actual.get("expression")) else 0.0
    if name == "search_rules":
        e = _norm(expected.get("query"))
        a = _norm(actual.get("query"))
        if not e:
            return 1.0
        return 1.0 if (e in a or a in e) else 0.0
    return 1.0 if _norm(expected) == _norm(actual) else 0.0


def eval_tools(predicted: List[Dict[str, Any]], reference: List[Dict[str, Any]]) -> Dict[str, Any]:
    pred_names = [c["name"] for c in predicted]
    ref_names = [c["name"] for c in reference]
    pred_counter, ref_counter = Counter(pred_names), Counter(ref_names)

    selection = 1.0 if pred_counter == ref_counter else 0.0
    tp = sum((pred_counter & ref_counter).values())
    precision = tp / len(pred_names) if pred_names else (1.0 if not ref_names else 0.0)
    recall = tp / len(ref_names) if ref_names else 1.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    if not predicted and not reference:
        arg_accuracy = 1.0
    else:
        used = set()
        scores = []
        for ref in reference:
            for i, pred in enumerate(predicted):
                if i in used or pred["name"] != ref["name"]:
                    continue
                used.add(i)
                scores.append(_match_args(ref["name"], ref.get("args", {}), pred.get("args", {})))
                break
        arg_accuracy = sum(scores) / len(reference) if reference and scores else 0.0

    return {
        "tool_selection": selection,
        "tool_precision": round(precision, 4),
        "tool_recall": round(recall, 4),
        "tool_f1": round(f1, 4),
        "arg_accuracy": round(arg_accuracy, 4),
        "predicted_tools": pred_names,
        "reference_tools": ref_names,
    }


def ragas_tool_accuracy(question: str, predicted: List[Dict[str, Any]],
                        reference: List[Dict[str, Any]]) -> Optional[float]:
    # ragas 0.3.x divides by the reference length; handle the empty-reference
    # cases ourselves instead of letting it raise ZeroDivisionError.
    if not reference:
        return 1.0 if not predicted else 0.0
    if not predicted:
        return 0.0
    try:
        from ragas.dataset_schema import MultiTurnSample
        from ragas.messages import AIMessage, HumanMessage, ToolCall
        from ragas.metrics import ToolCallAccuracy

        predicted_calls = [
            ToolCall(name=c["name"], args=c.get("args", {})) for c in predicted
        ]
        reference_calls = [
            ToolCall(name=c["name"], args=c.get("args", {})) for c in reference
        ]
        sample = MultiTurnSample(
            user_input=[HumanMessage(content=question), AIMessage(content="", tool_calls=predicted_calls)],
            reference_tool_calls=reference_calls,
        )
        try:
            metric = ToolCallAccuracy(strict_order=False)
        except TypeError:  # ragas 0.3.x has no strict_order argument
            metric = ToolCallAccuracy()
        return float(asyncio.run(metric.multi_turn_ascore(sample)))
    except Exception as exc:
        print(f"  [ragas] skipped: {type(exc).__name__}: {exc}")
        return None


# ---------------------------------------------------------------- reporting
def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if not n:
        return {}
    def mean(key):
        vals = [r["metrics"][key] for r in rows if r["metrics"].get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None
    return {
        "tasks": n,
        "task_success_rate": round(sum(1 for r in rows if r["metrics"]["success"]) / n, 4),
        "tool_selection_accuracy": mean("tool_selection"),
        "tool_precision": mean("tool_precision"),
        "tool_recall": mean("tool_recall"),
        "tool_f1": mean("tool_f1"),
        "arg_accuracy": mean("arg_accuracy"),
        "ragas_tool_accuracy": mean("ragas_tool_accuracy"),
        "avg_tool_calls": round(sum(len(r["tool_calls"]) for r in rows) / n, 2),
        "avg_llm_turns": round(sum(r["llm_turns"] for r in rows) / n, 2),
        "avg_latency_s": round(sum(r["latency_s"] for r in rows) / n, 2),
        "total_tokens": sum(r.get("usage", {}).get("total_tokens", 0) for r in rows),
    }


def write_summary(rows: List[Dict[str, Any]], overall: Dict[str, Any],
                  by_category: Dict[str, Dict[str, Any]], model: str, out: Path) -> None:
    lines = [
        "# Agent 评测结果",
        "",
        f"- 模型：`{model}`",
        f"- 时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- 任务数：{overall.get('tasks', 0)}",
        "",
        "## 总体指标",
        "",
        "| 指标 | 数值 |",
        "|---|---|",
        f"| 任务成功率 | {overall.get('task_success_rate', 0):.1%} |",
        f"| 工具选择准确率 | {overall.get('tool_selection_accuracy', 0):.1%} |",
        f"| 工具调用 F1 | {overall.get('tool_f1', 0):.2f} |",
        f"| 参数准确率 | {overall.get('arg_accuracy', 0):.1%} |",
        f"| RAGAS ToolCallAccuracy | {overall.get('ragas_tool_accuracy') or 'N/A'} |",
        f"| 平均工具调用数 | {overall.get('avg_tool_calls', 0)} |",
        f"| 平均 LLM 轮数 | {overall.get('avg_llm_turns', 0)} |",
        f"| 平均时延 | {overall.get('avg_latency_s', 0)} s |",
        f"| 总 token | {overall.get('total_tokens', 0)} |",
        "",
        "## 分类指标",
        "",
        "| 类别 | 任务数 | 成功率 | 工具选择 | 参数准确率 |",
        "|---|---|---|---|---|",
    ]
    for cat, m in by_category.items():
        lines.append(
            f"| {cat} | {m['tasks']} | {m['task_success_rate']:.1%} | "
            f"{m['tool_selection_accuracy']:.1%} | {m['arg_accuracy']:.1%} |"
        )
    failed = [r for r in rows if not r["metrics"]["success"]]
    lines += ["", f"## 失败案例（{len(failed)}）", ""]
    for r in failed:
        lines.append(f"### {r['id']} [{r['category']}]")
        lines.append(f"- 问题：{r['question']}")
        lines.append(f"- 回答：{r['answer']}")
        lines.append(f"- 工具：{r['metrics']['predicted_tools']} / 参考：{r['metrics']['reference_tools']}")
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="only run the first N tasks")
    parser.add_argument("--model", default=os.getenv("MODEL", DEFAULT_MODEL))
    parser.add_argument("--base-url", default=os.getenv("BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key", default=os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--sleep", type=float, default=0.4, help="seconds between tasks")
    parser.add_argument("--output", default=str(HERE / "results"))
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("未找到 API key：请设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY")

    tasks = json.loads((HERE / "tasks.json").read_text(encoding="utf-8"))
    if args.limit:
        tasks = tasks[:args.limit]

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    for i, task in enumerate(tasks, 1):
        print(f"[{i}/{len(tasks)}] {task['id']} {task['category']} {task['question']}")
        started = time.time()
        result = run_agent(
            task["question"],
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            max_steps=args.max_steps,
        )
        metrics = eval_tools(result["tool_calls"], task["reference_tool_calls"])
        success, checks = check_answer(result["answer"], task)
        ragas_score = ragas_tool_accuracy(
            task["question"], result["tool_calls"], task["reference_tool_calls"]
        )
        row = {
            **task,
            "answer": result["answer"],
            "tool_calls": result["tool_calls"],
            "llm_turns": result["llm_turns"],
            "latency_s": result.get("latency_s", round(time.time() - started, 2)),
            "usage": result.get("usage", {}),
            "metrics": {
                **metrics,
                "success": success,
                "checks": checks,
                "ragas_tool_accuracy": ragas_score,
            },
        }
        rows.append(row)
        print(f"  success={success} selection={metrics['tool_selection']} "
              f"tools={metrics['predicted_tools']} answer={result['answer'][:80]!r}")
        if result["answer"].startswith("ERROR: 模型调用失败"):
            print("模型调用失败，提前停止。请检查 API key / 免费额度。")
            break
        if args.sleep:
            time.sleep(args.sleep)

    overall = aggregate(rows)
    categories = sorted({r["category"] for r in rows})
    by_category = {c: aggregate([r for r in rows if r["category"] == c]) for c in categories}

    payload = {
        "model": args.model,
        "base_url": args.base_url,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "overall": overall,
        "by_category": by_category,
        "rows": rows,
    }
    (out_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    write_summary(rows, overall, by_category, args.model, out_dir / "summary.md")

    print("\n=== overall ===")
    for k, v in overall.items():
        print(f"{k}: {v}")
    print(f"\nwritten: {out_dir / 'results.json'}")
    print(f"written: {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
