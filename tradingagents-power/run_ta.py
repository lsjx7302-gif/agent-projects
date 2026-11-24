# -*- coding: utf-8 -*-
"""Run the original TradingAgents graph once with Qwen3.8-Flash (DashScope CN).

This is the "reproduce the framework" step. The power-scenario secondary
development lives in power_graph.py.
"""
import json
import os
import time
import traceback
from pathlib import Path

os.environ["DASHSCOPE_CN_API_KEY"] = os.environ.get("DASHSCOPE_API_KEY", "")
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10808"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:10808"
os.environ["TRADINGAGENTS_NO_UPDATE_CHECK"] = "1"

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
CACHE = BASE / "cache"
MEMORY = BASE / "memory" / "trading_memory.md"
for p in (RESULTS, CACHE, MEMORY.parent):
    p.mkdir(parents=True, exist_ok=True)

config = DEFAULT_CONFIG.copy()
config.update({
    "llm_provider": "qwen-cn",
    "deep_think_llm": "qwen3.8-flash",
    "quick_think_llm": "qwen3.8-flash",
    "output_language": "Chinese",
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_tool_rounds": 3,
    "results_dir": str(RESULTS),
    "data_cache_dir": str(CACHE),
    "memory_log_path": str(MEMORY),
    "data_vendors": {
        "core_stock_apis": "yfinance",
        "technical_indicators": "yfinance",
        "fundamental_data": "yfinance",
        "news_data": "yfinance",
        "macro_data": "fred",
        "prediction_markets": "polymarket",
    },
})

print("provider:", config["llm_provider"], "model:", config["quick_think_llm"])
print("ticker: NVDA  date: 2026-11-20")
started = time.time()
try:
    ta = TradingAgentsGraph(
        selected_analysts=("market", "social", "news", "fundamentals"),
        debug=False,
        config=config,
    )
    state, decision = ta.propagate("NVDA", "2026-11-20")
    elapsed = round(time.time() - started, 1)
    print(f"elapsed: {elapsed}s")
    print("decision:", decision)

    (RESULTS / "decision.txt").write_text(str(decision), encoding="utf-8")
    serializable = {}
    for k, v in state.items():
        try:
            json.dumps(v)
            serializable[k] = v
        except TypeError:
            serializable[k] = str(v)
    (RESULTS / "state.json").write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("saved:", RESULTS / "decision.txt")
    print("saved:", RESULTS / "state.json")
    print("result files:", [p.name for p in RESULTS.iterdir()])
except Exception:
    traceback.print_exc()
    raise
