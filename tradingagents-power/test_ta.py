# -*- coding: utf-8 -*-
"""Smoke-test: LLM client + yfinance data through the local proxy."""
import os

os.environ["DASHSCOPE_CN_API_KEY"] = os.environ.get("DASHSCOPE_API_KEY", "")
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10808"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:10808"

from tradingagents.llm_clients import create_llm_client

print("--- LLM ---")
llm = create_llm_client("qwen-cn", "qwen3.8-flash").get_llm()
resp = llm.invoke("只回复两个字：可以")
print("content:", getattr(resp, "content", resp))

print("--- yfinance ---")
try:
    import yfinance as yf
    df = yf.download("NVDA", start="2026-11-01", end="2026-11-22", progress=False, auto_adjust=True)
    print("rows:", len(df))
    print(df.tail(3))
except Exception as exc:
    print("yfinance failed:", type(exc).__name__, exc)
