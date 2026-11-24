# -*- coding: utf-8 -*-
"""Deterministic synthetic data for the Agent evaluation harness.

No company data is used. Prices are generated with a fixed seed so every run
sees exactly the same 96-point series.
"""
from __future__ import annotations

import math
import random
from datetime import date, timedelta
from typing import Dict, List, Optional

PRICE_DATES = ["2026-11-20", "2026-11-21", "2026-11-22"]

WEATHER: Dict[str, Dict[str, object]] = {
    "北京": {"condition": "晴", "high": 25, "low": 12, "wind": "北风 3 级"},
    "上海": {"condition": "多云", "high": 22, "low": 16, "wind": "东风 2 级"},
    "广州": {"condition": "小雨", "high": 28, "low": 22, "wind": "南风 2 级"},
    "深圳": {"condition": "阴", "high": 27, "low": 23, "wind": "东南风 3 级"},
    "成都": {"condition": "多云", "high": 20, "low": 14, "wind": "微风"},
}

# Synthetic market-rule snippets. These are simplified for the harness and are
# NOT a substitute for the real rulebook.
RULES: List[Dict[str, str]] = [
    {
        "id": "R01",
        "text": "现货市场申报价格上限为 650 元/兆瓦时，下限为 40 元/兆瓦时。",
        "keywords": "申报 价格 上限 下限 650 40",
    },
    {
        "id": "R02",
        "text": "日前市场申报截止时间为每日 10:00，逾期不再受理。",
        "keywords": "日前 申报 截止 时间 10:00",
    },
    {
        "id": "R03",
        "text": "新能源机组在现货市场中优先出清，享有优先调度权。",
        "keywords": "新能源 优先 出清 调度",
    },
    {
        "id": "R04",
        "text": "机组检修计划应至少提前 3 个工作日向交易中心报备。",
        "keywords": "检修 计划 提前 3 个工作日 报备",
    },
    {
        "id": "R05",
        "text": "现货市场按 15 分钟为一个结算时段，全天共 96 个时段。",
        "keywords": "结算 时段 15 分钟 96",
    },
    {
        "id": "R06",
        "text": "日前市场出清结果在运行日前一天 18:00 前发布。",
        "keywords": "出清 结果 发布 18:00 前一天",
    },
]


def _time_label(index: int) -> str:
    """Index 0 -> 00:15, index 95 -> 24:00."""
    minutes = (index + 1) * 15
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def price_series(day: str) -> Optional[List[float]]:
    """Deterministic 96-point day-ahead price series, or None for unknown date."""
    if day not in PRICE_DATES:
        return None
    seed = 20261120 + PRICE_DATES.index(day)
    rng = random.Random(seed)
    values: List[float] = []
    for i in range(96):
        hour = (i + 1) / 4.0
        evening_peak = 190 * math.exp(-((hour - 19.0) ** 2) / (2 * 2.2 ** 2))
        midday_valley = -130 * math.exp(-((hour - 12.0) ** 2) / (2 * 2.8 ** 2))
        morning_peak = 70 * math.exp(-((hour - 8.0) ** 2) / (2 * 1.8 ** 2))
        noise = rng.gauss(0, 12)
        value = 300 + evening_peak + midday_valley + morning_peak + noise
        values.append(round(max(40.0, min(650.0, value)), 2))
    return values


def price_summary(day: str) -> Optional[Dict[str, object]]:
    values = price_series(day)
    if values is None:
        return None
    max_v = max(values)
    min_v = min(values)
    max_i = values.index(max_v)
    min_i = values.index(min_v)
    return {
        "day": day,
        "max": max_v,
        "max_time": _time_label(max_i),
        "min": min_v,
        "min_time": _time_label(min_i),
        "avg": round(sum(values) / len(values), 2),
        "points": len(values),
        "unit": "元/MWh",
    }


def daterange(start: str, days: int) -> List[str]:
    y, m, d = (int(x) for x in start.split("-"))
    return [(date(y, m, d) + timedelta(days=i)).isoformat() for i in range(days)]
