# -*- coding: utf-8 -*-
"""Synthetic 96-point power-market data for the power-scenario agent.

All data is deterministic and synthetic: load, wind, solar, outage, and price.
The agents only see features; the target-day actual price is used for scoring.
"""
from __future__ import annotations

import math
import random
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

DATES = [(date(2026, 11, 20) + timedelta(days=i)).isoformat() for i in range(6)]
TARGET_DATES = DATES[3:]  # 2026-11-23 .. 2026-11-25

PEAK = list(range(68, 84))    # 17:15 - 21:00
VALLEY = list(range(44, 60))  # 11:15 - 15:00


def _time_label(index: int) -> str:
    minutes = (index + 1) * 15
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _series(day: str) -> Dict[str, List[float]]:
    seed = int(day.replace("-", ""))
    rng = random.Random(seed)
    load, wind, solar, outage, price = [], [], [], [], []
    for i in range(96):
        h = (i + 1) / 4.0
        load_v = (
            2950
            + 430 * math.exp(-((h - 19.0) ** 2) / (2 * 2.0 ** 2))
            + 260 * math.exp(-((h - 8.0) ** 2) / (2 * 1.8 ** 2))
            - 300 * math.exp(-((h - 3.5) ** 2) / (2 * 2.0 ** 2))
            + rng.gauss(0, 25)
        )
        wind_v = 780 + 260 * math.sin((h - 3) / 24 * 2 * math.pi) + rng.gauss(0, 90)
        solar_v = 820 * math.exp(-((h - 12.2) ** 2) / (2 * 2.6 ** 2)) + rng.gauss(0, 25)
        solar_v = max(0.0, solar_v)
        outage_v = 180 + 120 * math.sin((h - 9) / 24 * 2 * math.pi) + rng.gauss(0, 30)
        net_load = load_v - wind_v - solar_v + 0.5 * outage_v
        evening = 140 * math.exp(-((h - 19.0) ** 2) / (2 * 2.2 ** 2))
        midday = -80 * math.exp(-((h - 12.5) ** 2) / (2 * 2.6 ** 2))
        price_v = 40 + 0.115 * net_load + evening + midday + rng.gauss(0, 3.5)
        load.append(round(load_v, 1))
        wind.append(round(wind_v, 1))
        solar.append(round(solar_v, 1))
        outage.append(round(outage_v, 1))
        price.append(round(max(40.0, min(650.0, price_v)), 2))
    # A 3-point moving average keeps adjacent jumps compatible with the ≤20 rule.
    smoothed = []
    for i in range(96):
        lo, hi = max(0, i - 1), min(95, i + 1)
        smoothed.append(round(sum(price[lo:hi + 1]) / (hi - lo + 1), 2))
    return {"load": load, "wind": wind, "solar": solar, "outage": outage, "price": smoothed}


_CACHE: Dict[str, Dict[str, List[float]]] = {}


def get_series(day: str) -> Optional[Dict[str, List[float]]]:
    if day not in DATES:
        return None
    if day not in _CACHE:
        _CACHE[day] = _series(day)
    return _CACHE[day]


def _stats(values: List[float]) -> Dict[str, float]:
    return {"max": max(values), "min": min(values), "avg": round(sum(values) / len(values), 2)}


def segment_avg(prices: List[float], segment: str) -> float:
    idx = PEAK if segment == "peak" else VALLEY if segment == "valley" else [
        i for i in range(96) if i not in PEAK and i not in VALLEY
    ]
    return round(sum(prices[i] for i in idx) / len(idx), 2)


def features(day: str) -> Dict[str, object]:
    """Features the agents are allowed to see (no target-day actual price)."""
    series = get_series(day)
    prev = get_series(DATES[DATES.index(day) - 1])
    assert series and prev
    load, wind, solar, outage = series["load"], series["wind"], series["solar"], series["outage"]
    prev_price = prev["price"]
    return {
        "day": day,
        "load": {
            **_stats(load),
            "max_time": _time_label(load.index(max(load))),
            "min_time": _time_label(load.index(min(load))),
        },
        "wind": _stats(wind),
        "solar": _stats(solar),
        "outage": _stats(outage),
        "prev_price": {
            **_stats(prev_price),
            "max_time": _time_label(prev_price.index(max(prev_price))),
            "min_time": _time_label(prev_price.index(min(prev_price))),
            "segments": {s: segment_avg(prev_price, s) for s in ("peak", "valley", "flat")},
        },
    }


def baseline_prices(day: str) -> List[float]:
    """Yesterday's 96-point price — the no-change baseline and the constraint anchor."""
    prev = get_series(DATES[DATES.index(day) - 1])
    assert prev
    return list(prev["price"])


def actual_prices(day: str) -> List[float]:
    series = get_series(day)
    assert series
    return list(series["price"])


def actual_directions(day: str, threshold: float = 0.02) -> Dict[str, str]:
    """Actual segment direction vs yesterday, used only for scoring."""
    base = baseline_prices(day)
    act = actual_prices(day)
    out = {}
    for seg in ("peak", "valley", "flat"):
        b, a = segment_avg(base, seg), segment_avg(act, seg)
        diff = (a - b) / b if b else 0
        out[seg] = "up" if diff > threshold else "down" if diff < -threshold else "flat"
    return out


def features_text(day: str) -> str:
    f = features(day)
    load, wind, solar, outage, prev = f["load"], f["wind"], f["solar"], f["outage"], f["prev_price"]
    return (
        f"日期：{day}\n"
        f"负荷预测：最高 {load['max']} MW（{load['max_time']}），最低 {load['min']} MW（{load['min_time']}），日均 {load['avg']} MW\n"
        f"风电预测：最高 {wind['max']} MW，最低 {wind['min']} MW，日均 {wind['avg']} MW\n"
        f"光伏预测：最高 {solar['max']} MW，日均 {solar['avg']} MW\n"
        f"检修容量：日均 {outage['avg']} MW，最大 {outage['max']} MW\n"
        f"昨日电价：最高 {prev['max']}（{prev['max_time']}），最低 {prev['min']}（{prev['min_time']}），均价 {prev['avg']}\n"
        f"昨日分段均价：峰段 {prev['segments']['peak']}，谷段 {prev['segments']['valley']}，平段 {prev['segments']['flat']}\n"
        f"时段定义：峰段 17:15-21:00，谷段 11:15-15:00，其余为平段。"
    )
