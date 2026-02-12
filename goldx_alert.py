#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import math
import html
import mimetypes
import textwrap
from dotenv import load_dotenv

import smtplib
import ssl
import sys
import time
from twelvedata import TDClient
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone, tzinfo
from email.message import EmailMessage
from typing import Any, Iterable, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
matplotlib.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "Hiragino Sans GB",
    "Microsoft YaHei",
    "Noto Sans CJK SC",
    "WenQuanYi Zen Hei",
    "SimHei",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False



from zoneinfo import ZoneInfo



@dataclass(frozen=True)
class PricePoint:
    ts: datetime
    price: float


_TD_CLIENTS: dict[str, TDClient] = {}


def _get_td_client(api_key: str) -> TDClient:
    client = _TD_CLIENTS.get(api_key)
    if client is None:
        client = TDClient(apikey=api_key)
        _TD_CLIENTS[api_key] = client
    return client


def _local_tz() -> tzinfo:
    return datetime.now().astimezone().tzinfo or timezone.utc


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_int(value: str | None, default: int) -> int:
    if value is None or value.strip() == "":
        return default
    return int(value)


def _parse_float(value: str | None, default: float) -> float:
    if value is None or value.strip() == "":
        return default
    return float(value)


def _parse_api_keys(raw: str | None) -> list[str]:
    if raw is None:
        return []
    # Allow comma / whitespace separated tokens.
    tokens: list[str] = []
    for part in raw.split(","):
        t = part.strip()
        if not t:
            continue
        tokens.append(t)
    return tokens


def _get_arg_value(argv: list[str], key: str, default: Optional[str] = None) -> Optional[str]:
    for i, arg in enumerate(argv):
        if arg == key and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(key + "="):
            return arg.split("=", 1)[1]
    return default


def _parse_date(value: str) -> date:
    return datetime.strptime(value.strip(), "%Y-%m-%d").date()


def _today_in_tz(tz: tzinfo) -> date:
    return datetime.now(tz).date()


def _sanitize_symbol_for_path(symbol: str) -> str:
    out = []
    for ch in symbol.strip():
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out).strip("_")
    return slug or "symbol"




def _parse_dt_raw(value: str) -> datetime:
    v = value.strip()
    if v.endswith("Z"):
        v = v[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(v)
    except ValueError:
        return datetime.strptime(v, "%Y-%m-%d %H:%M:%S")


class DataProviderError(RuntimeError):
    pass


def _format_utc_offset_name(offset: timedelta) -> str:
    total_minutes = int(offset.total_seconds() // 60)
    sign = "+" if total_minutes >= 0 else "-"
    total_minutes = abs(total_minutes)
    hours, minutes = divmod(total_minutes, 60)
    if minutes:
        return f"UTC{sign}{hours:02d}:{minutes:02d}"
    return f"UTC{sign}{hours}"



def fetch_twelvedata_1min_series(
    *,
    api_key: str,
    symbol: str,
    interval: str = "1min",
    outputsize: int = 200,
    tz: tzinfo,
    timezone_name: str = "Asia/Shanghai",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> list[PricePoint]:
    td = _get_td_client(api_key)
    request_kwargs: dict[str, Any] = {
        "symbol": symbol,
        "interval": interval,
        "timezone": timezone_name,
        "outputsize": outputsize,
        "order": "asc",
    }
    if start_date:
        request_kwargs["start_date"] = start_date
    if end_date:
        request_kwargs["end_date"] = end_date
    ts = td.time_series(**request_kwargs)
    data = ts.as_json()
    points = []
    for row in data:
        dt_raw = row.get("datetime")
        close_raw = row.get("close")
        if dt_raw is None or close_raw is None:
            continue
        try:
            date = datetime.strptime(dt_raw, "%Y-%m-%d %H:%M:%S").replace(tzinfo=tz)
            price = float(close_raw)
        except Exception:
            continue
        points.append(PricePoint(ts=date, price=price))

    return points


def ema(series: Iterable[float], period: int) -> list[float]:
    series_list = list(series)
    if not series_list:
        return []
    alpha = 2.0 / (period + 1)
    out: list[float] = []
    value = series_list[0]
    out.append(value)
    for x in series_list[1:]:
        value = alpha * x + (1 - alpha) * value
        out.append(value)
    return out


def macd_histogram(prices: list[float], fast: int = 12, slow: int = 26, signal: int = 9) -> list[float]:
    if len(prices) < slow + signal:
        return []
    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
    signal_line = ema(macd_line, signal)
    return [m - s for m, s in zip(macd_line, signal_line)]


def linear_slope(prices: list[float]) -> float:
    n = len(prices)
    if n < 2:
        return 0.0
    xs = list(range(n))
    x_mean = (n - 1) / 2.0
    y_mean = sum(prices) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, prices))
    den = sum((x - x_mean) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return num / den  # price change per bar (assumes 1 bar = 1 minute)

def _quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("values is empty")
    if q <= 0:
        return min(values)
    if q >= 100:
        return max(values)
    sorted_vals = sorted(values)
    pos = (len(sorted_vals) - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _session_start_anchor(ts: datetime, *, start_min: int, end_min: int) -> datetime:
    base = datetime(ts.year, ts.month, ts.day, 0, 0, 0, tzinfo=ts.tzinfo)
    if start_min == 0 and end_min == 1440:
        return base

    current = ts.hour * 60 + ts.minute + (ts.second / 60.0)
    if start_min < end_min:
        session_day = base if current >= start_min else base - timedelta(days=1)
    else:
        if current >= start_min:
            session_day = base
        elif current < end_min:
            session_day = base - timedelta(days=1)
        else:
            session_day = base - timedelta(days=1)
    return session_day + timedelta(minutes=start_min)


@dataclass(frozen=True)
class Signal:
    side: str
    should_alert: bool
    reason: str
    metrics: dict[str, float]


def compute_buy_signal(
    points: list[PricePoint],
    *,
    low_window_min: int,
    slope_window_min: int,
    quantile_window_min: int,
    global_window_min: int,
    monitor_start_min: int,
    monitor_end_min: int,
    min_bars_to_start: int,
    open_scout_min: int,
    open_scout_score_threshold: float,
    q_low: float,
    q_skip: float,
    absolute_max_price: float,
    stall_min: int,
    micro_rebound_pct: float,
    early_near_global_low_pct: float,
    max_dist_from_global_low_pct: float,
    max_dist_from_session_low_pct: float,
    session_low_guard_max_age_min: int,
    session_low_guard_lookback_min: int,
    rebound_pct: float,
    rebound_max_pct: float,
    low_max_age_min: int,
    low_zone_slope_min_pct_per_min: float,
    early_score_threshold: float,
    buy_htf_slope_window_min: int,
    buy_htf_slope_min_pct_per_min: float,
    buy_soft_downtrend_slope_pct_per_min: float,
    buy_soft_downtrend_rebound_boost_pct: float,
    buy_soft_downtrend_stall_boost_min: int,
    buy_soft_downtrend_score_boost: float,
    buy_soft_downtrend_short_slope_min_pct_per_min: float,
    buy_hard_filter_near_session_low_bypass_pct: float,
    buy_hard_filter_bypass_rebound_pct: float,
) -> Signal:
    hard_downtrend_enabled = buy_htf_slope_window_min > 1 and buy_htf_slope_min_pct_per_min > -0.5
    soft_downtrend_enabled = buy_htf_slope_window_min > 1 and buy_soft_downtrend_slope_pct_per_min > -0.5
    htf_required = buy_htf_slope_window_min if (hard_downtrend_enabled or soft_downtrend_enabled) else 0
    min_required = max(8, min_bars_to_start, slope_window_min, htf_required)
    if len(points) < min_required:
        return Signal("BUY", False, "数据不足", {})

    if rebound_max_pct > 0 and rebound_max_pct < rebound_pct:
        rebound_max_pct = rebound_pct
    if q_low < 0:
        q_low = 0.0
    if q_low > 100:
        q_low = 100.0
    if q_skip < 0:
        q_skip = 0.0
    if q_skip > 100:
        q_skip = 100.0
    if q_skip > 0 and q_low > 0 and q_skip < q_low:
        q_skip = q_low
    q_low_eff = q_low if q_low > 0 else 25.0
    q_skip_eff = q_skip if q_skip > 0 else 60.0

    closes = [p.price for p in points]

    low_window_points = points[-low_window_min:]
    low_point = min(low_window_points, key=lambda p: p.price)
    low = low_point.price
    last = closes[-1]
    rebound = (last - low) / low * 100.0
    low_age_min = (points[-1].ts - low_point.ts).total_seconds() / 60.0

    if absolute_max_price > 0 and last > absolute_max_price:
        return Signal(
            "BUY",
            False,
            f"最新价{last:.4f}高于绝对价格上限{absolute_max_price:.4f}过滤",
            {
                "last": last,
                "low": low,
                "rebound_pct": rebound,
            },
        )

    if quantile_window_min <= 0:
        quantile_window = closes
    else:
        quantile_window = closes[-quantile_window_min:]
    if not quantile_window:
        quantile_window = [last]
    q_low_value = _quantile(quantile_window, q_low_eff)
    q_skip_value = _quantile(quantile_window, q_skip_eff)

    if global_window_min <= 0:
        global_window_points = points
    else:
        global_window_points = points[-global_window_min:]
    if not global_window_points:
        global_window_points = [points[-1]]
    global_low_point = min(global_window_points, key=lambda p: p.price)
    global_low = global_low_point.price
    dist_from_global_low_pct = (last - global_low) / global_low * 100.0 if global_low != 0 else 0.0
    global_low_age_min = (points[-1].ts - global_low_point.ts).total_seconds() / 60.0

    session_start = _session_start_anchor(points[-1].ts, start_min=monitor_start_min, end_min=monitor_end_min)
    session_age_min = (points[-1].ts - session_start).total_seconds() / 60.0
    session_points = [p for p in points if p.ts >= session_start]
    if not session_points:
        session_points = [points[-1]]
    session_low_point = min(session_points, key=lambda p: p.price)
    session_low = session_low_point.price
    dist_from_session_low_pct = (last - session_low) / session_low * 100.0 if session_low != 0 else 0.0
    session_low_age_min = (points[-1].ts - session_low_point.ts).total_seconds() / 60.0
    session_guard_low_point = session_low_point
    if (
        session_low_guard_max_age_min > 0
        and session_low_age_min > session_low_guard_max_age_min
        and session_low_guard_lookback_min > 0
    ):
        guard_start = points[-1].ts - timedelta(minutes=session_low_guard_lookback_min)
        guard_candidates = [p for p in session_points if p.ts >= guard_start]
        if guard_candidates:
            session_guard_low_point = min(guard_candidates, key=lambda p: p.price)
    session_guard_low = session_guard_low_point.price
    dist_from_session_guard_low_pct = (
        (last - session_guard_low) / session_guard_low * 100.0 if session_guard_low != 0 else 0.0
    )
    session_guard_low_age_min = (points[-1].ts - session_guard_low_point.ts).total_seconds() / 60.0
    dist_score_ref_pct = dist_from_global_low_pct
    dist_score_ref_is_session_guard = 0.0
    if session_guard_low_point.ts != session_low_point.ts:
        dist_score_ref_pct = dist_from_session_guard_low_pct
        dist_score_ref_is_session_guard = 1.0
    session_guard_global_cap_pct = max_dist_from_global_low_pct * 0.7 if max_dist_from_global_low_pct > 0 else 0.0

    near_global_low = early_near_global_low_pct > 0 and dist_from_global_low_pct <= early_near_global_low_pct
    session_guard_ok = (
        max_dist_from_session_low_pct <= 0
        or dist_from_session_guard_low_pct <= max_dist_from_session_low_pct
    )
    session_guard_global_ok = (
        dist_score_ref_is_session_guard < 0.5
        or session_guard_global_cap_pct <= 0
        or dist_from_global_low_pct <= session_guard_global_cap_pct
    )

    slope_window = closes[-slope_window_min:]
    slope_abs = linear_slope(slope_window)
    slope_pct_per_min = (slope_abs / last) * 100.0 if last != 0 else 0.0

    htf_slope_pct_per_min = slope_pct_per_min
    if hard_downtrend_enabled or soft_downtrend_enabled:
        htf_window = closes[-buy_htf_slope_window_min:]
        htf_slope_abs = linear_slope(htf_window)
        htf_slope_pct_per_min = (htf_slope_abs / last) * 100.0 if last != 0 else 0.0
    htf_filter_enabled = hard_downtrend_enabled or soft_downtrend_enabled
    hard_downtrend = hard_downtrend_enabled and htf_slope_pct_per_min < buy_htf_slope_min_pct_per_min
    near_session_low_bypass = (
        buy_hard_filter_near_session_low_bypass_pct > 0
        and buy_hard_filter_bypass_rebound_pct > 0
        and dist_from_session_low_pct <= buy_hard_filter_near_session_low_bypass_pct
        and rebound >= buy_hard_filter_bypass_rebound_pct
    )
    if hard_downtrend and not near_session_low_bypass:
        metrics = {
            "last": last,
            "low": low,
            "global_low": global_low,
            "rebound_pct": rebound,
            "slope_pct_per_min": slope_pct_per_min,
            "htf_slope_pct_per_min": htf_slope_pct_per_min,
            "buy_htf_slope_window_min": float(buy_htf_slope_window_min),
            "buy_htf_slope_min_pct_per_min": buy_htf_slope_min_pct_per_min,
            "low_age_min": low_age_min,
            "global_low_age_min": global_low_age_min,
            "dist_from_global_low_pct": dist_from_global_low_pct,
            "session_low": session_low,
            "session_low_age_min": session_low_age_min,
            "dist_from_session_low_pct": dist_from_session_low_pct,
            "near_session_low_bypass": 1.0 if near_session_low_bypass else 0.0,
            "q_low_value": q_low_value,
            "q_skip_value": q_skip_value,
            "quantile_window_min": float(quantile_window_min),
            "global_window_min": float(global_window_min),
        }
        return Signal(
            "BUY",
            False,
            f"高周期仍偏下行（{htf_slope_pct_per_min:.4f}%/min < 阈值{buy_htf_slope_min_pct_per_min:.4f}）过滤",
            metrics,
        )

    soft_downtrend = soft_downtrend_enabled and htf_slope_pct_per_min < buy_soft_downtrend_slope_pct_per_min
    soft_short_slope_ok = slope_pct_per_min >= buy_soft_downtrend_short_slope_min_pct_per_min
    effective_rebound_pct = rebound_pct + buy_soft_downtrend_rebound_boost_pct if soft_downtrend else rebound_pct
    effective_stall_min = stall_min + buy_soft_downtrend_stall_boost_min if soft_downtrend else stall_min
    if effective_stall_min < 0:
        effective_stall_min = 0
    effective_score_threshold = early_score_threshold + buy_soft_downtrend_score_boost if soft_downtrend else early_score_threshold

    rebound_max_ok = rebound_max_pct <= 0 or rebound <= rebound_max_pct
    rebound_ok = rebound >= effective_rebound_pct and rebound_max_ok
    low_age_ok = low_max_age_min <= 0 or low_age_min <= low_max_age_min
    micro_rebound_ok = micro_rebound_pct <= 0 or rebound >= micro_rebound_pct
    stall_ok = True
    if effective_stall_min > 0:
        stall_window = points[-effective_stall_min:] if effective_stall_min <= len(points) else points
        stall_low = min(p.price for p in stall_window) if stall_window else last
        stall_ok = (points[-1].ts - low_point.ts).total_seconds() / 60.0 >= effective_stall_min and stall_low >= low_point.price

    stall_progress = 1.0
    if effective_stall_min > 0:
        stall_progress = max(0.0, min(1.0, low_age_min / float(effective_stall_min)))

    if q_skip > 0 and last > q_skip_value:
        metrics = {
            "last": last,
            "low": low,
            "global_low": global_low,
            "rebound_pct": rebound,
            "slope_pct_per_min": slope_pct_per_min,
            "low_age_min": low_age_min,
            "global_low_age_min": global_low_age_min,
            "dist_from_global_low_pct": dist_from_global_low_pct,
            "session_low": session_low,
            "session_low_age_min": session_low_age_min,
            "dist_from_session_low_pct": dist_from_session_low_pct,
            "session_guard_low": session_guard_low,
            "session_guard_low_age_min": session_guard_low_age_min,
            "dist_from_session_guard_low_pct": dist_from_session_guard_low_pct,
            "q_low_value": q_low_value,
            "q_skip_value": q_skip_value,
            "quantile_window_min": float(quantile_window_min),
            "global_window_min": float(global_window_min),
            "htf_slope_pct_per_min": htf_slope_pct_per_min,
            "buy_htf_slope_window_min": float(buy_htf_slope_window_min),
            "buy_htf_slope_min_pct_per_min": buy_htf_slope_min_pct_per_min,
        }
        return Signal("BUY", False, f"价格高于{q_skip:.1f}分位过滤", metrics)

    if max_dist_from_session_low_pct > 0 and not session_guard_ok:
        metrics = {
            "last": last,
            "low": low,
            "global_low": global_low,
            "rebound_pct": rebound,
            "slope_pct_per_min": slope_pct_per_min,
            "low_age_min": low_age_min,
            "global_low_age_min": global_low_age_min,
            "dist_from_global_low_pct": dist_from_global_low_pct,
            "session_low": session_low,
            "session_low_age_min": session_low_age_min,
            "dist_from_session_low_pct": dist_from_session_low_pct,
            "session_guard_low": session_guard_low,
            "session_guard_low_age_min": session_guard_low_age_min,
            "dist_from_session_guard_low_pct": dist_from_session_guard_low_pct,
            "q_low_value": q_low_value,
            "q_skip_value": q_skip_value,
            "quantile_window_min": float(quantile_window_min),
            "global_window_min": float(global_window_min),
            "htf_slope_pct_per_min": htf_slope_pct_per_min,
            "buy_htf_slope_window_min": float(buy_htf_slope_window_min),
            "buy_htf_slope_min_pct_per_min": buy_htf_slope_min_pct_per_min,
        }
        return Signal("BUY", False, f"距离会话低点超过阈值{max_dist_from_session_low_pct:.3f}%过滤", metrics)
    if not session_guard_global_ok:
        metrics = {
            "last": last,
            "low": low,
            "global_low": global_low,
            "rebound_pct": rebound,
            "slope_pct_per_min": slope_pct_per_min,
            "low_age_min": low_age_min,
            "global_low_age_min": global_low_age_min,
            "dist_from_global_low_pct": dist_from_global_low_pct,
            "session_low": session_low,
            "session_low_age_min": session_low_age_min,
            "dist_from_session_low_pct": dist_from_session_low_pct,
            "session_guard_low": session_guard_low,
            "session_guard_low_age_min": session_guard_low_age_min,
            "dist_from_session_guard_low_pct": dist_from_session_guard_low_pct,
            "dist_score_ref_pct": dist_score_ref_pct,
            "dist_score_ref_is_session_guard": dist_score_ref_is_session_guard,
            "session_guard_global_cap_pct": session_guard_global_cap_pct,
            "q_low_value": q_low_value,
            "q_skip_value": q_skip_value,
            "quantile_window_min": float(quantile_window_min),
            "global_window_min": float(global_window_min),
            "htf_slope_pct_per_min": htf_slope_pct_per_min,
            "buy_htf_slope_window_min": float(buy_htf_slope_window_min),
            "buy_htf_slope_min_pct_per_min": buy_htf_slope_min_pct_per_min,
        }
        return Signal(
            "BUY",
            False,
            f"会话低点过老后，距离全局低点过远({dist_from_global_low_pct:.3f}%>{session_guard_global_cap_pct:.3f}%)过滤",
            metrics,
        )

    def _ratio_score(value: float, target: float, full: float) -> float:
        if target <= 0:
            return full
        if value <= 0:
            return 0.0
        return max(0.0, min(full, full * value / target))

    def _slope_score(value: float, threshold: float, full: float, tolerance: float = 0.03) -> float:
        if value >= threshold:
            return full
        lower = threshold - tolerance
        if value <= lower:
            return 0.0
        return full * (value - lower) / tolerance

    dist_score = 20.0
    if max_dist_from_global_low_pct > 0:
        dist_score = max(0.0, 30.0 * (1.0 - dist_score_ref_pct / max_dist_from_global_low_pct))
    low_zone_score = 8.0 if (q_low > 0 and last <= q_low_value) else 0.0
    near_global_score = 10.0 if near_global_low else 0.0
    stall_score = 10.0 * stall_progress
    micro_score = 12.0 if micro_rebound_ok else _ratio_score(rebound, micro_rebound_pct, 12.0)
    rebound_cap_score = 8.0 if rebound_max_ok else 0.0
    low_age_score = 10.0
    if low_max_age_min > 0 and not low_age_ok:
        decay = max(0.0, 1.0 - (low_age_min - low_max_age_min) / float(max(1, low_max_age_min)))
        low_age_score = 10.0 * decay
    low_slope_score = _slope_score(slope_pct_per_min, low_zone_slope_min_pct_per_min, 20.0)
    fresh_falling_penalty = 0.0
    if effective_stall_min > 0 and low_age_min < float(effective_stall_min) and not stall_ok:
        fresh_falling_penalty = 20.0
    downtrend_short_slope_penalty = 0.0
    if soft_downtrend and not soft_short_slope_ok and not near_session_low_bypass:
        downtrend_short_slope_penalty = 20.0

    low_zone = q_low > 0 and last <= q_low_value
    early_zone = low_zone or near_global_low
    if not early_zone:
        metrics = {
            "last": last,
            "low": low,
            "global_low": global_low,
            "rebound_pct": rebound,
            "slope_pct_per_min": slope_pct_per_min,
            "low_age_min": low_age_min,
            "global_low_age_min": global_low_age_min,
            "dist_from_global_low_pct": dist_from_global_low_pct,
            "session_low": session_low,
            "session_low_age_min": session_low_age_min,
            "dist_from_session_low_pct": dist_from_session_low_pct,
            "session_guard_low": session_guard_low,
            "session_guard_low_age_min": session_guard_low_age_min,
            "dist_from_session_guard_low_pct": dist_from_session_guard_low_pct,
            "q_low_value": q_low_value,
            "q_skip_value": q_skip_value,
            "quantile_window_min": float(quantile_window_min),
            "global_window_min": float(global_window_min),
            "htf_slope_pct_per_min": htf_slope_pct_per_min,
            "buy_htf_slope_window_min": float(buy_htf_slope_window_min),
            "buy_htf_slope_min_pct_per_min": buy_htf_slope_min_pct_per_min,
        }
        return Signal("BUY", False, "中分位通道已关闭，且当前不在低位提前区", metrics)

    signal_score = (
        dist_score
        + low_zone_score
        + near_global_score
        + stall_score
        + micro_score
        + rebound_cap_score
        + low_age_score
        + low_slope_score
        - fresh_falling_penalty
        - downtrend_short_slope_penalty
    )
    score_threshold = effective_score_threshold
    zone_label = "低位提前"
    open_scout_active = open_scout_min > 0 and 0.0 <= session_age_min <= float(open_scout_min)
    open_scout_eligible = (
        open_scout_active
        and early_zone
        and session_guard_ok
        and rebound_max_ok
        and (micro_rebound_ok or rebound_ok)
    )
    open_scout_threshold = open_scout_score_threshold
    if open_scout_threshold <= 0:
        open_scout_threshold = score_threshold
    if soft_downtrend:
        open_scout_threshold += buy_soft_downtrend_score_boost
    open_scout_should = open_scout_eligible and signal_score >= open_scout_threshold
    open_scout_used = open_scout_should and signal_score < score_threshold
    if open_scout_used:
        zone_label = "开盘抢底"
    should = signal_score >= score_threshold or open_scout_used

    metrics = {
        "last": last,
        "low": low,
        "global_low": global_low,
        "rebound_pct": rebound,
        "slope_pct_per_min": slope_pct_per_min,
        "low_age_min": low_age_min,
        "global_low_age_min": global_low_age_min,
        "dist_from_global_low_pct": dist_from_global_low_pct,
        "session_low": session_low,
        "session_low_age_min": session_low_age_min,
        "session_age_min": session_age_min,
        "dist_from_session_low_pct": dist_from_session_low_pct,
        "session_guard_low": session_guard_low,
        "session_guard_low_age_min": session_guard_low_age_min,
        "dist_from_session_guard_low_pct": dist_from_session_guard_low_pct,
        "q_low_value": q_low_value,
        "q_skip_value": q_skip_value,
        "quantile_window_min": float(quantile_window_min),
        "global_window_min": float(global_window_min),
        "signal_score": signal_score,
        "score_threshold": score_threshold,
        "score_dist": dist_score,
        "score_stall": stall_score,
        "score_micro": micro_score,
        "score_low_age": low_age_score,
        "score_low_slope": low_slope_score,
        "score_fresh_falling_penalty": fresh_falling_penalty,
        "score_downtrend_short_slope_penalty": downtrend_short_slope_penalty,
        "dist_score_ref_pct": dist_score_ref_pct,
        "dist_score_ref_is_session_guard": dist_score_ref_is_session_guard,
        "soft_downtrend": 1.0 if soft_downtrend else 0.0,
        "htf_slope_pct_per_min": htf_slope_pct_per_min,
        "buy_htf_slope_window_min": float(buy_htf_slope_window_min),
        "buy_htf_slope_min_pct_per_min": buy_htf_slope_min_pct_per_min,
        "buy_soft_downtrend_slope_pct_per_min": buy_soft_downtrend_slope_pct_per_min,
        "buy_soft_downtrend_short_slope_min_pct_per_min": buy_soft_downtrend_short_slope_min_pct_per_min,
        "effective_rebound_pct": effective_rebound_pct,
        "effective_stall_min": float(effective_stall_min),
        "base_early_score_threshold": early_score_threshold,
        "near_session_low_bypass": 1.0 if near_session_low_bypass else 0.0,
        "open_scout_active": 1.0 if open_scout_active else 0.0,
        "open_scout_eligible": 1.0 if open_scout_eligible else 0.0,
        "open_scout_threshold": open_scout_threshold,
        "is_open_scout": 1.0 if open_scout_used else 0.0,
    }

    low_age_part = ""
    if low_max_age_min > 0:
        low_age_part = f"低点距今{low_age_min:.1f}min<=阈值{low_max_age_min}"
    q_part = f"分位q{q_low_eff:.0f}/q{q_skip_eff:.0f}={q_low_value:.4f}/{q_skip_value:.4f}"
    session_part = f"距会话参考低点{dist_from_session_guard_low_pct:.3f}%"
    if max_dist_from_session_low_pct > 0:
        session_part += f"<=阈值{max_dist_from_session_low_pct:.3f}%={session_guard_ok}"
    if session_guard_low_point.ts != session_low_point.ts:
        session_part += (
            f"，原会话低点距今{session_low_age_min:.1f}min过老，改用近{session_low_guard_lookback_min}min参考低点"
        )
    htf_part = f"高周期斜率{htf_slope_pct_per_min:.4f}%/min"
    if htf_filter_enabled:
        htf_part += f"(硬阈值{buy_htf_slope_min_pct_per_min:.4f})"
    regime_part = ""
    if soft_downtrend:
        regime_part = (
            f"，下跌态加严[反弹阈值{effective_rebound_pct:.3f}%/停跌{effective_stall_min}min/"
            f"分数阈值+{buy_soft_downtrend_score_boost:.1f}/短斜率>={buy_soft_downtrend_short_slope_min_pct_per_min:.4f}]"
        )
    global_part = f"距全局低点{dist_from_global_low_pct:.3f}%"
    if max_dist_from_global_low_pct > 0:
        anchor_label = "会话参考低点" if dist_score_ref_is_session_guard >= 0.5 else "全局低点"
        global_part += (
            f"(评分锚值{max_dist_from_global_low_pct:.3f}%，"
            f"打分锚点={anchor_label}:{dist_score_ref_pct:.3f}%)"
        )
    if early_near_global_low_pct > 0:
        global_part += f"，近全局低点({early_near_global_low_pct:.3f}%){near_global_low}"
    age_part = f"，{low_age_part}" if low_age_part else ""
    reason = (
        f"{zone_label}：score={signal_score:.1f}/阈值{score_threshold:.1f}，{q_part}{age_part}，{session_part}，{global_part}，{htf_part}{regime_part}，"
        f"组件[低分位{low_zone_score:.1f}+近全局{near_global_score:.1f}+停跌{stall_score:.1f}+微反弹{micro_score:.1f}+"
        f"反弹上限{rebound_cap_score:.1f}+时效{low_age_score:.1f}+斜率{low_slope_score:.1f}-新低惩罚{fresh_falling_penalty:.1f}-下跌短斜率惩罚{downtrend_short_slope_penalty:.1f}]"
    )
    return Signal("BUY", should, reason, metrics)


def compute_sell_signal(
    points: list[PricePoint],
    *,
    high_window_min: int,
    slope_window_min: int,
    htf_slope_window_min: int,
    quantile_window_min: int,
    global_window_min: int,
    monitor_start_min: int,
    monitor_end_min: int,
    min_bars_to_start: int,
    open_scout_min: int,
    open_scout_score_threshold: float,
    q_high: float,
    q_skip_low: float,
    absolute_min_price: float,
    stall_min: int,
    micro_pullback_pct: float,
    near_global_high_pct: float,
    max_dist_from_global_high_pct: float,
    max_dist_from_session_high_pct: float,
    pullback_pct: float,
    pullback_max_pct: float,
    high_max_age_min: int,
    high_zone_slope_max_pct_per_min: float,
    htf_slope_max_pct_per_min: float,
    early_score_threshold: float,
) -> Signal:
    min_required = max(8, min_bars_to_start, slope_window_min, htf_slope_window_min)
    if len(points) < min_required:
        return Signal("SELL", False, "数据不足", {})

    if pullback_max_pct > 0 and pullback_max_pct < pullback_pct:
        pullback_max_pct = pullback_pct

    q_high = max(0.0, min(100.0, q_high))
    q_skip_low = max(0.0, min(100.0, q_skip_low))
    if q_skip_low > 0 and q_high > 0 and q_skip_low > q_high:
        q_skip_low = q_high
    q_high_eff = q_high if q_high > 0 else 75.0
    q_skip_low_eff = q_skip_low if q_skip_low > 0 else 60.0

    closes = [p.price for p in points]
    high_window_points = points[-high_window_min:]
    high_point = max(high_window_points, key=lambda p: p.price)
    high = high_point.price
    last = closes[-1]
    pullback = (high - last) / high * 100.0 if high != 0 else 0.0
    high_age_min = (points[-1].ts - high_point.ts).total_seconds() / 60.0

    if absolute_min_price > 0 and last < absolute_min_price:
        return Signal(
            "SELL",
            False,
            f"最新价{last:.4f}低于绝对价格下限{absolute_min_price:.4f}过滤",
            {
                "last": last,
                "high": high,
                "pullback_pct": pullback,
            },
        )

    if quantile_window_min <= 0:
        quantile_window = closes
    else:
        quantile_window = closes[-quantile_window_min:]
    if not quantile_window:
        quantile_window = [last]
    q_high_value = _quantile(quantile_window, q_high_eff)
    q_skip_low_value = _quantile(quantile_window, q_skip_low_eff)

    if global_window_min <= 0:
        global_window_points = points
    else:
        global_window_points = points[-global_window_min:]
    if not global_window_points:
        global_window_points = [points[-1]]
    global_high_point = max(global_window_points, key=lambda p: p.price)
    global_high = global_high_point.price
    dist_from_global_high_pct = (global_high - last) / global_high * 100.0 if global_high != 0 else 0.0
    global_high_age_min = (points[-1].ts - global_high_point.ts).total_seconds() / 60.0

    session_start = _session_start_anchor(points[-1].ts, start_min=monitor_start_min, end_min=monitor_end_min)
    session_age_min = (points[-1].ts - session_start).total_seconds() / 60.0
    session_points = [p for p in points if p.ts >= session_start]
    if not session_points:
        session_points = [points[-1]]
    session_high_point = max(session_points, key=lambda p: p.price)
    session_high = session_high_point.price
    dist_from_session_high_pct = (session_high - last) / session_high * 100.0 if session_high != 0 else 0.0
    session_high_age_min = (points[-1].ts - session_high_point.ts).total_seconds() / 60.0

    near_global_high = near_global_high_pct > 0 and dist_from_global_high_pct <= near_global_high_pct
    session_guard_ok = (
        max_dist_from_session_high_pct <= 0
        or dist_from_session_high_pct <= max_dist_from_session_high_pct
    )

    slope_window = closes[-slope_window_min:]
    slope_abs = linear_slope(slope_window)
    slope_pct_per_min = (slope_abs / last) * 100.0 if last != 0 else 0.0
    htf_slope_pct_per_min = slope_pct_per_min
    if htf_slope_window_min > 1:
        htf_window = closes[-htf_slope_window_min:]
        htf_slope_abs = linear_slope(htf_window)
        htf_slope_pct_per_min = (htf_slope_abs / last) * 100.0 if last != 0 else 0.0
    htf_filter_enabled = htf_slope_window_min > 1
    htf_slope_ok = (not htf_filter_enabled) or htf_slope_pct_per_min <= htf_slope_max_pct_per_min

    pullback_max_ok = pullback_max_pct <= 0 or pullback <= pullback_max_pct
    pullback_ok = pullback >= pullback_pct and pullback_max_ok
    high_age_ok = high_max_age_min <= 0 or high_age_min <= high_max_age_min
    micro_pullback_ok = micro_pullback_pct <= 0 or pullback >= micro_pullback_pct

    stall_ok = True
    if stall_min > 0:
        stall_window = points[-stall_min:] if stall_min <= len(points) else points
        stall_high = max(p.price for p in stall_window) if stall_window else last
        stall_ok = (points[-1].ts - high_point.ts).total_seconds() / 60.0 >= stall_min and stall_high <= high_point.price

    stall_progress = 1.0
    if stall_min > 0:
        stall_progress = max(0.0, min(1.0, high_age_min / float(stall_min)))

    if q_skip_low > 0 and last < q_skip_low_value:
        metrics = {
            "last": last,
            "high": high,
            "global_high": global_high,
            "pullback_pct": pullback,
            "slope_pct_per_min": slope_pct_per_min,
            "high_age_min": high_age_min,
            "global_high_age_min": global_high_age_min,
            "dist_from_global_high_pct": dist_from_global_high_pct,
            "session_high": session_high,
            "session_high_age_min": session_high_age_min,
            "dist_from_session_high_pct": dist_from_session_high_pct,
            "q_high_value": q_high_value,
            "q_skip_low_value": q_skip_low_value,
            "quantile_window_min": float(quantile_window_min),
            "global_window_min": float(global_window_min),
            "htf_slope_pct_per_min": htf_slope_pct_per_min,
            "htf_slope_window_min": float(htf_slope_window_min),
            "htf_slope_max_pct_per_min": htf_slope_max_pct_per_min,
        }
        return Signal("SELL", False, f"价格低于{q_skip_low:.1f}分位过滤", metrics)

    if max_dist_from_session_high_pct > 0 and not session_guard_ok:
        metrics = {
            "last": last,
            "high": high,
            "global_high": global_high,
            "pullback_pct": pullback,
            "slope_pct_per_min": slope_pct_per_min,
            "high_age_min": high_age_min,
            "global_high_age_min": global_high_age_min,
            "dist_from_global_high_pct": dist_from_global_high_pct,
            "session_high": session_high,
            "session_high_age_min": session_high_age_min,
            "dist_from_session_high_pct": dist_from_session_high_pct,
            "q_high_value": q_high_value,
            "q_skip_low_value": q_skip_low_value,
            "quantile_window_min": float(quantile_window_min),
            "global_window_min": float(global_window_min),
            "htf_slope_pct_per_min": htf_slope_pct_per_min,
            "htf_slope_window_min": float(htf_slope_window_min),
            "htf_slope_max_pct_per_min": htf_slope_max_pct_per_min,
        }
        return Signal("SELL", False, f"距离会话高点超过阈值{max_dist_from_session_high_pct:.3f}%过滤", metrics)
    if not htf_slope_ok:
        metrics = {
            "last": last,
            "high": high,
            "global_high": global_high,
            "pullback_pct": pullback,
            "slope_pct_per_min": slope_pct_per_min,
            "htf_slope_pct_per_min": htf_slope_pct_per_min,
            "htf_slope_window_min": float(htf_slope_window_min),
            "htf_slope_max_pct_per_min": htf_slope_max_pct_per_min,
            "high_age_min": high_age_min,
            "global_high_age_min": global_high_age_min,
            "dist_from_global_high_pct": dist_from_global_high_pct,
            "session_high": session_high,
            "session_high_age_min": session_high_age_min,
            "dist_from_session_high_pct": dist_from_session_high_pct,
            "q_high_value": q_high_value,
            "q_skip_low_value": q_skip_low_value,
            "quantile_window_min": float(quantile_window_min),
            "global_window_min": float(global_window_min),
        }
        return Signal(
            "SELL",
            False,
            f"高周期仍偏上行（{htf_slope_pct_per_min:.4f}%/min > 阈值{htf_slope_max_pct_per_min:.4f}）过滤",
            metrics,
        )

    def _ratio_score(value: float, target: float, full: float) -> float:
        if target <= 0:
            return full
        if value <= 0:
            return 0.0
        return max(0.0, min(full, full * value / target))

    def _slope_down_score(value: float, threshold: float, full: float, tolerance: float = 0.03) -> float:
        if value <= threshold:
            return full
        upper = threshold + tolerance
        if value >= upper:
            return 0.0
        return full * (upper - value) / tolerance

    dist_score = 20.0
    if max_dist_from_global_high_pct > 0:
        dist_score = max(0.0, 30.0 * (1.0 - dist_from_global_high_pct / max_dist_from_global_high_pct))
    high_zone_score = 8.0 if (q_high > 0 and last >= q_high_value) else 0.0
    near_global_score = 10.0 if near_global_high else 0.0
    stall_score = 10.0 * stall_progress
    micro_score = 12.0 if micro_pullback_ok else _ratio_score(pullback, micro_pullback_pct, 12.0)
    pullback_cap_score = 8.0 if pullback_max_ok else 0.0
    high_age_score = 10.0
    if high_max_age_min > 0 and not high_age_ok:
        decay = max(0.0, 1.0 - (high_age_min - high_max_age_min) / float(max(1, high_max_age_min)))
        high_age_score = 10.0 * decay
    high_slope_score = _slope_down_score(slope_pct_per_min, high_zone_slope_max_pct_per_min, 20.0)
    fresh_rising_penalty = 0.0
    if stall_min > 0 and high_age_min < float(stall_min) and not stall_ok:
        fresh_rising_penalty = 20.0

    high_zone = q_high > 0 and last >= q_high_value
    early_zone = high_zone or near_global_high
    if not early_zone:
        metrics = {
            "last": last,
            "high": high,
            "global_high": global_high,
            "pullback_pct": pullback,
            "slope_pct_per_min": slope_pct_per_min,
            "high_age_min": high_age_min,
            "global_high_age_min": global_high_age_min,
            "dist_from_global_high_pct": dist_from_global_high_pct,
            "session_high": session_high,
            "session_high_age_min": session_high_age_min,
            "dist_from_session_high_pct": dist_from_session_high_pct,
            "q_high_value": q_high_value,
            "q_skip_low_value": q_skip_low_value,
            "quantile_window_min": float(quantile_window_min),
            "global_window_min": float(global_window_min),
            "htf_slope_pct_per_min": htf_slope_pct_per_min,
            "htf_slope_window_min": float(htf_slope_window_min),
            "htf_slope_max_pct_per_min": htf_slope_max_pct_per_min,
        }
        return Signal("SELL", False, "中分位通道已关闭，且当前不在高位提前区", metrics)

    signal_score = (
        dist_score
        + high_zone_score
        + near_global_score
        + stall_score
        + micro_score
        + pullback_cap_score
        + high_age_score
        + high_slope_score
        - fresh_rising_penalty
    )
    score_threshold = early_score_threshold
    zone_label = "高位提前"

    open_scout_active = open_scout_min > 0 and 0.0 <= session_age_min <= float(open_scout_min)
    open_scout_eligible = (
        open_scout_active
        and early_zone
        and session_guard_ok
        and pullback_max_ok
        and (micro_pullback_ok or pullback_ok)
    )
    open_scout_threshold = open_scout_score_threshold
    if open_scout_threshold <= 0:
        open_scout_threshold = score_threshold
    open_scout_should = open_scout_eligible and signal_score >= open_scout_threshold
    open_scout_used = open_scout_should and signal_score < score_threshold
    if open_scout_used:
        zone_label = "开盘抢卖"
    should = signal_score >= score_threshold or open_scout_used

    metrics = {
        "last": last,
        "high": high,
        "global_high": global_high,
        "pullback_pct": pullback,
        "slope_pct_per_min": slope_pct_per_min,
        "high_age_min": high_age_min,
        "global_high_age_min": global_high_age_min,
        "dist_from_global_high_pct": dist_from_global_high_pct,
        "session_high": session_high,
        "session_high_age_min": session_high_age_min,
        "session_age_min": session_age_min,
        "dist_from_session_high_pct": dist_from_session_high_pct,
        "q_high_value": q_high_value,
        "q_skip_low_value": q_skip_low_value,
        "quantile_window_min": float(quantile_window_min),
        "global_window_min": float(global_window_min),
        "signal_score": signal_score,
        "score_threshold": score_threshold,
        "score_dist": dist_score,
        "score_stall": stall_score,
        "score_micro": micro_score,
        "score_high_age": high_age_score,
        "score_high_slope": high_slope_score,
        "score_fresh_rising_penalty": fresh_rising_penalty,
        "htf_slope_pct_per_min": htf_slope_pct_per_min,
        "htf_slope_window_min": float(htf_slope_window_min),
        "htf_slope_max_pct_per_min": htf_slope_max_pct_per_min,
        "open_scout_active": 1.0 if open_scout_active else 0.0,
        "open_scout_eligible": 1.0 if open_scout_eligible else 0.0,
        "open_scout_threshold": open_scout_threshold,
        "is_open_scout": 1.0 if open_scout_used else 0.0,
    }

    high_age_part = ""
    if high_max_age_min > 0:
        high_age_part = f"高点距今{high_age_min:.1f}min<=阈值{high_max_age_min}"
    q_part = f"分位q{q_skip_low_eff:.0f}/q{q_high_eff:.0f}={q_skip_low_value:.4f}/{q_high_value:.4f}"
    session_part = f"距会话高点{dist_from_session_high_pct:.3f}%"
    if max_dist_from_session_high_pct > 0:
        session_part += f"<=阈值{max_dist_from_session_high_pct:.3f}%={session_guard_ok}"
    global_part = f"距全局高点{dist_from_global_high_pct:.3f}%"
    if max_dist_from_global_high_pct > 0:
        global_part += f"(评分锚值{max_dist_from_global_high_pct:.3f}%)"
    if near_global_high_pct > 0:
        global_part += f"，近全局高点({near_global_high_pct:.3f}%){near_global_high}"

    age_part = f"，{high_age_part}" if high_age_part else ""
    reason = (
        f"{zone_label}：score={signal_score:.1f}/阈值{score_threshold:.1f}，{q_part}{age_part}，{session_part}，{global_part}，"
        f"组件[高分位{high_zone_score:.1f}+近全局{near_global_score:.1f}+停涨{stall_score:.1f}+微回撤{micro_score:.1f}+"
        f"回撤上限{pullback_cap_score:.1f}+时效{high_age_score:.1f}+斜率{high_slope_score:.1f}-新高惩罚{fresh_rising_penalty:.1f}]"
    )
    return Signal("SELL", should, reason, metrics)


@dataclass(frozen=True)
class BacktestAlert:
    side: str
    ts: datetime
    price: float
    low_window_ts: datetime
    low_window_price: float
    low_so_far_ts: datetime
    low_so_far_price: float
    reason: str
    metrics: dict[str, float]


def _min_price_point(points: list[PricePoint]) -> PricePoint:
    if not points:
        raise ValueError("points is empty")
    return min(points, key=lambda p: p.price)


def _write_csv(path: str, header: list[str], rows: list[list[Any]]) -> None:
    import csv

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _load_points_cache(path: str, *, tz: tzinfo) -> Optional[list[PricePoint]]:
    """从缓存加载数据点"""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            items = raw.get("points", [])
        else:
            return None

        points: list[PricePoint] = []
        for row in items:
            if not isinstance(row, dict):
                continue
            raw_ts = row.get("ts")
            price = row.get("price")
            if raw_ts is None or price is None:
                continue
            points.append(PricePoint(ts=datetime.fromisoformat(raw_ts).replace(tzinfo=tz), price=float(price)))
        if not points:
            return None
        return points
    except Exception as e:
        print(f"缓存加载失败：{e}",file=sys.stderr)
        return None


def _save_points_cache(path: str, points: list[PricePoint],tz="Asia/Shanghai") -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    raw = {
        "saved_at": datetime.now(ZoneInfo(tz)).isoformat(),
        "points": [{"ts": p.ts.isoformat(), "price": p.price} for p in points],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False)





def _plot_full_day(
    *,
    points: list[PricePoint],
    alerts: list[BacktestAlert],
    out_path: str,
    title: str,
    tz: tzinfo,
) -> bool:


    if not points:
        return False

    xs = [p.ts for p in points]
    ys = [p.price for p in points]
    low_p = _min_price_point(points)

    plt.figure(figsize=(12, 5))
    plt.plot(xs, ys, linewidth=1.2)
    plt.scatter([low_p.ts], [low_p.price], color="#2ca02c", s=40, zorder=5, label="Low")

    if alerts:
        buy_alerts = [a for a in alerts if a.side == "BUY"]
        sell_alerts = [a for a in alerts if a.side == "SELL"]
        if buy_alerts:
            plt.scatter([a.ts for a in buy_alerts], [a.price for a in buy_alerts], color="#d62728", s=30, zorder=6, label="Buy Alert")
        if sell_alerts:
            plt.scatter([a.ts for a in sell_alerts], [a.price for a in sell_alerts], color="#9467bd", s=30, zorder=6, label="Sell Alert")

    ax = plt.gca()
    try:
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=list(range(0, 24, 3)), tz=tz))
    except Exception:
        pass
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def _plot_alert_snapshot(
    *,
    points: list[PricePoint],
    alert: BacktestAlert,
    out_path: str,
    title: str,
    tz: tzinfo,
) -> bool:

    if not points:
        return False

    xs = [p.ts for p in points]
    ys = [p.price for p in points]

    plt.figure(figsize=(12, 5))
    plt.plot(xs, ys, linewidth=1.2)
    if alert.side == "SELL":
        plt.scatter([alert.low_so_far_ts], [alert.low_so_far_price], color="#2ca02c", s=45, zorder=6, label="High so far")
        plt.scatter([alert.low_window_ts], [alert.low_window_price], color="#1f77b4", s=40, zorder=6, label="High in window")
        plt.scatter([alert.ts], [alert.price], color="#9467bd", s=55, zorder=7, label="Sell Alert")
    else:
        plt.scatter([alert.low_so_far_ts], [alert.low_so_far_price], color="#2ca02c", s=45, zorder=6, label="Low so far")
        plt.scatter([alert.low_window_ts], [alert.low_window_price], color="#1f77b4", s=40, zorder=6, label="Low in window")
        plt.scatter([alert.ts], [alert.price], color="#d62728", s=55, zorder=7, label="Buy Alert")

    ax = plt.gca()
    try:
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=list(range(0, 24, 3)), tz=tz))
    except Exception:
        pass
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")

    movement = alert.metrics.get("rebound_pct", float("nan"))
    movement_label = "rebound"
    if alert.side == "SELL":
        movement = alert.metrics.get("pullback_pct", float("nan"))
        movement_label = "pullback"
    score = alert.metrics.get("signal_score")
    threshold = alert.metrics.get("score_threshold")
    margin = None
    if isinstance(score, (int, float)) and isinstance(threshold, (int, float)):
        margin = float(score) - float(threshold)
    dist_global = alert.metrics.get("dist_from_global_low_pct")
    if alert.side == "SELL":
        dist_global = alert.metrics.get("dist_from_global_high_pct")
    dist_session = alert.metrics.get("dist_from_session_guard_low_pct")
    if alert.side == "SELL":
        dist_session = alert.metrics.get("dist_from_session_high_pct")
    reason_tag = alert.reason.split("：", 1)[0] if "：" in alert.reason else "signal"
    line1 = (
        f"side={alert.side}  tag={reason_tag}  "
        f"{movement_label}={movement:.3f}%  "
        f"slope={alert.metrics.get('slope_pct_per_min', float('nan')):.4f}%/min"
    )
    line2 = "score=N/A"
    if margin is not None:
        line2 = f"score={float(score):.1f}/{float(threshold):.1f}  margin={margin:+.1f}"
    if isinstance(dist_global, (int, float)):
        line2 += f"  dist_global={float(dist_global):.3f}%"
    if isinstance(dist_session, (int, float)):
        line2 += f"  dist_session={float(dist_session):.3f}%"
    line3 = "reason_brief=" + textwrap.shorten(
        alert.reason.replace("\n", " "),
        width=140,
        placeholder=" ...",
    )
    info = "\n".join([line1, line2, line3])
    plt.gcf().text(0.01, 0.01, info, fontsize=8.5, va="bottom", ha="left")

    plt.tight_layout(rect=(0, 0.10, 1, 1))
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def _build_realtime_alert(points: list[PricePoint], cfg: RuntimeConfig, signal: Signal) -> BacktestAlert:
    if not points:
        raise ValueError("points is empty")
    last = points[-1]
    low_window_points = points[-cfg.low_window_min:] if cfg.low_window_min <= len(points) else points
    if signal.side == "SELL":
        low_window = max(low_window_points, key=lambda p: p.price)
        low_so_far = max(points, key=lambda p: p.price)
    else:
        low_window = min(low_window_points, key=lambda p: p.price)
        low_so_far = min(points, key=lambda p: p.price)
    return BacktestAlert(
        side=signal.side,
        ts=last.ts,
        price=last.price,
        low_window_ts=low_window.ts,
        low_window_price=low_window.price,
        low_so_far_ts=low_so_far.ts,
        low_so_far_price=low_so_far.price,
        reason=signal.reason,
        metrics=signal.metrics,
    )


def _render_realtime_alert_plot(points: list[PricePoint], cfg: RuntimeConfig, signal: Signal) -> Optional[str]:
    if not points:
        return None
    alert = _build_realtime_alert(points, cfg, signal)
    date_str = alert.ts.astimezone(cfg.monitor.tz).strftime("%Y-%m-%d")
    symbol_slug = _sanitize_symbol_for_path(cfg.symbol)
    out_dir = os.path.join(cfg.alert_plot_dir, symbol_slug, date_str)
    os.makedirs(out_dir, exist_ok=True)
    fn = f"alert_{signal.side.lower()}_{alert.ts.astimezone(cfg.monitor.tz).strftime('%Y-%m-%d_%H%M')}.png"
    out_path = os.path.join(out_dir, fn)
    ok = _plot_alert_snapshot(
        points=points,
        alert=alert,
        out_path=out_path,
        title=f"{cfg.symbol} realtime {signal.side} alert {alert.ts.strftime('%Y-%m-%d %H:%M %Z')}",
        tz=cfg.monitor.tz,
    )
    return out_path if ok else None


@dataclass
class EmailConfig:
    host: str
    port: int
    username: str
    password: str
    from_addr: str
    to_addrs: list[str]
    use_ssl: bool = True
    use_starttls: bool = False


def send_email(
    cfg: EmailConfig,
    *,
    subject: str,
    body: str,
    body_html: Optional[str] = None,
    attachments: Optional[list[str]] = None,
    inline_images: Optional[list[tuple[str, str]]] = None,
) -> None:
    msg = EmailMessage()
    msg["From"] = cfg.from_addr
    msg["To"] = ", ".join(cfg.to_addrs)
    msg["Subject"] = subject
    msg.set_content(body)
    if body_html:
        msg.add_alternative(body_html, subtype="html")
        if inline_images:
            html_part = msg.get_payload()[-1]
            for cid, path in inline_images:
                if not cid or not path:
                    continue
                if not os.path.exists(path):
                    print(f"内嵌图片不存在，已跳过：{path}", file=sys.stderr)
                    continue
                ctype, _ = mimetypes.guess_type(path)
                if not ctype:
                    ctype = "application/octet-stream"
                maintype, subtype = ctype.split("/", 1)
                with open(path, "rb") as f:
                    html_part.add_related(
                        f.read(),
                        maintype=maintype,
                        subtype=subtype,
                        cid=f"<{cid}>",
                        filename=os.path.basename(path),
                        disposition="inline",
                    )
    if attachments:
        for path in attachments:
            if not path:
                continue
            if not os.path.exists(path):
                print(f"附件不存在，已跳过：{path}", file=sys.stderr)
                continue
            ctype, _ = mimetypes.guess_type(path)
            if not ctype:
                ctype = "application/octet-stream"
            maintype, subtype = ctype.split("/", 1)
            with open(path, "rb") as f:
                msg.add_attachment(f.read(), maintype=maintype, subtype=subtype, filename=os.path.basename(path))

    if cfg.use_ssl:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(cfg.host, cfg.port, context=context, timeout=20) as s:
            s.login(cfg.username, cfg.password)
            s.send_message(msg)
        return

    with smtplib.SMTP(cfg.host, cfg.port, timeout=20) as s:
        s.ehlo()
        if cfg.use_starttls:
            context = ssl.create_default_context()
            s.starttls(context=context)
            s.ehlo()
        s.login(cfg.username, cfg.password)
        s.send_message(msg)


@dataclass
class RuntimeConfig:
    provider: str
    poll_seconds: int
    cooldown_minutes: int
    dry_run: bool
    symbol: str
    twelvedata_timezone: str
    state_file: str
    monitor: "MonitorWindow"
    alert_plot_on_email: bool
    alert_plot_dir: str
    # strategy
    low_window_min: int
    slope_window_min: int
    min_bars_to_start: int
    quantile_window_min: int
    global_window_min: int
    q_low: float
    q_skip: float
    absolute_max_price: float
    stall_min: int
    micro_rebound_pct: float
    early_near_global_low_pct: float
    max_dist_from_global_low_pct: float
    max_dist_from_session_low_pct: float
    buy_session_low_guard_max_age_min: int
    buy_session_low_guard_lookback_min: int
    rebound_pct: float
    rebound_max_pct: float
    low_max_age_min: int
    low_zone_slope_min_pct_per_min: float
    early_score_threshold: float
    open_scout_min: int
    open_scout_score_threshold: float
    open_scout_max_alerts_per_session: int
    buy_cooldown_bypass_on_fresh_low: bool
    buy_bypass_low_max_age_min: int
    buy_bypass_dist_from_session_low_pct: float
    buy_bypass_min_new_low_drop_pct: float
    buy_htf_slope_window_min: int
    buy_htf_slope_min_pct_per_min: float
    buy_soft_downtrend_slope_pct_per_min: float
    buy_soft_downtrend_rebound_boost_pct: float
    buy_soft_downtrend_stall_boost_min: int
    buy_soft_downtrend_score_boost: float
    buy_soft_downtrend_short_slope_min_pct_per_min: float
    buy_hard_filter_near_session_low_bypass_pct: float
    buy_hard_filter_bypass_rebound_pct: float
    buy_first_alert_min_rebound_pct: float
    buy_first_alert_max_dist_from_session_low_pct: float
    buy_first_alert_min_low_age_min: int
    buy_first_alert_max_dist_with_age_pct: float
    # sell strategy
    sell_enabled: bool
    sell_q_high: float
    sell_q_skip_low: float
    sell_absolute_min_price: float
    sell_stall_min: int
    sell_micro_pullback_pct: float
    sell_near_global_high_pct: float
    sell_max_dist_from_global_high_pct: float
    sell_max_dist_from_session_high_pct: float
    sell_pullback_pct: float
    sell_pullback_max_pct: float
    sell_high_max_age_min: int
    sell_high_zone_slope_max_pct_per_min: float
    sell_htf_slope_window_min: int
    sell_htf_slope_max_pct_per_min: float
    sell_rearm_on_new_high_pct: float
    sell_early_score_threshold: float
    sell_open_scout_min: int
    sell_open_scout_score_threshold: float
    sell_open_scout_max_alerts_per_session: int


@dataclass(frozen=True)
class MonitorWindow:
    tz: tzinfo
    tz_name: str
    start_min: int  # minutes from midnight; 0..1440
    end_min: int  # minutes from midnight; 0..1440 (allow 1440 as 24:00)


def _parse_hhmm_to_minutes(value: str) -> int:
    v = value.strip()
    if v == "":
        raise ValueError("时间不能为空")
    if v == "24:00":
        return 1440
    parts = v.split(":")
    if len(parts) != 2:
        raise ValueError(f"时间格式应为 HH:MM（如 09:00），收到：{value!r}")
    hour = int(parts[0])
    minute = int(parts[1])
    if not (0 <= hour <= 23):
        raise ValueError(f"小时应在 0..23，收到：{value!r}")
    if not (0 <= minute <= 59):
        raise ValueError(f"分钟应在 0..59，收到：{value!r}")
    return hour * 60 + minute


def _resolve_tz(name: str) -> tzinfo:
    raw = name.strip()
    if raw == "" or raw.lower() == "local":
        return _local_tz()

    if ZoneInfo is not None:
        try:
            return ZoneInfo(raw)
        except Exception:
            pass

    s = raw.upper()
    if s.startswith("UTC"):
        s = s[3:]
    if not s:
        return timezone.utc

    sign = 1
    if s[0] == "+":
        s = s[1:]
    elif s[0] == "-":
        sign = -1
        s = s[1:]

    if ":" in s:
        h_str, m_str = s.split(":", 1)
        hours = int(h_str)
        minutes = int(m_str)
    else:
        hours = int(s)
        minutes = 0
    if not (0 <= hours <= 23) or not (0 <= minutes <= 59):
        raise ValueError(f"无法解析 MONITOR_TZ：{name!r}")
    return timezone(sign * timedelta(hours=hours, minutes=minutes))


def _load_monitor_window() -> MonitorWindow:
    tz_name = os.environ.get("MONITOR_TZ", "").strip() or "local"
    start_raw = os.environ.get("MONITOR_START", "").strip()
    end_raw = os.environ.get("MONITOR_END", "").strip()

    if start_raw == "" and end_raw == "":
        start_min = 0
        end_min = 1440
    else:
        start_min = _parse_hhmm_to_minutes(start_raw) if start_raw else 0
        end_min = _parse_hhmm_to_minutes(end_raw) if end_raw else 1440

    tz = _resolve_tz(tz_name)
    return MonitorWindow(tz=tz, tz_name=tz_name, start_min=start_min, end_min=end_min)


def _is_within_window(now: datetime, window: MonitorWindow) -> bool:
    start_min = window.start_min
    end_min = window.end_min
    if start_min == 0 and end_min == 1440:
        return True

    current = now.hour * 60 + now.minute + (now.second / 60.0)
    if start_min < end_min:
        return start_min <= current < end_min
    return current >= start_min or current < end_min


def _dt_at_minutes(now: datetime, *, minutes: int) -> datetime:
    base = datetime(now.year, now.month, now.day, 0, 0, 0, tzinfo=now.tzinfo)
    if minutes == 1440:
        return base + timedelta(days=1)
    return base + timedelta(minutes=minutes)


def _format_minutes(minutes: int) -> str:
    if minutes == 1440:
        return "24:00"
    return f"{minutes//60:02d}:{minutes%60:02d}"


def _next_window_start(now: datetime, window: MonitorWindow) -> datetime:
    start_min = window.start_min
    end_min = window.end_min
    current = now.hour * 60 + now.minute + (now.second / 60.0)
    if start_min == 0 and end_min == 1440:
        return now

    # Non-wrapping window: [start, end)
    if start_min < end_min:
        if current < start_min:
            return _dt_at_minutes(now, minutes=start_min)
        # current >= end_min (or between end and start, but that's outside)
        return _dt_at_minutes(now, minutes=start_min) + timedelta(days=1)

    # Wrapping window: [start, 24:00) U [0, end)
    # Outside means end <= current < start -> next start is today at start.
    return _dt_at_minutes(now, minutes=start_min)


def _load_runtime_config() -> Tuple[RuntimeConfig, dict[str, str]]:
    provider = os.environ.get("PROVIDER", "twelvedata").strip().lower()
    poll_seconds = _parse_int(os.environ.get("POLL_SECONDS"), 60)
    cooldown_minutes = _parse_int(os.environ.get("COOLDOWN_MINUTES"), 120)
    dry_run = _parse_bool(os.environ.get("DRY_RUN"), False)

    symbol = os.environ.get("TWELVEDATA_SYMBOL", "XAU/USD")
    twelvedata_timezone = os.environ.get("TWELVEDATA_TIMEZONE", "").strip()
    state_file = os.environ.get("STATE_FILE", ".goldx_state.json").strip()
    monitor = _load_monitor_window()

    low_window_min = _parse_int(os.environ.get("LOW_WINDOW_MIN"), 60)
    slope_window_min = _parse_int(os.environ.get("SLOPE_WINDOW_MIN"), 10)
    min_bars_to_start = _parse_int(os.environ.get("MIN_BARS_TO_START"), 20)
    quantile_window_min = _parse_int(os.environ.get("QUANTILE_WINDOW_MIN"), 240)
    global_window_min = _parse_int(os.environ.get("GLOBAL_WINDOW_MIN"), 1440)
    q_low = _parse_float(os.environ.get("Q_LOW"), 25)
    q_skip = _parse_float(os.environ.get("Q_SKIP"), 60)
    absolute_max_price = _parse_float(os.environ.get("ABSOLUTE_MAX_PRICE"), 0)
    stall_min = _parse_int(os.environ.get("STALL_MIN"), 4)
    micro_rebound_pct = _parse_float(os.environ.get("MICRO_REBOUND_PCT"), 0.03)
    early_near_global_low_pct = _parse_float(os.environ.get("EARLY_NEAR_GLOBAL_LOW_PCT"), 0.10)
    max_dist_from_global_low_pct = _parse_float(os.environ.get("MAX_DIST_FROM_GLOBAL_LOW_PCT"), 0.25)
    max_dist_from_session_low_pct = _parse_float(
        os.environ.get("MAX_DIST_FROM_SESSION_LOW_PCT"),
        max_dist_from_global_low_pct,
    )
    buy_session_low_guard_max_age_min = _parse_int(os.environ.get("BUY_SESSION_LOW_GUARD_MAX_AGE_MIN"), 240)
    buy_session_low_guard_lookback_min = _parse_int(os.environ.get("BUY_SESSION_LOW_GUARD_LOOKBACK_MIN"), 240)
    rebound_pct = _parse_float(os.environ.get("REBOUND_PCT"), 0.15)
    rebound_max_pct = _parse_float(os.environ.get("REBOUND_MAX_PCT"), 0.5)
    low_max_age_min = _parse_int(os.environ.get("LOW_MAX_AGE_MIN"), 20)
    low_zone_slope_min = _parse_float(os.environ.get("LOW_ZONE_SLOPE_MIN_PCT_PER_MIN"), -0.003)
    early_score_threshold = _parse_float(os.environ.get("EARLY_SCORE_THRESHOLD"), 75)
    open_scout_min = _parse_int(os.environ.get("OPEN_SCOUT_MIN"), 45)
    open_scout_score_threshold = _parse_float(os.environ.get("OPEN_SCOUT_SCORE_THRESHOLD"), 72)
    open_scout_max_alerts_per_session = _parse_int(os.environ.get("OPEN_SCOUT_MAX_ALERTS_PER_SESSION"), 1)
    buy_cooldown_bypass_on_fresh_low = _parse_bool(os.environ.get("BUY_COOLDOWN_BYPASS_ON_FRESH_LOW"), False)
    buy_bypass_low_max_age_min = _parse_int(os.environ.get("BUY_BYPASS_LOW_MAX_AGE_MIN"), low_max_age_min)
    buy_bypass_dist_from_session_low_pct = _parse_float(
        os.environ.get("BUY_BYPASS_DIST_FROM_SESSION_LOW_PCT"),
        max_dist_from_session_low_pct,
    )
    buy_bypass_min_new_low_drop_pct = _parse_float(os.environ.get("BUY_BYPASS_MIN_NEW_LOW_DROP_PCT"), 0.05)
    buy_htf_slope_window_min = _parse_int(os.environ.get("BUY_HTF_SLOPE_WINDOW_MIN"), 60)
    buy_htf_slope_min_pct_per_min = _parse_float(os.environ.get("BUY_HTF_SLOPE_MIN_PCT_PER_MIN"), -1.0)
    buy_soft_downtrend_slope_pct_per_min = _parse_float(
        os.environ.get("BUY_SOFT_DOWNTREND_SLOPE_PCT_PER_MIN"),
        -1.0,
    )
    buy_soft_downtrend_rebound_boost_pct = _parse_float(
        os.environ.get("BUY_SOFT_DOWNTREND_REBOUND_BOOST_PCT"),
        0.0,
    )
    buy_soft_downtrend_stall_boost_min = _parse_int(
        os.environ.get("BUY_SOFT_DOWNTREND_STALL_BOOST_MIN"),
        0,
    )
    buy_soft_downtrend_score_boost = _parse_float(
        os.environ.get("BUY_SOFT_DOWNTREND_SCORE_BOOST"),
        0,
    )
    buy_soft_downtrend_short_slope_min_pct_per_min = _parse_float(
        os.environ.get("BUY_SOFT_DOWNTREND_SHORT_SLOPE_MIN_PCT_PER_MIN"),
        -1.0,
    )
    buy_hard_filter_near_session_low_bypass_pct = _parse_float(
        os.environ.get("BUY_HARD_FILTER_NEAR_SESSION_LOW_BYPASS_PCT"),
        0.05,
    )
    buy_hard_filter_bypass_rebound_pct = _parse_float(
        os.environ.get("BUY_HARD_FILTER_BYPASS_REBOUND_PCT"),
        0.12,
    )
    buy_first_alert_min_rebound_pct = _parse_float(os.environ.get("BUY_FIRST_ALERT_MIN_REBOUND_PCT"), 0.12)
    buy_first_alert_max_dist_from_session_low_pct = _parse_float(
        os.environ.get("BUY_FIRST_ALERT_MAX_DIST_FROM_SESSION_LOW_PCT"),
        0.05,
    )
    buy_first_alert_min_low_age_min = _parse_int(os.environ.get("BUY_FIRST_ALERT_MIN_LOW_AGE_MIN"), 20)
    buy_first_alert_max_dist_with_age_pct = _parse_float(
        os.environ.get("BUY_FIRST_ALERT_MAX_DIST_WITH_AGE_PCT"),
        0.10,
    )
    sell_enabled = _parse_bool(os.environ.get("SELL_ENABLED"), False)
    sell_q_high = _parse_float(os.environ.get("SELL_Q_HIGH"), 80)
    sell_q_skip_low = _parse_float(os.environ.get("SELL_Q_SKIP_LOW"), 60)
    sell_absolute_min_price = _parse_float(os.environ.get("SELL_ABSOLUTE_MIN_PRICE"), 0)
    sell_stall_min = _parse_int(os.environ.get("SELL_STALL_MIN"), stall_min)
    sell_micro_pullback_pct = _parse_float(os.environ.get("SELL_MICRO_PULLBACK_PCT"), micro_rebound_pct)
    sell_near_global_high_pct = _parse_float(os.environ.get("SELL_NEAR_GLOBAL_HIGH_PCT"), early_near_global_low_pct)
    sell_max_dist_from_global_high_pct = _parse_float(
        os.environ.get("SELL_MAX_DIST_FROM_GLOBAL_HIGH_PCT"),
        max_dist_from_global_low_pct,
    )
    sell_max_dist_from_session_high_pct = _parse_float(
        os.environ.get("SELL_MAX_DIST_FROM_SESSION_HIGH_PCT"),
        max_dist_from_session_low_pct,
    )
    sell_pullback_pct = _parse_float(os.environ.get("SELL_PULLBACK_PCT"), rebound_pct)
    sell_pullback_max_pct = _parse_float(os.environ.get("SELL_PULLBACK_MAX_PCT"), rebound_max_pct)
    sell_high_max_age_min = _parse_int(os.environ.get("SELL_HIGH_MAX_AGE_MIN"), low_max_age_min)
    sell_high_zone_slope_max_pct_per_min = _parse_float(
        os.environ.get("SELL_HIGH_ZONE_SLOPE_MAX_PCT_PER_MIN"),
        -low_zone_slope_min,
    )
    sell_htf_slope_window_min = _parse_int(os.environ.get("SELL_HTF_SLOPE_WINDOW_MIN"), 60)
    sell_htf_slope_max_pct_per_min = _parse_float(os.environ.get("SELL_HTF_SLOPE_MAX_PCT_PER_MIN"), 0.01)
    sell_rearm_on_new_high_pct = _parse_float(os.environ.get("SELL_REARM_ON_NEW_HIGH_PCT"), 0.10)
    sell_early_score_threshold = _parse_float(os.environ.get("SELL_EARLY_SCORE_THRESHOLD"), early_score_threshold)
    sell_open_scout_min = _parse_int(os.environ.get("SELL_OPEN_SCOUT_MIN"), open_scout_min)
    sell_open_scout_score_threshold = _parse_float(
        os.environ.get("SELL_OPEN_SCOUT_SCORE_THRESHOLD"),
        open_scout_score_threshold,
    )
    sell_open_scout_max_alerts_per_session = _parse_int(
        os.environ.get("SELL_OPEN_SCOUT_MAX_ALERTS_PER_SESSION"),
        open_scout_max_alerts_per_session,
    )
    alert_plot_on_email = _parse_bool(os.environ.get("ALERT_PLOT_ON_EMAIL"), True)
    alert_plot_dir = os.environ.get("ALERT_PLOT_DIR", "alert_out").strip() or "alert_out"

    cfg = RuntimeConfig(
        provider=provider,
        poll_seconds=poll_seconds,
        cooldown_minutes=cooldown_minutes,
        dry_run=dry_run,
        symbol=symbol,
        twelvedata_timezone=twelvedata_timezone,
        state_file=state_file,
        monitor=monitor,
        alert_plot_on_email=alert_plot_on_email,
        alert_plot_dir=alert_plot_dir,
        low_window_min=low_window_min,
        slope_window_min=slope_window_min,
        min_bars_to_start=min_bars_to_start,
        quantile_window_min=quantile_window_min,
        global_window_min=global_window_min,
        q_low=q_low,
        q_skip=q_skip,
        absolute_max_price=absolute_max_price,
        stall_min=stall_min,
        micro_rebound_pct=micro_rebound_pct,
        early_near_global_low_pct=early_near_global_low_pct,
        max_dist_from_global_low_pct=max_dist_from_global_low_pct,
        max_dist_from_session_low_pct=max_dist_from_session_low_pct,
        buy_session_low_guard_max_age_min=buy_session_low_guard_max_age_min,
        buy_session_low_guard_lookback_min=buy_session_low_guard_lookback_min,
        rebound_pct=rebound_pct,
        rebound_max_pct=rebound_max_pct,
        low_max_age_min=low_max_age_min,
        low_zone_slope_min_pct_per_min=low_zone_slope_min,
        early_score_threshold=early_score_threshold,
        open_scout_min=open_scout_min,
        open_scout_score_threshold=open_scout_score_threshold,
        open_scout_max_alerts_per_session=open_scout_max_alerts_per_session,
        buy_cooldown_bypass_on_fresh_low=buy_cooldown_bypass_on_fresh_low,
        buy_bypass_low_max_age_min=buy_bypass_low_max_age_min,
        buy_bypass_dist_from_session_low_pct=buy_bypass_dist_from_session_low_pct,
        buy_bypass_min_new_low_drop_pct=buy_bypass_min_new_low_drop_pct,
        buy_htf_slope_window_min=buy_htf_slope_window_min,
        buy_htf_slope_min_pct_per_min=buy_htf_slope_min_pct_per_min,
        buy_soft_downtrend_slope_pct_per_min=buy_soft_downtrend_slope_pct_per_min,
        buy_soft_downtrend_rebound_boost_pct=buy_soft_downtrend_rebound_boost_pct,
        buy_soft_downtrend_stall_boost_min=buy_soft_downtrend_stall_boost_min,
        buy_soft_downtrend_score_boost=buy_soft_downtrend_score_boost,
        buy_soft_downtrend_short_slope_min_pct_per_min=buy_soft_downtrend_short_slope_min_pct_per_min,
        buy_hard_filter_near_session_low_bypass_pct=buy_hard_filter_near_session_low_bypass_pct,
        buy_hard_filter_bypass_rebound_pct=buy_hard_filter_bypass_rebound_pct,
        buy_first_alert_min_rebound_pct=buy_first_alert_min_rebound_pct,
        buy_first_alert_max_dist_from_session_low_pct=buy_first_alert_max_dist_from_session_low_pct,
        buy_first_alert_min_low_age_min=buy_first_alert_min_low_age_min,
        buy_first_alert_max_dist_with_age_pct=buy_first_alert_max_dist_with_age_pct,
        sell_enabled=sell_enabled,
        sell_q_high=sell_q_high,
        sell_q_skip_low=sell_q_skip_low,
        sell_absolute_min_price=sell_absolute_min_price,
        sell_stall_min=sell_stall_min,
        sell_micro_pullback_pct=sell_micro_pullback_pct,
        sell_near_global_high_pct=sell_near_global_high_pct,
        sell_max_dist_from_global_high_pct=sell_max_dist_from_global_high_pct,
        sell_max_dist_from_session_high_pct=sell_max_dist_from_session_high_pct,
        sell_pullback_pct=sell_pullback_pct,
        sell_pullback_max_pct=sell_pullback_max_pct,
        sell_high_max_age_min=sell_high_max_age_min,
        sell_high_zone_slope_max_pct_per_min=sell_high_zone_slope_max_pct_per_min,
        sell_htf_slope_window_min=sell_htf_slope_window_min,
        sell_htf_slope_max_pct_per_min=sell_htf_slope_max_pct_per_min,
        sell_rearm_on_new_high_pct=sell_rearm_on_new_high_pct,
        sell_early_score_threshold=sell_early_score_threshold,
        sell_open_scout_min=sell_open_scout_min,
        sell_open_scout_score_threshold=sell_open_scout_score_threshold,
        sell_open_scout_max_alerts_per_session=sell_open_scout_max_alerts_per_session,
    )

    secrets: dict[str, str] = {}
    if provider == "twelvedata":
        secrets["TWELVEDATA_API_KEYS"] = os.environ.get("TWELVEDATA_API_KEYS", "").strip()
        secrets["TWELVEDATA_API_KEY"] = os.environ.get("TWELVEDATA_API_KEY", "").strip()
    return cfg, secrets


def _load_email_config() -> EmailConfig:
    host = os.environ.get("SMTP_HOST", "").strip()
    port = _parse_int(os.environ.get("SMTP_PORT"), 465)
    username = os.environ.get("SMTP_USER", "").strip()
    password = os.environ.get("SMTP_PASSWORD", "").strip()
    from_addr = os.environ.get("SMTP_FROM", username).strip()
    to_raw = os.environ.get("SMTP_TO", "").strip()
    to_addrs = [a.strip() for a in to_raw.split(",") if a.strip()]

    use_ssl = _parse_bool(os.environ.get("SMTP_SSL"), True)
    use_starttls = _parse_bool(os.environ.get("SMTP_STARTTLS"), False)
    if use_ssl and use_starttls:
        raise ValueError("请只启用 SMTP_SSL 或 SMTP_STARTTLS 其中一个")

    if not host or not username or not password or not from_addr or not to_addrs:
        raise ValueError("缺少邮箱配置：SMTP_HOST/SMTP_USER/SMTP_PASSWORD/SMTP_FROM/SMTP_TO")

    return EmailConfig(
        host=host,
        port=port,
        username=username,
        password=password,
        from_addr=from_addr,
        to_addrs=to_addrs,
        use_ssl=use_ssl,
        use_starttls=use_starttls,
    )


def _metric_number(metrics: dict[str, float], key: str) -> Optional[float]:
    value = metrics.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _signal_recommendation(signal: Signal) -> tuple[str, str]:
    score = _metric_number(signal.metrics, "signal_score")
    threshold = _metric_number(signal.metrics, "score_threshold")
    if score is None or threshold is None:
        return "N", "信息不足"
    margin = score - threshold
    if margin >= 12:
        return "A", "强"
    if margin >= 7:
        return "B", "较强"
    if margin >= 3:
        return "C", "中等"
    return "D", "谨慎"


def _recommendation_action(side: str, grade: str) -> str:
    if side == "SELL":
        if grade in {"A", "B"}:
            return "建议优先分批止盈/减仓。"
        if grade == "C":
            return "建议小幅减仓并等待后续确认。"
        return "建议继续观察，避免过早离场。"
    if grade in {"A", "B"}:
        return "建议分批买入，避免一次性重仓。"
    if grade == "C":
        return "建议小仓位试探，等待二次确认。"
    return "建议继续观察，等待更强确认。"


def _score_breakdown_items(signal: Signal) -> list[tuple[str, float]]:
    if signal.side == "SELL":
        items = [
            ("位置贴近高位", "score_dist", 1.0),
            ("停涨确认", "score_stall", 1.0),
            ("微回撤确认", "score_micro", 1.0),
            ("高点时效", "score_high_age", 1.0),
            ("斜率确认", "score_high_slope", 1.0),
            ("新高惩罚", "score_fresh_rising_penalty", -1.0),
        ]
    else:
        items = [
            ("位置贴近低位", "score_dist", 1.0),
            ("停跌确认", "score_stall", 1.0),
            ("微反弹确认", "score_micro", 1.0),
            ("低点时效", "score_low_age", 1.0),
            ("斜率确认", "score_low_slope", 1.0),
            ("新低惩罚", "score_fresh_falling_penalty", -1.0),
            ("下跌短斜率惩罚", "score_downtrend_short_slope_penalty", -1.0),
        ]
    breakdown: list[tuple[str, float]] = []
    for label, key, sign in items:
        value = _metric_number(signal.metrics, key)
        if value is None:
            continue
        breakdown.append((label, sign * value))
    return breakdown


def _score_breakdown_lines(signal: Signal) -> list[str]:
    lines: list[str] = []
    for label, shown in _score_breakdown_items(signal):
        lines.append(f"- {label}: {shown:+.1f}")
    return lines


def _format_email_body(
    *,
    now: datetime,
    symbol: str,
    points: list[PricePoint],
    signal: Signal,
    tz: tzinfo,
) -> str:
    last_point = points[-1]
    now_tz = now.astimezone(tz)
    ts_str = last_point.ts.astimezone(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
    grade, grade_text = _signal_recommendation(signal)
    score = _metric_number(signal.metrics, "signal_score")
    threshold = _metric_number(signal.metrics, "score_threshold")
    margin_str = "N/A"
    score_line = "综合得分：N/A"
    if score is not None and threshold is not None:
        margin = score - threshold
        margin_str = f"{margin:+.1f}"
        score_line = f"综合得分：{score:.1f} / 阈值{threshold:.1f}（差值 {margin_str}）"
    channel = signal.reason.split("：", 1)[0] if "：" in signal.reason else "未知通道"
    side_cn = "卖出" if signal.side == "SELL" else "买入"
    lines = [
        "【黄金交易提醒】",
        f"方向：{side_cn}（{signal.side}）",
        f"推荐等级：{grade}（{grade_text}）",
        score_line,
        f"触发通道：{channel}",
        f"时间：{now_tz.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"数据源：{symbol}（1min）",
        f"最新价格：{signal.metrics.get('last', float('nan')):.6f}",
    ]
    if signal.side == "SELL":
        lines.extend(
            [
                f"近{int(os.environ.get('LOW_WINDOW_MIN','60'))}分钟最高：{signal.metrics.get('high', float('nan')):.6f}",
                f"回撤幅度：{signal.metrics.get('pullback_pct', float('nan')):.3f}%",
                f"短周期斜率：{signal.metrics.get('slope_pct_per_min', float('nan')):.4f}%/min",
            ]
        )
    else:
        lines.extend(
            [
                f"近{int(os.environ.get('LOW_WINDOW_MIN','60'))}分钟最低：{signal.metrics.get('low', float('nan')):.6f}",
                f"反弹幅度：{signal.metrics.get('rebound_pct', float('nan')):.3f}%",
                f"短周期斜率：{signal.metrics.get('slope_pct_per_min', float('nan')):.4f}%/min",
            ]
        )
    lines.append("")
    lines.append("分项得分：")
    score_lines = _score_breakdown_lines(signal)
    if score_lines:
        lines.extend(score_lines)
    else:
        lines.append("- 暂无分项得分")
    lines.extend(
        [
            "",
            f"推荐动作：{_recommendation_action(signal.side, grade)}",
            f"触发细则：{signal.reason}",
            f"(最近一根K线时间戳：{ts_str})",
        ]
    )
    return "\n".join(lines)


def _format_email_html(
    *,
    now: datetime,
    symbol: str,
    points: list[PricePoint],
    signal: Signal,
    tz: tzinfo,
    plot_cid: Optional[str] = None,
) -> str:
    last_point = points[-1]
    now_tz = now.astimezone(tz)
    ts_str = last_point.ts.astimezone(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
    grade, grade_text = _signal_recommendation(signal)
    score = _metric_number(signal.metrics, "signal_score")
    threshold = _metric_number(signal.metrics, "score_threshold")
    score_line = "N/A"
    if score is not None and threshold is not None:
        score_line = f"{score:.1f} / {threshold:.1f}（差值 {score-threshold:+.1f}）"
    side_cn = "卖出" if signal.side == "SELL" else "买入"
    side_color = "#b91c1c" if signal.side == "SELL" else "#166534"
    card_bg = "#fff5f5" if signal.side == "SELL" else "#f0fdf4"
    channel = signal.reason.split("：", 1)[0] if "：" in signal.reason else "未知通道"
    movement_label = "回撤幅度" if signal.side == "SELL" else "反弹幅度"
    movement_value = signal.metrics.get("pullback_pct", float("nan")) if signal.side == "SELL" else signal.metrics.get("rebound_pct", float("nan"))
    extreme_label = "近窗口最高" if signal.side == "SELL" else "近窗口最低"
    extreme_value = signal.metrics.get("high", float("nan")) if signal.side == "SELL" else signal.metrics.get("low", float("nan"))
    breakdown_rows = "".join(
        f"<tr><td style='padding:6px 10px;border-bottom:1px solid #e5e7eb;'>{html.escape(label)}</td>"
        f"<td style='padding:6px 10px;border-bottom:1px solid #e5e7eb;text-align:right;'>{shown:+.1f}</td></tr>"
        for label, shown in _score_breakdown_items(signal)
    )
    if not breakdown_rows:
        breakdown_rows = (
            "<tr><td colspan='2' style='padding:8px 10px;color:#6b7280;'>暂无分项得分</td></tr>"
        )
    image_html = ""
    if plot_cid:
        image_html = (
            "<div style='margin-top:16px;'>"
            "<div style='font-size:14px;font-weight:600;margin-bottom:8px;color:#111827;'>本次提醒图</div>"
            f"<img src='cid:{html.escape(plot_cid)}' alt='alert plot' "
            "style='max-width:100%;border:1px solid #d1d5db;border-radius:8px;'/>"
            "</div>"
        )
    return (
        "<html><body style='margin:0;padding:16px;background:#f3f4f6;font-family:-apple-system,BlinkMacSystemFont,\"Segoe UI\",Arial,sans-serif;'>"
        f"<div style='max-width:860px;margin:0 auto;background:#ffffff;border:1px solid #e5e7eb;border-radius:10px;overflow:hidden;'>"
        f"<div style='background:{card_bg};padding:14px 16px;border-bottom:1px solid #e5e7eb;'>"
        "<div style='font-size:20px;font-weight:700;color:#111827;'>黄金交易提醒</div>"
        f"<div style='margin-top:6px;color:{side_color};font-size:16px;font-weight:700;'>方向：{side_cn}（{html.escape(signal.side)}）</div>"
        "</div>"
        "<div style='padding:16px;'>"
        f"<div style='font-size:14px;line-height:1.8;color:#111827;'>"
        f"<div><b>推荐等级：</b><span style='color:{side_color};font-weight:700'>{html.escape(grade)}（{html.escape(grade_text)}）</span></div>"
        f"<div><b>综合得分/阈值：</b>{html.escape(score_line)}</div>"
        f"<div><b>触发通道：</b>{html.escape(channel)}</div>"
        f"<div><b>时间：</b>{html.escape(now_tz.strftime('%Y-%m-%d %H:%M:%S %Z'))}</div>"
        f"<div><b>数据源：</b>{html.escape(symbol)}（1min）</div>"
        f"<div><b>最新价格：</b>{signal.metrics.get('last', float('nan')):.6f}</div>"
        f"<div><b>{extreme_label}：</b>{extreme_value:.6f}</div>"
        f"<div><b>{movement_label}：</b>{movement_value:.3f}%</div>"
        f"<div><b>短周期斜率：</b>{signal.metrics.get('slope_pct_per_min', float('nan')):.4f}%/min</div>"
        "</div>"
        "<div style='margin-top:14px;font-size:14px;font-weight:600;color:#111827;'>分项得分</div>"
        "<table style='width:100%;margin-top:8px;border-collapse:collapse;font-size:13px;color:#111827;'>"
        "<thead><tr><th style='text-align:left;padding:6px 10px;background:#f9fafb;border-bottom:1px solid #e5e7eb;'>项</th>"
        "<th style='text-align:right;padding:6px 10px;background:#f9fafb;border-bottom:1px solid #e5e7eb;'>分值</th></tr></thead>"
        f"<tbody>{breakdown_rows}</tbody></table>"
        f"<div style='margin-top:14px;font-size:14px;'><b>推荐动作：</b>{html.escape(_recommendation_action(signal.side, grade))}</div>"
        f"<div style='margin-top:10px;font-size:13px;color:#374151;line-height:1.6;'><b>触发细则：</b>{html.escape(signal.reason)}</div>"
        f"{image_html}"
        f"<div style='margin-top:14px;font-size:12px;color:#6b7280;'>最近一根K线时间戳：{html.escape(ts_str)}</div>"
        "</div></div></body></html>"
    )


def main(argv: list[str]) -> int:
    load_dotenv()
    once = "--once" in argv
    test_email = "--test-email" in argv
    backtest = "--backtest" in argv
    show_help = ("-h" in argv) or ("--help" in argv)

    cfg, secrets = _load_runtime_config()

    if show_help:
        print(
            "用法：\n"
            "  监测并发邮件：python3 goldx_alert.py\n"
            "  只跑一次：  python3 goldx_alert.py --once\n"
            "  测试邮箱：  python3 goldx_alert.py --test-email\n"
            "  回测（模拟）：python3 goldx_alert.py --backtest [--date YYYY-MM-DD] [--start HH:MM] [--end HH:MM]\n"
            "\n"
            "回测常用参数：\n"
            "  --outdir backtest_out     输出目录（默认 backtest_out）\n"
            "  --outputsize 5000         历史拉取条数上限（默认 5000）\n"
            "  --warmup-min 0            计算预热分钟数（默认 0）\n"
            "  --refresh-cache           忽略缓存，强制重新拉取数据\n"
            "  --no-cache                不读写缓存\n"
        )
        return 0

    if test_email:
        try:
            email_cfg = _load_email_config()
        except Exception as e:
            print(f"邮箱配置错误：{e}", file=sys.stderr)
            return 2
        now = datetime.now(cfg.monitor.tz)
        subject = "goldx-alert 测试邮件"
        body = (
            "这是一封测试邮件，用于验证 SMTP 配置是否可用。\n"
            f"时间：{now.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
        )
        if cfg.dry_run:
            print("DRY_RUN=1：将发送测试邮件但已跳过\n" + body)
        else:
            send_email(email_cfg, subject=subject, body=body)
            print("已发送测试邮件")
        return 0

    if cfg.provider != "twelvedata":
        print(f"暂仅支持 PROVIDER=twelvedata；当前={cfg.provider}", file=sys.stderr)
        return 2

    api_keys = _parse_api_keys(secrets.get("TWELVEDATA_API_KEYS")) or _parse_api_keys(secrets.get("TWELVEDATA_API_KEY"))
    if not api_keys:
        print("缺少 TWELVEDATA_API_KEYS 或 TWELVEDATA_API_KEY（建议放到 .env）", file=sys.stderr)
        return 2
    api_key_index = 0

    def _is_key_likely_bad(err: Exception) -> bool:
        msg = str(err).lower()
        keywords = [
            "limit",
            "quota",
            "credits",
            "rate",
            "too many",
            "429",
            "unauthorized",
            "forbidden",
            "invalid",
            "apikey",
            "api key",
        ]
        return any(k in msg for k in keywords)

    def fetch_points_with_key_failover(
        api_keys: list[str],
        cfg: RuntimeConfig,
        outputsize: int,
        api_key_index: int = 0,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> tuple[list[PricePoint], int]:
        last_err: Optional[Exception] = None
        for offset in range(len(api_keys)):
            idx = (api_key_index + offset) % len(api_keys)
            key = api_keys[idx]
            try:
                points = fetch_twelvedata_1min_series(
                    api_key=key,
                    symbol=cfg.symbol,
                    interval="1min",
                    outputsize=outputsize,
                    tz=cfg.monitor.tz,
                    timezone_name=cfg.twelvedata_timezone or "Asia/Shanghai",
                    start_date=start_date,
                    end_date=end_date,
                )
                # Success: move to next key on next poll to spread quota usage.
                next_idx = (idx + 1) % len(api_keys)
                return points, next_idx
            except Exception as e:
                last_err = e
                stamp = datetime.now(cfg.monitor.tz).strftime("%H:%M:%S %Z")
                if _is_key_likely_bad(e):
                    print(f"{stamp},API_KEY{idx} 可能受限/失效，：{e}", file=sys.stderr)
                    continue
                # 非配额/鉴权类错误也尝试下一个 key（可能是临时网络问题）
                print(f"{stamp},API_KEY{idx} 请求失败，尝试切换：{e}", file=sys.stderr)
                continue
        assert last_err is not None
        raise last_err

    def _signal_margin(signal: Signal) -> float:
        score = signal.metrics.get("signal_score")
        threshold = signal.metrics.get("score_threshold")
        if isinstance(score, (int, float)) and isinstance(threshold, (int, float)):
            return float(score) - float(threshold)
        return 0.0

    def _compute_buy(points: list[PricePoint]) -> Signal:
        return compute_buy_signal(
            points,
            low_window_min=cfg.low_window_min,
            slope_window_min=cfg.slope_window_min,
            min_bars_to_start=cfg.min_bars_to_start,
            quantile_window_min=cfg.quantile_window_min,
            global_window_min=cfg.global_window_min,
            monitor_start_min=cfg.monitor.start_min,
            monitor_end_min=cfg.monitor.end_min,
            open_scout_min=cfg.open_scout_min,
            open_scout_score_threshold=cfg.open_scout_score_threshold,
            q_low=cfg.q_low,
            q_skip=cfg.q_skip,
            absolute_max_price=cfg.absolute_max_price,
            stall_min=cfg.stall_min,
            micro_rebound_pct=cfg.micro_rebound_pct,
            early_near_global_low_pct=cfg.early_near_global_low_pct,
            max_dist_from_global_low_pct=cfg.max_dist_from_global_low_pct,
            max_dist_from_session_low_pct=cfg.max_dist_from_session_low_pct,
            session_low_guard_max_age_min=cfg.buy_session_low_guard_max_age_min,
            session_low_guard_lookback_min=cfg.buy_session_low_guard_lookback_min,
            rebound_pct=cfg.rebound_pct,
            rebound_max_pct=cfg.rebound_max_pct,
            low_max_age_min=cfg.low_max_age_min,
            low_zone_slope_min_pct_per_min=cfg.low_zone_slope_min_pct_per_min,
            early_score_threshold=cfg.early_score_threshold,
            buy_htf_slope_window_min=cfg.buy_htf_slope_window_min,
            buy_htf_slope_min_pct_per_min=cfg.buy_htf_slope_min_pct_per_min,
            buy_soft_downtrend_slope_pct_per_min=cfg.buy_soft_downtrend_slope_pct_per_min,
            buy_soft_downtrend_rebound_boost_pct=cfg.buy_soft_downtrend_rebound_boost_pct,
            buy_soft_downtrend_stall_boost_min=cfg.buy_soft_downtrend_stall_boost_min,
            buy_soft_downtrend_score_boost=cfg.buy_soft_downtrend_score_boost,
            buy_soft_downtrend_short_slope_min_pct_per_min=cfg.buy_soft_downtrend_short_slope_min_pct_per_min,
            buy_hard_filter_near_session_low_bypass_pct=cfg.buy_hard_filter_near_session_low_bypass_pct,
            buy_hard_filter_bypass_rebound_pct=cfg.buy_hard_filter_bypass_rebound_pct,
        )

    def _compute_sell(points: list[PricePoint]) -> Signal:
        return compute_sell_signal(
            points,
            high_window_min=cfg.low_window_min,
            slope_window_min=cfg.slope_window_min,
            htf_slope_window_min=cfg.sell_htf_slope_window_min,
            min_bars_to_start=cfg.min_bars_to_start,
            quantile_window_min=cfg.quantile_window_min,
            global_window_min=cfg.global_window_min,
            monitor_start_min=cfg.monitor.start_min,
            monitor_end_min=cfg.monitor.end_min,
            open_scout_min=cfg.sell_open_scout_min,
            open_scout_score_threshold=cfg.sell_open_scout_score_threshold,
            q_high=cfg.sell_q_high,
            q_skip_low=cfg.sell_q_skip_low,
            absolute_min_price=cfg.sell_absolute_min_price,
            stall_min=cfg.sell_stall_min,
            micro_pullback_pct=cfg.sell_micro_pullback_pct,
            near_global_high_pct=cfg.sell_near_global_high_pct,
            max_dist_from_global_high_pct=cfg.sell_max_dist_from_global_high_pct,
            max_dist_from_session_high_pct=cfg.sell_max_dist_from_session_high_pct,
            pullback_pct=cfg.sell_pullback_pct,
            pullback_max_pct=cfg.sell_pullback_max_pct,
            high_max_age_min=cfg.sell_high_max_age_min,
            high_zone_slope_max_pct_per_min=cfg.sell_high_zone_slope_max_pct_per_min,
            htf_slope_max_pct_per_min=cfg.sell_htf_slope_max_pct_per_min,
            early_score_threshold=cfg.sell_early_score_threshold,
        )

    def _choose_signal(points: list[PricePoint]) -> Signal:
        buy_signal = _compute_buy(points)
        if not cfg.sell_enabled:
            return buy_signal
        sell_signal = _compute_sell(points)
        if sell_signal.should_alert and not buy_signal.should_alert:
            return sell_signal
        if buy_signal.should_alert and not sell_signal.should_alert:
            return buy_signal
        if buy_signal.should_alert and sell_signal.should_alert:
            return sell_signal if _signal_margin(sell_signal) > _signal_margin(buy_signal) else buy_signal
        return buy_signal

    def _can_bypass_buy_cooldown(signal: Signal, last_buy_alert_session_low: Optional[float]) -> bool:
        if signal.side != "BUY" or not cfg.buy_cooldown_bypass_on_fresh_low:
            return False
        session_low_value = signal.metrics.get("session_low")
        if not isinstance(session_low_value, (int, float)):
            return False
        current_session_low = float(session_low_value)
        dist_value = signal.metrics.get("dist_from_session_low_pct")
        if cfg.buy_bypass_dist_from_session_low_pct > 0:
            if not isinstance(dist_value, (int, float)) or float(dist_value) > cfg.buy_bypass_dist_from_session_low_pct:
                return False
        low_age_value = signal.metrics.get("low_age_min")
        if cfg.buy_bypass_low_max_age_min > 0:
            if not isinstance(low_age_value, (int, float)) or float(low_age_value) > cfg.buy_bypass_low_max_age_min:
                return False
        if cfg.buy_bypass_min_new_low_drop_pct > 0 and last_buy_alert_session_low is not None:
            need_low = last_buy_alert_session_low * (1 - cfg.buy_bypass_min_new_low_drop_pct / 100.0)
            if current_session_low > need_low:
                return False
        return True

    def _can_trigger_first_buy_in_session(signal: Signal, has_buy_alert_in_session: bool) -> tuple[bool, str]:
        if signal.side != "BUY" or has_buy_alert_in_session:
            return True, ""
        rebound_value = signal.metrics.get("rebound_pct")
        dist_value = signal.metrics.get("dist_from_session_guard_low_pct")
        if not isinstance(dist_value, (int, float)):
            dist_value = signal.metrics.get("dist_from_session_low_pct")
        low_age_value = signal.metrics.get("low_age_min")
        if not isinstance(rebound_value, (int, float)) or not isinstance(dist_value, (int, float)):
            return True, ""
        rebound_pct_value = float(rebound_value)
        dist_pct_value = float(dist_value)
        low_age_min_value = float(low_age_value) if isinstance(low_age_value, (int, float)) else -1.0
        if rebound_pct_value >= cfg.buy_first_alert_min_rebound_pct:
            return True, ""
        if dist_pct_value <= cfg.buy_first_alert_max_dist_from_session_low_pct:
            return True, ""
        if (
            cfg.buy_first_alert_min_low_age_min > 0
            and low_age_min_value >= cfg.buy_first_alert_min_low_age_min
            and dist_pct_value <= cfg.buy_first_alert_max_dist_with_age_pct
        ):
            return True, ""
        return (
            False,
            (
                f"首个买点保护未通过：rebound={rebound_pct_value:.3f}%/"
                f"dist={dist_pct_value:.3f}%/low_age={low_age_min_value:.1f}min"
            ),
        )

    def _handle_backtest(argv: list[str], cfg: RuntimeConfig, api_keys: list[str]) -> int:
        """处理回测逻辑"""
        date_str = _get_arg_value(argv, "--date")
        start_str = _get_arg_value(argv, "--start", "09:00") or "09:00"
        end_str = _get_arg_value(argv, "--end", "22:00") or "22:00"
        out_root = _get_arg_value(argv, "--outdir", "backtest_out") or "backtest_out"
        outputsize = int(_get_arg_value(argv, "--outputsize", "5000") or "5000")
        warmup_min = int(_get_arg_value(argv, "--warmup-min", "0") or "0")
        refresh_cache = "--refresh-cache" in argv
        no_cache = "--no-cache" in argv
        cache_root = _get_arg_value(argv, "--cache-dir", "backtest_cache") or "backtest_cache"

        tz = cfg.monitor.tz
        bt_date = _parse_date(date_str) if date_str else _today_in_tz(tz)
        start_min = _parse_hhmm_to_minutes(start_str)
        end_min = _parse_hhmm_to_minutes(end_str)
        base = datetime(bt_date.year, bt_date.month, bt_date.day, 0, 0, 0, tzinfo=tz)
        eval_start = base + timedelta(minutes=start_min)
        # Include the candle at END time.
        if end_min == 1440:
            eval_end_excl = base + timedelta(days=1)
        else:
            eval_end_excl = base + timedelta(minutes=end_min) + timedelta(minutes=1)
        prefetch_min = max(
            0,
            warmup_min,
            cfg.low_window_min,
            cfg.slope_window_min,
            cfg.quantile_window_min,
            cfg.global_window_min,
            cfg.stall_min,
            cfg.min_bars_to_start,
        )
        warmup_start = eval_start - timedelta(minutes=max(0, warmup_min))
        fetch_start = eval_start - timedelta(minutes=prefetch_min)
        fetch_end_inclusive = eval_end_excl - timedelta(minutes=1)
        fetch_start_str = fetch_start.strftime("%Y-%m-%d %H:%M:%S")
        fetch_end_str = fetch_end_inclusive.strftime("%Y-%m-%d %H:%M:%S")

        symbol_slug = _sanitize_symbol_for_path(cfg.symbol)
        out_dir = os.path.join(out_root, symbol_slug, bt_date.isoformat())
        os.makedirs(out_dir, exist_ok=True)

        points: Optional[list[PricePoint]] = None
        
        cache_path = os.path.join(cache_root, symbol_slug, f"{bt_date.isoformat()}.json")
        if not no_cache and not refresh_cache:
            points = _load_points_cache(cache_path, tz=tz)
            if points:
                points.sort(key=lambda p: p.ts)
                cache_start = points[0].ts
                cache_end = points[-1].ts
                cache_ok = cache_start <= fetch_start and cache_end >= fetch_end_inclusive
                if cache_ok:
                    print(f"已从缓存加载数据：{cache_path}")
                else:
                    print(
                        f"缓存覆盖不足，将重拉：{cache_path} "
                        f"(cache={cache_start.isoformat()}~{cache_end.isoformat()} "
                        f"need={fetch_start.isoformat()}~{fetch_end_inclusive.isoformat()})"
                    )
                    points = None

        if not points:
            api_key_index = 0
            print(
                f"拉取K线：symbol={cfg.symbol} interval=1min outputsize={outputsize} "
                f"start_date={fetch_start_str} end_date={fetch_end_str}"
            )
            points, api_key_index = fetch_points_with_key_failover(
                api_keys,
                cfg,
                outputsize,
                start_date=fetch_start_str,
                end_date=fetch_end_str,
            )
            if not no_cache:
                _save_points_cache(cache_path, points, tz=cfg.monitor.tz_name)
                print(f"已写入缓存：{cache_path}")

 
        points = [p for p in points if warmup_start <= p.ts < eval_end_excl]
        points.sort(key=lambda p: p.ts)
        eval_points = [p for p in points if eval_start <= p.ts < eval_end_excl]
        if len(eval_points) < 10:
            print(
                "回测范围内数据太少，可能是 outputsize 不够或数据源缺失。\n"
                f"范围：{eval_start.isoformat()} ~ {eval_end_excl.isoformat()}  (points={len(eval_points)})\n"
                f"建议：提高 --outputsize，或换一个日期/品种。",
                file=sys.stderr,
            )
            return 2


        window_size = max(
            80,
            cfg.low_window_min,
            cfg.slope_window_min,
            cfg.quantile_window_min,
            cfg.global_window_min,
            cfg.stall_min,
        )
        cooldown = timedelta(minutes=max(0, cfg.cooldown_minutes))

        alerts: list[BacktestAlert] = []
        low_so_far: Optional[PricePoint] = None
        high_so_far: Optional[PricePoint] = None
        last_alert_ts: Optional[datetime] = None
        current_session_anchor: Optional[datetime] = None
        buy_open_scout_alerts_in_session = 0
        sell_open_scout_alerts_in_session = 0
        last_buy_alert_session_low: Optional[float] = None
        sell_rearm_high_in_session: Optional[float] = None
        seen: list[PricePoint] = []

        for p in points:
            if p.ts >= eval_end_excl:
                break
            seen.append(p)
            if p.ts < eval_start:
                continue
            if low_so_far is None or p.price < low_so_far.price:
                low_so_far = p
            if high_so_far is None or p.price > high_so_far.price:
                high_so_far = p

            session_anchor = _session_start_anchor(
                p.ts,
                start_min=cfg.monitor.start_min,
                end_min=cfg.monitor.end_min,
            )
            if current_session_anchor != session_anchor:
                current_session_anchor = session_anchor
                buy_open_scout_alerts_in_session = 0
                sell_open_scout_alerts_in_session = 0
                last_buy_alert_session_low = None
                sell_rearm_high_in_session = None

            window = seen[-window_size:]
            signal = _choose_signal(window)
            if not signal.should_alert:
                continue
            if (
                signal.side == "BUY"
                and last_buy_alert_session_low is not None
                and signal.metrics.get("dist_score_ref_is_session_guard", 0.0) >= 0.5
            ):
                continue
            first_buy_ok, _ = _can_trigger_first_buy_in_session(signal, last_buy_alert_session_low is not None)
            if not first_buy_ok:
                continue
            cooldown_active = last_alert_ts is not None and (p.ts - last_alert_ts) < cooldown
            if cooldown_active and not _can_bypass_buy_cooldown(signal, last_buy_alert_session_low):
                continue
            is_open_scout = signal.metrics.get("is_open_scout", 0.0) >= 0.5
            if is_open_scout and signal.side == "BUY":
                if (
                    cfg.open_scout_max_alerts_per_session > 0
                    and buy_open_scout_alerts_in_session >= cfg.open_scout_max_alerts_per_session
                ):
                    continue
            if is_open_scout and signal.side == "SELL":
                if (
                    cfg.sell_open_scout_max_alerts_per_session > 0
                    and sell_open_scout_alerts_in_session >= cfg.sell_open_scout_max_alerts_per_session
                ):
                    continue
            if signal.side == "SELL" and cfg.sell_rearm_on_new_high_pct > 0:
                session_high_value = signal.metrics.get("session_high")
                if isinstance(session_high_value, (int, float)):
                    current_session_high = float(session_high_value)
                    if sell_rearm_high_in_session is not None:
                        rearm_required = sell_rearm_high_in_session * (1 + cfg.sell_rearm_on_new_high_pct / 100.0)
                        if current_session_high < rearm_required:
                            continue

            if signal.side == "SELL":
                ref_window = max(window[-cfg.low_window_min:], key=lambda point: point.price)
                assert high_so_far is not None
                ref_so_far = high_so_far
            else:
                ref_window = _min_price_point(window[-cfg.low_window_min:])
                assert low_so_far is not None
                ref_so_far = low_so_far
            alert = BacktestAlert(
                side=signal.side,
                ts=p.ts,
                price=p.price,
                low_window_ts=ref_window.ts,
                low_window_price=ref_window.price,
                low_so_far_ts=ref_so_far.ts,
                low_so_far_price=ref_so_far.price,
                reason=signal.reason,
                metrics=signal.metrics,
            )
            alerts.append(alert)
            last_alert_ts = p.ts
            if signal.side == "BUY":
                session_low_value = signal.metrics.get("session_low")
                if isinstance(session_low_value, (int, float)):
                    last_buy_alert_session_low = float(session_low_value)
            if signal.side == "SELL":
                session_high_value = signal.metrics.get("session_high")
                if isinstance(session_high_value, (int, float)):
                    sell_rearm_high_in_session = float(session_high_value)
            if is_open_scout:
                if signal.side == "SELL":
                    sell_open_scout_alerts_in_session += 1
                else:
                    buy_open_scout_alerts_in_session += 1

        # Write outputs
        header = [
            "side",
            "alert_time",
            "price",
            "low_window_time",
            "low_window_price",
            "low_so_far_time",
            "low_so_far_price",
            "movement_pct",
            "slope_pct_per_min",
            "macd_hist_last",
            "reason",
        ]
        rows: list[list[Any]] = []
        for a in alerts:
            rows.append(
                [
                    a.side,
                    a.ts.strftime("%Y-%m-%d %H:%M:%S %Z"),
                    a.price,
                    a.low_window_ts.strftime("%Y-%m-%d %H:%M:%S %Z"),
                    a.low_window_price,
                    a.low_so_far_ts.strftime("%Y-%m-%d %H:%M:%S %Z"),
                    a.low_so_far_price,
                    a.metrics.get("rebound_pct") if a.side == "BUY" else a.metrics.get("pullback_pct"),
                    a.metrics.get("slope_pct_per_min"),
                    a.metrics.get("macd_hist_last"),
                    a.reason,
                ]
            )
        csv_path = os.path.join(out_dir, "alerts.csv")
        _write_csv(csv_path, header, rows)

        summary = {
            "symbol": cfg.symbol,
            "tz": cfg.monitor.tz_name,
            "date": bt_date.isoformat(),
            "start": start_str,
            "end": end_str,
            "warmup_min": warmup_min,
            "fetch_range": {
                "start_date": fetch_start_str,
                "end_date": fetch_end_str,
                "prefetch_min": prefetch_min,
                "outputsize": outputsize,
            },
            "data_range": {
                "first": points[0].ts.isoformat(),
                "last": points[-1].ts.isoformat(),
                "eval_first": eval_points[0].ts.isoformat(),
                "eval_last": eval_points[-1].ts.isoformat(),
            },
            "points_in_range": len(eval_points),
            "alerts": len(alerts),
            "window_size": window_size,
            "params": {
                "LOW_WINDOW_MIN": cfg.low_window_min,
                "SLOPE_WINDOW_MIN": cfg.slope_window_min,
                "MIN_BARS_TO_START": cfg.min_bars_to_start,
                "QUANTILE_WINDOW_MIN": cfg.quantile_window_min,
                "GLOBAL_WINDOW_MIN": cfg.global_window_min,
                "Q_LOW": cfg.q_low,
                "Q_SKIP": cfg.q_skip,
                "ABSOLUTE_MAX_PRICE": cfg.absolute_max_price,
                "STALL_MIN": cfg.stall_min,
                "MICRO_REBOUND_PCT": cfg.micro_rebound_pct,
                "EARLY_NEAR_GLOBAL_LOW_PCT": cfg.early_near_global_low_pct,
                "MAX_DIST_FROM_GLOBAL_LOW_PCT": cfg.max_dist_from_global_low_pct,
                "MAX_DIST_FROM_SESSION_LOW_PCT": cfg.max_dist_from_session_low_pct,
                "BUY_SESSION_LOW_GUARD_MAX_AGE_MIN": cfg.buy_session_low_guard_max_age_min,
                "BUY_SESSION_LOW_GUARD_LOOKBACK_MIN": cfg.buy_session_low_guard_lookback_min,
                "REBOUND_PCT": cfg.rebound_pct,
                "REBOUND_MAX_PCT": cfg.rebound_max_pct,
                "LOW_MAX_AGE_MIN": cfg.low_max_age_min,
                "LOW_ZONE_SLOPE_MIN_PCT_PER_MIN": cfg.low_zone_slope_min_pct_per_min,
                "EARLY_SCORE_THRESHOLD": cfg.early_score_threshold,
                "OPEN_SCOUT_MIN": cfg.open_scout_min,
                "OPEN_SCOUT_SCORE_THRESHOLD": cfg.open_scout_score_threshold,
                "OPEN_SCOUT_MAX_ALERTS_PER_SESSION": cfg.open_scout_max_alerts_per_session,
                "BUY_COOLDOWN_BYPASS_ON_FRESH_LOW": cfg.buy_cooldown_bypass_on_fresh_low,
                "BUY_BYPASS_LOW_MAX_AGE_MIN": cfg.buy_bypass_low_max_age_min,
                "BUY_BYPASS_DIST_FROM_SESSION_LOW_PCT": cfg.buy_bypass_dist_from_session_low_pct,
                "BUY_BYPASS_MIN_NEW_LOW_DROP_PCT": cfg.buy_bypass_min_new_low_drop_pct,
                "BUY_HTF_SLOPE_WINDOW_MIN": cfg.buy_htf_slope_window_min,
                "BUY_HTF_SLOPE_MIN_PCT_PER_MIN": cfg.buy_htf_slope_min_pct_per_min,
                "BUY_SOFT_DOWNTREND_SLOPE_PCT_PER_MIN": cfg.buy_soft_downtrend_slope_pct_per_min,
                "BUY_SOFT_DOWNTREND_REBOUND_BOOST_PCT": cfg.buy_soft_downtrend_rebound_boost_pct,
                "BUY_SOFT_DOWNTREND_STALL_BOOST_MIN": cfg.buy_soft_downtrend_stall_boost_min,
                "BUY_SOFT_DOWNTREND_SCORE_BOOST": cfg.buy_soft_downtrend_score_boost,
                "BUY_SOFT_DOWNTREND_SHORT_SLOPE_MIN_PCT_PER_MIN": cfg.buy_soft_downtrend_short_slope_min_pct_per_min,
                "BUY_HARD_FILTER_NEAR_SESSION_LOW_BYPASS_PCT": cfg.buy_hard_filter_near_session_low_bypass_pct,
                "BUY_HARD_FILTER_BYPASS_REBOUND_PCT": cfg.buy_hard_filter_bypass_rebound_pct,
                "BUY_FIRST_ALERT_MIN_REBOUND_PCT": cfg.buy_first_alert_min_rebound_pct,
                "BUY_FIRST_ALERT_MAX_DIST_FROM_SESSION_LOW_PCT": cfg.buy_first_alert_max_dist_from_session_low_pct,
                "BUY_FIRST_ALERT_MIN_LOW_AGE_MIN": cfg.buy_first_alert_min_low_age_min,
                "BUY_FIRST_ALERT_MAX_DIST_WITH_AGE_PCT": cfg.buy_first_alert_max_dist_with_age_pct,
                "SELL_ENABLED": cfg.sell_enabled,
                "SELL_Q_HIGH": cfg.sell_q_high,
                "SELL_Q_SKIP_LOW": cfg.sell_q_skip_low,
                "SELL_ABSOLUTE_MIN_PRICE": cfg.sell_absolute_min_price,
                "SELL_STALL_MIN": cfg.sell_stall_min,
                "SELL_MICRO_PULLBACK_PCT": cfg.sell_micro_pullback_pct,
                "SELL_NEAR_GLOBAL_HIGH_PCT": cfg.sell_near_global_high_pct,
                "SELL_MAX_DIST_FROM_GLOBAL_HIGH_PCT": cfg.sell_max_dist_from_global_high_pct,
                "SELL_MAX_DIST_FROM_SESSION_HIGH_PCT": cfg.sell_max_dist_from_session_high_pct,
                "SELL_PULLBACK_PCT": cfg.sell_pullback_pct,
                "SELL_PULLBACK_MAX_PCT": cfg.sell_pullback_max_pct,
                "SELL_HIGH_MAX_AGE_MIN": cfg.sell_high_max_age_min,
                "SELL_HIGH_ZONE_SLOPE_MAX_PCT_PER_MIN": cfg.sell_high_zone_slope_max_pct_per_min,
                "SELL_HTF_SLOPE_WINDOW_MIN": cfg.sell_htf_slope_window_min,
                "SELL_HTF_SLOPE_MAX_PCT_PER_MIN": cfg.sell_htf_slope_max_pct_per_min,
                "SELL_REARM_ON_NEW_HIGH_PCT": cfg.sell_rearm_on_new_high_pct,
                "SELL_EARLY_SCORE_THRESHOLD": cfg.sell_early_score_threshold,
                "SELL_OPEN_SCOUT_MIN": cfg.sell_open_scout_min,
                "SELL_OPEN_SCOUT_SCORE_THRESHOLD": cfg.sell_open_scout_score_threshold,
                "SELL_OPEN_SCOUT_MAX_ALERTS_PER_SESSION": cfg.sell_open_scout_max_alerts_per_session,
                "COOLDOWN_MINUTES": cfg.cooldown_minutes,
            },
        }


        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        full_points = [p for p in points if eval_start <= p.ts < eval_end_excl]
        ok = _plot_full_day(
            points=full_points,
            alerts=alerts,
            out_path=os.path.join(out_dir, "full.png"),
            title=f"{cfg.symbol} backtest {bt_date.isoformat()} ({start_str}-{end_str} {cfg.monitor.tz_name})",
            tz=tz,
        )
        if not ok:
            print("未检测到 matplotlib，已跳过绘图（可先安装 matplotlib 再跑回测）。", file=sys.stderr)

        low_day = _min_price_point(eval_points)
        print(f"回测完成：输出目录={out_dir}")
        print(f"- 区间最低：{low_day.ts.strftime('%Y-%m-%d %H:%M %Z')} price={low_day.price}")
        print(f"- 提醒次数：{len(alerts)}（详情见 {csv_path}）")
        return 0

    if backtest:
        return _handle_backtest(argv, cfg, api_keys)

    try:
        email_cfg = _load_email_config()
    except Exception as e:
        print(f"邮箱配置错误：{e}", file=sys.stderr)
        return 2

    last_alert_at: Optional[datetime] = None
    if cfg.state_file:
        try:
            if os.path.exists(cfg.state_file):
                with open(cfg.state_file, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, dict) and raw.get("last_alert_at"):
                    last_alert_at = datetime.fromisoformat(raw["last_alert_at"]).replace(tzinfo=cfg.monitor.tz)
        except Exception as e:
            print(f"读取状态失败（将忽略）：{e}", file=sys.stderr)

    def can_alert(now: datetime) -> bool:
        nonlocal last_alert_at
        if last_alert_at is None:
            return True
        return now - last_alert_at >= timedelta(minutes=cfg.cooldown_minutes)
    
    api_key_index = 0
    idle_until: Optional[datetime] = None
    current_session_anchor: Optional[datetime] = None
    buy_open_scout_alerts_in_session = 0
    sell_open_scout_alerts_in_session = 0
    last_buy_alert_session_low: Optional[float] = None
    sell_rearm_high_in_session: Optional[float] = None

    while True:
        now = datetime.now(cfg.monitor.tz)
        if not _is_within_window(now, cfg.monitor):
            next_start = _next_window_start(now, cfg.monitor)
            api_key_index = 0
            current_session_anchor = None
            buy_open_scout_alerts_in_session = 0
            sell_open_scout_alerts_in_session = 0
            last_buy_alert_session_low = None
            sell_rearm_high_in_session = None
            if idle_until is None or next_start != idle_until:
                print(
                    f"[{now.strftime('%Y-%m-%d %H:%M:%S %Z')}] 不在监测时段 "
                    f"({cfg.monitor.tz_name} {_format_minutes(cfg.monitor.start_min)}"
                    f"-{_format_minutes(cfg.monitor.end_min)})，"
                    f"休眠至 {next_start.strftime('%Y-%m-%d %H:%M:%S %Z')}"
                )
                idle_until = next_start
            if once:
                return 0
            sleep_s = (next_start - now).total_seconds()
            time.sleep(max(5, min(sleep_s, 3600)))
            continue

        idle_until = None
        try:
            outputsize = max(
                80,
                cfg.low_window_min + 30,
                cfg.slope_window_min + 5,
                cfg.quantile_window_min + 5,
                cfg.global_window_min + 5,
                cfg.stall_min + 5,
            )
            points, api_key_index = fetch_points_with_key_failover(api_keys, cfg, outputsize, api_key_index)
            
            signal = _choose_signal(points)
            movement = signal.metrics.get("rebound_pct")
            movement_label = "rebound"
            if signal.side == "SELL":
                movement = signal.metrics.get("pullback_pct")
                movement_label = "pullback"
            print(
                f"[{now.strftime('%H:%M:%S %Z')}] side={signal.side} key={api_key_index+1}/{len(api_keys)} last={signal.metrics.get('last')} "
                f"{movement_label}={movement}% slope={signal.metrics.get('slope_pct_per_min')}%/min "
                f"alert={signal.should_alert}"
            )

            session_anchor = _session_start_anchor(
                points[-1].ts,
                start_min=cfg.monitor.start_min,
                end_min=cfg.monitor.end_min,
            )
            if current_session_anchor != session_anchor:
                current_session_anchor = session_anchor
                buy_open_scout_alerts_in_session = 0
                sell_open_scout_alerts_in_session = 0
                last_buy_alert_session_low = None
                sell_rearm_high_in_session = None

            if signal.should_alert:
                if (
                    signal.side == "BUY"
                    and last_buy_alert_session_low is not None
                    and signal.metrics.get("dist_score_ref_is_session_guard", 0.0) >= 0.5
                ):
                    print(f"[{now.strftime('%H:%M:%S %Z')}] 会话参考低点通道仅允许首个买点，跳过重复提醒")
                    if once:
                        return 0
                    time.sleep(max(5, cfg.poll_seconds))
                    continue
                first_buy_ok, first_buy_reason = _can_trigger_first_buy_in_session(
                    signal,
                    last_buy_alert_session_low is not None,
                )
                if not first_buy_ok:
                    print(f"[{now.strftime('%H:%M:%S %Z')}] {first_buy_reason}，跳过提醒")
                    if once:
                        return 0
                    time.sleep(max(5, cfg.poll_seconds))
                    continue
                cooldown_ok = can_alert(now)
                bypass_buy_cooldown = (not cooldown_ok) and _can_bypass_buy_cooldown(signal, last_buy_alert_session_low)
                if not cooldown_ok and not bypass_buy_cooldown:
                    if once:
                        return 0
                    time.sleep(max(5, cfg.poll_seconds))
                    continue
                if bypass_buy_cooldown:
                    print(f"[{now.strftime('%H:%M:%S %Z')}] BUY 新低豁免冷却，允许提前提醒")
                is_open_scout = signal.metrics.get("is_open_scout", 0.0) >= 0.5
                if is_open_scout and signal.side == "BUY":
                    if (
                        cfg.open_scout_max_alerts_per_session > 0
                        and buy_open_scout_alerts_in_session >= cfg.open_scout_max_alerts_per_session
                    ):
                        print(
                            f"[{now.strftime('%H:%M:%S %Z')}] 买入开盘通道已达会话上限"
                            f"({cfg.open_scout_max_alerts_per_session})，跳过提醒"
                        )
                        continue
                if is_open_scout and signal.side == "SELL":
                    if (
                        cfg.sell_open_scout_max_alerts_per_session > 0
                        and sell_open_scout_alerts_in_session >= cfg.sell_open_scout_max_alerts_per_session
                    ):
                        print(
                            f"[{now.strftime('%H:%M:%S %Z')}] 卖出开盘通道已达会话上限"
                            f"({cfg.sell_open_scout_max_alerts_per_session})，跳过提醒"
                        )
                        continue
                if signal.side == "SELL" and cfg.sell_rearm_on_new_high_pct > 0:
                    session_high_value = signal.metrics.get("session_high")
                    if isinstance(session_high_value, (int, float)):
                        current_session_high = float(session_high_value)
                        if sell_rearm_high_in_session is not None:
                            rearm_required = sell_rearm_high_in_session * (1 + cfg.sell_rearm_on_new_high_pct / 100.0)
                            if current_session_high < rearm_required:
                                print(
                                    f"[{now.strftime('%H:%M:%S %Z')}] 卖出新高重置未满足，"
                                    f"当前会话高点{current_session_high:.4f} < 触发线{rearm_required:.4f}，跳过提醒"
                                )
                                continue
                grade, grade_text = _signal_recommendation(signal)
                if signal.side == "SELL":
                    subject = f"黄金提醒【卖出 {grade}级-{grade_text}】（{cfg.symbol}）"
                else:
                    subject = f"黄金提醒【买入 {grade}级-{grade_text}】（{cfg.symbol}）"
                body = _format_email_body(now=now, symbol=cfg.symbol, points=points, signal=signal, tz=cfg.monitor.tz)
                inline_images: list[tuple[str, str]] = []
                plot_cid: Optional[str] = None
                if cfg.alert_plot_on_email:
                    try:
                        plot_path = _render_realtime_alert_plot(points, cfg, signal)
                        if plot_path:
                            plot_cid = "alert_plot"
                            inline_images.append((plot_cid, plot_path))
                    except Exception as e:
                        print(f"生成提醒图失败（将继续发送邮件）：{e}", file=sys.stderr)
                body_html = _format_email_html(
                    now=now,
                    symbol=cfg.symbol,
                    points=points,
                    signal=signal,
                    tz=cfg.monitor.tz,
                    plot_cid=plot_cid,
                )
                if cfg.dry_run:
                    msg = "DRY_RUN=1：将发送邮件但已跳过\n" + body
                    if inline_images:
                        msg += f"\n(内嵌图片：{', '.join(path for _, path in inline_images)})"
                    print(msg)
                else:
                    send_email(
                        email_cfg,
                        subject=subject,
                        body=body,
                        body_html=body_html,
                        inline_images=inline_images,
                    )
                    print("已发送提醒邮件")
                last_alert_at = now
                if signal.side == "BUY":
                    session_low_value = signal.metrics.get("session_low")
                    if isinstance(session_low_value, (int, float)):
                        last_buy_alert_session_low = float(session_low_value)
                if signal.side == "SELL":
                    session_high_value = signal.metrics.get("session_high")
                    if isinstance(session_high_value, (int, float)):
                        sell_rearm_high_in_session = float(session_high_value)
                if is_open_scout:
                    if signal.side == "SELL":
                        sell_open_scout_alerts_in_session += 1
                    else:
                        buy_open_scout_alerts_in_session += 1
                if cfg.state_file:
                    try:
                        with open(cfg.state_file, "w", encoding="utf-8") as f:
                            json.dump({"last_alert_at": now.isoformat()}, f, ensure_ascii=False, indent=2)
                    except Exception as e:
                        print(f"写入状态失败（将忽略）：{e}", file=sys.stderr)
        except Exception as e:
            print(f"[{now.strftime('%H:%M:%S %Z')}] 获取/计算失败：{e}", file=sys.stderr)

        if once:
            return 0
        time.sleep(max(5, cfg.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
