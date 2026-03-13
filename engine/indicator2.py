# =============================================================================
# engine/indicator2.py
# Structure & ATR — SMA 144 zero line with 144(-12) trend filter.
# Replicates Pine Script v6 Structure & ATR v2 exactly in Python.
#
# Trend logic:
#   trendUp   = SMA144 > SMA144[shiftBars]  → buy touches valid
#   trendDown = SMA144 < SMA144[shiftBars]  → sell touches valid
#   flat      = SMA144 == SMA144[shiftBars] → no trades
# =============================================================================

import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path

from config.pairs import (
    ACTIVE_PAIRS,
    IND2_PARAMS,
)
from engine.strength import compute_atr

logger = logging.getLogger(__name__)

# ── COOLDOWN STATE FILE ───────────────────────────────────────────────────────

STATE_PATH = Path("data/state.json")


def load_cooldown_state() -> dict:
    if STATE_PATH.exists():
        with open(STATE_PATH, "r") as f:
            return json.load(f)
    return {}


def save_cooldown_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def _cooldown_key(pair: str, line: str, interval: str) -> str:
    return f"{pair}_{line}_{interval}"


# ── SMA ───────────────────────────────────────────────────────────────────────

def compute_sma(series: pd.Series, span: int) -> pd.Series:
    """Simple Moving Average — matches Pine Script ta.sma()."""
    return series.rolling(span).mean()


# ── EMA ───────────────────────────────────────────────────────────────────────

def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """EMA matching Pine Script ta.ema(). Uses adjust=False."""
    return series.ewm(span=span, adjust=False).mean()


# ── TREND FILTER ──────────────────────────────────────────────────────────────

def compute_trend(
    sma_series: pd.Series,
    shift_bars: int = None,
) -> pd.Series:
    """
    Determine trend direction by comparing SMA to its shifted value.

    Returns pd.Series of strings:
        "up"   — SMA144 > SMA144[shiftBars] → buy touches valid
        "down" — SMA144 < SMA144[shiftBars] → sell touches valid
        "flat" — SMA144 == SMA144[shiftBars] → no trades
    """
    shift_bars = shift_bars or IND2_PARAMS["shift_bars"]
    shifted    = sma_series.shift(shift_bars)

    conditions = [
        sma_series > shifted,
        sma_series < shifted,
    ]
    choices = ["up", "down"]
    return pd.Series(
        np.select(conditions, choices, default="flat"),
        index=sma_series.index,
    )


# ── FAST EMA SLOPE ────────────────────────────────────────────────────────────

def compute_slope(
    ema_series: pd.Series,
    atr_series: pd.Series,
    slope_len: int = None,
) -> pd.Series:
    """ATR-normalised slope for the fast EMA only."""
    slope_len = slope_len or IND2_PARAMS["slope_len"]
    return (ema_series - ema_series.shift(slope_len)) / atr_series


# ── BB %WIDTH ─────────────────────────────────────────────────────────────────

def compute_bb_pct(
    close: pd.Series,
    bb_len: int = None,
    bb_mult: float = None,
    bb_norm_len: int = None,
) -> pd.Series:
    """BB %Width normalised to 0.0-1.0 rolling range."""
    bb_len      = bb_len      or IND2_PARAMS["bb_len"]
    bb_mult     = bb_mult     or IND2_PARAMS["bb_mult"]
    bb_norm_len = bb_norm_len or IND2_PARAMS["bb_norm_len"]

    sma    = close.rolling(bb_len).mean()
    std    = close.rolling(bb_len).std(ddof=0)
    upper  = sma + bb_mult * std
    lower  = sma - bb_mult * std

    bb_width = (upper - lower) / sma.replace(0, np.nan)
    roll_min = bb_width.rolling(bb_norm_len, min_periods=1).min()
    roll_max = bb_width.rolling(bb_norm_len, min_periods=1).max()
    rng      = (roll_max - roll_min).replace(0, 1e-10)

    return (bb_width - roll_min) / rng


def bb_state(bb_pct: float) -> str:
    if bb_pct is None or np.isnan(bb_pct):
        return "neutral"
    if bb_pct < 0.3:
        return "squeeze"
    if bb_pct > 0.7:
        return "expanding"
    return "neutral"


# ── TOUCH DETECTION ───────────────────────────────────────────────────────────

def detect_touches(
    df: pd.DataFrame,
    line_values: np.ndarray,
    atr_values: np.ndarray,
    touch_zone: float = None,
    filter_series: pd.Series = None,
    cooldown: int = None,
    pair: str = "",
    line_label: str = "",
    interval: str = "H4",
    state: dict = None,
) -> pd.Series:
    """
    Detect line touch events with cooldown and optional filter.

    A touch fires when:
      1. abs(dist) <= touch_zone
      2. First bar entering the zone
      3. Cooldown elapsed
      4. filter_series is True
    """
    touch_zone = touch_zone or IND2_PARAMS["touch_zone"]
    cooldown   = cooldown   or IND2_PARAMS["cooldown"]
    state      = state if state is not None else {}

    close   = df["close"].values
    dist    = (close - line_values) / atr_values
    in_zone = np.abs(dist) <= touch_zone

    prev_in_zone = np.concatenate([[False], in_zone[:-1]])
    first_entry  = in_zone & ~prev_in_zone

    if filter_series is not None:
        filter_vals = filter_series.values
    else:
        filter_vals = np.ones(len(df), dtype=bool)

    key        = _cooldown_key(pair, line_label, interval)
    bars_since = state.get(key, cooldown + 1)

    touch_flags = np.zeros(len(df), dtype=bool)

    for i in range(len(df)):
        bars_since += 1
        if first_entry[i] and filter_vals[i] and bars_since > cooldown:
            touch_flags[i] = True
            bars_since = 0

    state[key] = bars_since
    return pd.Series(touch_flags, index=df.index)


# ── CANDLE CONFIRMATION ───────────────────────────────────────────────────────

def candle_confirmation(
    df: pd.DataFrame,
    line_values: np.ndarray,
    atr_values: np.ndarray,
    touch_zone: float = None,
) -> pd.Series:
    """Candle close confirmation — wick touched zone, body closed on correct side."""
    touch_zone = touch_zone or IND2_PARAMS["touch_zone"]

    bull = (
        (df["low"].values   <= line_values + touch_zone * atr_values)
        & (df["close"].values > line_values)
        & (df["open"].values  > line_values)
    )
    bear = (
        (df["high"].values  >= line_values - touch_zone * atr_values)
        & (df["close"].values < line_values)
        & (df["open"].values  < line_values)
    )
    return pd.Series(bull | bear, index=df.index)


# ── SINGLE PAIR OUTPUT ────────────────────────────────────────────────────────

def run_indicator2_pair(
    df: pd.DataFrame,
    pair: str,
    interval: str = "H4",
    params: dict = None,
    state: dict = None,
) -> dict:
    """
    Run Indicator 2 for a single pair. Returns latest bar state.

    Returns:
        {
            "pair"           : str,
            "touch_144"      : bool,
            "touch_72"       : bool,
            "candle_conf_144": bool,
            "candle_conf_72" : bool,
            "trend"          : str,   # "up" | "down" | "flat"
            "is_trending"    : bool,
            "is_sloped_72"   : bool,
            "slope_72"       : float,
            "dist_144"       : float,
            "dist_72"        : float,
            "bb_pct"         : float,
            "bb_state"       : str,
            "is_ranging"     : bool,
        }
    """
    p     = {**IND2_PARAMS, **(params or {})}
    state = state if state is not None else {}

    df = df.set_index("datetime").sort_index().copy()

    min_bars = p["sma_slow"] + p["bb_norm_len"] + p["shift_bars"] + 10
    if len(df) < min_bars:
        logger.warning(
            f"{pair}: insufficient bars ({len(df)}) for Indicator 2 (need {min_bars})"
        )
        return _empty_output2(pair)

    try:
        close  = df["close"]
        atr    = compute_atr(df, p["atr_len"])
        sma144 = compute_sma(close, p["sma_slow"])
        ema72  = compute_ema(close, p["ema_fast"])

        dist144 = (close - sma144) / atr
        dist72  = (close - ema72)  / atr

        # Trend filter
        trend       = compute_trend(sma144, p["shift_bars"])
        is_trending = (trend != "flat")

        # Fast slope
        slope72     = compute_slope(ema72, atr, p["slope_len"])
        is_sloped72 = slope72.abs() >= p["slope_thresh"]

        df_reset = df.reset_index()

        touches144 = detect_touches(
            df_reset, sma144.values, atr.values,
            p["touch_zone"], is_trending,
            p["cooldown"], pair, "144", interval, state,
        )
        touches72 = detect_touches(
            df_reset, ema72.values, atr.values,
            p["touch_zone"], is_sloped72,
            p["cooldown"], pair, "72", interval, state,
        )

        conf144 = candle_confirmation(df_reset, sma144.values, atr.values, p["touch_zone"])
        conf72  = candle_confirmation(df_reset, ema72.values,  atr.values, p["touch_zone"])

        bb_pct_series = compute_bb_pct(close, p["bb_len"], p["bb_mult"], p["bb_norm_len"])

        idx = -1
        return {
            "pair"           : pair,
            "touch_144"      : bool(touches144.iloc[idx]),
            "touch_72"       : bool(touches72.iloc[idx]),
            "candle_conf_144": bool(conf144.iloc[idx]),
            "candle_conf_72" : bool(conf72.iloc[idx]),
            "trend"          : str(trend.iloc[idx]),
            "is_trending"    : bool(is_trending.iloc[idx]),
            "is_sloped_72"   : bool(is_sloped72.iloc[idx]),
            "slope_72"       : round(float(slope72.iloc[idx]),       4),
            "dist_144"       : round(float(dist144.iloc[idx]),       4),
            "dist_72"        : round(float(dist72.iloc[idx]),        4),
            "bb_pct"         : round(float(bb_pct_series.iloc[idx]), 4),
            "bb_state"       : bb_state(float(bb_pct_series.iloc[idx])),
            "is_ranging"     : not bool(is_trending.iloc[idx]),
        }

    except Exception as e:
        logger.error(f"{pair} ({interval}): error in indicator2 — {e}")
        return _empty_output2(pair)


def _empty_output2(pair: str) -> dict:
    return {
        "pair"           : pair,
        "touch_144"      : False,
        "touch_72"       : False,
        "candle_conf_144": False,
        "candle_conf_72" : False,
        "trend"          : "flat",
        "is_trending"    : False,
        "is_sloped_72"   : False,
        "slope_72"       : None,
        "dist_144"       : None,
        "dist_72"        : None,
        "bb_pct"         : None,
        "bb_state"       : "neutral",
        "is_ranging"     : True,
    }


# ── ALL PAIRS ─────────────────────────────────────────────────────────────────

def run_indicator2_all(
    pair_data: dict[str, pd.DataFrame],
    interval: str = "H4",
    pairs: list[str] = None,
    params: dict = None,
) -> dict[str, dict]:
    """Run Indicator 2 for all active pairs."""
    pairs = pairs or ACTIVE_PAIRS
    state = load_cooldown_state()

    results = {}
    for pair in pairs:
        if pair not in pair_data:
            logger.warning(f"{pair}: no data available for Indicator 2")
            results[pair] = _empty_output2(pair)
            continue
        logger.debug(f"Indicator 2: {pair} ({interval})")
        results[pair] = run_indicator2_pair(
            pair_data[pair], pair, interval, params, state
        )

    save_cooldown_state(state)

    valid  = [p for p, r in results.items() if r["slope_72"] is not None]
    failed = [p for p, r in results.items() if r["slope_72"] is None]
    logger.info(f"Indicator 2 complete: {len(valid)} valid, {len(failed)} failed")
    if failed:
        logger.warning(f"Failed pairs: {failed}")

    return results


# ── MAIN (for manual testing) ─────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from engine.fetch import fetch_active_pairs

    print("Fetching active pairs H4...")
    pair_data = fetch_active_pairs(interval="H4")

    print("\nRunning Indicator 2 for all active pairs (H4)...")
    results = run_indicator2_all(pair_data, interval="H4")

    print("\n── Indicator 2 Results ──────────────────────────────────────────────")
    print(f"{'Pair':<10} {'T144':>5} {'T72':>5} {'Trend':<6} {'Slp72':>7} {'BB':<12} {'Ranging'}")
    print("─" * 70)
    for pair, r in results.items():
        if r["slope_72"] is None:
            print(f"{pair:<10} ERROR")
            continue
        print(
            f"{r['pair']:<10} "
            f"{'✅' if r['touch_144'] else '  ':>5} "
            f"{'✅' if r['touch_72']  else '  ':>5} "
            f"{r['trend']:<6} "
            f"{r['slope_72']:>7.4f} "
            f"{r['bb_state']:<12} "
            f"{'⚠️  ranging' if r['is_ranging'] else ''}"
        )
