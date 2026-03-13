# =============================================================================
# engine/indicator2.py
# Structure & ATR — EMA touch detection with slope filter and candle
# confirmation. Replicates Pine Script v6 Indicator 4 exactly in Python.
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
# Tracks bars-since-last-touch per pair per EMA per timeframe.
# Persisted to disk between GitHub Actions runs.

STATE_PATH = Path("data/state.json")


def load_cooldown_state() -> dict:
    """Load cooldown state from disk. Returns empty dict if file missing."""
    if STATE_PATH.exists():
        with open(STATE_PATH, "r") as f:
            return json.load(f)
    return {}


def save_cooldown_state(state: dict) -> None:
    """Save cooldown state to disk."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def _cooldown_key(pair: str, ema: str, interval: str) -> str:
    """e.g. 'EURUSD_200_H4'"""
    return f"{pair}_{ema}_{interval}"


# ── EMA ───────────────────────────────────────────────────────────────────────

def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """
    EMA matching Pine Script ta.ema().
    Uses adjust=False for recursive definition.
    """
    return series.ewm(span=span, adjust=False).mean()


# ── SLOPE ─────────────────────────────────────────────────────────────────────

def compute_slope(
    ema_series: pd.Series,
    atr_series: pd.Series,
    slope_len: int = None,
) -> pd.Series:
    """
    ATR-normalised EMA slope over slope_len bars.

    Replicates Pine Script:
        slope200 = (ema200 - ema200[slopeLen]) / atr

    Self-calibrating across all pairs and timeframes — 0.1 means the same
    thing on GBPJPY as on EURGBP regardless of pip value.
    """
    slope_len = slope_len or IND2_PARAMS["slope_len"]
    return (ema_series - ema_series.shift(slope_len)) / atr_series


# ── BB %WIDTH ─────────────────────────────────────────────────────────────────

def compute_bb_pct(
    close: pd.Series,
    bb_len: int = None,
    bb_mult: float = None,
    bb_norm_len: int = None,
) -> pd.Series:
    """
    Bollinger Band %Width normalised to 0.0–1.0 rolling range.

    bbWidth  = (BB_upper - BB_lower) / BB_mid
    bbPctRaw = (bbWidth - lowest(bbWidth, norm_len)) /
               (highest(bbWidth, norm_len) - lowest(bbWidth, norm_len))

    Returns:
        pd.Series of float 0.0–1.0
        0.0 = most compressed (squeeze)
        1.0 = most expanded
    """
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
    """
    Classify BB %Width into squeeze / neutral / expanding.
    Matches Pine Script colour logic:
        < 0.3 → squeeze   (gold)
        > 0.7 → expanding (white)
        else  → neutral   (grey)
    """
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
    ema_series: pd.Series,
    atr_series: pd.Series,
    touch_zone: float = None,
    slope_series: pd.Series = None,
    slope_thresh: float = None,
    cooldown: int = None,
    pair: str = "",
    ema_label: str = "",
    interval: str = "H4",
    state: dict = None,
) -> pd.Series:
    """
    Detect EMA touch events with cooldown and slope filter.

    A touch fires when:
      1. abs(dist) <= touch_zone  (price enters zone around EMA)
      2. This is the FIRST bar entering the zone (not touch[1])
      3. Cooldown bars have elapsed since last touch
      4. isSloped = abs(slope) >= slope_thresh

    Args:
        df:           OHLCV DataFrame
        ema_series:   EMA price series
        atr_series:   ATR series
        touch_zone:   ATR multiplier for zone width
        slope_series: ATR-normalised slope series
        slope_thresh: minimum slope to be considered trending
        cooldown:     minimum bars between touch events
        pair:         pair name (for state key)
        ema_label:    "200" or "50" (for state key)
        interval:     "H4" or "D1" (for state key)
        state:        mutable dict for cooldown tracking (modified in place)

    Returns:
        pd.Series of bool — True on bars where a valid touch fires
    """
    touch_zone   = touch_zone   or IND2_PARAMS["touch_zone"]
    slope_thresh = slope_thresh or IND2_PARAMS["slope_thresh"]
    cooldown     = cooldown     or IND2_PARAMS["cooldown"]
    state        = state if state is not None else {}

    close = df["close"]
    dist  = (close - ema_series) / atr_series

    in_zone      = dist.abs() <= touch_zone
    first_entry  = in_zone & ~in_zone.shift(1).fillna(False)
    is_sloped    = slope_series.abs() >= slope_thresh if slope_series is not None else pd.Series(True, index=df.index)

    # Apply cooldown using state from previous runs
    key           = _cooldown_key(pair, ema_label, interval)
    bars_since    = state.get(key, cooldown + 1)

    touch_flags   = pd.Series(False, index=df.index)

    for i in range(len(df)):
        bars_since += 1
        if (
            first_entry.iloc[i]
            and is_sloped.iloc[i]
            and bars_since > cooldown
        ):
            touch_flags.iloc[i] = True
            bars_since = 0

    # Save updated cooldown state
    state[key] = bars_since

    return touch_flags


# ── CANDLE CLOSE CONFIRMATION ─────────────────────────────────────────────────

def candle_confirmation(
    df: pd.DataFrame,
    ema_series: pd.Series,
    atr_series: pd.Series,
    touch_zone: float = None,
) -> pd.Series:
    """
    Candle close confirmation — distinguishes a wick touch (rejection) from
    a body close through (potential break).

    Bull confirmation:
        low  <= EMA + touchZone*ATR  (wick touched zone)
        AND close > EMA              (closed above EMA)
        AND open  > EMA              (opened above EMA)

    Bear confirmation:
        high >= EMA - touchZone*ATR  (wick touched zone)
        AND close < EMA              (closed below EMA)
        AND open  < EMA              (opened below EMA)

    Returns:
        pd.Series of bool — True when candle confirms a rejection
    """
    touch_zone = touch_zone or IND2_PARAMS["touch_zone"]

    bull = (
        (df["low"]  <= ema_series + touch_zone * atr_series)
        & (df["close"] > ema_series)
        & (df["open"]  > ema_series)
    )
    bear = (
        (df["high"] >= ema_series - touch_zone * atr_series)
        & (df["close"] < ema_series)
        & (df["open"]  < ema_series)
    )
    return bull | bear


# ── SINGLE PAIR OUTPUT ────────────────────────────────────────────────────────

def run_indicator2_pair(
    df: pd.DataFrame,
    pair: str,
    interval: str = "H4",
    params: dict = None,
    state: dict = None,
) -> dict:
    """
    Run Indicator 2 (Structure & ATR) for a single pair.
    Returns the latest bar state.

    Args:
        df:       OHLCV DataFrame for the pair
        pair:     pair symbol e.g. "EURUSD"
        interval: "H4" or "D1"
        params:   override IND2_PARAMS
        state:    mutable cooldown state dict (modified in place)

    Returns:
        dict with latest bar values:
        {
            "pair"          : str,
            "touch_200"     : bool,
            "touch_50"      : bool,
            "candle_conf_200": bool,
            "candle_conf_50" : bool,
            "is_sloped_200" : bool,
            "is_sloped_50"  : bool,
            "slope_200"     : float,
            "slope_50"      : float,
            "dist_200"      : float,  # ATR-normalised distance from 200 EMA
            "dist_50"       : float,  # ATR-normalised distance from 50 EMA
            "bb_pct"        : float,  # 0.0–1.0
            "bb_state"      : str,    # "squeeze" | "neutral" | "expanding"
            "is_ranging"    : bool,   # True when 200 EMA is flat
        }
    """
    p     = {**IND2_PARAMS, **(params or {})}
    state = state if state is not None else {}

    df = df.set_index("datetime").sort_index().copy()

    if len(df) < p["ema_slow"] + p["bb_norm_len"]:
        logger.warning(f"{pair}: insufficient bars ({len(df)}) for Indicator 2")
        return _empty_output2(pair)

    try:
        close = df["close"]
        atr   = compute_atr(df, p["atr_len"])

        # EMAs
        ema200 = compute_ema(close, p["ema_slow"])
        ema50  = compute_ema(close, p["ema_fast"])

        # ATR-normalised distances
        dist200 = (close - ema200) / atr
        dist50  = (close - ema50)  / atr

        # Slopes
        slope200 = compute_slope(ema200, atr, p["slope_len"])
        slope50  = compute_slope(ema50,  atr, p["slope_len"])

        is_sloped200 = slope200.abs() >= p["slope_thresh"]
        is_sloped50  = slope50.abs()  >= p["slope_thresh"]

        # Touch detection with cooldown
        df_reset = df.reset_index()
        touches200 = detect_touches(
            df_reset, ema200.values, atr.values,
            p["touch_zone"], slope200, p["slope_thresh"],
            p["cooldown"], pair, "200", interval, state,
        )
        touches50 = detect_touches(
            df_reset, ema50.values, atr.values,
            p["touch_zone"], slope50, p["slope_thresh"],
            p["cooldown"], pair, "50", interval, state,
        )

        # Candle confirmation
        conf200 = candle_confirmation(df_reset, ema200.values, atr.values, p["touch_zone"])
        conf50  = candle_confirmation(df_reset, ema50.values,  atr.values, p["touch_zone"])

        # BB %Width oscillator
        bb_pct_series = compute_bb_pct(close, p["bb_len"], p["bb_mult"], p["bb_norm_len"])

        # Latest bar values
        idx = -1

        latest_touch200 = bool(touches200.iloc[idx])
        latest_touch50  = bool(touches50.iloc[idx])
        latest_conf200  = bool(conf200.iloc[idx])
        latest_conf50   = bool(conf50.iloc[idx])
        latest_sloped200= bool(is_sloped200.iloc[idx])
        latest_sloped50 = bool(is_sloped50.iloc[idx])
        latest_slope200 = float(slope200.iloc[idx])
        latest_slope50  = float(slope50.iloc[idx])
        latest_dist200  = float(dist200.iloc[idx])
        latest_dist50   = float(dist50.iloc[idx])
        latest_bb_pct   = float(bb_pct_series.iloc[idx])

        return {
            "pair"           : pair,
            "touch_200"      : latest_touch200,
            "touch_50"       : latest_touch50,
            "candle_conf_200": latest_conf200,
            "candle_conf_50" : latest_conf50,
            "is_sloped_200"  : latest_sloped200,
            "is_sloped_50"   : latest_sloped50,
            "slope_200"      : round(latest_slope200, 4),
            "slope_50"       : round(latest_slope50,  4),
            "dist_200"       : round(latest_dist200,  4),
            "dist_50"        : round(latest_dist50,   4),
            "bb_pct"         : round(latest_bb_pct,   4),
            "bb_state"       : bb_state(latest_bb_pct),
            "is_ranging"     : not latest_sloped200,
        }

    except Exception as e:
        logger.error(f"{pair} ({interval}): error in indicator2 — {e}")
        return _empty_output2(pair)


def _empty_output2(pair: str) -> dict:
    return {
        "pair"           : pair,
        "touch_200"      : False,
        "touch_50"       : False,
        "candle_conf_200": False,
        "candle_conf_50" : False,
        "is_sloped_200"  : False,
        "is_sloped_50"   : False,
        "slope_200"      : None,
        "slope_50"       : None,
        "dist_200"       : None,
        "dist_50"        : None,
        "bb_pct"         : None,
        "bb_state"       : "neutral",
        "is_ranging"     : True,
    }


# ── ALL PAIRS OUTPUT ──────────────────────────────────────────────────────────

def run_indicator2_all(
    pair_data: dict[str, pd.DataFrame],
    interval: str = "H4",
    pairs: list[str] = None,
    params: dict = None,
) -> dict[str, dict]:
    """
    Run Indicator 2 for all active pairs.

    Cooldown state is loaded from disk at the start, updated during the run,
    and saved back to disk at the end.

    Args:
        pair_data: dict of pair -> OHLCV DataFrame
        interval:  "H4" or "D1"
        pairs:     list of pairs (default: ACTIVE_PAIRS)
        params:    override IND2_PARAMS

    Returns:
        dict of pair -> indicator2 output dict
    """
    pairs = pairs or ACTIVE_PAIRS

    # Load persistent cooldown state
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

    # Save updated cooldown state
    save_cooldown_state(state)

    valid  = [p for p, r in results.items() if r["slope_200"] is not None]
    failed = [p for p, r in results.items() if r["slope_200"] is None]
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

    print("\n── Indicator 2 Results ─────────────────────────────────────────────────")
    print(f"{'Pair':<10} {'T200':>5} {'T50':>5} {'Conf':>5} {'Slp200':>8} {'Slp50':>7} {'BB':>10} {'Ranging':>8}")
    print("─" * 75)
    for pair, r in results.items():
        if r["slope_200"] is None:
            print(f"{pair:<10} ERROR")
            continue
        print(
            f"{r['pair']:<10} "
            f"{'✅' if r['touch_200']       else '  ':>5} "
            f"{'✅' if r['touch_50']        else '  ':>5} "
            f"{'✅' if r['candle_conf_200'] or r['candle_conf_50'] else '  ':>5} "
            f"{r['slope_200']:>8.4f} "
            f"{r['slope_50']:>7.4f} "
            f"{r['bb_state']:>10} "
            f"{'⚠️  ranging' if r['is_ranging'] else '':>8}"
        )