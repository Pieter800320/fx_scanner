# =============================================================================
# engine/strength.py
# ATR-normalised currency strength scores for all 8 major currencies.
# Replicates Pine Script v6 Indicator 1 currency engine exactly.
# =============================================================================

import logging
import numpy as np
import pandas as pd

from config.pairs import (
    CURRENCIES,
    STRENGTH_COMPOSITION,
    STRENGTH_PAIRS,
    IND1_PARAMS,
)

logger = logging.getLogger(__name__)

# ── ATR ───────────────────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """
    Compute ATR (Average True Range) using Wilder's smoothing.
    Matches Pine Script ta.atr() behaviour.
    """
    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Wilder's smoothing (equivalent to EMA with alpha = 1/length)
    atr = tr.ewm(alpha=1.0 / length, adjust=False).mean()
    return atr

# ── F_NORM ────────────────────────────────────────────────────────────────────

def f_norm(
    df: pd.DataFrame,
    lookback: int = None,
    atr_len: int = None,
) -> pd.Series:
    """
    ATR-normalised return over lookback bars.
    Replicates Pine Script:
        f_norm(sym) = (close_D1 - close_D1[lookback]) / ATR_D1(14)

    Args:
        df:       OHLCV DataFrame for a single pair
        lookback: number of bars to look back (default from IND1_PARAMS)
        atr_len:  ATR period (default from IND1_PARAMS)

    Returns:
        pd.Series of normalised returns, same index as df
    """
    lookback = lookback or IND1_PARAMS["lookback"]
    atr_len  = atr_len  or IND1_PARAMS["atr_len"]

    close = df["close"]
    atr   = compute_atr(df, atr_len)

    return (close - close.shift(lookback)) / atr

# ── EMA SMOOTHING ─────────────────────────────────────────────────────────────

def ema_smooth(series: pd.Series, span: int) -> pd.Series:
    """
    EMA smoothing matching Pine Script ta.ema().
    Uses adjust=False to match Pine's recursive EMA definition.
    """
    return series.ewm(span=span, adjust=False).mean()

# ── CURRENCY STRENGTH SCORES ──────────────────────────────────────────────────

def compute_currency_scores(
    pair_data: dict[str, pd.DataFrame],
    lookback: int = None,
    atr_len:  int = None,
    smooth:   int = None,
) -> dict[str, pd.Series]:
    """
    Compute EMA-smoothed ATR-normalised strength scores for all 8 currencies.

    Replicates Pine Script:
        EUR = ta.ema((f_norm(EURUSD) + f_norm(EURGBP) + ... ) / 7, smooth)
        GBP = ta.ema((f_norm(GBPUSD) - f_norm(EURGBP) + ... ) / 5, smooth)
        ... etc.

    Args:
        pair_data: dict of pair -> OHLCV DataFrame (must include all 17 pairs)
        lookback:  return lookback period
        atr_len:   ATR period
        smooth:    EMA smoothing period

    Returns:
        dict of currency -> pd.Series (strength score, same index as input data)

    Note:
        All series are aligned to a common datetime index before scoring.
        Pairs missing from pair_data are skipped with a warning.
    """
    lookback = lookback or IND1_PARAMS["lookback"]
    atr_len  = atr_len  or IND1_PARAMS["atr_len"]
    smooth   = smooth   or IND1_PARAMS["smooth"]

    # ── Step 1: compute f_norm for every available pair ──────────────────────
    norms: dict[str, pd.Series] = {}

    for pair in STRENGTH_PAIRS:
        if pair not in pair_data:
            logger.warning(f"Missing pair data: {pair} — strength scores may be inaccurate")
            continue
        df = pair_data[pair].copy()
        df = df.set_index("datetime").sort_index()
        norms[pair] = f_norm(df, lookback, atr_len)

    if not norms:
        raise ValueError("No pair data available — cannot compute currency scores")

    # ── Step 2: align all series to a common index ───────────────────────────
    # Use intersection of all available datetime indices
    common_index = None
    for series in norms.values():
        if common_index is None:
            common_index = series.index
        else:
            common_index = common_index.intersection(series.index)

    if common_index is None or len(common_index) == 0:
        raise ValueError("No common datetime index across strength pairs")

    norms = {pair: s.reindex(common_index) for pair, s in norms.items()}

    # ── Step 3: compute composite score per currency ──────────────────────────
    scores: dict[str, pd.Series] = {}

    for currency in CURRENCIES:
        composition = STRENGTH_COMPOSITION.get(currency, [])
        available   = [(p, sign) for p, sign in composition if p in norms]

        if not available:
            logger.error(f"No pairs available for {currency} strength — skipping")
            continue

        if len(available) < len(composition):
            missing = [p for p, _ in composition if p not in norms]
            logger.warning(f"{currency}: missing pairs {missing} — using {len(available)}/{len(composition)}")

        # Sum signed contributions, divide by count of available pairs
        total = pd.Series(0.0, index=common_index)
        for pair, sign in available:
            total = total + sign * norms[pair]

        composite = total / len(available)

        # EMA smoothing — matches Pine Script ta.ema()
        scores[currency] = ema_smooth(composite, smooth)

    logger.info(
        f"Currency scores computed: {list(scores.keys())} "
        f"over {len(common_index)} bars"
    )

    return scores

# ── PSL & MSL ─────────────────────────────────────────────────────────────────

def compute_psl_msl(
    scores: dict[str, pd.Series],
    base: str,
    quote: str,
) -> tuple[pd.Series, pd.Series]:
    """
    Compute PSL (Pair Strength Line) and MSL (Market Strength Line).

    PSL_raw = strength(base) - strength(quote)
    MSL_raw = mean of all 8 currency scores

    Args:
        scores: dict of currency -> strength Series
        base:   base currency of the pair  (e.g. "EUR" for EURUSD)
        quote:  quote currency of the pair (e.g. "USD" for EURUSD)

    Returns:
        (PSL_raw, MSL_raw) as pd.Series
    """
    if base not in scores:
        raise ValueError(f"Base currency '{base}' not in scores")
    if quote not in scores:
        raise ValueError(f"Quote currency '{quote}' not in scores")

    psl_raw = scores[base] - scores[quote]

    available_currencies = [c for c in CURRENCIES if c in scores]
    msl_raw = pd.concat(
        [scores[c] for c in available_currencies], axis=1
    ).mean(axis=1)

    return psl_raw, msl_raw

# ── NORMALISATION ─────────────────────────────────────────────────────────────

def normalise_0_100(
    series: pd.Series,
    norm_len: int = None,
) -> pd.Series:
    """
    Normalise a series to 0–100 using a rolling min/max window.
    Matches Pine Script rolling normalisation.
    """
    norm_len = norm_len or IND1_PARAMS["norm_len"]
    roll_min = series.rolling(norm_len, min_periods=1).min()
    roll_max = series.rolling(norm_len, min_periods=1).max()
    rng      = (roll_max - roll_min).replace(0, 1e-10)
    return 100.0 * (series - roll_min) / rng


def normalise_psl_msl(
    psl_raw: pd.Series,
    msl_raw: pd.Series,
    norm_len: int = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Shared normalisation of PSL and MSL to 0–100 scale.

    Uses combined min/max across BOTH series — exactly matching Pine Script:
        combLow  = min(lowest(PSL_raw, normLen), lowest(MSL_raw, normLen))
        combHigh = max(highest(PSL_raw, normLen), highest(MSL_raw, normLen))

    This keeps PSL and MSL on the same scale so they are directly comparable.

    Returns:
        (PSL_norm, MSL_norm) both on 0–100 scale
    """
    norm_len = norm_len or IND1_PARAMS["norm_len"]

    comb_low  = pd.concat([
        psl_raw.rolling(norm_len, min_periods=1).min(),
        msl_raw.rolling(norm_len, min_periods=1).min(),
    ], axis=1).min(axis=1)

    comb_high = pd.concat([
        psl_raw.rolling(norm_len, min_periods=1).max(),
        msl_raw.rolling(norm_len, min_periods=1).max(),
    ], axis=1).max(axis=1)

    rng = (comb_high - comb_low).replace(0, 1e-10)

    psl_norm = 100.0 * (psl_raw - comb_low) / rng
    msl_norm = 100.0 * (msl_raw - comb_low) / rng

    return psl_norm, msl_norm

# ── PAIR RANK ─────────────────────────────────────────────────────────────────

def compute_pair_rank(
    scores: dict[str, pd.Series],
    psl_raw: pd.Series,
) -> pd.Series:
    """
    Rank PSL_raw among all 28 possible currency pair differentials.

    Replicates Pine Script rank logic:
        lowerCount = count of how many of the 28 pairs are below PSL_raw
        pslRank    = lowerCount + 1  (1 = weakest, 28 = strongest)

    Returns:
        pd.Series of integer ranks (1–28)
    """
    # All 28 pairs = all unique combinations of 8 currencies
    from itertools import combinations

    available = [c for c in CURRENCIES if c in scores]
    all_pairs = [
        (base, quote)
        for base, quote in combinations(available, 2)
    ]

    # Build DataFrame of all 28 differentials
    diffs = pd.DataFrame({
        f"{b}{q}": scores[b] - scores[q]
        for b, q in all_pairs
    })

    # Count how many are strictly below PSL_raw at each bar
    lower_count = (diffs.lt(psl_raw, axis=0)).sum(axis=1)
    return (lower_count + 1).astype(int)

# ── MSL TIDE STATE ────────────────────────────────────────────────────────────

def compute_tide_state(
    msl_norm: pd.Series,
    msl_slope_len: int = None,
) -> pd.Series:
    """
    Compute MSL tide state at each bar.

    Four states matching Pine Script:
        longFuel  : MSL > 50 AND rising  → market elevated and going up   (sell bias)
        shortFuel : MSL < 50 AND falling → market depressed and going down (buy bias)
        longFade  : MSL > 50 AND falling → market elevated but fading
        shortFade : MSL < 50 AND rising  → market depressed but recovering

    Returns:
        pd.Series of strings: "longFuel" | "shortFuel" | "longFade" | "shortFade"
    """
    msl_slope_len = msl_slope_len or IND1_PARAMS["msl_slope_len"]

    msl_above = msl_norm > 50
    msl_slope = msl_norm - msl_norm.shift(msl_slope_len)

    conditions = [
        msl_above  & (msl_slope > 0),   # longFuel
        ~msl_above & (msl_slope < 0),   # shortFuel
        msl_above  & (msl_slope <= 0),  # longFade
        ~msl_above & (msl_slope >= 0),  # shortFade
    ]
    choices = ["longFuel", "shortFuel", "longFade", "shortFade"]

    return pd.Series(
        np.select(conditions, choices, default="shortFade"),
        index=msl_norm.index,
    )

# ── MAIN (for manual testing) ─────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from engine.fetch import fetch_strength_pairs

    print("Fetching strength pairs (D1)...")
    pair_data = fetch_strength_pairs(interval="D1")
    print(f"Fetched: {list(pair_data.keys())}")

    print("\nComputing currency scores...")
    scores = compute_currency_scores(pair_data)
    print(f"Scores computed for: {list(scores.keys())}")

    # Show latest value for each currency
    print("\n── Latest Currency Strength Values ────────")
    for ccy, series in scores.items():
        latest = series.dropna().iloc[-1] if len(series.dropna()) > 0 else float("nan")
        print(f"  {ccy}: {latest:+.4f}")

    # Test PSL/MSL for EURUSD
    print("\n── EURUSD PSL & MSL (last 5 bars) ────────")
    psl_raw, msl_raw = compute_psl_msl(scores, "EUR", "USD")
    psl_norm, msl_norm = normalise_psl_msl(psl_raw, msl_raw)
    rank = compute_pair_rank(scores, psl_raw)
    tide = compute_tide_state(msl_norm)

    df_out = pd.DataFrame({
        "PSL_raw" : psl_raw,
        "MSL_raw" : msl_raw,
        "PSL_norm": psl_norm,
        "MSL_norm": msl_norm,
        "Rank"    : rank,
        "Tide"    : tide,
    }).dropna().tail(5)

    print(df_out.to_string())