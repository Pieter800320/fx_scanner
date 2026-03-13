# =============================================================================
# engine/indicator1.py
# Normalized PSL & MSL — full Indicator 1 output per pair.
# Assembles currency scores, PSL/MSL, rank, tide state into dashboard output.
# =============================================================================

import logging
import pandas as pd

from config.pairs import (
    ACTIVE_PAIRS,
    CURRENCIES,
    IND1_PARAMS,
    get_base,
    get_quote,
)
from engine.strength import (
    compute_currency_scores,
    compute_psl_msl,
    normalise_psl_msl,
    compute_pair_rank,
    compute_tide_state,
)

logger = logging.getLogger(__name__)


# ── RANK DOT FLAGS ────────────────────────────────────────────────────────────

def compute_rank_dots(
    rank: pd.Series,
    rank_threshold: int = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Determine buy/sell rank dot flags.

    buyRankDot  = rank <= threshold        (pair is among weakest N — oversold)
    sellRankDot = rank >= (29 - threshold) (pair is among strongest N — overbought)

    Returns:
        (buy_rank_dot, sell_rank_dot) as boolean Series
    """
    rank_threshold = rank_threshold or IND1_PARAMS["rank_threshold"]
    buy_rank_dot  = rank <= rank_threshold
    sell_rank_dot = rank >= (29 - rank_threshold)
    return buy_rank_dot, sell_rank_dot


# ── SINGLE PAIR OUTPUT ────────────────────────────────────────────────────────

def run_indicator1_pair(
    scores: dict[str, pd.Series],
    pair: str,
    params: dict = None,
) -> dict:
    """
    Run Indicator 1 for a single pair and return the latest bar state.

    Args:
        scores: dict of currency -> strength Series (from compute_currency_scores)
        pair:   pair symbol e.g. "EURUSD"
        params: override default IND1_PARAMS

    Returns:
        dict with latest bar values:
        {
            "pair"         : str,
            "psl_norm"     : float,   # 0–100
            "msl_norm"     : float,   # 0–100
            "psl_raw"      : float,
            "msl_raw"      : float,
            "rank"         : int,     # 1–28
            "buy_rank_dot" : bool,
            "sell_rank_dot": bool,
            "tide_state"   : str,     # longFuel | shortFuel | longFade | shortFade
            "msl_slope"    : float,   # raw slope value
            "bars"         : int,     # number of valid bars computed
        }
    """
    p = {**IND1_PARAMS, **(params or {})}

    base  = get_base(pair)
    quote = get_quote(pair)

    if base not in scores or quote not in scores:
        logger.error(f"{pair}: base '{base}' or quote '{quote}' missing from scores")
        return _empty_output(pair)

    try:
        psl_raw, msl_raw       = compute_psl_msl(scores, base, quote)
        psl_norm, msl_norm     = normalise_psl_msl(psl_raw, msl_raw, p["norm_len"])
        rank                   = compute_pair_rank(scores, psl_raw)
        tide                   = compute_tide_state(msl_norm, p["msl_slope_len"])
        buy_dot, sell_dot      = compute_rank_dots(rank, p["rank_threshold"])

        msl_slope = msl_norm - msl_norm.shift(p["msl_slope_len"])

        # Drop NaN rows for clean output
        valid = psl_norm.dropna()
        if len(valid) == 0:
            logger.warning(f"{pair}: no valid bars after dropna")
            return _empty_output(pair)

        # Latest bar
        idx = valid.index[-1]

        return {
            "pair"         : pair,
            "psl_norm"     : round(float(psl_norm.loc[idx]),  2),
            "msl_norm"     : round(float(msl_norm.loc[idx]),  2),
            "psl_raw"      : round(float(psl_raw.loc[idx]),   5),
            "msl_raw"      : round(float(msl_raw.loc[idx]),   5),
            "rank"         : int(rank.loc[idx]),
            "buy_rank_dot" : bool(buy_dot.loc[idx]),
            "sell_rank_dot": bool(sell_dot.loc[idx]),
            "tide_state"   : str(tide.loc[idx]),
            "msl_slope"    : round(float(msl_slope.loc[idx]), 4),
            "bars"         : len(valid),
        }

    except Exception as e:
        logger.error(f"{pair}: error in indicator1 — {e}")
        return _empty_output(pair)


def _empty_output(pair: str) -> dict:
    return {
        "pair"         : pair,
        "psl_norm"     : None,
        "msl_norm"     : None,
        "psl_raw"      : None,
        "msl_raw"      : None,
        "rank"         : None,
        "buy_rank_dot" : False,
        "sell_rank_dot": False,
        "tide_state"   : None,
        "msl_slope"    : None,
        "bars"         : 0,
    }


# ── ALL PAIRS OUTPUT ──────────────────────────────────────────────────────────

def run_indicator1_all(
    pair_data: dict[str, pd.DataFrame],
    pairs: list[str] = None,
    params: dict = None,
) -> dict[str, dict]:
    """
    Run Indicator 1 for all active pairs.

    Args:
        pair_data: dict of pair -> OHLCV DataFrame (strength engine pairs)
        pairs:     list of pairs to compute (default: ACTIVE_PAIRS)
        params:    override IND1_PARAMS

    Returns:
        dict of pair -> indicator1 output dict
    """
    pairs = pairs or ACTIVE_PAIRS
    p     = {**IND1_PARAMS, **(params or {})}

    # Compute currency scores once — shared across all pairs
    logger.info("Computing currency strength scores...")
    scores = compute_currency_scores(
        pair_data,
        lookback = p["lookback"],
        atr_len  = p["atr_len"],
        smooth   = p["smooth"],
    )

    results = {}
    for pair in pairs:
        logger.debug(f"Indicator 1: {pair}")
        results[pair] = run_indicator1_pair(scores, pair, p)

    # Summary log
    valid   = [p for p, r in results.items() if r["psl_norm"] is not None]
    failed  = [p for p, r in results.items() if r["psl_norm"] is None]
    logger.info(f"Indicator 1 complete: {len(valid)} valid, {len(failed)} failed")
    if failed:
        logger.warning(f"Failed pairs: {failed}")

    return results


# ── MSL GLOBAL SUMMARY ────────────────────────────────────────────────────────

def get_msl_summary(results: dict[str, dict]) -> dict:
    """
    Extract global MSL state from indicator1 results.
    MSL is the same across all pairs so we read it from the first valid result.

    Returns:
        {
            "msl_norm"  : float,
            "msl_raw"   : float,
            "msl_slope" : float,
            "tide_state": str,
        }
    """
    for pair, result in results.items():
        if result["msl_norm"] is not None:
            return {
                "msl_norm"  : result["msl_norm"],
                "msl_raw"   : result["msl_raw"],
                "msl_slope" : result["msl_slope"],
                "tide_state": result["tide_state"],
            }
    return {
        "msl_norm"  : None,
        "msl_raw"   : None,
        "msl_slope" : None,
        "tide_state": None,
    }


# ── CURRENCY RANKING TABLE ────────────────────────────────────────────────────

def get_currency_ranking(
    pair_data: dict[str, pd.DataFrame],
    params: dict = None,
) -> list[dict]:
    """
    Rank all 8 currencies by their latest strength score.
    Used in the Sunday summary and dashboard currency table.

    Returns:
        list of dicts sorted strongest → weakest:
        [{"currency": "GBP", "score": 0.234, "rank": 1}, ...]
    """
    p = {**IND1_PARAMS, **(params or {})}

    scores = compute_currency_scores(
        pair_data,
        lookback = p["lookback"],
        atr_len  = p["atr_len"],
        smooth   = p["smooth"],
    )

    ranking = []
    for currency in CURRENCIES:
        if currency not in scores:
            continue
        series = scores[currency].dropna()
        if len(series) == 0:
            continue
        latest = float(series.iloc[-1])
        ranking.append({"currency": currency, "score": round(latest, 4)})

    ranking.sort(key=lambda x: x["score"], reverse=True)
    for i, item in enumerate(ranking):
        item["rank"] = i + 1

    return ranking


# ── MAIN (for manual testing) ─────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from engine.fetch import fetch_strength_pairs

    print("Fetching D1 strength pairs...")
    pair_data = fetch_strength_pairs(interval="D1")

    print("\nRunning Indicator 1 for all active pairs...")
    results = run_indicator1_all(pair_data)

    print("\n── Indicator 1 Results ────────────────────────────────────")
    print(f"{'Pair':<10} {'PSL':>6} {'MSL':>6} {'Rank':>5} {'Buy':>5} {'Sell':>5} {'Tide':<12}")
    print("─" * 65)
    for pair, r in sorted(results.items(), key=lambda x: x[1]["rank"] or 99):
        if r["psl_norm"] is None:
            print(f"{pair:<10} {'ERROR':>6}")
            continue
        print(
            f"{r['pair']:<10} "
            f"{r['psl_norm']:>6.1f} "
            f"{r['msl_norm']:>6.1f} "
            f"{r['rank']:>5} "
            f"{'✅' if r['buy_rank_dot']  else '  ':>5} "
            f"{'✅' if r['sell_rank_dot'] else '  ':>5} "
            f"{r['tide_state']:<12}"
        )

    print("\n── MSL Global State ───────────────────────────────────────")
    msl = get_msl_summary(results)
    for k, v in msl.items():
        print(f"  {k}: {v}")

    print("\n── Currency Ranking ───────────────────────────────────────")
    ranking = get_currency_ranking(pair_data)
    for item in ranking:
        print(f"  {item['rank']}. {item['currency']}  {item['score']:+.4f}")