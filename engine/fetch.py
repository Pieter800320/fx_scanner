# =============================================================================
# engine/fetch.py
# Twelvedata API wrapper with batching, caching, and error handling.
# =============================================================================

import os
import json
import time
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
import pandas as pd

from config.pairs import (
    TWELVEDATA_BASE_URL,
    FETCH_OUTPUTSIZE,
    INTERVALS,
)

logger = logging.getLogger(__name__)

# ── CACHE SETTINGS ────────────────────────────────────────────────────────────

CACHE_DIR         = Path("data/cache")
CACHE_TTL_MINUTES = {
    "D1": 60,    # D1 data: cache valid for 60 minutes
    "H4": 15,    # H4 data: cache valid for 15 minutes
}

# Twelvedata free tier: 800 credits/day, 8 per minute
# Each symbol in a batch call costs 1 credit
# Batch up to 120 symbols per call (API limit)
MAX_SYMBOLS_PER_BATCH = 55  # conservative — stays well within limits
RATE_LIMIT_DELAY      = 0.5  # seconds between batch calls

# ── API KEY ───────────────────────────────────────────────────────────────────

def get_api_key() -> str:
    """
    Reads the Twelvedata API key from environment variable.
    Set as a GitHub Actions secret: TWELVEDATA_API_KEY
    """
    key = os.environ.get("TWELVEDATA_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "TWELVEDATA_API_KEY environment variable not set. "
            "Add it as a GitHub Actions secret."
        )
    return key

# ── CACHE HELPERS ─────────────────────────────────────────────────────────────

def _cache_path(pair: str, interval: str) -> Path:
    """Returns the cache file path for a given pair and interval."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{pair}_{interval}.json"


def _cache_valid(path: Path, interval: str) -> bool:
    """Returns True if the cache file exists and is within TTL."""
    if not path.exists():
        return False
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    ttl = CACHE_TTL_MINUTES.get(interval, 60)
    return datetime.now(timezone.utc) - modified < timedelta(minutes=ttl)


def _read_cache(path: Path) -> pd.DataFrame:
    """Reads a cached DataFrame from JSON."""
    with open(path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _write_cache(path: Path, df: pd.DataFrame) -> None:
    """Writes a DataFrame to cache as JSON."""
    data = df.copy()
    data["datetime"] = data["datetime"].astype(str)
    with open(path, "w") as f:
        json.dump(data.to_dict(orient="records"), f)

# ── SINGLE PAIR FETCH (with cache) ───────────────────────────────────────────

def fetch_pair(
    pair: str,
    interval: str,
    outputsize: int = FETCH_OUTPUTSIZE,
    force: bool = False,
) -> pd.DataFrame | None:
    """
    Fetch OHLCV data for a single pair.
    Returns a DataFrame sorted ascending by datetime, or None on failure.

    Args:
        pair:       e.g. "EURUSD"
        interval:   "D1" or "H4"
        outputsize: number of bars to fetch
        force:      if True, bypass cache and fetch fresh data
    """
    td_interval = INTERVALS.get(interval)
    if not td_interval:
        raise ValueError(f"Unknown interval '{interval}'. Use 'D1' or 'H4'.")

    cache_path = _cache_path(pair, interval)

    if not force and _cache_valid(cache_path, interval):
        logger.debug(f"Cache hit: {pair} {interval}")
        return _read_cache(cache_path)

    logger.info(f"Fetching {pair} {interval} from Twelvedata...")

    try:
        resp = requests.get(
            TWELVEDATA_BASE_URL,
            params={
                "symbol"    : pair,
                "interval"  : td_interval,
                "outputsize": outputsize,
                "apikey"    : get_api_key(),
                "order"     : "ASC",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") == "error":
            logger.error(f"Twelvedata error for {pair}: {data.get('message')}")
            return None

        values = data.get("values", [])
        if not values:
            logger.warning(f"No data returned for {pair} {interval}")
            return None

        df = pd.DataFrame(values)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        _write_cache(cache_path, df)
        return df

    except requests.exceptions.Timeout:
        logger.error(f"Timeout fetching {pair} {interval}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error for {pair} {interval}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error for {pair} {interval}: {e}")
        return None

# ── BATCH FETCH ───────────────────────────────────────────────────────────────

def fetch_pairs(
    pairs: list[str],
    interval: str,
    outputsize: int = FETCH_OUTPUTSIZE,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for multiple pairs efficiently.

    Strategy:
      1. Check cache for each pair — use cached data if valid.
      2. Collect pairs that need fresh data.
      3. Batch the uncached pairs into groups of MAX_SYMBOLS_PER_BATCH.
      4. Each batch = one API call (multiple symbols, comma-separated).
      5. Parse response and cache each pair individually.

    Returns:
        dict mapping pair symbol -> DataFrame (or missing if fetch failed)
    """
    td_interval = INTERVALS.get(interval)
    if not td_interval:
        raise ValueError(f"Unknown interval '{interval}'. Use 'D1' or 'H4'.")

    result: dict[str, pd.DataFrame] = {}
    to_fetch: list[str] = []

    # Step 1 — check cache
    for pair in pairs:
        cache_path = _cache_path(pair, interval)
        if not force and _cache_valid(cache_path, interval):
            logger.debug(f"Cache hit: {pair} {interval}")
            result[pair] = _read_cache(cache_path)
        else:
            to_fetch.append(pair)

    if not to_fetch:
        logger.info(f"All {len(pairs)} pairs served from cache ({interval})")
        return result

    logger.info(
        f"Fetching {len(to_fetch)} pairs in batches of {MAX_SYMBOLS_PER_BATCH} ({interval})"
    )

    # Step 2 — batch fetch uncached pairs
    for i in range(0, len(to_fetch), MAX_SYMBOLS_PER_BATCH):
        batch = to_fetch[i : i + MAX_SYMBOLS_PER_BATCH]
        batch_result = _fetch_batch(batch, td_interval, outputsize)
        result.update(batch_result)

        # Rate limit between batches
        if i + MAX_SYMBOLS_PER_BATCH < len(to_fetch):
            time.sleep(RATE_LIMIT_DELAY)

    # Report any failures
    failed = [p for p in pairs if p not in result]
    if failed:
        logger.warning(f"Failed to fetch: {failed}")

    return result


def _fetch_batch(
    pairs: list[str],
    td_interval: str,
    outputsize: int,
) -> dict[str, pd.DataFrame]:
    """
    Fetch a single batch of pairs in one API call.
    Twelvedata returns a dict keyed by symbol when multiple symbols requested,
    or a flat response when only one symbol is requested.
    """
    symbol_str = ",".join(pairs)
    logger.debug(f"Batch fetch: {symbol_str} ({td_interval})")

    try:
        resp = requests.get(
            TWELVEDATA_BASE_URL,
            params={
                "symbol"    : symbol_str,
                "interval"  : td_interval,
                "outputsize": outputsize,
                "apikey"    : get_api_key(),
                "order"     : "ASC",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        # Top-level error check (e.g. invalid API key, quota exceeded)
        if isinstance(data, dict) and data.get("status") == "error":
            logger.error(
                f"Twelvedata API error: {data.get('message')} "
                f"(code {data.get('code')})"
            )
            return {}

    except requests.exceptions.Timeout:
        logger.error(f"Timeout on batch: {symbol_str}")
        return {}
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error on batch: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error on batch: {e}")
        return {}

    # Twelvedata returns a flat dict when only 1 symbol requested
    if len(pairs) == 1:
        data = {pairs[0]: data}

    result = {}
    for pair in pairs:
        pair_data = data.get(pair, {})

        if pair_data.get("status") == "error":
            logger.error(f"Twelvedata error for {pair}: {pair_data.get('message')}")
            continue

        values = pair_data.get("values", [])
        if not values:
            logger.warning(f"No values for {pair}")
            continue

        try:
            df = pd.DataFrame(values)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime").reset_index(drop=True)
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Infer interval string from td_interval for cache key
            interval_key = "D1" if td_interval == "1day" else "H4"
            cache_path = _cache_path(pair, interval_key)
            _write_cache(cache_path, df)
            result[pair] = df

        except Exception as e:
            logger.error(f"Error parsing data for {pair}: {e}")
            continue

    return result

# ── CREDIT USAGE ESTIMATE ─────────────────────────────────────────────────────

def estimate_daily_credits(
    active_pairs: int = 12,
    strength_pairs: int = 17,
    h4_runs_per_day: int = 6,   # 00:05, 04:05, 08:05, 12:05, 16:05, 20:05
    d1_runs_per_day: int = 1,
) -> dict:
    """
    Estimates daily Twelvedata API credit usage.
    Assumes aggressive caching — only fetches on cache miss.

    Free tier: 800 credits/day.
    """
    # Strength pairs needed for Indicator 1 (D1 only)
    d1_credits = strength_pairs * d1_runs_per_day

    # Active pairs for Indicator 2 H4 (structure + ATR)
    h4_credits = active_pairs * h4_runs_per_day

    total = d1_credits + h4_credits

    return {
        "d1_credits"       : d1_credits,
        "h4_credits"       : h4_credits,
        "total_estimated"  : total,
        "free_tier_limit"  : 800,
        "headroom"         : 800 - total,
        "within_limit"     : total <= 800,
    }


# ── CONVENIENCE: FETCH ALL STRENGTH PAIRS ────────────────────────────────────

def fetch_strength_pairs(
    interval: str = "D1",
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fetch all 17 strength engine pairs."""
    from config.pairs import STRENGTH_PAIRS
    return fetch_pairs(STRENGTH_PAIRS, interval, force=force)


def fetch_active_pairs(
    interval: str = "H4",
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fetch all 12 active trading pairs."""
    from config.pairs import ACTIVE_PAIRS
    return fetch_pairs(ACTIVE_PAIRS, interval, force=force)


# ── MAIN (for manual testing) ─────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Credit estimate
    est = estimate_daily_credits()
    print("\n── Daily Credit Estimate ──────────────────")
    for k, v in est.items():
        print(f"  {k:<22}: {v}")

    # Test fetch single pair
    print("\n── Test: fetch EURUSD H4 ──────────────────")
    df = fetch_pair("EURUSD", "H4")
    if df is not None:
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Latest bar: {df.iloc[-1]['datetime']}  close={df.iloc[-1]['close']}")
    else:
        print("  Fetch failed — check API key")
