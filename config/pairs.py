# =============================================================================
# config/pairs.py
# Pair list, currency mapping, session rules, signal grades
# =============================================================================

from datetime import datetime, timezone

# ── ACTIVE TRADING PAIRS (12) ─────────────────────────────────────────────────
# These pairs are used for signal generation and dashboard display.

ACTIVE_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD",  # 7 majors
    "EURJPY", "GBPJPY", "AUDJPY", "EURGBP", "GBPAUD",                        # 5 crosses
]

# ── STRENGTH ENGINE PAIRS (17) ────────────────────────────────────────────────
# All 17 pairs needed to compute accurate currency scores for all 8 currencies.
# Not all of these are traded — they are data inputs only.

STRENGTH_PAIRS = [
    "EURUSD", "EURGBP", "EURAUD", "EURNZD", "EURCAD", "EURCHF", "EURJPY",
    "GBPUSD", "GBPAUD", "GBPCAD", "GBPJPY",
    "AUDUSD", "AUDCAD", "AUDJPY", "AUDNZD",
    "NZDUSD", "NZDJPY",
    "USDCAD", "USDCHF", "USDJPY",
]

# ── 8 MAJOR CURRENCIES ────────────────────────────────────────────────────────

CURRENCIES = ["EUR", "GBP", "AUD", "NZD", "USD", "CAD", "CHF", "JPY"]

# ── CURRENCY STRENGTH COMPOSITION ────────────────────────────────────────────
# Defines which pairs contribute to each currency's strength score,
# and the sign (+1 or -1) of that contribution.
# Format: { currency: [(pair, sign), ...] }

STRENGTH_COMPOSITION = {
    "EUR": [
        ("EURUSD", +1), ("EURGBP", +1), ("EURAUD", +1), ("EURNZD", +1),
        ("EURCAD", +1), ("EURCHF", +1), ("EURJPY", +1),
    ],
    "GBP": [
        ("GBPUSD", +1), ("EURGBP", -1), ("GBPAUD", +1),
        ("GBPCAD", +1), ("GBPJPY", +1),
    ],
    "AUD": [
        ("AUDUSD", +1), ("EURAUD", -1), ("GBPAUD", -1),
        ("AUDCAD", +1), ("AUDJPY", +1), ("AUDNZD", +1),
    ],
    "NZD": [
        ("NZDUSD", +1), ("EURNZD", -1), ("AUDNZD", -1), ("NZDJPY", +1),
    ],
    "USD": [
        ("EURUSD", -1), ("GBPUSD", -1), ("AUDUSD", -1), ("NZDUSD", -1),
        ("USDCAD", +1), ("USDCHF", +1), ("USDJPY", +1),
    ],
    "CAD": [
        ("USDCAD", -1), ("EURCAD", -1), ("GBPCAD", -1), ("AUDCAD", -1),
    ],
    "CHF": [
        ("USDCHF", -1), ("EURCHF", -1),
    ],
    "JPY": [
        ("USDJPY", -1), ("EURJPY", -1), ("GBPJPY", -1),
        ("AUDJPY", -1), ("NZDJPY", -1),
    ],
}

# ── PAIR → BASE / QUOTE CURRENCY ─────────────────────────────────────────────

def get_base(pair: str) -> str:
    """Return base currency of a pair. e.g. EURUSD -> EUR"""
    return pair[:3]

def get_quote(pair: str) -> str:
    """Return quote currency of a pair. e.g. EURUSD -> USD"""
    return pair[3:]

# ── JPY PAIRS (extended session hours) ───────────────────────────────────────

JPY_PAIRS = [p for p in ACTIVE_PAIRS if "JPY" in p]

# ── SESSION RULES ─────────────────────────────────────────────────────────────

SESSION_START_UTC = 7   # 07:00 UTC
SESSION_END_UTC   = 20  # 20:00 UTC (exclusive)

def is_active_session(pair: str = None) -> bool:
    """
    Returns True if the current UTC time is within the active trading session.

    JPY pairs are valid all weekday hours (Asian session active for JPY).
    All other pairs: 07:00–20:00 UTC Monday–Friday only.
    Saturday and Sunday are always blocked (except Sunday summary at 08:00).
    """
    now = datetime.now(timezone.utc)
    weekday = now.weekday()  # 0=Monday, 6=Sunday

    if weekday == 5:  # Saturday — always blocked
        return False
    if weekday == 6:  # Sunday — always blocked for signals
        return False

    # JPY pairs valid all weekday hours
    if pair and pair in JPY_PAIRS:
        return True

    # All other pairs: session window only
    return SESSION_START_UTC <= now.hour < SESSION_END_UTC


def is_sunday_summary_time() -> bool:
    """Returns True if it is Sunday between 08:00 and 08:59 UTC."""
    now = datetime.now(timezone.utc)
    return now.weekday() == 6 and now.hour == 8


def is_weekend() -> bool:
    """Returns True if it is Saturday or Sunday."""
    return datetime.now(timezone.utc).weekday() >= 5

# ── SIGNAL GRADE THRESHOLDS ───────────────────────────────────────────────────
# Grade is determined by how many conditions are met in the combined signal
# logic (see engine/signals.py).

GRADE_A_MIN_CONDITIONS = 6   # All conditions — full conviction, send alert
GRADE_B_MIN_CONDITIONS = 5   # Good setup — send alert
GRADE_C_MIN_CONDITIONS = 4   # Watch only — show in dashboard, no alert

# ── INDICATOR PARAMETERS (defaults matching Pine Script v6) ──────────────────

IND1_PARAMS = {
    "lookback"       : 10,
    "atr_len"        : 14,
    "smooth"         : 5,
    "norm_len"       : 100,
    "upper_threshold": 99.0,
    "lower_threshold": 1.0,
    "msl_slope_len"  : 10,
    "rank_threshold" : 3,
}

IND2_PARAMS = {
    "atr_len"       : 14,
    "ema_fast"      : 72,    # EMA 72 — fast line
    "sma_slow"      : 144,   # SMA 144 — slow line / zero line
    "shift_bars"    : 12,    # trend filter: SMA144 > SMA144[12]
    "touch_zone"    : 0.5,
    "cooldown"      : 8,
    "slope_len"     : 5,     # fast EMA slope only
    "slope_thresh"  : 0.1,   # fast EMA slope threshold only
    "bb_len"        : 20,
    "bb_mult"       : 2.0,
    "bb_norm_len"   : 100,
}

# ── TWELVEDATA API SETTINGS ───────────────────────────────────────────────────

TWELVEDATA_BASE_URL = "https://api.twelvedata.com/time_series"
FETCH_OUTPUTSIZE    = 300   # 200 EMA warmup + 100 BB norm lookback + buffer
INTERVALS           = {
    "D1": "1day",
    "H4": "4h",
}
