# =============================================================================
# engine/signals.py
# Combined signal logic — merges Indicator 1 and Indicator 2 outputs into
# graded A/B/C buy/sell signals per pair per timeframe.
# =============================================================================

import logging
import json
from datetime import datetime, timezone
from pathlib import Path

from config.pairs import (
    ACTIVE_PAIRS,
    GRADE_A_MIN_CONDITIONS,
    GRADE_B_MIN_CONDITIONS,
    GRADE_C_MIN_CONDITIONS,
    is_active_session,
)

logger = logging.getLogger(__name__)

# ── TIDE ALIGNMENT ────────────────────────────────────────────────────────────

# Which tide states are aligned for each direction
BUY_TIDE_STATES  = {"shortFuel", "shortFade"}
SELL_TIDE_STATES = {"longFuel",  "longFade"}

# Full conviction tide states (fuel > fade)
BUY_FUEL_STATES  = {"shortFuel"}
SELL_FUEL_STATES = {"longFuel"}


# ── SINGLE PAIR SIGNAL ────────────────────────────────────────────────────────

def evaluate_signal(
    ind1: dict,
    ind2: dict,
    interval: str = "H4",
) -> dict:
    """
    Evaluate combined signal for a single pair.

    Checks 6 conditions for BUY and SELL independently.
    Assigns grade based on how many conditions are met.

    BUY conditions:
      1. is_trending = True           (SMA 144 trending up or down)
      2. trend == "up"                (SMA144 > SMA144[12])
      3. touch fired this bar         (144 SMA or 72 EMA touch)
      4. candle_conf = True           (wick rejection confirmed)
      5. tide aligned for buy         (shortFuel or shortFade)
      6. buy_rank_dot = True          (pair in bottom N of 28)

    SELL conditions:
      1. is_trending = True
      2. trend == "down"              (SMA144 < SMA144[12])
      3. touch fired this bar
      4. candle_conf = True
      5. tide aligned for sell        (longFuel or longFade)
      6. sell_rank_dot = True         (pair in top N of 28)

    Grade:
      A = 6 conditions  → alert fires
      B = 5 conditions  → alert fires
      C = 4 conditions  → dashboard only, no alert
      None = < 4        → no signal

    Args:
        ind1:     output dict from indicator1.run_indicator1_pair()
        ind2:     output dict from indicator2.run_indicator2_pair()
        interval: "H4" or "D1"

    Returns:
        {
            "pair"          : str,
            "signal"        : "BUY" | "SELL" | None,
            "grade"         : "A" | "B" | "C" | None,
            "conditions_met": int,
            "send_alert"    : bool,
            "direction"     : "BUY" | "SELL" | None,
            "ema_touched"   : "200" | "50" | None,
            "tide_state"    : str,
            "tide_aligned"  : bool,
            "is_ranging"    : bool,
            "conditions"    : dict,  # individual condition breakdown
        }
    """
    pair = ind1.get("pair") or ind2.get("pair", "UNKNOWN")

    # ── Guard: ranging market ─────────────────────────────────────────────────
    if ind2.get("is_ranging", True):
        return _no_signal(pair, reason="ranging", ind1=ind1, ind2=ind2)

    # ── Guard: missing data ───────────────────────────────────────────────────
    if ind1.get("psl_norm") is None or ind2.get("slope_72") is None:
        return _no_signal(pair, reason="missing_data", ind1=ind1, ind2=ind2)

    # ── Guard: session filter ─────────────────────────────────────────────────
    if not is_active_session(pair):
        return _no_signal(pair, reason="outside_session", ind1=ind1, ind2=ind2)

    # ── Extract values ────────────────────────────────────────────────────────
    is_sloped_200  = ind2["is_trending"]
    slope_200      = ind2["slope_72"] if ind2["trend"] == "up" else (
                     -abs(ind2["slope_72"]) if ind2["trend"] == "down" else 0
                     ) if ind2["slope_72"] is not None else None
    touch_fired    = ind2["touch_144"] or ind2["touch_72"]
    ema_touched    = "144" if ind2["touch_144"] else ("72" if ind2["touch_72"] else None)
    candle_conf    = (
        (ind2["touch_144"] and ind2["candle_conf_144"]) or
        (ind2["touch_72"]  and ind2["candle_conf_72"])
    )
    tide_state     = ind1.get("tide_state", "")
    buy_rank_dot   = ind1.get("buy_rank_dot",  False)
    sell_rank_dot  = ind1.get("sell_rank_dot", False)

    # ── Evaluate BUY conditions ───────────────────────────────────────────────
    buy_conditions = {
        "c1_sloped"     : is_sloped_200,
        "c2_slope_up"   : ind2.get("trend") == "up",
        "c3_touch"      : touch_fired,
        "c4_candle_conf": candle_conf,
        "c5_tide"       : tide_state in BUY_TIDE_STATES,
        "c6_rank"       : buy_rank_dot,
    }
    buy_count = sum(buy_conditions.values())

    # ── Evaluate SELL conditions ──────────────────────────────────────────────
    sell_conditions = {
        "c1_sloped"     : is_sloped_200,
        "c2_slope_down" : ind2.get("trend") == "down",
        "c3_touch"      : touch_fired,
        "c4_candle_conf": candle_conf,
        "c5_tide"       : tide_state in SELL_TIDE_STATES,
        "c6_rank"       : sell_rank_dot,
    }
    sell_count = sum(sell_conditions.values())

    # ── Determine direction ───────────────────────────────────────────────────
    # Take the direction with more conditions met.
    # If tied, prefer the one with tide fuel (stronger conviction).
    # If still tied, no signal.

    if buy_count >= GRADE_C_MIN_CONDITIONS or sell_count >= GRADE_C_MIN_CONDITIONS:
        if buy_count > sell_count:
            direction   = "BUY"
            cond_count  = buy_count
            conditions  = buy_conditions
        elif sell_count > buy_count:
            direction   = "SELL"
            cond_count  = sell_count
            conditions  = sell_conditions
        else:
            # Tied — prefer fuel tide
            buy_fuel  = tide_state in BUY_FUEL_STATES
            sell_fuel = tide_state in SELL_FUEL_STATES
            if buy_fuel and not sell_fuel:
                direction  = "BUY"
                cond_count = buy_count
                conditions = buy_conditions
            elif sell_fuel and not buy_fuel:
                direction  = "SELL"
                cond_count = sell_count
                conditions = sell_conditions
            else:
                return _no_signal(pair, reason="tied", ind1=ind1, ind2=ind2)
    else:
        return _no_signal(pair, reason="insufficient_conditions", ind1=ind1, ind2=ind2)

    # ── Assign grade ──────────────────────────────────────────────────────────
    if cond_count >= GRADE_A_MIN_CONDITIONS:
        grade = "A"
    elif cond_count >= GRADE_B_MIN_CONDITIONS:
        grade = "B"
    elif cond_count >= GRADE_C_MIN_CONDITIONS:
        grade = "C"
    else:
        return _no_signal(pair, reason="insufficient_conditions", ind1=ind1, ind2=ind2)

    send_alert = grade in ("A", "B")

    return {
        "pair"          : pair,
        "signal"        : direction,
        "grade"         : grade,
        "conditions_met": cond_count,
        "send_alert"    : send_alert,
        "direction"     : direction,
        "ema_touched"   : ema_touched,
        "tide_state"    : tide_state,
        "tide_aligned"  : conditions["c5_tide"],
        "is_ranging"    : False,
        "psl_norm"      : ind1.get("psl_norm"),
        "msl_norm"      : ind1.get("msl_norm"),
        "rank"          : ind1.get("rank"),
        "slope_72"      : ind2.get("slope_72"),
        "bb_state"      : ind2.get("bb_state"),
        "conditions"    : conditions,
    }


def _no_signal(pair: str, reason: str = "", ind1: dict = None, ind2: dict = None) -> dict:
    """Return a no-signal result with context preserved for dashboard."""
    ind1 = ind1 or {}
    ind2 = ind2 or {}
    return {
        "pair"          : pair,
        "signal"        : None,
        "grade"         : None,
        "conditions_met": 0,
        "send_alert"    : False,
        "direction"     : None,
        "ema_touched"   : None,
        "tide_state"    : ind1.get("tide_state"),
        "tide_aligned"  : False,
        "is_ranging"    : ind2.get("is_ranging", True),
        "psl_norm"      : ind1.get("psl_norm"),
        "msl_norm"      : ind1.get("msl_norm"),
        "rank"          : ind1.get("rank"),
        "slope_72"      : ind2.get("slope_72"),
        "bb_state"      : ind2.get("bb_state"),
        "reason"        : reason,
        "conditions"    : {},
    }


# ── ALL PAIRS ─────────────────────────────────────────────────────────────────

def evaluate_all_signals(
    ind1_results: dict[str, dict],
    ind2_results: dict[str, dict],
    interval: str = "H4",
    pairs: list[str] = None,
) -> dict[str, dict]:
    """
    Evaluate signals for all active pairs.

    Args:
        ind1_results: output from indicator1.run_indicator1_all()
        ind2_results: output from indicator2.run_indicator2_all()
        interval:     "H4" or "D1"
        pairs:        list of pairs (default: ACTIVE_PAIRS)

    Returns:
        dict of pair -> signal result dict
    """
    pairs = pairs or ACTIVE_PAIRS
    results = {}

    for pair in pairs:
        ind1 = ind1_results.get(pair, {})
        ind2 = ind2_results.get(pair, {})
        results[pair] = evaluate_signal(ind1, ind2, interval)

    # Summary
    grade_a = [p for p, r in results.items() if r["grade"] == "A"]
    grade_b = [p for p, r in results.items() if r["grade"] == "B"]
    grade_c = [p for p, r in results.items() if r["grade"] == "C"]
    ranging  = [p for p, r in results.items() if r.get("is_ranging")]

    logger.info(
        f"Signals ({interval}): "
        f"Grade A={grade_a}, Grade B={grade_b}, "
        f"Grade C={len(grade_c)}, Ranging={len(ranging)}"
    )

    return results


# ── DASHBOARD JSON ────────────────────────────────────────────────────────────

def build_dashboard_json(
    signal_results: dict[str, dict],
    ind1_results:   dict[str, dict],
    ind2_results:   dict[str, dict],
    interval: str = "H4",
) -> dict:
    """
    Build the dashboard JSON payload from signal, indicator1 and indicator2
    results. This is written to data/d1_dashboard.json or h4_dashboard.json.

    Schema matches Section 10 of the system design document.
    """
    from engine.indicator1 import get_msl_summary

    msl = get_msl_summary(ind1_results)

    pairs_output = []
    for pair in ACTIVE_PAIRS:
        sig  = signal_results.get(pair, {})
        ind1 = ind1_results.get(pair, {})
        ind2 = ind2_results.get(pair, {})

        pairs_output.append({
            "symbol"       : pair,
            "psl"          : ind1.get("psl_norm"),
            "msl"          : ind1.get("msl_norm"),
            "rank"         : ind1.get("rank"),
            "buy_rank_dot" : ind1.get("buy_rank_dot",  False),
            "sell_rank_dot": ind1.get("sell_rank_dot", False),
            "tide_state"   : ind1.get("tide_state"),
            "tide_aligned" : sig.get("tide_aligned", False),
            "is_trending"  : ind2.get("is_trending", False),
            "trend"        : ind2.get("trend"),
            "slope_72"     : ind2.get("slope_72"),
            "touch_144"    : ind2.get("touch_144", False),
            "touch_72"     : ind2.get("touch_72",  False),
            "candle_conf"  : (
                (ind2.get("touch_144") and ind2.get("candle_conf_144")) or
                (ind2.get("touch_72")  and ind2.get("candle_conf_72"))
            ),
            "bb_state"     : ind2.get("bb_state", "neutral"),
            "signal"       : sig.get("signal"),
            "grade"        : sig.get("grade"),
        })

    # Sort: signals first (A→B→C), then by rank
    def sort_key(p):
        grade_order = {"A": 0, "B": 1, "C": 2, None: 3}
        return (grade_order.get(p["grade"], 3), p["rank"] or 99)

    pairs_output.sort(key=sort_key)

    return {
        "updated"   : datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "timeframe" : interval,
        "msl"       : msl,
        "pairs"     : pairs_output,
    }


def save_dashboard_json(payload: dict, interval: str = "H4") -> None:
    """Write dashboard JSON to data/ directory."""
    path = Path("data") / f"{'d1' if interval == 'D1' else 'h4'}_dashboard.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Dashboard JSON saved: {path}")


# ── MAIN (for manual testing) ─────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from engine.fetch      import fetch_strength_pairs, fetch_active_pairs
    from engine.indicator1 import run_indicator1_all
    from engine.indicator2 import run_indicator2_all

    INTERVAL = "H4"

    print(f"Fetching data ({INTERVAL})...")
    strength_data = fetch_strength_pairs(interval="D1")
    active_data   = fetch_active_pairs(interval=INTERVAL)

    print("Running Indicator 1...")
    ind1 = run_indicator1_all(strength_data)

    print(f"Running Indicator 2 ({INTERVAL})...")
    ind2 = run_indicator2_all(active_data, interval=INTERVAL)

    print("Evaluating signals...")
    signals = evaluate_all_signals(ind1, ind2, interval=INTERVAL)

    print(f"\n── Signal Results ({INTERVAL}) ─────────────────────────────────────────")
    print(f"{'Pair':<10} {'Signal':<6} {'Grade':<6} {'Conds':<6} {'Tide':<12} {'BB':<12} {'Alert'}")
    print("─" * 72)

    for pair, r in signals.items():
        if r["signal"]:
            arrow = "▲" if r["signal"] == "BUY" else "▼"
            print(
                f"{pair:<10} "
                f"{arrow} {r['signal']:<4} "
                f"{r['grade']:<6} "
                f"{r['conditions_met']}/6    "
                f"{r.get('tide_state', ''):<12} "
                f"{r.get('bb_state', ''):<12} "
                f"{'📲' if r['send_alert'] else ''}"
            )

    no_signal = [p for p, r in signals.items() if not r["signal"]]
    if no_signal:
        print(f"\nNo signal: {', '.join(no_signal)}")

    print("\nBuilding dashboard JSON...")
    payload = build_dashboard_json(signals, ind1, ind2, interval=INTERVAL)
    save_dashboard_json(payload, interval=INTERVAL)
    print("Done.")
