# =============================================================================
# alerts/telegram.py
# Formats and sends signal alerts to Telegram.
# Supports Grade A/B signal alerts and a plain status ping.
# =============================================================================

import os
import logging
import requests
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ── CREDENTIALS ───────────────────────────────────────────────────────────────

def get_bot_token() -> str:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        raise EnvironmentError(
            "TELEGRAM_BOT_TOKEN environment variable not set. "
            "Add it as a GitHub Actions secret."
        )
    return token


def get_chat_id() -> str:
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not chat_id:
        raise EnvironmentError(
            "TELEGRAM_CHAT_ID environment variable not set. "
            "Add it as a GitHub Actions secret."
        )
    return chat_id


# ── SEND ──────────────────────────────────────────────────────────────────────

def send_message(text: str, parse_mode: str = "HTML") -> bool:
    """
    Send a message via Telegram Bot API.

    Args:
        text:       message text (HTML formatted)
        parse_mode: "HTML" or "Markdown"

    Returns:
        True if sent successfully, False otherwise.
    """
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{get_bot_token()}/sendMessage",
            json={
                "chat_id"   : get_chat_id(),
                "text"      : text,
                "parse_mode": parse_mode,
            },
            timeout=10,
        )
        resp.raise_for_status()
        logger.info("Telegram message sent successfully")
        return True

    except requests.exceptions.RequestException as e:
        logger.error(f"Telegram send failed: {e}")
        return False


# ── FORMATTERS ────────────────────────────────────────────────────────────────

def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M UTC")


def _grade_badge(grade: str) -> str:
    return {"A": "🏆 Grade A", "B": "⭐ Grade B", "C": "👁 Grade C"}.get(grade, "")


def _tide_emoji(tide_state: str) -> str:
    return {
        "longFuel" : "🔴 LONG FUEL",
        "shortFuel": "🟢 SHORT FUEL",
        "longFade" : "🟠 LONG FADE",
        "shortFade": "🟡 SHORT FADE",
    }.get(tide_state, tide_state or "—")


def _bb_emoji(bb_state: str) -> str:
    return {
        "squeeze"  : "🟡 Squeeze",
        "expanding": "⚪ Expanding",
        "neutral"  : "⬛ Neutral",
    }.get(bb_state, "—")


def _direction_emoji(direction: str) -> str:
    return "🟢 BUY" if direction == "BUY" else "🔴 SELL"


def _ema_label(ema_touched: str) -> str:
    if ema_touched == "144":
        return "144 SMA touch"
    if ema_touched == "72":
        return "72 EMA touch"
    return "EMA touch"


# ── SIGNAL ALERT ─────────────────────────────────────────────────────────────

def format_signal_alert(signal: dict, interval: str = "H4") -> str:
    """
    Format a Grade A or B signal alert.

    Example output:
        ⚡ H4 SIGNAL — 12:05 UTC
        🟢 EURUSD — BUY  ⭐ Grade B

        200 EMA touch ✅  Candle confirmed ✅
        Tide: 🟢 SHORT FUEL ✅
        BB: 🟡 Squeeze
        PSL: 18.4  MSL: 52.3  Rank: 3/28
        Slope: +0.14 ATR/bar
    """
    pair      = signal.get("pair", "")
    direction = signal.get("direction", "")
    grade     = signal.get("grade", "")
    ema       = signal.get("ema_touched")
    candle    = signal.get("conditions", {}).get("c4_candle_conf", False)
    tide      = signal.get("tide_state", "")
    tide_ok   = signal.get("tide_aligned", False)
    bb        = signal.get("bb_state", "neutral")
    psl       = signal.get("psl_norm")
    msl       = signal.get("msl_norm")
    rank      = signal.get("rank")
    slope     = signal.get("slope_72")

    header_emoji = "⚡" if interval == "H4" else "📊"

    lines = [
        f"{header_emoji} <b>{interval} SIGNAL — {_now_utc()}</b>",
        f"{_direction_emoji(direction)}  <b>{pair}</b>  {_grade_badge(grade)}",
        "",
        f"{_ema_label(ema)}  {'✅' if ema else '—'}   "
        f"Candle confirmed {'✅' if candle else '❌'}",
        f"Tide: {_tide_emoji(tide)} {'✅' if tide_ok else '⚠️'}",
        f"BB: {_bb_emoji(bb)}",
    ]

    # PSL / MSL / Rank line
    stats = []
    if psl  is not None: stats.append(f"PSL: {psl:.1f}")
    if msl  is not None: stats.append(f"MSL: {msl:.1f}")
    if rank is not None: stats.append(f"Rank: {rank}/28")
    if stats:
        lines.append("  ".join(stats))

    if slope is not None:
        lines.append(f"Slope: {slope:+.4f} ATR/bar")

    return "\n".join(lines)


def send_signal_alert(signal: dict, interval: str = "H4") -> bool:
    """
    Send a signal alert for a Grade A or B signal.
    Silently skips Grade C and no-signal results.
    """
    if not signal.get("send_alert"):
        logger.debug(f"{signal.get('pair')}: no alert (grade {signal.get('grade')})")
        return False

    text = format_signal_alert(signal, interval)
    logger.info(f"Sending {interval} signal alert: {signal['pair']} {signal['direction']} Grade {signal['grade']}")
    return send_message(text)


def send_all_signal_alerts(
    signal_results: dict[str, dict],
    interval: str = "H4",
) -> int:
    """
    Send alerts for all Grade A/B signals in the results dict.

    Returns:
        Number of alerts sent.
    """
    sent = 0
    # Send Grade A first, then Grade B
    for grade in ("A", "B"):
        for pair, signal in signal_results.items():
            if signal.get("grade") == grade and signal.get("send_alert"):
                if send_signal_alert(signal, interval):
                    sent += 1
    return sent


# ── NO-SIGNAL CANDLE (silent) ─────────────────────────────────────────────────

def send_run_summary(
    signal_results: dict[str, dict],
    interval: str = "H4",
) -> bool:
    """
    Send a brief run summary after each workflow execution.
    Only sent if at least one Grade A/B signal fired, otherwise silent.
    Useful for confirming the pipeline is running without spamming.
    """
    grade_a = [p for p, r in signal_results.items() if r.get("grade") == "A"]
    grade_b = [p for p, r in signal_results.items() if r.get("grade") == "B"]
    grade_c = [p for p, r in signal_results.items() if r.get("grade") == "C"]

    if not grade_a and not grade_b:
        logger.info("No Grade A/B signals — run summary suppressed")
        return False

    lines = [
        f"📋 <b>{interval} Run Complete — {_now_utc()}</b>",
        "",
    ]
    if grade_a:
        lines.append(f"🏆 Grade A: {', '.join(grade_a)}")
    if grade_b:
        lines.append(f"⭐ Grade B: {', '.join(grade_b)}")
    if grade_c:
        lines.append(f"👁 Watching: {', '.join(grade_c)}")

    return send_message("\n".join(lines))


# ── MAIN (for manual testing) ─────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test with a dummy signal
    test_signal = {
        "pair"         : "GBPJPY",
        "signal"       : "BUY",
        "direction"    : "BUY",
        "grade"        : "A",
        "send_alert"   : True,
        "conditions_met": 6,
        "ema_touched"  : "200",
        "tide_state"   : "shortFuel",
        "tide_aligned" : True,
        "bb_state"     : "squeeze",
        "psl_norm"     : 6.2,
        "msl_norm"     : 38.1,
        "rank"         : 2,
        "slope_200"    : 0.142,
        "conditions"   : {
            "c1_sloped"     : True,
            "c2_slope_up"   : True,
            "c3_touch"      : True,
            "c4_candle_conf": True,
            "c5_tide"       : True,
            "c6_rank"       : True,
        },
    }

    print("── Formatted alert preview ─────────────────────────────")
    print(format_signal_alert(test_signal, interval="H4"))
    print("\nSending test message to Telegram...")
    result = send_signal_alert(test_signal, interval="H4")
    print(f"Sent: {result}")
