# =============================================================================
# alerts/summary.py
# Sunday 08:00 UTC weekly market summary builder and sender.
# Gives a full market snapshot at the start of each trading week.
# =============================================================================

import logging
from datetime import datetime, timezone

from alerts.telegram import send_message, _tide_emoji, _bb_emoji

logger = logging.getLogger(__name__)


# ── CURRENCY RANKING DISPLAY ──────────────────────────────────────────────────

def _format_currency_ranking(ranking: list[dict]) -> str:
    """
    Format currency strength ranking as a compact numbered list.

    Example:
        1. GBP  +0.2341
        2. USD  +0.1823
        ...
        8. JPY  -0.3210
    """
    lines = []
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    for item in ranking:
        rank   = item["rank"]
        ccy    = item["currency"]
        score  = item["score"]
        prefix = medals.get(rank, f"{rank}.")
        lines.append(f"  {prefix} {ccy}  {score:+.4f}")
    return "\n".join(lines)


# ── PAIR TABLE ────────────────────────────────────────────────────────────────

def _format_pair_table(
    ind1_results: dict[str, dict],
    signal_results: dict[str, dict],
) -> str:
    """
    Format a compact pair strength table sorted by PSL rank.

    Example:
        Rank  Pair      PSL    Tide
         1    GBPJPY   81.2   🔴 LONG FUEL
         2    EURUSD   74.1   🔴 LONG FUEL
        ...
    """
    rows = []
    for pair, r in ind1_results.items():
        if r.get("psl_norm") is None:
            continue
        sig = signal_results.get(pair, {})
        rows.append({
            "pair"      : pair,
            "rank"      : r.get("rank") or 99,
            "psl_norm"  : r.get("psl_norm"),
            "tide_state": r.get("tide_state", ""),
            "grade"     : sig.get("grade"),
        })

    rows.sort(key=lambda x: x["rank"])

    lines = ["<pre>Rank  Pair      PSL    Signal</pre>"]
    for row in rows:
        grade_tag = f"[{row['grade']}]" if row["grade"] else ""
        lines.append(
            f"<pre>"
            f"{row['rank']:>4}  "
            f"{row['pair']:<10}"
            f"{row['psl_norm']:>5.1f}  "
            f"{grade_tag}"
            f"</pre>"
        )
    return "\n".join(lines)


# ── ACTIVE SIGNALS SECTION ────────────────────────────────────────────────────

def _format_active_signals(signal_results: dict[str, dict]) -> str:
    """
    Format active Grade A/B signals section.
    Returns empty string if no active signals.
    """
    grade_a = [(p, r) for p, r in signal_results.items() if r.get("grade") == "A"]
    grade_b = [(p, r) for p, r in signal_results.items() if r.get("grade") == "B"]

    if not grade_a and not grade_b:
        return ""

    lines = ["<b>Active Signals</b>"]
    for pair, r in grade_a:
        arrow = "▲" if r.get("direction") == "BUY" else "▼"
        lines.append(f"  🏆 {pair} {arrow} {r.get('direction')} — Grade A")
    for pair, r in grade_b:
        arrow = "▲" if r.get("direction") == "BUY" else "▼"
        lines.append(f"  ⭐ {pair} {arrow} {r.get('direction')} — Grade B")

    return "\n".join(lines)


# ── WATCHLIST SECTION ─────────────────────────────────────────────────────────

def _format_watchlist(signal_results: dict[str, dict]) -> str:
    """
    Format Grade C watchlist section.
    Returns empty string if nothing to watch.
    """
    grade_c = [
        p for p, r in signal_results.items()
        if r.get("grade") == "C"
    ]
    if not grade_c:
        return ""
    return "<b>Pairs to Watch (Grade C)</b>\n  " + ",  ".join(grade_c)


# ── RANGING PAIRS SECTION ─────────────────────────────────────────────────────

def _format_ranging(signal_results: dict[str, dict]) -> str:
    """List pairs currently ranging (flat 200 EMA) — avoid these."""
    ranging = [
        p for p, r in signal_results.items()
        if r.get("is_ranging")
    ]
    if not ranging:
        return ""
    return "⚠️ <b>Ranging — avoid:</b>  " + ",  ".join(ranging)


# ── FULL SUMMARY BUILDER ──────────────────────────────────────────────────────

def build_weekly_summary(
    ind1_results:    dict[str, dict],
    signal_results:  dict[str, dict],
    currency_ranking: list[dict],
    interval: str = "D1",
) -> str:
    """
    Build the full Sunday weekly summary message.

    Args:
        ind1_results:      output from indicator1.run_indicator1_all()
        signal_results:    output from signals.evaluate_all_signals()
        currency_ranking:  output from indicator1.get_currency_ranking()
        interval:          timeframe used for signals (default D1)

    Returns:
        Formatted HTML string ready to send via Telegram.
    """
    now = datetime.now(timezone.utc).strftime("%d %b %Y")

    # MSL global state
    msl_norm   = None
    tide_state = None
    for r in ind1_results.values():
        if r.get("msl_norm") is not None:
            msl_norm   = r["msl_norm"]
            tide_state = r["tide_state"]
            break

    sections = []

    # ── Header ────────────────────────────────────────────────────────────────
    sections.append(
        f"📊 <b>WEEKLY MARKET SUMMARY</b>\n"
        f"Sunday {now} — 08:00 UTC\n"
        f"Timeframe: {interval}"
    )

    # ── MSL Tide ──────────────────────────────────────────────────────────────
    msl_line = f"MSL: {msl_norm:.1f}" if msl_norm is not None else "MSL: —"
    tide_line = _tide_emoji(tide_state) if tide_state else "—"
    sections.append(
        f"<b>Market Tide</b>\n"
        f"  {tide_line}\n"
        f"  {msl_line}"
    )

    # ── Currency Ranking ──────────────────────────────────────────────────────
    if currency_ranking:
        sections.append(
            "<b>Currency Strength</b>\n" +
            _format_currency_ranking(currency_ranking)
        )

    # ── Active Signals ────────────────────────────────────────────────────────
    active = _format_active_signals(signal_results)
    if active:
        sections.append(active)

    # ── Watchlist ─────────────────────────────────────────────────────────────
    watchlist = _format_watchlist(signal_results)
    if watchlist:
        sections.append(watchlist)

    # ── Ranging pairs ─────────────────────────────────────────────────────────
    ranging = _format_ranging(signal_results)
    if ranging:
        sections.append(ranging)

    # ── Pair Table ────────────────────────────────────────────────────────────
    sections.append(
        "<b>All Pairs — by PSL Rank</b>\n" +
        _format_pair_table(ind1_results, signal_results)
    )

    # ── Footer ────────────────────────────────────────────────────────────────
    sections.append(
        "─────────────────────\n"
        "Next update: Monday 08:05 UTC\n"
        "Trade safely. Protect capital first."
    )

    return "\n\n".join(sections)


def send_weekly_summary(
    ind1_results:     dict[str, dict],
    signal_results:   dict[str, dict],
    currency_ranking: list[dict],
    interval: str = "D1",
) -> bool:
    """
    Build and send the Sunday weekly summary to Telegram.

    Returns:
        True if sent successfully.
    """
    logger.info("Building Sunday weekly summary...")
    text = build_weekly_summary(
        ind1_results, signal_results, currency_ranking, interval
    )
    logger.info("Sending Sunday weekly summary...")
    return send_message(text)


# ── MAIN (for manual testing) ─────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from engine.fetch      import fetch_strength_pairs
    from engine.indicator1 import run_indicator1_all, get_currency_ranking
    from engine.indicator2 import run_indicator2_all
    from engine.signals    import evaluate_all_signals
    from engine.fetch      import fetch_active_pairs

    print("Fetching data...")
    strength_data = fetch_strength_pairs(interval="D1")
    active_data   = fetch_active_pairs(interval="D1")

    print("Running indicators...")
    ind1     = run_indicator1_all(strength_data)
    ind2     = run_indicator2_all(active_data, interval="D1")
    signals  = evaluate_all_signals(ind1, ind2, interval="D1")
    ranking  = get_currency_ranking(strength_data)

    print("\n── Weekly Summary Preview ──────────────────────────────")
    preview = build_weekly_summary(ind1, signals, ranking, interval="D1")
    # Strip HTML tags for terminal preview
    import re
    print(re.sub(r"<[^>]+>", "", preview))

    print("\nSend to Telegram? (y/n): ", end="")
    if input().strip().lower() == "y":
        result = send_weekly_summary(ind1, signals, ranking)
        print(f"Sent: {result}")