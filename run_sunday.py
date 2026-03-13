# =============================================================================
# run_sunday.py
# Entry point for the Sunday weekly summary workflow.
# Called by .github/workflows/sunday_summary.yml
# =============================================================================

import logging
from engine.fetch      import fetch_strength_pairs, fetch_active_pairs
from engine.indicator1 import run_indicator1_all, get_currency_ranking
from engine.indicator2 import run_indicator2_all
from engine.signals    import evaluate_all_signals
from alerts.summary    import send_weekly_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    logger.info("═══ Sunday Weekly Summary ═══")

    logger.info("Step 1/5 — Fetching D1 strength pairs...")
    strength_data = fetch_strength_pairs(interval="D1")

    logger.info("Step 2/5 — Fetching D1 active pairs...")
    active_data = fetch_active_pairs(interval="D1")

    logger.info("Step 3/5 — Running Indicator 1...")
    ind1 = run_indicator1_all(strength_data)

    logger.info("Step 4/5 — Running Indicator 2...")
    ind2 = run_indicator2_all(active_data, interval="D1")

    logger.info("Step 5/5 — Evaluating signals + sending summary...")
    signals = evaluate_all_signals(ind1, ind2, interval="D1")
    ranking = get_currency_ranking(strength_data)

    result = send_weekly_summary(ind1, signals, ranking, interval="D1")
    logger.info(f"Summary sent: {result}")
    logger.info("═══ Sunday Summary complete ═══")

if __name__ == "__main__":
    main()