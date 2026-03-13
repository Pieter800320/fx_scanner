# =============================================================================
# run_h4.py
# Entry point for the H4 signal engine workflow.
# Called by .github/workflows/h4_signals.yml
# =============================================================================

import logging
from engine.fetch      import fetch_strength_pairs, fetch_active_pairs
from engine.indicator1 import run_indicator1_all
from engine.indicator2 import run_indicator2_all
from engine.signals    import evaluate_all_signals, build_dashboard_json, save_dashboard_json
from alerts.telegram   import send_all_signal_alerts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    logger.info("═══ H4 Signal Engine ═══")

    logger.info("Step 1/5 — Fetching D1 strength pairs...")
    strength_data = fetch_strength_pairs(interval="D1")

    logger.info("Step 2/5 — Fetching H4 active pairs...")
    active_data = fetch_active_pairs(interval="H4")

    logger.info("Step 3/5 — Running Indicator 1 (PSL & MSL)...")
    ind1 = run_indicator1_all(strength_data)

    logger.info("Step 4/5 — Running Indicator 2 (Structure & ATR)...")
    ind2 = run_indicator2_all(active_data, interval="H4")

    logger.info("Step 5/5 — Evaluating signals...")
    signals = evaluate_all_signals(ind1, ind2, interval="H4")

    payload = build_dashboard_json(signals, ind1, ind2, interval="H4")
    save_dashboard_json(payload, interval="H4")

    sent = send_all_signal_alerts(signals, interval="H4")
    logger.info(f"Alerts sent: {sent}")
    logger.info("═══ H4 Engine complete ═══")

if __name__ == "__main__":
    main()