"""
Daily Scraper Orchestrator:
Runs all scrapers sequentially, then copies the resulting data folders
to a cloud-synced directory for backup. Finally, sends a summary report via Telegram.

Scrapers executed:
1. headline_scraper.py (Groww news)
2. ohlc_scraper.py (TradingView daily OHLCV)
3. nse_preopen_scraper.py (NSE F&O pre-open data)
"""

import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import telebot

# ── Resolve project root ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import (
    CLOUD_SYNC_DIR,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    now_local,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("orchestrator")


def run_command(cmd: list[str], task_name: str) -> tuple[bool, str, float]:
    """Run a shell command, stream output to console, and return (success, logs, duration)."""
    logger.info("Starting task: %s", task_name)
    start_time = time.time()
    output_lines = []
    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        
        # Read output line by line as it is generated
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            output_lines.append(line)
            
        process.wait()
        duration = time.time() - start_time
        success = process.returncode == 0
        output = "".join(output_lines)
        
        if success:
            logger.info("✅ %s completed in %.1fs", task_name, duration)
        else:
            logger.error("❌ %s failed with exit code %d", task_name, process.returncode)
            
        return success, output.strip(), duration
    except Exception as e:
        duration = time.time() - start_time
        logger.exception("Failed to execute %s", task_name)
        return False, str(e), duration


def sync_to_cloud(folders: list[str]) -> tuple[bool, str, float]:
    """Copy specified data folders to the cloud-synced directory."""
    logger.info("Starting task: Cloud Sync")
    start_time = time.time()
    try:
        if not CLOUD_SYNC_DIR:
            return False, "CLOUD_SYNC_DIR is not set in config.", time.time() - start_time

        os.makedirs(CLOUD_SYNC_DIR, exist_ok=True)
        
        copied_details = []
        for folder in folders:
            src = os.path.join(PROJECT_ROOT, folder)
            dst = os.path.join(CLOUD_SYNC_DIR, os.path.basename(folder))
            
            if not os.path.exists(src):
                logger.warning("Source folder %s does not exist, skipping.", src)
                continue
                
            # Remove destination if it exists to cleanly mirror
            if os.path.exists(dst):
                shutil.rmtree(dst)
                
            shutil.copytree(src, dst)
            copied_details.append(f"Copied {folder} -> {dst}")
            
        duration = time.time() - start_time
        logger.info("✅ Cloud sync completed in %.1fs", duration)
        return True, "\n".join(copied_details), duration
    except Exception as e:
        duration = time.time() - start_time
        logger.exception("Failed to sync to cloud")
        return False, str(e), duration


def send_telegram_summary(summary_text: str):
    """Send a markdown message via Telegram using pyTelegramBotAPI."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials not set. Skipping Telegram alert.")
        print("\n--- TELEGRAM MESSAGE (Dry Run) ---\n" + summary_text + "\n----------------------------------")
        return

    try:
        bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
        bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=summary_text,
            parse_mode="Markdown",
            disable_web_page_preview=True,
        )
        logger.info("Telegram summary sent successfully.")
    except Exception as e:
        logger.error("Failed to send Telegram message: %s", e)


def main():
    logger.info("Starting daily scraper orchestration...")
    
    tasks = [
        (
            "Headline Scraper",
            [sys.executable, "-m", "role1_data_engineering.scrapers.headline_scraper"]
        ),
        (
            "OHLC Scraper",
            [sys.executable, "-m", "role1_data_engineering.scrapers.ohlc_scraper"]
        ),
        (
            "NSE Pre-open Scraper",
            [sys.executable, "-m", "role1_data_engineering.scrapers.nse_preopen_scraper"]
        ),
    ]

    report_lines = ["📊 *Daily Data Engineering Pipeline*"]
    report_lines.append(f"Date: `{now_local().strftime('%Y-%m-%d %H:%M:%S %Z')}`\n")

    all_success = True

    # 1. Run the scrapers
    for task_name, cmd in tasks:
        success, output, duration = run_command(cmd, task_name)
        status_icon = "✅" if success else "❌"
        report_lines.append(f"{status_icon} *{task_name}* ({duration:.1f}s)")
        
        if not success:
            all_success = False
            error_snippet = output[-500:] if len(output) > 500 else output
            report_lines.append(f"```\n{error_snippet}\n```")
            break  # Stop pipeline on first critical failure

    # 2. Run the cloud sync if scrapers succeeded
    if all_success:
        folders_to_sync = ["data/stock_news", "data/ohlc_data", "data/preopen_csv"]
        success, output, duration = sync_to_cloud(folders_to_sync)
        status_icon = "✅" if success else "❌"
        report_lines.append(f"{status_icon} *Cloud Sync* ({duration:.1f}s)")
        
        if not success:
            all_success = False
            report_lines.append(f"```\n{output}\n```")

    report_lines.append(f"\n*Overall Status*: {'✅ SUCCESS' if all_success else '❌ FAILED'}")
    
    summary_text = "\n".join(report_lines)
    send_telegram_summary(summary_text)

    if not all_success:
        sys.exit(1)


if __name__ == "__main__":
    main()
