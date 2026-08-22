#!/bin/bash

# Cron uses a minimal PATH; ensure uv and other user-local tools are reachable
export PATH="/Users/kgarg/.local/bin:/usr/local/bin:$PATH"

# Define paths
REPO_DIR="/Users/kgarg/extras/personal_github/Financial-Time-Series-MLOps"
ENV_FILE="$REPO_DIR/.env"
LOGFILE="/tmp/scraper_log_$(date +%Y%m%d_%H%M%S).txt"
source $REPO_DIR/.venv/bin/activate
# Load environment variables securely from .env
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
else
    echo "Error: .env file not found at $ENV_FILE" > "$LOGFILE"
    exit 1
fi

# Ensure the Telegram variables are available
if [ -z "$DISCORD_WEBHOOK_URL" ]; then
    echo "Error: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID or DISCORD_WEBHOOK_URL is missing from .env" >> "$LOGFILE"
    exit 1
fi

# Navigate to the repository directory
cd "$REPO_DIR"
source .venv/bin/activate
# Run the python job and capture output
# Make sure uv is in PATH or specify its absolute path if cron doesn't find it
# In the crontab PATH is defined, but it's good practice
echo "Running daily scraper orchestrator" >> "$LOGFILE"
uv run python -m role1_data_engineering.scrapers.daily_scraper_orchestrator > "$LOGFILE" 2>&1
STATUS=$?

# Create summary messages
if [ $STATUS -eq 0 ]; then
    TG_SUMMARY="✅ *Daily Scraper Success*%0AThe financial data scraper completed successfully."
    DISCORD_SUMMARY="✅ **Daily Scraper Success**\nThe financial data scraper completed successfully."
else
    TG_SUMMARY="❌ *Daily Scraper Failed*%0AThe financial data scraper encountered an error (Exit code: $STATUS)."
    DISCORD_SUMMARY="❌ **Daily Scraper Failed**\nThe financial data scraper encountered an error (Exit code: $STATUS)."
fi

# ── Telegram ──────────────────────────────────────────────────────────────────
# Send Summary text via Telegram
# curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
#     -d chat_id="${TELEGRAM_CHAT_ID}" \
#     -d text="${TG_SUMMARY}" \
#     -d parse_mode="Markdown" > /dev/null

# # Send the Log File as a document
# curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendDocument" \
#     -F chat_id="${TELEGRAM_CHAT_ID}" \
#     -F document=@"$LOGFILE" > /dev/null

# ── Discord ───────────────────────────────────────────────────────────────────
if [ -n "$DISCORD_WEBHOOK_URL" ]; then
    # Send summary message
    curl -s -X POST "$DISCORD_WEBHOOK_URL" \
        -H "Content-Type: application/json" \
        -d "{\"content\": \"${DISCORD_SUMMARY}\"}" > /dev/null

    # Send log file as attachment
    curl -s -X POST "$DISCORD_WEBHOOK_URL" \
        -F "file=@${LOGFILE}" \
        -F "payload_json={\"content\": \"📄 Scraper log attached\"}" > /dev/null
fi

# Clean up log file if successful (optional, you can comment this out to keep local logs)
# if [ $STATUS -eq 0 ]; then
#     rm "$LOGFILE"
# fi
