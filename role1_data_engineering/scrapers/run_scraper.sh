#!/bin/bash

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
if [ -z "$TELEGRAM_BOT_TOKEN" ] || [ -z "$TELEGRAM_CHAT_ID" ]; then
    echo "Error: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing from .env" >> "$LOGFILE"
    exit 1
fi

# Navigate to the repository directory
cd "$REPO_DIR"
source .venv/bin/activate
# Run the python job and capture output
# Make sure uv is in PATH or specify its absolute path if cron doesn't find it
# In the crontab PATH is defined, but it's good practice
uv run python -m role1_data_engineering.scrapers.daily_scraper_orchestrator > "$LOGFILE" 2>&1
STATUS=$?

# Create a URL-encoded summary message for curl
if [ $STATUS -eq 0 ]; then
    SUMMARY="✅ *Daily Scraper Success*%0AThe financial data scraper completed successfully."
else
    SUMMARY="❌ *Daily Scraper Failed*%0AThe financial data scraper encountered an error (Exit code: $STATUS)."
fi

# Send Summary text via Telegram
curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
    -d chat_id="${TELEGRAM_CHAT_ID}" \
    -d text="${SUMMARY}" \
    -d parse_mode="Markdown" > /dev/null

# Send the Log File as a document
curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendDocument" \
    -F chat_id="${TELEGRAM_CHAT_ID}" \
    -F document=@"$LOGFILE" > /dev/null

# Clean up log file if successful (optional, you can comment this out to keep local logs)
if [ $STATUS -eq 0 ]; then
    rm "$LOGFILE"
fi
