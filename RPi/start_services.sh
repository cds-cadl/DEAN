#!/bin/bash

# Enable debugging
set -x

# -------------------------
# Load Environment Variables
# -------------------------
source /home/admin/cronenv

# Activate the virtual environment
source /home/admin/venv/bin/activate

# Update PATH to prioritize the virtual environment's bin directory
export PATH=/home/admin/venv/bin:/home/admin/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/games:/usr/games

LOG_FILE=/home/admin/server/script_log.txt

# Log the current environment (optional, for debugging)
printenv > /home/admin/server/env_log.txt

# -------------------------
# Function to Start a Screen Session
# -------------------------
start_screen_session() {
    local SESSION_NAME=$1
    local COMMAND=$2
    local SLEEP_TIME=$3
    local LOG_SUCCESS=$4
    local LOG_FAILURE=$5

    echo "$(date '+%Y-%m-%d %H:%M:%S'): Attempting to start $SESSION_NAME" >> "$LOG_FILE"
    screen -dmS "$SESSION_NAME" bash -c "$COMMAND"
    sleep "$SLEEP_TIME"

    # Check if the session started
    local PID
    PID=$(screen -list | grep "$SESSION_NAME" | awk -F '.' '{print $1}' | awk '{print $1}')
    if [ -z "$PID" ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S'): $LOG_FAILURE" >> "$LOG_FILE"
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S'): $LOG_SUCCESS with session ID $PID" >> "$LOG_FILE"
    fi
}

# -------------------------
# Start Uvicorn in a Screen Session
# -------------------------
echo "$(date '+%Y-%m-%d %H:%M:%S'): Starting Uvicorn" >> "$LOG_FILE"

# Define the Uvicorn command with explicit activation and directory change
UVICORN_CMD="source /home/admin/venv/bin/activate && cd /home/admin/server && uvicorn rasp_new_fast:app --host 0.0.0.0 --port 5000 >> /home/admin/server/uvicorn.log 2>&1"

# Start Uvicorn
start_screen_session "uvicorn_session" "$UVICORN_CMD" 10 "Uvicorn started successfully" "Failed to start Uvicorn server"

# -------------------------
# Start Ngrok in another Screen Session
# -------------------------
echo "$(date '+%Y-%m-%d %H:%M:%S'): Connecting to Ngrok" >> "$LOG_FILE"

# Define the Ngrok command
NGROK_CMD="cd /home/admin/server && ngrok http --domain=humane-marmot-entirely.ngrok-free.app 5000 >> /home/admin/server/ngrok.log 2>&1"

# Start Ngrok
start_screen_session "ngrok_session" "$NGROK_CMD" 5 "Ngrok connected successfully" "Failed to connect to Ngrok"

# -------------------------
# Optional: List Running Sessions
# -------------------------
echo "$(date '+%Y-%m-%d %H:%M:%S'): Currently running screen sessions:" >> "$LOG_FILE"
screen -list >> "$LOG_FILE"


