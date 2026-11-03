#!/bin/bash
# Start Celery worker and FastAPI app together.
# Usage: ./run.sh
# Stop: Ctrl+C (kills both processes)

set -e

# Activate venv if present
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

echo "Starting Celery worker..."
celery -A celery_app worker --loglevel=info &
CELERY_PID=$!

echo "Starting FastAPI app..."
uvicorn app:app --host 0.0.0.0 --port 8000 --reload &
UVICORN_PID=$!

# Kill both on Ctrl+C
trap "echo 'Stopping...'; kill $CELERY_PID $UVICORN_PID 2>/dev/null; exit 0" INT TERM

wait
