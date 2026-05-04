#!/bin/bash
# stop.sh — Arrête le worker Celery
cd "$(dirname "$0")"

if [ -f data/celery.pid ]; then
  echo "▸ Arrêt du worker Celery..."
  kill "$(cat data/celery.pid)" 2>/dev/null && echo "  ✓ Worker arrêté"
  rm -f data/celery.pid
else
  echo "  Aucun worker actif."
fi
