#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# start.sh — Lance la plateforme en local
# ─────────────────────────────────────────────────────────────────────────────
# Utilisation :
#   ./start.sh
#
# Pré-requis :
#   - Redis installé et lancé (sudo systemctl start redis ou redis-server &)
#   - Environnement conda rapids_gpu activé
#   - pip install -r requirements.txt déjà exécuté
# ─────────────────────────────────────────────────────────────────────────────

set -e

cd "$(dirname "$0")"

# ─── Vérifications ───
echo "▸ Vérification de Redis..."
if ! redis-cli ping > /dev/null 2>&1; then
  echo "  ✗ Redis ne répond pas. Lancer : sudo systemctl start redis-server"
  echo "  (ou : redis-server &  pour un lancement manuel)"
  exit 1
fi
echo "  ✓ Redis OK"

# ─── Lancement ───
mkdir -p data/inputs data/outputs data/logs

echo ""
echo "▸ Démarrage du worker Celery..."
celery -A backend.celery_app worker \
  --loglevel=info \
  --concurrency=1 \
  --logfile=data/logs/celery.log \
  --detach \
  --pidfile=data/celery.pid

echo "  ✓ Worker démarré (logs : data/logs/celery.log)"

echo ""
echo "▸ Démarrage de l'API FastAPI..."
echo "  → http://localhost:8000"
echo "  → http://localhost:8000/docs (Swagger)"
echo ""
echo "  Pour arrêter : Ctrl+C puis ./stop.sh"
echo ""

uvicorn backend.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload
