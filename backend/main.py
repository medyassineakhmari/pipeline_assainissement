"""
Point d'entrée FastAPI de la plateforme.

Lancement local :
    uvicorn backend.main:app --reload --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from .routes import pipeline, jobs, files
from . import config

app = FastAPI(
    title="Pipeline Assainissement — Plateforme",
    version="0.1.0",
    description="API pour orchestrer le pipeline ML/SIG sur les réseaux d'assainissement.",
)

# CORS (utile si frontend servi séparément, sinon inutile)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes API
app.include_router(pipeline.router, prefix="/api")
app.include_router(jobs.router,     prefix="/api")
app.include_router(files.router,    prefix="/api")


@app.get("/api/health")
def health():
    return {"status": "ok", "version": app.version}


# Sert le frontend statique
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
