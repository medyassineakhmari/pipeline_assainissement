"""
Routes API : gestion des fichiers (parcours, upload, téléchargement).

Sécurité : les chemins sont validés contre ALLOWED_ROOTS pour empêcher
de lire/écrire ailleurs que dans les zones autorisées.
"""
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from fastapi.responses import FileResponse

from .. import config

router = APIRouter(prefix="/files", tags=["files"])


def _is_allowed(path: Path) -> bool:
    """Vérifie qu'un chemin est sous l'une des racines autorisées."""
    try:
        resolved = path.resolve()
    except Exception:
        return False
    for root in config.ALLOWED_ROOTS:
        try:
            resolved.relative_to(Path(root).resolve())
            return True
        except ValueError:
            continue
    return False


@router.get("/browse")
def browse(path: str = Query(default="")):
    """
    Liste les fichiers et sous-dossiers d'un répertoire.
    Sans `path`, retourne les racines autorisées.
    """
    if not path:
        return {
            "current": None,
            "parent":  None,
            "entries": [
                {"name": Path(r).name or r, "path": r, "type": "directory"}
                for r in config.ALLOWED_ROOTS
            ],
        }

    p = Path(path)
    if not _is_allowed(p):
        raise HTTPException(403, f"Accès refusé : {path}")
    if not p.exists():
        raise HTTPException(404, f"Chemin inexistant : {path}")
    if not p.is_dir():
        raise HTTPException(400, f"N'est pas un dossier : {path}")

    entries = []
    try:
        for child in sorted(p.iterdir()):
            if child.name.startswith("."):
                continue
            entries.append({
                "name": child.name,
                "path": str(child),
                "type": "directory" if child.is_dir() else "file",
                "size": child.stat().st_size if child.is_file() else None,
            })
    except PermissionError:
        raise HTTPException(403, "Permission refusée sur ce dossier")

    return {
        "current": str(p),
        "parent":  str(p.parent) if _is_allowed(p.parent) else None,
        "entries": entries,
    }


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload un fichier dans data/inputs/."""
    if not file.filename:
        raise HTTPException(400, "Nom de fichier vide")

    target = config.INPUTS_DIR / file.filename
    content = await file.read()
    target.write_bytes(content)

    return {
        "filename": file.filename,
        "path":     str(target),
        "size":     len(content),
    }


@router.get("/download")
def download_file(path: str):
    """Télécharge un fichier (validé contre les racines autorisées)."""
    p = Path(path)
    if not _is_allowed(p):
        raise HTTPException(403, f"Accès refusé : {path}")
    if not p.exists() or not p.is_file():
        raise HTTPException(404, f"Fichier introuvable : {path}")
    return FileResponse(p, filename=p.name)
