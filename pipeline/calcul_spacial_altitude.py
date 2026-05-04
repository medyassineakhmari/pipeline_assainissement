"""
enrich_cote_tn.py
─────────────────
Remplit Cote_TN_am, Cote_TN_av et Cote_TN dans un shapefile de collecteurs
à partir de dalles raster ASC/TIF.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from shapely.geometry import box, Point
from tqdm import tqdm

import os, sys

def _env_required(name):
    """Lit une variable d'environnement obligatoire. Sort proprement si absente/vide."""
    val = os.environ.get(name, "").strip()
    if not val:
        sys.exit(f"[ERREUR] Variable d'environnement '{name}' requise et non fournie.")
    return val

def _env_optional(name, default=""):
    """Lit une variable d'environnement optionnelle avec valeur par défaut."""
    return os.environ.get(name, default).strip()


# =============================================================================
# PARAMÈTRES — à modifier ici
# =============================================================================

# PATH_SHP      = r"I:\HYDRAU\THEMES\02-Assainissement\05-Documentation\06- GPASS\15- Modele predictif\02-BDD_ SIG\CAPI\shp_a_collec_line - Copie.shp"
# PATH_DALLES   = r"I:\HYDRAU\THEMES\02-Assainissement\05-Documentation\06- GPASS\15- Modele predictif\02-BDD_ SIG\CAPI\liste_dalles.txt"
# PATH_RASTERS  = r"C:\Users\invite-0108\Downloads\dep-38\RGEALTI_2-0_5M_ASC_LAMB93-IGN69_D038_2020-11-13\RGEALTI\1_DONNEES_LIVRAISON_2021-10-00009\RGEALTI_MNT_5M_ASC_LAMB93_IGN69_D038"
# PATH_OUTPUT   = PATH_SHP

PATH_SHP     = _env_required("PATH_SHP")
PATH_DALLES = _env_optional("PATH_DALLES", "")
PATH_RASTERS = _env_required("PATH_RASTERS")
PATH_OUTPUT  = _env_optional("PATH_OUTPUT", PATH_SHP)   # par défaut, écrit sur le fichier d'entrée


COL_COTE_AM  = "Cote_TN_am"
COL_COTE_AV  = "Cote_TN_av"
COL_COTE_MOY = "COTE_TN"

# =============================================================================


def lire_liste_dalles():
    extensions = {".asc", ".tif", ".tiff"}

    # Mode 1 : fichier fourni → lire la liste
    if PATH_DALLES and os.path.exists(PATH_DALLES):
        with open(PATH_DALLES, "r", encoding="utf-8") as f:
            noms = [ligne.strip() for ligne in f if ligne.strip()]

        index_dispo = {}
        for root, _, files in os.walk(PATH_RASTERS):
            for fname in files:
                if os.path.splitext(fname)[1].lower() in extensions:
                    cle = os.path.splitext(fname)[0].lower()
                    index_dispo[cle] = os.path.join(root, fname)

        chemins, non_trouves = [], []
        for nom in noms:
            cle = os.path.splitext(nom)[0].lower()
            if cle in index_dispo:
                chemins.append(index_dispo[cle])
            else:
                non_trouves.append(nom)
        if non_trouves:
            print(f"[ATTENTION] {len(non_trouves)} dalle(s) non trouvée(s)")
        print(f"[INFO] {len(chemins)}/{len(noms)} dalles trouvées (depuis liste)")
        return chemins

    # Mode 2 : pas de fichier → parcourir tous les rasters du dossier
    print("[INFO] Pas de liste fournie — scan complet de PATH_RASTERS")
    chemins = []
    for root, _, files in os.walk(PATH_RASTERS):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in extensions:
                chemins.append(os.path.join(root, fname))
    print(f"[INFO] {len(chemins)} rasters trouvés (mode complet)")
    return chemins


def construire_index_rasters(chemins):
    records = []
    for path in chemins:
        try:
            with rasterio.open(path) as src:
                b = src.bounds
                records.append({
                    "path":    path,
                    "nodata":  src.nodata,
                    "crs_wkt": src.crs.to_wkt() if src.crs else None,
                    "geometry": box(b.left, b.bottom, b.right, b.top)
                })
        except Exception as e:
            print(f"[ATTENTION] Impossible de lire {path} : {e}")

    crs_dalles = records[0]["crs_wkt"] if records and records[0]["crs_wkt"] else "EPSG:2154"

    gdf_index = gpd.GeoDataFrame(records, crs=crs_dalles)
    print(f"[INFO] Index spatial construit sur {len(gdf_index)} dalles")
    return gdf_index


def echantillonner_points(gdf_points, gdf_index):
    joined = gpd.sjoin(gdf_points, gdf_index, how="left", predicate="within")

    resultats = {}
    couverts = joined[joined["index_right"].notna()].copy()

    for idx_dalle, groupe in tqdm(
        couverts.groupby("index_right"),
        desc="Lecture dalles",
        total=couverts["index_right"].nunique()
    ):
        path   = gdf_index.loc[idx_dalle, "path"]
        nodata = gdf_index.loc[idx_dalle, "nodata"]

        try:
            with rasterio.open(path) as src:
                xs = groupe["x"].values
                ys = groupe["y"].values
                rows, cols = rowcol(src.transform, xs, ys)
                rows = np.array(rows)
                cols = np.array(cols)
                data = src.read(1).astype(float)
                height, width = data.shape

                for i, pt_id in enumerate(groupe["pt_id"].values):
                    r, c = int(rows[i]), int(cols[i])
                    if 0 <= r < height and 0 <= c < width:
                        val = data[r, c]
                        if nodata is not None and val == nodata:
                            val = np.nan
                        resultats[pt_id] = float(val)
                    else:
                        resultats[pt_id] = np.nan

        except Exception as e:
            print(f"[ATTENTION] Erreur lecture {path} : {e}")
            for pt_id in groupe["pt_id"].values:
                resultats[pt_id] = np.nan

    for pt_id in joined[joined["index_right"].isna()]["pt_id"].values:
        resultats[pt_id] = np.nan

    n_ok  = sum(1 for v in resultats.values() if not np.isnan(v))
    n_nan = sum(1 for v in resultats.values() if np.isnan(v))
    print(f"[INFO] Points échantillonnés : {n_ok} valides | {n_nan} sans valeur")
    return resultats


def build_points_gdf(gdf_troncons, suffixe):
    """
    Crée un GDF de points à partir de la géométrie des tronçons
    (Premier point si am, Dernier point si av)
    """
    print(f"[DEBUG] Extraction des points {suffixe} depuis la géométrie...")
    
    # Extraction sécurisée du premier ou dernier point
    def get_coord(geom, mode):
        if geom is None or geom.is_empty:
            return None
        # On prend le premier point (0) ou le dernier (-1)
        idx = 0 if mode == "am" else -1
        return Point(geom.coords[idx])

    points_geom = gdf_troncons.geometry.apply(lambda g: get_coord(g, suffixe))
    
    df = pd.DataFrame({
        "pt_id": gdf_troncons.index.astype(str) + f"_{suffixe}",
        "troncon_idx": gdf_troncons.index,
        "x": points_geom.apply(lambda p: p.x if p else np.nan),
        "y": points_geom.apply(lambda p: p.y if p else np.nan),
    }).dropna(subset=["x", "y"])

    return gpd.GeoDataFrame(
        df,
        geometry=points_geom[df.index],
        crs=gdf_troncons.crs
    )

def main():
    # ── Chargement shapefile ──────────────────────────────────────────────────
    print(f"[INFO] Lecture du shapefile : {PATH_SHP}")
    gdf = gpd.read_file(PATH_SHP)
    print(f"   → {len(gdf):,} tronçons | CRS : {gdf.crs}")

    # ── Création des colonnes si absentes ─────────────────────────────────────
    for col in [COL_COTE_AM, COL_COTE_AV, COL_COTE_MOY]:
        if col not in gdf.columns:
            gdf[col] = np.nan
            print(f"   → Colonne '{col}' créée")

    # ── Dalles ────────────────────────────────────────────────────────────────
    chemins = lire_liste_dalles()
    if not chemins:
        print("[ERREUR] Aucune dalle trouvée. Abandon.")
        return

    gdf_index = construire_index_rasters(chemins)

    # ── Reprojection si CRS différents ────────────────────────────────────────
    if gdf_index.crs is None:
        print("[ATTENTION] CRS des dalles inconnu → on suppose EPSG:2154 (Lambert 93)")
        gdf_index = gdf_index.set_crs("EPSG:2154")

    if gdf.crs and gdf_index.crs and gdf.crs != gdf_index.crs:
        print(f"[INFO] Reprojection collecteurs {gdf.crs} → {gdf_index.crs}")
        gdf = gdf.to_crs(gdf_index.crs)

    # ── Points amont et aval (CORRECTION ICI : On utilise la géométrie) ──────
    print("[INFO] Préparation des points amont/aval depuis la géométrie...")
    gdf_am = build_points_gdf(gdf, "am")
    gdf_av = build_points_gdf(gdf, "av")
    print(f"   → {len(gdf_am):,} points amont | {len(gdf_av):,} points aval")

    # ── Echantillonnage ───────────────────────────────────────────────────────
    print("\n[INFO] Echantillonnage des cotes amont...")
    val_am = echantillonner_points(gdf_am, gdf_index)

    print("\n[INFO] Echantillonnage des cotes aval...")
    val_av = echantillonner_points(gdf_av, gdf_index)

    # ── Réinjection ───────────────────────────────────────────────────────────
    print("\n[INFO] Réinjection des valeurs dans le shapefile...")
    for row in gdf_am.itertuples():
        cote = val_am.get(row.pt_id, np.nan)
        # On remplit si la valeur est valide
        if not np.isnan(cote):
            gdf.at[row.troncon_idx, COL_COTE_AM] = cote

    for row in gdf_av.itertuples():
        cote = val_av.get(row.pt_id, np.nan)
        if not np.isnan(cote):
            gdf.at[row.troncon_idx, COL_COTE_AV] = cote

    gdf[COL_COTE_MOY] = (
        pd.to_numeric(gdf[COL_COTE_AM], errors="coerce") +
        pd.to_numeric(gdf[COL_COTE_AV], errors="coerce")
    ) / 2

    # ── Statistiques ─────────────────────────────────────────────────────────
    total = len(gdf)
    for col in [COL_COTE_AM, COL_COTE_AV, COL_COTE_MOY]:
        n = gdf[col].notna().sum()
        print(f"   {col:<15} : {n:,}/{total:,} ({n/total*100:.1f}%)")

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    print(f"\n[INFO] Sauvegarde → {PATH_OUTPUT}")
    gdf.to_file(PATH_OUTPUT, encoding="utf-8")
    print("[INFO] Terminé.")


if __name__ == "__main__":
    main()