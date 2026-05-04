"""
Jointures spatiales — Enrichissement de geom_troncon
Exécuter sur Windows avec : python jointures_spatiales.py
Prérequis : pip install geopandas pyogrio
"""

import geopandas as gpd
import pandas as pd
import time
import os
import fiona

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
# CONFIGURATION
# =============================================================================
# Couche cible (copie de geom_troncon)
# INPUT_GPKG = r"I:\HYDRAU\COMMUN\Stagiaires\M-yassine\test\800k\output_gpkg_800k.gpkg"
# INPUT_LAYER = "geom_troncon"

# # Sortie (copie enrichie)
# OUTPUT_GPKG = r"I:\HYDRAU\COMMUN\Stagiaires\M-yassine\test\800k\output_gpkg_800k.gpkg"
# OUTPUT_LAYER = "geom_troncon_enrichi"

# # Sources externes
# ARGILES_SHP = r"I:\HYDRAU\COMMUN\Stagiaires\M-yassine\test\AleaRG_2025_Fxx_L93\AleaRG_2025_Fxx_L93.shp"

INPUT_GPKG   = _env_required("INPUT_GPKG")
INPUT_LAYER  = _env_required("INPUT_LAYER")
OUTPUT_GPKG  = _env_required("OUTPUT_GPKG")
OUTPUT_LAYER = _env_required("OUTPUT_LAYER")
ARGILES_SHP  = _env_required("ARGILES_SHP")
NAPPES_DIR   = _env_required("NAPPES_DIR")



# =============================================================================
# 1. CHARGEMENT DE GEOM_TRONCON
# =============================================================================
print("\n" + "=" * 70)
print("[1] Chargement de geom_troncon...")
print("=" * 70)

t0 = time.time()
gdf = gpd.read_file(INPUT_GPKG, layer=INPUT_LAYER)
print(f"    ✓ Couche '{INPUT_LAYER}' chargée")
print(f"    {len(gdf):,} tronçons en {time.time() - t0:.1f}s")
print(f"    CRS : {gdf.crs}")
print(f"    Colonnes : {list(gdf.columns)}")
crs_cible = gdf.crs


# =============================================================================
# 2. JOINTURE — Aléa retrait-gonflement des argiles
# =============================================================================
print("\n" + "=" * 70)
print("[2] Jointure — Aléa retrait-gonflement des argiles")
print("=" * 70)


# ... tout le bloc argiles existant ...
if os.path.exists(ARGILES_SHP):
    t0 = time.time()
    argiles = gpd.read_file(ARGILES_SHP)
    print(f"    {len(argiles):,} polygones chargés en {time.time() - t0:.1f}s")

    if argiles.crs != crs_cible:
        print(f"    Reprojection {argiles.crs} → {crs_cible}...")
        argiles = argiles.to_crs(crs_cible)

    argiles = argiles[['geometry', 'niveau']].copy()
    argiles = argiles.rename(columns={'niveau': 'alea_argiles'})

    # Index spatial pour accélérer
    argiles_sindex = argiles.sindex

    print("    Jointure en cours (within puis intersection)...")
    t0 = time.time()
    resultats = []

    for idx, row in gdf.iterrows():
        line = row.geometry
        if line is None or line.is_empty:
            resultats.append(None)
            continue

        # Candidats via bbox
        candidates_idx = list(argiles_sindex.intersection(line.bounds))
        if not candidates_idx:
            resultats.append(None)
            continue

        candidates = argiles.iloc[candidates_idx]

        # CAS 1 : tronçon entièrement contenu dans un seul polygone
        containing = candidates[candidates.geometry.contains(line)]
        if len(containing) == 1:
            resultats.append(containing.iloc[0]['alea_argiles'])
            continue
        elif len(containing) > 1:
            # Rare : contenu dans plusieurs polygones superposés → prendre le plus restrictif (max)
            resultats.append(containing['alea_argiles'].max())
            continue

        # CAS 2 : tronçon à cheval sur plusieurs polygones → plus grande intersection
        best_val = None
        best_len = 0
        for _, poly_row in candidates.iterrows():
            if not line.intersects(poly_row.geometry):
                continue
            try:
                inter = line.intersection(poly_row.geometry)
                inter_len = inter.length
                if inter_len > best_len:
                    best_len = inter_len
                    best_val = poly_row['alea_argiles']
            except Exception:
                continue

        resultats.append(best_val)

        # Progression
        if (idx + 1) % 5000 == 0:
            print(f"      {idx + 1:,} / {len(gdf):,} tronçons traités...")

    gdf['alea_argiles'] = resultats
    print(f"    ✓ Jointure terminée en {time.time() - t0:.1f}s")

    # Statistiques
    filled = gdf['alea_argiles'].notna().sum()
    total = len(gdf)
    n_within = sum(1 for r in resultats if r is not None)
    print(f"\n    Résultat :")
    print(f"      Total tronçons  : {total:,}")
    print(f"      Avec aléa       : {filled:,} ({filled / total * 100:.1f}%)")
    print(f"      Sans aléa       : {total - filled:,} ({(total - filled) / total * 100:.1f}%)")

    print(f"\n    Distribution des niveaux :")
    for niv, count in gdf['alea_argiles'].value_counts().sort_index().items():
        print(f"      Niveau {niv} : {count:,}")
else:
    print(f"    ⚠ Fichier non trouvé : {ARGILES_SHP}")

# =============================================================================
# 3. JOINTURE — Remontée de nappes
# =============================================================================
print("\n" + "=" * 70)
print("[3] Jointure — Remontée de nappes")
print("=" * 70)


# NAPPES_DIR = r"I:\HYDRAU\COMMUN\Stagiaires\M-yassine\test\nappes"

if os.path.exists(NAPPES_DIR):
    import zipfile
    import glob
    import tempfile
    import shutil

    # Lire et fusionner tous les shapefiles des départements
    print("    Lecture des fichiers départementaux...")
    t0 = time.time()
    all_nappes = []

    zip_files = glob.glob(os.path.join(NAPPES_DIR, "Dept_*.zip"))
    print(f"    {len(zip_files)} départements trouvés")

    for zip_path in sorted(zip_files):
        dep_name = os.path.basename(zip_path).replace('.zip', '')
        try:
            # Extraire dans un dossier temporaire
            tmp_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(tmp_dir)

            # Chercher le shapefile Re_Nappe_fr.shp
            shp_candidates = glob.glob(os.path.join(tmp_dir, '**', 'Re_Nappe_fr.shp'), recursive=True)
            if not shp_candidates:
                # Essayer d'autres noms possibles
                shp_candidates = glob.glob(os.path.join(tmp_dir, '**', '*.shp'), recursive=True)

            if shp_candidates:
                nappes_dep = gpd.read_file(shp_candidates[0])
                all_nappes.append(nappes_dep)
            else:
                print(f"      ⚠ Pas de .shp trouvé dans {dep_name}")

            # Nettoyer le dossier temporaire
            shutil.rmtree(tmp_dir, ignore_errors=True)

        except Exception as e:
            print(f"      ⚠ Erreur {dep_name} : {e}")
            continue

    if all_nappes:
        nappes = pd.concat(all_nappes, ignore_index=True)
        nappes = gpd.GeoDataFrame(nappes, geometry='geometry')
        print(f"    {len(nappes):,} polygones nappes chargés en {time.time() - t0:.1f}s")

        if nappes.crs != crs_cible:
            print(f"    Reprojection {nappes.crs} → {crs_cible}...")
            nappes = nappes.to_crs(crs_cible)

        nappes = nappes[['geometry', 'gridcode']].copy()
        nappes = nappes.rename(columns={'gridcode': 'alea_nappes'})

        # Index spatial
        nappes_sindex = nappes.sindex

        print("    Jointure en cours (within puis intersection)...")
        t0 = time.time()
        resultats_nappes = []

        for idx, row in gdf.iterrows():
            line = row.geometry
            if line is None or line.is_empty:
                resultats_nappes.append(None)
                continue

            candidates_idx = list(nappes_sindex.intersection(line.bounds))
            if not candidates_idx:
                resultats_nappes.append(0)  # hors zone = pas de risque
                continue

            candidates = nappes.iloc[candidates_idx]

            # CAS 1 : tronçon entièrement contenu dans un polygone
            containing = candidates[candidates.geometry.contains(line)]
            if len(containing) == 1:
                resultats_nappes.append(int(containing.iloc[0]['alea_nappes']))
                continue
            elif len(containing) > 1:
                resultats_nappes.append(int(containing['alea_nappes'].max()))
                continue

            # CAS 2 : à cheval → plus grande intersection
            best_val = 0
            best_len = 0
            for _, poly_row in candidates.iterrows():
                if not line.intersects(poly_row.geometry):
                    continue
                try:
                    inter = line.intersection(poly_row.geometry)
                    inter_len = inter.length
                    if inter_len > best_len:
                        best_len = inter_len
                        best_val = int(poly_row['alea_nappes'])
                except Exception:
                    continue

            resultats_nappes.append(best_val)

            if (idx + 1) % 5000 == 0:
                print(f"      {idx + 1:,} / {len(gdf):,} tronçons traités...")

        gdf['alea_nappes'] = resultats_nappes
        print(f"    ✓ Jointure terminée en {time.time() - t0:.1f}s")

        # Statistiques
        total = len(gdf)
        print(f"\n    Distribution des niveaux :")
        for niv, count in gdf['alea_nappes'].value_counts().sort_index().items():
            label = {0: 'Pas de risque', 1: 'Débordement nappe', 2: 'Inondation cave'}.get(niv, str(niv))
            print(f"      {niv} ({label}) : {count:,}")
    else:
        print("    ⚠ Aucun fichier nappe chargé")
else:
    print(f"    ⚠ Dossier non trouvé : {NAPPES_DIR}")


# =============================================================================
# 99. SAUVEGARDE
# =============================================================================
print("\n" + "=" * 70)
print("[99] Sauvegarde du GeoPackage enrichi...")
print("=" * 70)

t0 = time.time()

# Supprimer la couche existante avant de réécrire
try:
    from osgeo import ogr
    ds = ogr.Open(OUTPUT_GPKG, 1)
    if ds:
        for i in range(ds.GetLayerCount()):
            if ds.GetLayer(i).GetName() == OUTPUT_LAYER:
                ds.DeleteLayer(i)
                print(f"    ✓ Ancienne couche '{OUTPUT_LAYER}' supprimée")
                break
        ds = None
except Exception:
    pass

gdf.to_file(OUTPUT_GPKG, layer=OUTPUT_LAYER, driver='GPKG', mode='w')
print(f"    ✓ Sauvegardé en {time.time() - t0:.1f}s")
print(f"    ✓ Sauvegardé en {time.time() - t0:.1f}s")
print(f"    Fichier : {OUTPUT_GPKG}")
print(f"    Couche  : {OUTPUT_LAYER}")
print(f"    Colonnes finales : {list(gdf.columns)}")

# Résumé des colonnes ajoutées
print("\n" + "=" * 70)
print("RÉSUMÉ DES ENRICHISSEMENTS")
print("=" * 70)
colonnes_ajoutees = []
if 'alea_argiles' in gdf.columns:
    colonnes_ajoutees.append(('alea_argiles', gdf['alea_argiles'].notna().sum()))
if 'alea_nappes' in gdf.columns:
    colonnes_ajoutees.append(('alea_nappes', gdf['alea_nappes'].notna().sum()))
if 'type_sol' in gdf.columns:
    colonnes_ajoutees.append(('type_sol', gdf['type_sol'].notna().sum()))

for col, filled in colonnes_ajoutees:
    print(f"    {col:<20} : {filled:,} / {len(gdf):,} ({filled / len(gdf) * 100:.1f}%)")

print("\n✓ Terminé")