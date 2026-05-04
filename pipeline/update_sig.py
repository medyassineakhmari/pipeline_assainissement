import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial
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


# --- CONFIGURATION DES CHEMINS ---
# PATH_BASE = r"I:\HYDRAU\THEMES\02-Assainissement\05-Documentation\06- GPASS\15- Modele predictif\02-BDD_ SIG"
# PATH_EXCEL_MAPPING = r"Synthèse_SIG - Copie.xlsx"

# FILE_TREE = r"I:\HYDRAU\THEMES\02-Assainissement\05-Documentation\06- GPASS\15- Modele predictif\04- Covariables\Mode occupation des sols\Taux de couverture arboré\Tree_cover.tif"
# FILE_IMPER = r"I:\HYDRAU\THEMES\02-Assainissement\05-Documentation\06- GPASS\15- Modele predictif\04- Covariables\Mode occupation des sols\Taux imperméabilisation des sols\impervious surface ratio.tif"

PATH_BASE          = _env_required("PATH_BASE")
PATH_EXCEL_MAPPING = _env_required("PATH_EXCEL_MAPPING")
FILE_TREE          = _env_required("FILE_TREE")
FILE_IMPER         = _env_required("FILE_IMPER")

# --- FONCTION DE CALCUL CORRIGÉE ---
def compute_pixel_mean(geom, raster_path):
    try:
        with rasterio.open(raster_path) as src:
            # Utilisation de all_touched=True pour capturer les pixels même si le centre n'est pas dans le buffer
            out_image, _ = mask(src, [geom], crop=True, all_touched=True)
            data = out_image[0]
            
            # Récupération dynamique du NoData du fichier
            nodata = src.nodata if src.nodata is not None else -999
            
            # On garde toutes les valeurs qui ne sont pas du NoData (y compris les 0 valides)
            valid_pixels = data[data != nodata]
            
            if valid_pixels.size == 0:
                return 0.0
            
            # On prend la moyenne des pixels touchés
            return float(np.mean(valid_pixels))
    except Exception:
        return 0.0

def process_city(row):
    ville = str(row['Agglo']).strip()
    if pd.isna(ville) or ville == 'nan': return

    dossier = str(row['Nom dossier'])
    fichier = str(row['Nom_BDD_troncons'])
    path_shp = os.path.join(PATH_BASE, dossier, f"{fichier}.shp")

    if not os.path.exists(path_shp):
        print(f"Fichier introuvable : {path_shp}")
        return

    print(f"\n--- Analyse Spatiale : {ville} ---")
    
    try:
        gdf = gpd.read_file(path_shp)


        # Vérifie si les colonnes existent ET si elles contiennent déjà des données
        if 'taux_arbo' in gdf.columns and 'taux_imper' in gdf.columns:
            if gdf['taux_arbo'].notna().any() and gdf['taux_imper'].notna().any():
                print(f"[SKIP] {ville} : Taux déjà calculés. Passage à la ville suivante.")
                return
            
        gdf = gdf[gdf.geometry.type.isin(['LineString', 'MultiLineString'])]

        # 2. On supprime les géométries None ou vides
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]

        # 3. On calcule la longueur et on supprime tout ce qui fait 0 mètre
        #    (C'est souvent là que l'erreur "point array" se cache)
        gdf = gdf[gdf.geometry.length > 0]

        # 4. On répare les erreurs de topologie légères
        gdf['geometry'] = gdf.geometry.make_valid()
        
        # Buffer de 5m en Lambert 93 (Pour travailler en mètres réels)
        gdf_buffer = gdf.to_crs(epsg=2154).copy()
        gdf_buffer['geometry'] = gdf_buffer.geometry.buffer(5)

        with rasterio.open(FILE_TREE) as src:
            raster_crs = src.crs
        
        # Alignement parfait sur le CRS du Raster
        gdf_buffer = gdf_buffer.to_crs(raster_crs)

        worker_tree = partial(compute_pixel_mean, raster_path=FILE_TREE)
        worker_imper = partial(compute_pixel_mean, raster_path=FILE_IMPER)

        # Calcul Parallèle
        with ProcessPoolExecutor() as executor:
            print(f"Calcul Taux Arboré...")
            gdf['taux_arbo'] = list(tqdm(executor.map(worker_tree, gdf_buffer.geometry), total=len(gdf)))

            print(f"Calcul Taux Imper...")
            gdf['taux_imper'] = list(tqdm(executor.map(worker_imper, gdf_buffer.geometry), total=len(gdf)))

        # Sauvegarde
        gdf.to_file(path_shp)
        print(f"{ville} mis à jour.")

    except Exception as e:
        print(f"Erreur sur {ville} : {e}")

if __name__ == "__main__":
    try:
        df_mapping = pd.read_excel(PATH_EXCEL_MAPPING, sheet_name="Resumé variables", header=2)
        for _, row in df_mapping.iterrows():
            process_city(row)
    except Exception as e:
        print(f"Erreur Excel : {e}")