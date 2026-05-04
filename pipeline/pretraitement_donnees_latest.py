import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import warnings
import geopandas as gpd
from shapely.geometry import Point
import tqdm
from tqdm import tqdm
import os
import matplotlib.ticker as ticker

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


# Ignorer les avertissements de fragmentation de Pandas pour garder une console propre
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# --- CONFIGURATION ---
# PATH_BASE = r"I:\HYDRAU\THEMES\02-Assainissement\05-Documentation\06- GPASS\15- Modele predictif\02-BDD_ SIG"    # (Windows)   
# # PATH_BASE = r"/home/user-ia/modele_predictif/02-BDD_ SIG"   # (Linux)
# PATH_RUE = r"I:\HYDRAU\THEMES\02-Assainissement\05-Documentation\06- GPASS\15- Modele predictif\04- Covariables\Rue"
# PATH_EXCEL_MAPPING = "Synthèse_SIG - Copie.xlsx"
# OUTPUT_FILE = "BDD_Pretraitee_Finale_plastique_divisé.csv"
# OUTPUT_GPKG = "BDD_Pretraitee_Finaleplastique_divisé.gpkg"
# OUTPUT_PLOT = "Stats_Completude_Pretraitement.png"

PATH_BASE          = _env_required("PATH_BASE")
PATH_EXCEL_MAPPING = _env_required("PATH_EXCEL_MAPPING")
OUTPUT_FILE        = _env_required("OUTPUT_FILE")
OUTPUT_GPKG        = _env_required("OUTPUT_GPKG")

# Optionnels (avec défauts)
PATH_RUE    = _env_optional("PATH_RUE",    "")
OUTPUT_PLOT = _env_optional("OUTPUT_PLOT", "Stats_Completude_Pretraitement.png")


# Liste d'attributs pour le mapping SIG
ATTRIBUTS_UNIFIES = [
    'Nature_effluent', 'Diametre', 'Longueur', 'Materiau', 
    'Age', 'File_amont', 'File_aval', 'Cote_TN_am', 'Cote_TN_av', 'Cote_TN',
    'Pente', 'Prof_Amont', 'Prof_Aval', 'id_regard_am', 'id_regard_av', 
    'nom_rue', 'id_troncon', 'X_amont', 'Y_amont', 'X_aval', 'Y_aval', 'taux_arbo', 'taux_imper'
]

# Liste des valeurs considérées comme nulles
VALEURS_NULLES = ['INCONNU', 'Inconnu', 'inconnu', 'None', 'nan', 'nan.0', ' ', '', 'NULL', 'null', 'NC', 'ND']

# Colonnes numériques attendues
COLS_NUMERIQUES = [
    'Longueur', 'File_amont', 'File_aval', 'Cote_TN_am', 
    'Cote_TN_av', 'Cote_TN', 'Pente', 'Prof_Amont', 'Prof_Aval', 
    'X_amont', 'Y_amont', 'X_aval', 'Y_aval', 'taux_arbo', 'taux_imper'
]

# Colonnes où la valeur 0.0 est une vraie donnée et non une absence de donnée
COLS_ZERO_VALIDE = ['taux_arbo', 'taux_imper', 'Pente']

def verifier_mappings_rapide(df_map_villes, dict_mat, dict_eff, dict_dia):
    """
    Lit tous les shapefiles rapidement (sans calcul spatial)
    et vérifie les modalités manquantes dans les dictionnaires de mapping.
    S'arrête avant le traitement lourd (profondeur, jointure spatiale...).
    """
    print("\n" + "="*70)
    print("[VÉRIFICATION RAPIDE] Lecture des modalités avant traitement...")
    print("="*70)

    all_mat, all_eff, all_dia = [], [], []

    for _, row in tqdm(df_map_villes.iterrows(),
                       total=df_map_villes.shape[0],
                       desc="Lecture rapide des villes"):

        ville_nom = str(row['Agglo']).strip()
        if pd.isna(ville_nom) or ville_nom == 'nan':
            continue

        dossier_name    = str(row['Nom dossier'])
        fichier_troncons = str(row['Nom_BDD_troncons'])
        path_t = os.path.join(PATH_BASE, dossier_name, f"{fichier_troncons}.shp")

        if not os.path.exists(path_t):
            continue

        try:
            # ← Lire uniquement les colonnes utiles, pas la géométrie
            gdf_t = gpd.read_file(path_t, ignore_geometry=True)

            # Récupérer les noms des colonnes brutes via le mapping Excel
            col_mat = row.get('Materiau')
            col_eff = row.get('Nature_effluent')
            col_dia = row.get('Diametre')

            if pd.notna(col_mat) and col_mat in gdf_t.columns:
                all_mat.extend(gdf_t[col_mat].dropna().astype(str).unique().tolist())

            if pd.notna(col_eff) and col_eff in gdf_t.columns:
                all_eff.extend(gdf_t[col_eff].dropna().astype(str).unique().tolist())

            if pd.notna(col_dia) and col_dia in gdf_t.columns:
                vals_dia = gdf_t[col_dia].dropna().apply(normalize_dia).unique().tolist()
                all_dia.extend(vals_dia)

        except Exception as e:
            print(f"    ⚠ Erreur lecture {ville_nom} : {e}")

    # Dédupliquer
    all_mat = pd.Series(all_mat).drop_duplicates()
    all_eff = pd.Series(all_eff).drop_duplicates()
    all_dia = pd.Series(all_dia).drop_duplicates()

    print(f"\n    Modalités uniques trouvées :")
    print(f"    Matériau        : {len(all_mat)}")
    print(f"    Nature_effluent : {len(all_eff)}")
    print(f"    Diamètre        : {len(all_dia)}")

    print("\n--- MATÉRIAU ---")
    detect_missing_mappings(all_mat, dict_mat, "Materiau")

    print("\n--- NATURE EFFLUENT ---")
    detect_missing_mappings(all_eff, dict_eff, "Nature_effluent")

    print("\n--- DIAMÈTRE ---")
    detect_missing_mappings(all_dia, dict_dia, "Diametre")

    print("\n" + "="*70)
    print("[VÉRIFICATION RAPIDE] Terminée — corrige le fichier Excel avant de relancer")
    print("="*70)


def calculer_profondeur_moyenne_troncons_gdf(troncons, regards, c_rad, c_tn, c_prof,c_id_regard_am, c_id_regard_av, distance_max=2.0):
    """
    Version modifiée qui prend des GeoDataFrames en entrée
    """
    print("=" * 80)
    print("CALCUL DE LA PROFONDEUR MOYENNE DES TRONÇONS")
    print("=" * 80)
    
    print(f"\n1. Données reçues...")
    print(f"   - Tronçons : {len(troncons)}")
    print(f"   - Regards : {len(regards)}")
    print(f"   - CRS tronçons : {troncons.crs}")
    print(f"   - CRS regards : {regards.crs}")
    
    # Vérification du CRS
    if troncons.crs != regards.crs:
        print("   ⚠ Reprojection des regards dans le CRS des tronçons...")
        regards = regards.to_crs(troncons.crs)
    
    # === ÉTAPE 1 : Création des points aux extrémités ===
    print("\n2. Création des points aux extrémités des tronçons...")
    

    
    def get_point_amont(geom):
        if geom is None or geom.is_empty: return None
        line = list(geom.geoms)[0] if geom.geom_type == 'MultiLineString' else geom
        return Point(line.coords[0])

    def get_point_aval(geom):
        if geom is None or geom.is_empty: return None
        line = list(geom.geoms)[0] if geom.geom_type == 'MultiLineString' else geom
        return Point(line.coords[-1])

    
    gdf_amont = gpd.GeoDataFrame(
        {'id_troncon': troncons.index},
        geometry=troncons.geometry.apply(get_point_amont),
        crs=troncons.crs
    ).dropna(subset=['geometry'])

    gdf_aval = gpd.GeoDataFrame(
        {'id_troncon': troncons.index},
        geometry=troncons.geometry.apply(get_point_aval),
        crs=troncons.crs
    ).dropna(subset=['geometry'])

    
    print(f"   - Points créés : {len(gdf_amont)} amont, {len(gdf_aval)} aval")
    
    # === ÉTAPE 2 : Jointure spatiale par proximité ===
    print(f"\n3. Jointure spatiale (distance max = {distance_max}m)...")
    
    join_amont = gpd.sjoin_nearest(
        gdf_amont,
        regards,
        how='left',
        max_distance=distance_max,
        distance_col='dist_amont'
    )
    
    join_aval = gpd.sjoin_nearest(
        gdf_aval,
        regards,
        how='left',
        max_distance=distance_max,
        distance_col='dist_aval'
    )
    
    join_amont = join_amont.sort_values('dist_amont').groupby('id_troncon').first()
    join_aval = join_aval.sort_values('dist_aval').groupby('id_troncon').first()
    
    print(f"   - Jointures réussies amont : {join_amont['dist_amont'].notna().sum()}/{len(join_amont)}")
    print(f"   - Jointures réussies aval : {join_aval['dist_aval'].notna().sum()}/{len(join_aval)}")
    
    # === ÉTAPE 3 : Ajout des champs ===
    print("\n4. Ajout des champs aux tronçons...")
    
    champ_radier = c_rad if (pd.notna(c_rad) and c_rad in regards.columns) else None
    champ_tn = c_tn if (pd.notna(c_tn) and c_tn in regards.columns) else None
    champ_prof = c_prof if (pd.notna(c_prof) and c_prof in regards.columns) else None
    
    result = troncons.copy()

    champ_id_reg_am = c_id_regard_am if (pd.notna(c_id_regard_am) and c_id_regard_am in regards.columns) else None
    champ_id_reg_av = c_id_regard_av if (pd.notna(c_id_regard_av) and c_id_regard_av in regards.columns) else None

    if champ_id_reg_am:
        result['id_regard_am'] = result.index.map(lambda x: join_amont.loc[x, champ_id_reg_am] if x in join_amont.index else np.nan)
    if champ_id_reg_av:
        result['id_regard_av'] = result.index.map(lambda x: join_aval.loc[x, champ_id_reg_av] if x in join_aval.index else np.nan)
    
    # On définit les colonnes à créer et les sources
    mappings = [
        ('radier_amont', champ_radier, join_amont), ('radier_aval', champ_radier, join_aval),
        ('tn_amont', champ_tn, join_amont), ('tn_aval', champ_tn, join_aval),
        ('prof_amont', champ_prof, join_amont), ('prof_aval', champ_prof, join_aval)
    ]

    for col_dest, col_src, join_df in mappings:
        if col_src:
            result[col_dest] = result.index.map(lambda x: join_df.loc[x, col_src] if x in join_df.index else np.nan)
            result[col_dest] = pd.to_numeric(result[col_dest], errors='coerce')
    
    # === ÉTAPE 4 : Tests de vraisemblance ===
    print("\n5. Tests de vraisemblance des données...")
    
    result['test_coherence_radier'] = True
    result['test_coherence_tn'] = True
    result['test_profondeur'] = True
    result['alerte_vraisemblance'] = ''
    
    if champ_radier:
        mask_radier = result['radier_amont'].notna() & result['radier_aval'].notna()
        pente_negative = result.loc[mask_radier, 'radier_amont'] < result.loc[mask_radier, 'radier_aval']
        result.loc[mask_radier & pente_negative, 'test_coherence_radier'] = False
        result.loc[mask_radier & pente_negative, 'alerte_vraisemblance'] += 'PENTE_NEGATIVE; '
        print(f"   - Tronçons avec pente négative : {pente_negative.sum()}")
    
    if champ_radier and champ_tn:
        mask_tn = result['radier_amont'].notna() & result['tn_amont'].notna()
        tn_inf_radier_amont = result.loc[mask_tn, 'tn_amont'] < result.loc[mask_tn, 'radier_amont']
        
        mask_tn_aval = result['radier_aval'].notna() & result['tn_aval'].notna()
        tn_inf_radier_aval = result.loc[mask_tn_aval, 'tn_aval'] < result.loc[mask_tn_aval, 'radier_aval']
        
        result.loc[mask_tn & tn_inf_radier_amont, 'test_coherence_tn'] = False
        result.loc[mask_tn & tn_inf_radier_amont, 'alerte_vraisemblance'] += 'TN<RADIER_AMONT; '
        
        result.loc[mask_tn_aval & tn_inf_radier_aval, 'test_coherence_tn'] = False
        result.loc[mask_tn_aval & tn_inf_radier_aval, 'alerte_vraisemblance'] += 'TN<RADIER_AVAL; '
        
        print(f"   - Tronçons avec TN < Radier : {(tn_inf_radier_amont.sum() + tn_inf_radier_aval.sum())}")
        
    
    # === ÉTAPE 5 : Calcul profondeur moyenne ===
    print("\n6. Calcul de la profondeur moyenne...")

    for c in ['prof_amont', 'prof_aval', 'tn_amont', 'tn_aval', 'radier_amont', 'radier_aval']:
        if c in result.columns:
            result[c] = result[c].astype(float)
    
    result['profondeur_moy'] = np.nan
    result['methode_calcul'] = ''
    
    if champ_prof:
        mask_prof = result['prof_amont'].notna() & result['prof_aval'].notna()
        result.loc[mask_prof, 'profondeur_moy'] = (result.loc[mask_prof, 'prof_amont'] + result.loc[mask_prof, 'prof_aval']) / 2
        result.loc[mask_prof, 'methode_calcul'] = 'MOYENNE_PROFONDEURS'
        nb_methode1 = mask_prof.sum()
    else:
        nb_methode1 = 0
    
    if champ_tn and champ_radier:
        mask_calcul = (result['profondeur_moy'].isna()) & \
                      (result['tn_amont'].notna()) & (result['radier_amont'].notna()) & \
                      (result['tn_aval'].notna()) & (result['radier_aval'].notna())
        
        prof_amont_calc = result.loc[mask_calcul, 'tn_amont'] - result.loc[mask_calcul, 'radier_amont']
        prof_aval_calc = result.loc[mask_calcul, 'tn_aval'] - result.loc[mask_calcul, 'radier_aval']
        
        result.loc[mask_calcul, 'profondeur_moy'] = (prof_amont_calc + prof_aval_calc) / 2
        result.loc[mask_calcul, 'methode_calcul'] = 'CALCUL_TN_RADIER'
        nb_methode2 = mask_calcul.sum()
    else:
        nb_methode2 = 0
    
    tests_ok = result['test_coherence_radier'] & result['test_coherence_tn']
    result.loc[~tests_ok, 'profondeur_moy'] = np.nan
    result.loc[~tests_ok, 'methode_calcul'] += '_DONNEES_DOUTEUSES'
    
    nb_tests_echec = (~tests_ok).sum()
    
    # === RÉSUMÉ ===
    print("\n" + "=" * 80)
    print("RÉSUMÉ DES CALCULS")
    print("=" * 80)
    print(f"Total tronçons : {len(result)}")
    print(f"Profondeur calculée (méthode 1 - profondeurs) : {nb_methode1}")
    print(f"Profondeur calculée (méthode 2 - TN/Radier) : {nb_methode2}")
    print(f"Données douteuses (tests échoués) : {nb_tests_echec}")
    print(f"Tronçons sans calcul possible : {result['profondeur_moy'].isna().sum()}")
    
    if result['profondeur_moy'].notna().any():
        print(f"\nProfondeur moyenne min : {result['profondeur_moy'].min():.2f}m")
        print(f"Profondeur moyenne max : {result['profondeur_moy'].max():.2f}m")
        print(f"Profondeur moyenne globale : {result['profondeur_moy'].mean():.2f}m")
    
    print("\n" + "=" * 80)
    
    return result


def detect_missing_mappings(series, mapping_dict, label):
        """Identifie les valeurs presentes dans les donnees mais absentes du dictionnaire."""
        unique_vals = set(series.dropna().unique())
        mapped_keys = set(mapping_dict.keys())
        missing = unique_vals - mapped_keys
        
        missing = {m for m in missing if m not in VALEURS_NULLES}
        
        if missing:
            print(f"[ALERTE] Modalites de '{label}' non trouvees dans le mapping Excel :")
            for m in sorted(list(missing)):
                print(f"   - {m}")
        return missing

def extract_classes_to_bins(df_classes, col_max, col_label):
    thresholds = pd.to_numeric(df_classes[col_max], errors='coerce').dropna().tolist()
    bins = [float('-inf')] + sorted(thresholds) + [float('inf')]
    
    labels = df_classes[col_label].dropna().astype(str).tolist()
    
    # Vérification
    assert len(labels) == len(bins) - 1, \
        f"Mismatch : {len(labels)} labels vs {len(bins)-1} intervalles"
    
    return bins, labels

def process_complex_dates(series):
    """
    Convertit en datetime puis extrait l'annee. 
    Gere : 2013-09-02T00:00:00, 15/10/2000, 1969/01/31, 1990.0
    """
    s = series.astype(str).str.strip().replace(VALEURS_NULLES, np.nan)
    
    parsed_dates = pd.to_datetime(s, dayfirst=True, format='mixed', errors='coerce', utc=True)
    years_dt = parsed_dates.dt.year

    years_regex = s.str.extract(r'((?:18|19|20)\d{2})')[0]
    years_regex = pd.to_numeric(years_regex, errors='coerce')

    return years_dt.fillna(years_regex)


def plot_statistiques_completes(df, save_path_completude):
    """Génère le graphique de complétude + statistiques descriptives par variable."""

    # --- CORRECTION ICI : On force l'utilisation du dossier dicté par Celery ---
    output_dir_base = os.getenv("OUTPUT_DIR", ".")
    save_path_completude = os.path.join(output_dir_base, "Stats_Completude_Pretraitement.png")
    
    save_dir = os.path.join(output_dir_base, "statistiques")
    os.makedirs(save_dir, exist_ok=True)
    print(f"[INFO] Dossier de sortie des stats : {save_dir}")

    # ── GRAPHE 1 : Complétude (inchangé) ──────────────────────────────────────
    print("[INFO] Génération du graphique de complétude...")
    colonnes_a_exclure = [
        'Y_aval', 'X_aval', 'Y_amont', 'X_amont', 'id_troncon', 'id_regard_av', 
        'id_regard_am', 'geometry', 'Ville', 'classe_profondeur', 'Age_extracted',
        'Diametre', 'Age', 'File_amont', 'File_aval', 'Longueur',
        'Prof_Aval', 'Prof_Amont', 'Cote_TN_am', 'Cote_TN_av', 'Cote_TN', 'nom_rue'
    ]
    attributs_analyse = [col for col in df.columns if col not in colonnes_a_exclure]
    stats = df[attributs_analyse].notna().mean() * 100
    stats = stats.sort_values(ascending=True)

    plt.figure(figsize=(14, 10))
    colors = ['skyblue' if x >= 50 else 'salmon' for x in stats]
    ax = stats.plot(kind='barh', color=colors)
    plt.title("Taux de complétude des données après prétraitement", fontsize=16, fontweight='bold')
    plt.xlabel("Taux de remplissage (%)", fontsize=12)
    plt.xlim(0, 110)
    for i, v in enumerate(stats):
        ax.text(v + 1, i, f"{v:.1f}%", color='black', va='center', fontweight='bold')
    plt.axvline(x=50, color='red', linestyle='--', linewidth=2, label='Seuil (50%)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path_completude, dpi=150)
    plt.close()
    print(f"   → Sauvegardé : {save_path_completude}")



    # ── GRAPHE 2 : longueur_calc — Histogramme + Boîte à moustaches ───────────
    if 'longueur_calc' in df.columns:
        print("[INFO] Génération graphique longueur_calc...")
        data = df['longueur_calc'].dropna()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.hist(data, bins=60, color='steelblue', edgecolor='white')
        ax1.set_title("Distribution — Longueur calculée", fontweight='bold')
        ax1.set_xlabel("Longueur (m)")
        ax1.set_ylabel("Effectif")
        ax1.axvline(data.median(), color='red', linestyle='--', label=f"Médiane : {data.median():.1f}m")
        ax1.legend()

        ax2.boxplot(data, vert=True, patch_artist=True,
                    boxprops=dict(facecolor='steelblue', alpha=0.6),
                    medianprops=dict(color='red', linewidth=2))
        ax2.set_title("Boîte à moustaches — Longueur calculée", fontweight='bold')
        ax2.set_ylabel("Longueur (m)")
        ax2.set_xticks([])

        stats_txt = f"n={len(data):,}  |  moy={data.mean():.1f}m  |  méd={data.median():.1f}m  |  max={data.max():.1f}m"
        fig.suptitle(stats_txt, fontsize=10, y=0.02)
        plt.tight_layout()
        path = os.path.join(save_dir, "Stats_longueur_calc.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"   → Sauvegardé : {path}")

    # ── GRAPHE 3 : Pente — Histogramme + Boîte à moustaches ──────────────────
    if 'Pente' in df.columns:
        print("[INFO] Génération graphique Pente...")
        data = df['Pente'].dropna()
        # Filtrage des outliers visuels (>99e percentile) pour lisibilité
        p1, p99 = data.quantile(0.01), data.quantile(0.99)
        data_clip = data.clip(p1, p99)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.hist(data_clip, bins=60, color='darkorange', edgecolor='white')
        ax1.set_title("Distribution — Pente (valeurs entre P1 et P99)", fontweight='bold')
        ax1.set_xlabel("Pente (%)")
        ax1.set_ylabel("Effectif")
        ax1.axvline(0, color='black', linestyle='-', linewidth=1, label='Pente = 0')
        ax1.axvline(data.median(), color='red', linestyle='--', label=f"Médiane : {data.median():.2f}%")
        ax1.legend()

        ax2.boxplot(data_clip, vert=True, patch_artist=True,
                    boxprops=dict(facecolor='darkorange', alpha=0.6),
                    medianprops=dict(color='red', linewidth=2))
        ax2.set_title("Boîte à moustaches — Pente (P1-P99)", fontweight='bold')
        ax2.set_ylabel("Pente (%)")
        ax2.set_xticks([])

        pct_neg = (data < 0).mean() * 100
        stats_txt = (f"n={len(data):,}  |  moy={data.mean():.2f}%  |  méd={data.median():.2f}%  "
                     f"|  contre-pentes : {pct_neg:.1f}%")
        fig.suptitle(stats_txt, fontsize=10, y=0.02)
        plt.tight_layout()
        path = os.path.join(save_dir, "Stats_Pente.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"   → Sauvegardé : {path}")

    # ── GRAPHE 4 : Diametre_clean — Histogramme + Boîte à moustaches ─────────
    if 'Diametre_clean' in df.columns:
        print("[INFO] Génération graphique Diametre_clean...")
        data = df['Diametre_clean'].dropna()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.hist(data, bins=40, color='mediumseagreen', edgecolor='white')
        ax1.set_title("Distribution — Diamètre (mm)", fontweight='bold')
        ax1.set_xlabel("Diamètre (mm)")
        ax1.set_ylabel("Effectif")
        ax1.axvline(data.median(), color='red', linestyle='--', label=f"Médiane : {data.median():.0f}mm")
        ax1.legend()

        ax2.boxplot(data, vert=True, patch_artist=True,
                    boxprops=dict(facecolor='mediumseagreen', alpha=0.6),
                    medianprops=dict(color='red', linewidth=2))
        ax2.set_title("Boîte à moustaches — Diamètre (mm)", fontweight='bold')
        ax2.set_ylabel("Diamètre (mm)")
        ax2.set_xticks([])

        stats_txt = (f"n={len(data):,}  |  moy={data.mean():.0f}mm  |  méd={data.median():.0f}mm  "
                     f"|  min={data.min():.0f}mm  |  max={data.max():.0f}mm")
        fig.suptitle(stats_txt, fontsize=10, y=0.02)
        plt.tight_layout()
        path = os.path.join(save_dir, "Stats_Diametre_clean.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"   → Sauvegardé : {path}")

    # ── GRAPHE 5 : Nature_effluent — Camembert + Barres ───────────────────────
    if 'Nature_effluent' in df.columns:
        print("[INFO] Génération graphique Nature_effluent...")
        counts = df['Nature_effluent'].value_counts(dropna=True)
        couleurs = [plt.cm.Set2(i / len(counts)) for i in range(len(counts))]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        ax1.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
                startangle=90, colors=couleurs)
        ax1.set_title("Répartition — Nature de l'effluent", fontweight='bold')

        ax2.barh(range(len(counts)), counts.values, color=couleurs, edgecolor='white')
        ax2.set_yticks(range(len(counts)))
        ax2.set_yticklabels(counts.index)
        ax2.set_title("Effectifs — Nature de l'effluent", fontweight='bold')
        ax2.set_xlabel("Nombre de tronçons")
        for i, v in enumerate(counts.values):
            ax2.text(v + counts.max() * 0.01, i, f"{v:,}", va='center')

        plt.tight_layout()
        path = os.path.join(save_dir, "Stats_Nature_effluent.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"   → Sauvegardé : {path}")

    # ── GRAPHE 6 : Materiau — Barres horizontales triées ─────────────────────
    if 'Materiau' in df.columns:
        print("[INFO] Génération graphique Materiau...")
        counts = df['Materiau'].value_counts(dropna=True)
        couleurs = [plt.cm.tab10(i / len(counts)) for i in range(len(counts))]

        fig, ax = plt.subplots(figsize=(12, max(5, len(counts) * 0.5)))
        ax.barh(range(len(counts)), counts.values, color=couleurs, edgecolor='white')
        ax.set_yticks(range(len(counts)))
        ax.set_yticklabels(counts.index)
        ax.invert_yaxis()
        ax.set_title("Répartition des matériaux", fontweight='bold', fontsize=14)
        ax.set_xlabel("Nombre de tronçons")
        for i, v in enumerate(counts.values):
            ax.text(v + counts.max() * 0.005, i, f"{v:,}  ({v/len(df)*100:.1f}%)", va='center')

        plt.tight_layout()
        path = os.path.join(save_dir, "Stats_Materiau.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"   → Sauvegardé : {path}")

    # ── GRAPHE 7 : classe_age — Barres verticales ordonnées ──────────────────
    if 'classe_age' in df.columns:
        print("[INFO] Génération graphique classe_age...")
        counts = df['classe_age'].value_counts(dropna=True).sort_index()

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(counts.index.astype(str), counts.values,
                      color='cornflowerblue', edgecolor='white')
        ax.set_title("Répartition par classe d'âge", fontweight='bold', fontsize=14)
        ax.set_xlabel("Classe d'âge")
        ax.set_ylabel("Nombre de tronçons")
        for bar, v in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + counts.max() * 0.01,
                    f"{v:,}\n({v/counts.sum()*100:.1f}%)", ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        path = os.path.join(save_dir, "Stats_classe_age.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"   → Sauvegardé : {path}")

    # ── GRAPHE 8 : Distribution par année de pose ─────────────────────────────
    if 'Age_extracted' in df.columns:
        print("[INFO] Génération graphique distribution par année de pose...")
        data = df['Age_extracted'].dropna()
        # Filtrer les années aberrantes (avant 1880 ou après l'année courante)
        annee_courante = pd.Timestamp.now().year
        data = data[(data >= 1880) & (data <= annee_courante)]

        # Comptage par année
        counts_year = data.astype(int).value_counts().sort_index()

        # Calcul de l'âge moyen (en années depuis aujourd'hui)
        age_moyen_annees = annee_courante - data.mean()
        annee_moyenne_pose = data.mean()

        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(counts_year.index, counts_year.values,
                color='steelblue', linewidth=1.5, marker='o', markersize=4)
        ax.fill_between(counts_year.index, counts_year.values,
                        alpha=0.15, color='steelblue')
        
        ax.xaxis.set_major_locator(ticker.MultipleLocator(20))

        ax.set_title("Distribution du nombre de conduites par année de pose",
                     fontweight='bold', fontsize=14)
        ax.set_xlabel("Année de pose")
        ax.set_ylabel("Nombre de conduites")
        ax.grid(True, alpha=0.3)

        # Ligne de l'année moyenne
        ax.axvline(annee_moyenne_pose, color='red', linestyle='--', linewidth=1.5,
                   label=f"Année moyenne de pose : {annee_moyenne_pose:.0f}\n"
                         f"(Âge moyen du réseau : {age_moyen_annees:.0f} ans)")
        ax.legend(fontsize=10)

        stats_txt = (f"n={len(data):,} conduites datées  |  "
                     f"Année min : {int(data.min())}  |  "
                     f"Année max : {int(data.max())}  |  "
                     f"Âge moyen : {age_moyen_annees:.1f} ans  |  "
                     f"Médiane pose : {int(data.median())}")
        fig.suptitle(stats_txt, fontsize=9, y=0.01)

        plt.tight_layout()
        path = os.path.join(save_dir, "Stats_distribution_annee_pose.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"   → Sauvegardé : {path}")
        print(f"   ℹ Âge moyen du réseau : {age_moyen_annees:.1f} ans (pose moyenne : {annee_moyenne_pose:.0f})")

    # ── GRAPHE 9 : Matrice de corrélation ─────────────────────────────────────
    print("[INFO] Génération de la matrice de corrélation...")
    
    COLS_CORR = [
        'longueur_calc', 'Pente', 'Profondeur_finale', 'Diametre_clean',
        'taux_arbo', 'taux_imper', 'Age_extracted',
        'X_centroid', 'Y_centroid',
        # 'Traffic', 'speedLimit',   # ← décommenter quand disponibles
    ]
    cols_corr_present = [c for c in COLS_CORR if c in df.columns]
    
    if len(cols_corr_present) >= 2:
        corr_matrix = df[cols_corr_present].corr(method='pearson')
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        plt.colorbar(im, ax=ax, label='Corrélation de Pearson', shrink=0.8)
        
        ax.set_xticks(range(len(cols_corr_present)))
        ax.set_yticks(range(len(cols_corr_present)))
        ax.set_xticklabels(cols_corr_present, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(cols_corr_present, fontsize=10)
        
        # Valeurs dans chaque cellule
        for i in range(len(cols_corr_present)):
            for j in range(len(cols_corr_present)):
                val = corr_matrix.iloc[i, j]
                # Texte blanc si fond foncé, noir sinon
                color = 'white' if abs(val) > 0.6 else 'black'
                ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                        fontsize=9, fontweight='bold', color=color)
        
        ax.set_title("Matrice de corrélation — Variables numériques",
                     fontweight='bold', fontsize=14, pad=20)
        
        plt.tight_layout()
        path = os.path.join(save_dir, "Stats_correlation_matrix.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   → Sauvegardé : {path}")
    else:
        print("   ⚠ Pas assez de colonnes numériques pour la matrice de corrélation")





def normalize_dia(val):
    """Normalise un diamètre vers string minimal : '000000001'→'1', '9.5600'→'9.56'"""
    val_str = str(val).strip()
    if val_str in VALEURS_NULLES + ['nan']:
        return val_str
    try:
        num = float(val_str)
        return str(int(num)) if num == int(num) else str(round(num, 6)).rstrip('0').rstrip('.')
    except ValueError:
        return val_str


def main_preprocessing():
    print("[INFO] Demarrage du pretraitement detaille et optimise...")

    # try:
    #     df_map_villes = pd.read_excel(PATH_EXCEL_MAPPING, sheet_name="Resumé variables", header=2)
    # except Exception as e:
    #     print(f"[ERREUR] Impossible de lire l'onglet 'Resumé variables' : {e}")
    #     return
    
    # 1. LECTURE DES DICTIONNAIRES DE MAPPING ET DES CLASSES
    try:
        df_mat = pd.read_excel(PATH_EXCEL_MAPPING, sheet_name="Modalites_Materiaux")
        dict_mat = dict(zip(df_mat['Materiau_uni'].astype(str).str.strip(), df_mat['Recodage 3']))

        df_eff = pd.read_excel(PATH_EXCEL_MAPPING, sheet_name="Modalites_Effluents")
        dict_eff = dict(zip(df_eff['Nature_effluent_uni'].astype(str).str.strip(), df_eff['Recodage 2']))

        df_dia = pd.read_excel(PATH_EXCEL_MAPPING, sheet_name="Modalites_Diametres_3")
        dict_dia = {
            normalize_dia(k): v
            for k, v in zip(df_dia['Diametre_unii'].astype(str).str.strip(), df_dia['Recodage_22'])
        }

        # verifier_mappings_rapide(df_map_villes, dict_mat, dict_eff, dict_dia)

        df_classes = pd.read_excel(PATH_EXCEL_MAPPING, sheet_name="Definition_classes_old")
        
        bins_age, labels_age = extract_classes_to_bins(df_classes, 'max', 'classe age')
        bins_prof, labels_prof = extract_classes_to_bins(df_classes, 'max.1', 'classe profondeur')
        
    except Exception as e:
        print(f"[ERREUR] Erreur lors de la lecture des mappings : {e}")
        return

    # 2. LECTURE DU MAPPING DES VILLES ET CHARGEMENT DES SHAPEFILES
    try:
        df_map_villes = pd.read_excel(PATH_EXCEL_MAPPING, sheet_name="Resumé variables", header=2)
    except Exception as e:
        print(f"[ERREUR] Impossible de lire l'onglet 'Resumé variables' : {e}")
        return

    list_dfs = []
    print("[INFO] Chargement des Shapefiles SIG par ville...")
    
    for _, row in tqdm(df_map_villes.iterrows(), total=df_map_villes.shape[0], desc="Chargement des villes"):
        ville_nom = str(row['Agglo']).strip()
        if pd.isna(ville_nom) or ville_nom == 'nan': continue 
        
        dossier_name = str(row['Nom dossier'])
        fichier_troncons = str(row['Nom_BDD_troncons'])
        fichier_regards = str(row['Nom_BDD_regards'])
        col_id_regard_am = row['id_regard_am'] if pd.notna(row['id_regard_am']) else None
        col_id_regard_av = row['id_regard_av'] if pd.notna(row['id_regard_av']) else None
        
        path_t = os.path.join(PATH_BASE, dossier_name, f"{fichier_troncons}.shp")
        path_r = os.path.join(PATH_BASE, dossier_name, f"{fichier_regards}.shp")
        
        if not os.path.exists(path_t):
            print(f"[ATTENTION] Tronçons introuvables pour {ville_nom}")
            continue

        try:
            try:
                gdf_t = gpd.read_file(path_t)
            except:
                gdf_t = gpd.read_file(path_t, encoding='cp1252')

            # --- AJOUT DU FILTRE ANTI-CRASH BIC ICI ---
            # On supprime les géométries None, vides, les points isolés et les lignes de 0m
            gdf_t = gdf_t[gdf_t.geometry.notna() & ~gdf_t.geometry.is_empty].copy()
            gdf_t = gdf_t[gdf_t.geometry.type.isin(['LineString', 'MultiLineString'])]
            gdf_t = gdf_t[gdf_t.geometry.length > 0.001].copy()

            # Filtre supplémentaire — éliminer les géométries avec < 2 points
            def has_enough_coords(geom):
                try:
                    if geom is None or geom.is_empty:
                        return False
                    line = list(geom.geoms)[0] if geom.geom_type == 'MultiLineString' else geom
                    return len(line.coords) >= 2
                except Exception:
                    return False

            nb_avant = len(gdf_t)
            gdf_t = gdf_t[gdf_t.geometry.apply(has_enough_coords)].copy()
            nb_filtre = nb_avant - len(gdf_t)
            if nb_filtre > 0:
                print(f"   ⚠ {nb_filtre} géométries dégénérées supprimées pour {ville_nom}")

            # make_valid dans un try/except pour ne pas crasher
            def safe_make_valid(geom):
                try:
                    return geom.make_valid()
                except Exception:
                    return geom

            gdf_t['geometry'] = gdf_t.geometry.apply(safe_make_valid)
            # -------------------------------------------

            col_radier_regard = row['cote_radier_r']
            col_tn_regard = row['cote_tn_r']
            col_prof_regard = row['profondeur_r']

            if gdf_t.crs is None:
                gdf_t.set_crs("EPSG:2154", inplace=True)

            if os.path.exists(path_r):
                try:
                    gdf_r = gpd.read_file(path_r)
                except:
                    gdf_r = gpd.read_file(path_r, encoding='cp1252')

                if gdf_r.crs is None:
                    gdf_r.set_crs("EPSG:2154", inplace=True)

                gdf_t = calculer_profondeur_moyenne_troncons_gdf(gdf_t, gdf_r, col_radier_regard, col_tn_regard, col_prof_regard, col_id_regard_am, col_id_regard_av)
            else:
                print(f"   ! Regards absents pour {ville_nom}, calcul spatial impossible.")
                gdf_t['profondeur_moy'] = np.nan
                gdf_t['radier_amont']   = np.nan
                gdf_t['tn_amont']       = np.nan
                gdf_t['radier_aval']    = np.nan
                gdf_t['tn_aval']        = np.nan

            # Application du mapping
            mapping_local = {row[attr]: attr for attr in ATTRIBUTS_UNIFIES if pd.notna(row[attr]) and row[attr] in gdf_t.columns}
            df_ville = gdf_t.rename(columns=mapping_local).copy()
            colonnes_utiles = [c for c in ATTRIBUTS_UNIFIES if c in df_ville.columns] + ['geometry', 'radier_amont', 'tn_amont', 'radier_aval', 'tn_aval']
            df_ville = df_ville[colonnes_utiles]

            # --- DÉTECTION DES DOUBLONS POUR LE DEBUG ---
            cols_dupliquees = df_ville.columns[df_ville.columns.duplicated()].unique().tolist()
            if cols_dupliquees:
                print(f"\n[DÉBOGAGE] ⚠️ Doublons détectés dans la ville : {ville_nom}")
                for col in cols_dupliquees:
                    print(f"   -> La colonne '{col}' est présente plusieurs fois.")
                    nb_cols = list(df_ville.columns).count(col)
                    print(f"      (Apparaît {nb_cols} fois après le renommage)")

            colonnes_speciales = ['profondeur_moy']
            for col in colonnes_speciales:
                if col in gdf_t.columns:
                    df_ville[col] = gdf_t[col]
                    
            if 'profondeur_moy' in gdf_t.columns:
                df_ville['Profondeur_finale'] = gdf_t['profondeur_moy']

            for attr in ATTRIBUTS_UNIFIES:
                if attr not in df_ville.columns: df_ville[attr] = np.nan
            
            df_ville['Ville'] = ville_nom
            df_ville = df_ville.to_crs("EPSG:2154")
            list_dfs.append(df_ville)
            
        except Exception as e:
            print(f"[ERREUR] Sur {ville_nom} : {e}")

    if not list_dfs:
        print("[ERREUR] Aucune donnée chargée.")
        return

    df_global = gpd.GeoDataFrame(pd.concat(list_dfs, ignore_index=True), crs="EPSG:2154")
    
    print("[INFO] Calcul de la longueur géométrique des tronçons...")
    df_global['longueur_calc'] = df_global.geometry.length.round(3)

    count_avant_longueur = len(df_global)
    df_global = df_global[df_global['longueur_calc'] >= 1.0].copy()
    print(f"[INFO] Tronçons supprimés (longueur_calc < 1m) : {count_avant_longueur - len(df_global)}")


    # ── Coordonnées du centroïde (moyenne amont/aval) ────────────────────────
    print("[INFO] Calcul des coordonnées centroïdes des tronçons...")
    df_global['X_centroid'] = (pd.to_numeric(df_global['X_amont'], errors='coerce') +
                                 pd.to_numeric(df_global['X_aval'], errors='coerce')) / 2
    df_global['Y_centroid'] = (pd.to_numeric(df_global['Y_amont'], errors='coerce') +
                                pd.to_numeric(df_global['Y_aval'], errors='coerce')) / 2
    print(f"   → X_centroid/Y_centroid renseignés : {df_global['X_centroid'].notna().sum():,} tronçons")


    # 3. NETTOYAGE TEXTUEL GLOBAL
    print("[INFO] Nettoyage des valeurs manquantes...")
    df_global = df_global.replace(VALEURS_NULLES, np.nan)

    # 4. RECODAGE DES MODALITES
    print("[INFO] Recodage des materiaux, effluents et diametres...")
    if 'Materiau' in df_global.columns:
        clean_mat = df_global['Materiau'].astype(str).str.strip()
        detect_missing_mappings(clean_mat, dict_mat, "Materiau")
        df_global['Materiau'] = df_global['Materiau'].astype(str).str.strip().map(dict_mat)
        df_global['Materiau'] = df_global['Materiau'].replace(['INCONNU', 'inconnu', 'None', 'nan'], np.nan)
        
    if 'Nature_effluent' in df_global.columns:
        clean_eff = df_global['Nature_effluent'].astype(str).str.strip()
        detect_missing_mappings(clean_eff, dict_eff, "Nature_effluent")
        df_global['Nature_effluent'] = df_global['Nature_effluent'].astype(str).str.strip().map(dict_eff)
        df_global['Nature_effluent'] = df_global['Nature_effluent'].replace(['INCONNU', 'inconnu', 'None', 'nan'], np.nan)

    if 'Diametre' in df_global.columns:
        df_global['Diametre'] = df_global['Diametre'].apply(
            lambda x: normalize_dia(x) if pd.notna(x) else np.nan
        )
        df_global['Diametre'] = df_global['Diametre'].replace(VALEURS_NULLES + ['nan'], np.nan)
        
        detect_missing_mappings(df_global['Diametre'], dict_dia, "Diametre")
        df_global['Diametre_clean'] = df_global['Diametre'].map(dict_dia)
        df_global['Diametre_clean'] = df_global['Diametre_clean'].replace(['INCONNU', 'inconnu', 'None', 'nan'], np.nan)

        df_global['Diametre_clean'] = pd.to_numeric(df_global['Diametre_clean'], errors='coerce')
        df_global.loc[df_global['Diametre_clean'] < 100,  'Diametre_clean'] = np.nan
        df_global.loc[df_global['Diametre_clean'] > 1500, 'Diametre_clean'] = np.nan
        print(f"[INFO] Diametre_clean : {df_global['Diametre_clean'].notna().sum():,} valeurs valides "
            f"({df_global['Diametre_clean'].notna().mean()*100:.1f}%)")

    # 5. CONVERSION EN NUMERIQUE ET GESTION DES ZEROS
    print("[INFO] Conversion des types numeriques...")
    for col in COLS_NUMERIQUES:
        if col in df_global.columns:
            df_global[col] = pd.to_numeric(df_global[col], errors='coerce')
            if col not in COLS_ZERO_VALIDE:
                df_global[col] = df_global[col].replace(0, np.nan)

    # --- AJOUT DU FILTRAGE DES TAUX ARBO ICI ---
    if 'taux_arbo' in df_global.columns:
        df_global.loc[df_global['taux_arbo'] > 100, 'taux_arbo'] = np.nan
    # --------------------------------------------

    # 6. TRAITEMENT DE L'AGE (TOUS FORMATS) ET CLASSEMENT DIRECT
    # Age = année de pose, on extrait l'année et on classe directement
    print("[INFO] Traitement de l'age...")
    if 'Age' in df_global.columns:
        df_global['Age_extracted'] = process_complex_dates(df_global['Age'])
        df_global['classe_age'] = pd.cut(df_global['Age_extracted'], bins=bins_age, labels=labels_age, right=False)
    
    # 7. FINALISATION DE LA PROFONDEUR ET CLASSEMENT
    print("[INFO] Finalisation et classement des profondeurs...")
    
    if 'Profondeur_finale' not in df_global.columns:
        df_global['Profondeur_finale'] = np.nan

    # Fallback 1 : Prof_Amont / Prof_Aval du tronçon
    backup_prof = df_global[['Prof_Amont', 'Prof_Aval']].mean(axis=1)
    df_global['Profondeur_finale'] = df_global['Profondeur_finale'].fillna(backup_prof)

    # ── Fallback 2 : Cotes TN et files d'eau du tronçon ──────────────────────
    file_moy = (pd.to_numeric(df_global['File_amont'], errors='coerce') +
                pd.to_numeric(df_global['File_aval'],  errors='coerce')) / 2

    # Cas A : Cote_TN disponible → profondeur = Cote_TN − moyenne des files
    if 'Cote_TN' in df_global.columns:
        cote_tn = pd.to_numeric(df_global['Cote_TN'], errors='coerce')
        backup_tn = cote_tn - file_moy
        df_global['Profondeur_finale'] = df_global['Profondeur_finale'].fillna(
            backup_tn.where(cote_tn.notna())
        )

    # Cas B : Cote_TN_am + Cote_TN_av → moyenne TN − moyenne des files
    if 'Cote_TN_am' in df_global.columns and 'Cote_TN_av' in df_global.columns:
        cote_tn_moy = (pd.to_numeric(df_global['Cote_TN_am'], errors='coerce') +
                       pd.to_numeric(df_global['Cote_TN_av'],  errors='coerce')) / 2
        backup_tn_moy = cote_tn_moy - file_moy
        df_global['Profondeur_finale'] = df_global['Profondeur_finale'].fillna(
            backup_tn_moy.where(cote_tn_moy.notna())
        )
    # ─────────────────────────────────────────────────────────────────────────

    df_global['classe_profondeur'] = pd.cut(df_global['Profondeur_finale'], bins=bins_prof, labels=labels_prof, right=True)

    if 'Profondeur_finale' in df_global.columns:
        df_global.loc[df_global['Profondeur_finale'] > 10, 'Profondeur_finale'] = np.nan
        df_global.loc[df_global['Profondeur_finale'] <= 0, 'Profondeur_finale'] = np.nan

    # 8. TRAITEMENT DE LA PENTE ET VERIFICATION DE COHERENCE
    print("[INFO] Calcul et verification de la coherence des pentes...")

    if 'radier_amont' in df_global.columns and 'radier_aval' in df_global.columns:
        df_global['File_amont'] = df_global['File_amont'].fillna(df_global['radier_amont'])
        df_global['File_aval'] = df_global['File_aval'].fillna(df_global['radier_aval'])

    if all(col in df_global.columns for col in ['Pente', 'File_amont', 'File_aval', 'Longueur']):
        
        mask_long = df_global['Longueur'] > 0
        
        mask_suspect_am = df_global['Pente'].notna() & \
                          df_global['File_amont'].notna() & \
                          (df_global['File_aval'].isna() | (df_global['File_aval'] == 0)) & \
                          mask_long
        
        pente_theorique_am = (df_global['File_amont'] / df_global['Longueur']) * 100
        is_broken_am = (df_global['Pente'].abs() - pente_theorique_am.abs()).abs() < 0.01
        df_global.loc[mask_suspect_am & is_broken_am, 'Pente'] = np.nan

        mask_suspect_av = df_global['Pente'].notna() & \
                          df_global['File_aval'].notna() & \
                          (df_global['File_amont'].isna() | (df_global['File_amont'] == 0)) & \
                          mask_long
        
        pente_theorique_av = (df_global['File_aval'] / df_global['Longueur']) * 100
        is_broken_av = (df_global['Pente'].abs() - pente_theorique_av.abs()).abs() < 0.01
        df_global.loc[mask_suspect_av & is_broken_av, 'Pente'] = np.nan

        mask_calc = df_global['Pente'].isna() & \
                      df_global['File_amont'].notna() & \
                      df_global['File_aval'].notna() & \
                      (df_global['File_aval'] != 0) & \
                      mask_long
        
        df_global.loc[mask_calc, 'Pente'] = \
            ((df_global['File_amont'] - df_global['File_aval']) / df_global['Longueur']) * 100

        mask_signe = (df_global['File_amont'] > df_global['File_aval']) & (df_global['Pente'] < 0)
        df_global.loc[mask_signe, 'Pente'] = df_global.loc[mask_signe, 'Pente'].abs()
        
        mask_contre_pente = (df_global['File_amont'] < df_global['File_aval']) & (df_global['Pente'] > 0)
        df_global.loc[mask_contre_pente, 'Pente'] = -df_global.loc[mask_contre_pente, 'Pente'].abs()

        df_global.loc[df_global['Pente'].abs() > 10, 'Pente'] = np.nan
        
    # --- FILTRATION FINALE (Filtres metiers) ---
    print("[INFO] Application des filtres de suppression (Classes 1/6 et Extremes)...")
    
    initial_count = len(df_global)

    mask_age_extreme = df_global['classe_age'].astype(str).isin(['1', '6'])
    df_global.loc[mask_age_extreme, 'classe_age'] = np.nan
    
    mask_prof_extreme = df_global['classe_profondeur'].astype(str).isin(['1', '5'])
    df_global.loc[mask_prof_extreme, 'classe_profondeur'] = np.nan

    print(f"[INFO] Classes d'âge '1' et '6' passées en NaN : {mask_age_extreme.sum()}")
    print(f"[INFO] Classes de profondeur '1' et '5' passees en INCONNU : {mask_prof_extreme.sum()}")

    # =========================================================================
    # 11. EXTRACTION DES NŒUDS — fait ICI, avant le filtrage des colonnes,
    #     car on a encore besoin de id_regard_am/av, tn_*, radier_*, X/Y
    # =========================================================================
    print("[INFO] Extraction des nœuds (regards) uniques pour le GCN...")
    
    nodes_am = df_global[['id_regard_am', 'tn_amont', 'radier_amont', 'X_amont', 'Y_amont']].rename(
        columns={'id_regard_am': 'id_node', 'tn_amont': 'tn', 'radier_amont': 'radier', 'X_amont': 'x', 'Y_amont': 'y'})
    
    nodes_av = df_global[['id_regard_av', 'tn_aval', 'radier_aval', 'X_aval', 'Y_aval']].rename(
        columns={'id_regard_av': 'id_node', 'tn_aval': 'tn', 'radier_aval': 'radier', 'X_aval': 'x', 'Y_aval': 'y'})

    df_nodes = pd.concat([nodes_am, nodes_av]).dropna(subset=['id_node']).drop_duplicates(subset=['id_node'])

    df_nodes['id_node'] = df_nodes['id_node'].astype(str)
    df_nodes = df_nodes.dropna(subset=['id_node']).drop_duplicates(subset=['id_node'])
    df_nodes = df_nodes.reset_index(drop=True)
    df_nodes['node_index'] = df_nodes.index

    # === FILTRAGE FINAL DES COLONNES ===
    # On garde : ATTRIBUTS_UNIFIES sans id_regard_am, id_regard_av, id_troncon
    # (car ces IDs peuvent se répéter entre villes)
    # + les colonnes calculées demandées
    print(f"[INFO] Nettoyage du surplus de colonnes ({len(df_global.columns)} colonnes détectées)...")

    ATTRIBUTS_FILTRE = [
        a for a in ATTRIBUTS_UNIFIES
        if a not in ('id_troncon', 'nom_rue', 'Longueur', 'Age', 'Diametre')          # 'id_regard_am', 'id_regard_av'
    ]

    COLS_A_GARDER = ATTRIBUTS_FILTRE + [
        'Ville', 'Profondeur_finale', 'Age_extracted', 'longueur_calc', 'Traffic', 'speedLimit',
        'classe_age', 'Diametre_clean', 'classe_profondeur', 'geometry', 'X_centroid', 'Y_centroid',
    ]

    colonnes_finales = [c for c in COLS_A_GARDER if c in df_global.columns]
    df_global = df_global[colonnes_finales]

    print(f"[INFO] Colonnes conservées : {len(df_global.columns)}")
    print(f"[INFO] Colonnes : {colonnes_finales}")


    # 9. GENERATION DES SORTIES
    plot_statistiques_completes(df_global, OUTPUT_PLOT)


    # 10. SAUVEGARDES FINALES
    # --- Sécurité anti-crash PyOGRIO : déduplication finale des colonnes ---
    cols_dup = df_global.columns[df_global.columns.duplicated()].unique().tolist()
    if cols_dup:
        print(f"[ATTENTION] Colonnes dupliquées détectées avant sauvegarde : {cols_dup} → suppression des doublons")
        df_global = df_global.loc[:, ~df_global.columns.duplicated(keep='first')]

    if os.path.exists(OUTPUT_GPKG):
        os.remove(OUTPUT_GPKG)

    # 1. SAUVEGARDE DU CSV EN PREMIER (Sécurité absolue)
    print(f"\n[INFO] Sauvegarde du CSV ({OUTPUT_FILE})...")
    df_csv = df_global.drop(columns='geometry', errors='ignore')
    # Convertir les dates aberrantes (2999) ou objets complexes en texte pour ne pas planter
    for col in df_csv.columns:
        if df_csv[col].dtype == 'datetime64[ns]' or df_csv[col].dtype == 'object':
            df_csv[col] = df_csv[col].astype(str)
    
    df_csv.to_csv(OUTPUT_FILE, index=False, sep=';', encoding='utf-8-sig')
    print("   ✓ CSV sauvegardé avec succès")

    # 2. SAUVEGARDE DU GEOPACKAGE (Uniquement les tronçons)
    print(f"[INFO] Sauvegarde du GeoPackage ({OUTPUT_GPKG})...")
    try:
        # On force aussi les types object/datetime en string pour le GPKG
        for col in df_global.columns:
            if col != 'geometry' and (df_global[col].dtype == 'object' or df_global[col].dtype == 'datetime64[ns]'):
                df_global[col] = df_global[col].astype(str)
                
        df_global.to_file(OUTPUT_GPKG, layer='edges', driver="GPKG", engine="pyogrio")
        print("   ✓ GeoPackage sauvegardé avec succès")
    except Exception as e:
        print(f"   [ERREUR FATALE] Impossible de sauvegarder le GeoPackage : {e}")

if __name__ == "__main__":
     main_preprocessing()