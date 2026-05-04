# conda activate rapids_gpu
"""
═══════════════════════════════════════════════════════════════════════════════
COMPLETION DE L'ETAT DE SANTE (EDS) DES TRONCONS
═══════════════════════════════════════════════════════════════════════════════

Reprend les modeles entraines pour EDS4_j1 (par defaut) et complete les
troncons sans etat de sante via prediction.

Pipeline :
    1. Chargement de la BDD source (avec ou sans etat)
    2. Application du meme pretraitement que l'entrainement
        (filtrage completude, encodage categoriel, log transform, scaling)
    3. Imputation KNN avec le KNN_imputer.pkl sauvegarde a l'entrainement
    4. Prediction avec le modele choisi (ExtraTrees retenu)
    5. Calcul de la confiance (probabilite max)
    6. Sauvegarde de la BDD enrichie + statistiques

Sortie : un GeoPackage (et CSV) avec colonnes
    - <NIVEAU>_pred       : valeur predite (ou originale)
    - <NIVEAU>_source     : 'original' ou 'predit_<modele>' ou 'non_predit_incomplet'
    - <NIVEAU>_confiance  : probabilite associee (1.0 si original)
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

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


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

# # Source des troncons a completer
# INPUT_FILE  = "/run/user/1000/gvfs/smb-share:server=172.16.39.24,share=groupe-merlin/Lyon-Siege/Hors_Affaires/HYDRAU/COMMUN/Stagiaires/M-yassine/test/1m200k/output_gpkg_1m200k.gpkg"
# INPUT_LAYER = "geom_troncon_final"

# # Dossier contenant les modeles entraines (target choisie)
# TARGET_NAME    = "EDS4_j1"
# MODELS_DIR     = "/home/user-ia/ML2/resultats_multi_eds/EDS4_j1"

# # Modele a utiliser pour la prediction
# MODEL_FILENAME = "model_extratrees.pkl"
# MODEL_LABEL    = "ExtraTrees"

# Niveaux a completer (commenter pour desactiver)
NIVEAUX_A_COMPLETER = {
    # nom du niveau (= sous-dossier dans MODELS_DIR) : (nom_colonne_sortie, mapping post-prediction)
    "niveau_1_multiclasse": {
        "col_sortie":   "EDS_n1",
        "names":        ["Bon", "Moyen", "Mauvais", "Tres mauvais"],
        "mapping_back": lambda y: y + 1,   # 0..3 -> 1..4
    },
    "niveau_2_isolation_4": {
        "col_sortie":   "EDS_n2",
        "names":        ["Non critique", "Tres mauvais"],
        "mapping_back": lambda y: y,       # 0/1
    },
    "niveau_3_bon_vs_degrade": {
        "col_sortie":   "EDS_n3",
        "names":        ["Bon", "Degrade"],
        "mapping_back": lambda y: y,       # 0/1
    },
}

# # Sortie
# OUTPUT_DIR  = "/home/user-ia/ML2/resultats_completion_eds"
# OUTPUT_GPKG = os.path.join(OUTPUT_DIR, "BDD_Finale_Complete_EDS.gpkg")
# OUTPUT_CSV  = os.path.join(OUTPUT_DIR, "BDD_Finale_Complete_EDS.csv")


INPUT_FILE     = _env_required("INPUT_FILE")
INPUT_LAYER    = _env_required("INPUT_LAYER")
TARGET_NAME    = _env_required("TARGET_NAME")
MODELS_DIR     = _env_required("MODELS_DIR")
MODEL_FILENAME = _env_required("MODEL_FILENAME")
OUTPUT_DIR     = _env_required("OUTPUT_DIR")

# Champs dérivés (calculés depuis OUTPUT_DIR) — pas besoin d'env vars
MODEL_LABEL = _env_optional("MODEL_LABEL", "ExtraTrees")
OUTPUT_GPKG = os.path.join(OUTPUT_DIR, "BDD_Finale_Complete_EDS.gpkg")
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, "BDD_Finale_Complete_EDS.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Seuil de completude minimal pour autoriser une prediction
COMPLETUDE_MIN = 0.5   # 50%

# Features (DOIT correspondre exactement a l'entrainement)
COLS_NUM = [
    "X_centroid", "Y_centroid",
    "prof_moy", "diametre", "pente",
    "classe_age", "taux_arbo", "taux_imper",
    "length", "alea_argiles", "alea_nappes",
]
COLS_CAT = ["materiau_affecte", "nature_effluent"]


# ═════════════════════════════════════════════════════════════════════════════
# RECODAGE TARGET (pour identifier les "originaux" si la cible existe deja)
# ═════════════════════════════════════════════════════════════════════════════

def recoder_target(y_raw, niveau):
    """Recode 1-4 selon le niveau."""
    if niveau == "niveau_1_multiclasse":
        return y_raw - 1
    elif niveau == "niveau_2_isolation_4":
        return (y_raw == 4).astype(int)
    elif niveau == "niveau_3_bon_vs_degrade":
        return (y_raw >= 3).astype(int)
    else:
        raise ValueError(niveau)


# ═════════════════════════════════════════════════════════════════════════════
# [1] CHARGEMENT
# ═════════════════════════════════════════════════════════════════════════════

print("=" * 75)
print(f"COMPLETION EDS — Target = {TARGET_NAME} | Modele = {MODEL_LABEL}")
print("=" * 75)

print(f"\n[1] Chargement du GeoPackage source...")
gdf = gpd.read_file(INPUT_FILE, layer=INPUT_LAYER)
df  = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
print(f"    {len(df):,} troncons charges")

# Calcul centroides
gdf["X_centroid"] = gdf.geometry.centroid.x
gdf["Y_centroid"] = gdf.geometry.centroid.y
df["X_centroid"]  = gdf["X_centroid"].values
df["Y_centroid"]  = gdf["Y_centroid"].values

# COALESCE
if "profondeur_affectee" in df.columns:
    df["prof_moy"] = df["prof_moy"].fillna(df["profondeur_affectee"])
if "diametre_affecte" in df.columns:
    df["diametre"] = df["diametre"].fillna(df["diametre_affecte"])
if "nature_effluent" in df.columns:
    vals_ok = df.loc[
        df["nature_effluent"].notna() &
        (df["nature_effluent"] != "INCONNU"), "nature_effluent"
    ]
    if len(vals_ok) > 0:
        mode_eff = vals_ok.mode()[0]
        df["nature_effluent"] = df["nature_effluent"].replace("INCONNU", mode_eff)

# Log transform (numerique uniquement, identique a l'entrainement)
cols_num_present = [c for c in COLS_NUM if c in df.columns]
for col in ["length", "diametre"]:
    if col in df.columns and col in cols_num_present:
        df[col] = np.log1p(df[col].clip(lower=0))

# Pre-remplissage des categorielles avec INCONNU (encodage fait par niveau)
cols_cat_present = [c for c in COLS_CAT if c in df.columns]
for col in cols_cat_present:
    df[col] = df[col].fillna("INCONNU").astype(str)

print(f"    Colonnes numeriques  : {cols_num_present}")
print(f"    Colonnes categorielles : {cols_cat_present}")

# Calcul du taux de remplissage par troncon (decide si on peut predire)
COLS_TOUTES = cols_num_present + cols_cat_present
taux_remplissage = df[COLS_TOUTES].notna().mean(axis=1)
mask_completude = taux_remplissage >= COMPLETUDE_MIN
n_incomplets = (~mask_completude).sum()
print(f"    Troncons avec completude < {COMPLETUDE_MIN*100:.0f}% : {n_incomplets:,} "
      f"(non predits)")


# ═════════════════════════════════════════════════════════════════════════════
# [2] BOUCLE SUR LES NIVEAUX A COMPLETER
# ═════════════════════════════════════════════════════════════════════════════

resultats_completion = {}

for niveau, params in NIVEAUX_A_COMPLETER.items():

    print(f"\n{'='*75}")
    print(f"[NIVEAU] {niveau}")
    print(f"{'='*75}")

    niveau_dir         = os.path.join(MODELS_DIR, niveau)
    model_path         = os.path.join(niveau_dir, MODEL_FILENAME)
    imputer_path       = os.path.join(niveau_dir, "knn_imputer.pkl")
    encoders_path      = os.path.join(niveau_dir, "label_encoders.pkl")
    features_path      = os.path.join(niveau_dir, "features_finales.pkl")

    # Verification de la presence de tous les artefacts requis
    artefacts_requis = {
        "modele":          model_path,
        "imputer":         imputer_path,
        "label_encoders":  encoders_path,
        "features_finales": features_path,
    }
    manquants = [nom for nom, path in artefacts_requis.items() if not os.path.exists(path)]
    if manquants:
        print(f"    ⚠ Artefacts absents : {manquants} — niveau ignore")
        continue

    # Chargement de tous les artefacts entraines
    print(f"    Chargement modele           : {model_path}")
    model = joblib.load(model_path)
    print(f"    Chargement imputer          : {imputer_path}")
    imputer = joblib.load(imputer_path)
    print(f"    Chargement label_encoders   : {encoders_path}")
    label_encoders = joblib.load(encoders_path)
    print(f"    Chargement features_finales : {features_path}")
    FEATURES_FINALES = joblib.load(features_path)

    # Application des encoders sauvegardes (avec gestion des modalites inconnues)
    df_niveau = df.copy()
    for col in cols_cat_present:
        if col not in label_encoders:
            print(f"    ⚠ Encoder absent pour {col} — niveau ignore")
            FEATURES_FINALES = None
            break
        le = label_encoders[col]
        known    = set(le.classes_)
        fallback = le.classes_[0]
        # Remplace les modalites inconnues par le fallback (premiere modalite connue)
        col_safe = df_niveau[col].apply(lambda x: x if x in known else fallback)
        df_niveau[col + "_enc"] = le.transform(col_safe)
        n_unknown = (col_safe != df_niveau[col]).sum()
        if n_unknown > 0:
            print(f"      {col} : {n_unknown:,} modalites inconnues mappees sur '{fallback}'")

    if FEATURES_FINALES is None:
        continue

    col_sortie    = params["col_sortie"]
    class_names   = params["names"]
    mapping_back  = params["mapping_back"]
    col_source    = col_sortie + "_source"
    col_confiance = col_sortie + "_confiance"

    # Initialisation des colonnes de sortie
    df[col_sortie]    = np.nan
    df[col_source]    = "non_predit_incomplet"
    df[col_confiance] = 0.0

    # ── Identifier les "originaux" (target deja presente dans la BDD) ──
    n_original = 0
    if TARGET_NAME in df.columns:
        mask_original = df[TARGET_NAME].notna() & df[TARGET_NAME].isin([1, 2, 3, 4])
        if mask_original.sum() > 0:
            y_raw = df.loc[mask_original, TARGET_NAME].astype(int).values
            y_recode = recoder_target(y_raw, niveau)
            df.loc[mask_original, col_sortie]    = y_recode
            df.loc[mask_original, col_source]    = "original"
            df.loc[mask_original, col_confiance] = 1.0
            n_original = mask_original.sum()
            print(f"    Troncons avec etat original : {n_original:,}")

    # ── Identifier les troncons a predire ──
    if TARGET_NAME in df.columns:
        mask_a_predire = (~mask_original) & mask_completude
    else:
        mask_a_predire = mask_completude

    n_a_predire = mask_a_predire.sum()
    print(f"    Troncons a predire           : {n_a_predire:,}")

    if n_a_predire == 0:
        print(f"    ⚠ Aucun troncon a predire — niveau ignore")
        continue

    # ── Preparer X pour les troncons a predire ──
    X_predire = df_niveau.loc[mask_a_predire, FEATURES_FINALES].values
    
    # Imputation KNN avec l'imputer entraine
    print(f"    Imputation KNN sur {n_a_predire:,} troncons...")
    t0 = time.time()
    X_imputed = imputer.transform(X_predire)
    print(f"    ✓ Imputation terminee en {time.time()-t0:.1f}s")

    # ── Prediction ──
    print(f"    Prediction...")
    t0 = time.time()
    y_pred = model.predict(X_imputed).astype(int)
    y_proba = model.predict_proba(X_imputed)
    confiance = y_proba.max(axis=1)
    print(f"    ✓ Prediction terminee en {time.time()-t0:.1f}s")

    # Application du mapping (0..3 -> 1..4 en niveau 1, sinon identite)
    y_pred_mapped = mapping_back(y_pred)

    # Ecriture dans le df
    df.loc[mask_a_predire, col_sortie]    = y_pred_mapped
    df.loc[mask_a_predire, col_source]    = f"predit_{MODEL_LABEL}"
    df.loc[mask_a_predire, col_confiance] = confiance

    # ── Statistiques ──
    print(f"\n    --- STATISTIQUES {niveau} ---")
    print(f"    Originaux            : {n_original:,}")
    print(f"    Predits              : {n_a_predire:,}")
    print(f"    Non predits          : {n_incomplets:,}")

    # Distribution des predictions
    print(f"\n    Distribution des predictions :")
    pred_counts = pd.Series(y_pred_mapped).value_counts().sort_index()
    for cls_idx, n in pred_counts.items():
        # Recuperer le nom de classe lisible
        if niveau == "niveau_1_multiclasse":
            label_name = class_names[int(cls_idx) - 1]
        else:
            label_name = class_names[int(cls_idx)]
        pct = n / n_a_predire * 100
        print(f"      {label_name:<15} : {n:>8,}  ({pct:5.1f}%)")

    # Confiance par classe predite
    print(f"\n    Confiance moyenne par classe :")
    for cls_idx in pred_counts.index:
        if niveau == "niveau_1_multiclasse":
            label_name = class_names[int(cls_idx) - 1]
        else:
            label_name = class_names[int(cls_idx)]
        mask_cls = (y_pred_mapped == cls_idx)
        conf_cls = confiance[mask_cls]
        print(f"      {label_name:<15} : moy={conf_cls.mean():.1%}  "
              f"min={conf_cls.min():.1%}  n={mask_cls.sum():,}")

    # Distribution de la confiance
    seuils = [0.5, 0.6, 0.7, 0.8, 0.9]
    print(f"\n    Repartition de la confiance :")
    for s in seuils:
        n_sup = (confiance >= s).sum()
        pct = n_sup / n_a_predire * 100
        print(f"      Confiance >= {s:.1f} : {n_sup:>8,}  ({pct:5.1f}%)")

    resultats_completion[niveau] = {
        "n_original":   int(n_original),
        "n_predit":     int(n_a_predire),
        "n_incomplet":  int(n_incomplets),
        "confiance_moy": float(confiance.mean()),
        "confiance_med": float(np.median(confiance)),
    }

    # ── Figure : distribution de la confiance ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogramme global
    axes[0].hist(confiance, bins=40, color="steelblue", alpha=0.85, edgecolor="white")
    axes[0].axvline(confiance.mean(), color="red", linestyle="--",
                    label=f"Moyenne : {confiance.mean():.1%}")
    axes[0].axvline(0.7, color="orange", linestyle="--",
                    label="Seuil 70%")
    axes[0].set_xlabel("Confiance")
    axes[0].set_ylabel("Nombre de troncons")
    axes[0].set_title(f"Distribution confiance — {niveau}",
                      fontweight="bold", fontsize=11)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Barplot par classe
    classes_uniques = sorted(np.unique(y_pred_mapped))
    moyennes = []
    labels = []
    counts_b = []
    for cls in classes_uniques:
        if niveau == "niveau_1_multiclasse":
            lbl = class_names[int(cls) - 1]
        else:
            lbl = class_names[int(cls)]
        mask = (y_pred_mapped == cls)
        moyennes.append(confiance[mask].mean())
        labels.append(lbl)
        counts_b.append(mask.sum())

    couleurs_b = ["#e74c3c" if m < 0.7 else "#f39c12" if m < 0.8 else "#2ecc71"
                  for m in moyennes]
    bars = axes[1].barh(labels, moyennes, color=couleurs_b, alpha=0.85,
                         edgecolor="white")
    for i, (m, n) in enumerate(zip(moyennes, counts_b)):
        axes[1].text(m + 0.02, i, f"{m:.1%} (n={n:,})",
                     va="center", fontsize=9)
    axes[1].axvline(0.7, color="red", linestyle="--", alpha=0.7,
                    label="Seuil 70%")
    axes[1].set_xlim(0, 1.15)
    axes[1].set_xlabel("Confiance moyenne")
    axes[1].set_title(f"Confiance par classe predite — {niveau}",
                      fontweight="bold", fontsize=11)
    axes[1].legend()
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"confiance_{niveau}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    ✓ Figure : confiance_{niveau}.png")


# ═════════════════════════════════════════════════════════════════════════════
# [3] SAUVEGARDE
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*75}")
print("[3] Sauvegarde de la BDD enrichie")
print(f"{'='*75}")

# Sauvegarde CSV (sans geometry)
df.to_csv(OUTPUT_CSV, sep=";", encoding="utf-8-sig", index=False)
print(f"    ✓ CSV  : {OUTPUT_CSV}")

# Sauvegarde GPKG (avec geometry)
gdf_out = gpd.GeoDataFrame(df, geometry=gdf.geometry.values, crs=gdf.crs)
if os.path.exists(OUTPUT_GPKG):
    os.remove(OUTPUT_GPKG)
gdf_out.to_file(OUTPUT_GPKG, driver="GPKG")
print(f"    ✓ GPKG : {OUTPUT_GPKG}")


# ═════════════════════════════════════════════════════════════════════════════
# [4] RAPPORT FINAL
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*75}")
print("RAPPORT FINAL")
print(f"{'='*75}")

rapport = []
rapport.append("=" * 75)
rapport.append(f"COMPLETION EDS — RAPPORT FINAL")
rapport.append(f"Target  : {TARGET_NAME}")
rapport.append(f"Modele  : {MODEL_LABEL}")
rapport.append(f"Source  : {INPUT_FILE}")
rapport.append("=" * 75)
rapport.append("")
rapport.append(f"Total troncons charges       : {len(df):,}")
rapport.append(f"Troncons incomplets exclus   : {n_incomplets:,}")
rapport.append("")

for niveau, stats in resultats_completion.items():
    rapport.append(f"--- {niveau} ---")
    rapport.append(f"  Originaux         : {stats['n_original']:>8,}")
    rapport.append(f"  Predits           : {stats['n_predit']:>8,}")
    rapport.append(f"  Non predits       : {stats['n_incomplet']:>8,}")
    rapport.append(f"  Confiance moyenne : {stats['confiance_moy']:.1%}")
    rapport.append(f"  Confiance mediane : {stats['confiance_med']:.1%}")
    rapport.append("")

rapport.append(f"Fichier de sortie : {OUTPUT_GPKG}")
rapport.append("=" * 75)

rapport_str = "\n".join(rapport)
print(rapport_str)

with open(os.path.join(OUTPUT_DIR, "rapport_completion.txt"), "w",
          encoding="utf-8") as f:
    f.write(rapport_str)

print(f"\n  ✓ Tous les fichiers dans : {OUTPUT_DIR}/")