"""
=============================================================================
COMPLÉTION DE LA BDD — PRÉDICTION DE LA CLASSE D'ÂGE
=============================================================================
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import os
import joblib
import time
from sklearn.impute import KNNImputer

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
# INPUT_FILE   = "/home/user-ia/ML2/resultats_comparatif_materiau/BDD_Completee_Materiau_sup_70.gpkg"
# OUTPUT_FILE  = "/home/user-ia/ML2/resultats_comparatif_age/BDD_Finale_Complete_Age_Materiau.csv"
# OUTPUT_GPKG  = "/home/user-ia/ML2/resultats_comparatif_age/BDD_Finale_Complete_Age_Materiau.gpkg"
# MODEL_DIR    = "/home/user-ia/ML2/resultats_comparatif_age"
# BEST_MODEL   = "model_rf.pkl"

INPUT_FILE  = _env_required("INPUT_FILE")
MODEL_DIR   = _env_required("MODEL_DIR")
BEST_MODEL  = _env_optional("BEST_MODEL", "model_rf.pkl")
OUTPUT_FILE = _env_required("OUTPUT_FILE")
OUTPUT_GPKG = _env_required("OUTPUT_GPKG")
KNN_K        = 7

LABELS_AGE = {2: '1950-1970', 3: '1970-1990', 4: '1990-2010', 5: '2010-2025'}

# =============================================================================
# 1. CHARGEMENT DES OUTILS SAUVEGARDÉS
# =============================================================================
print("\n" + "="*70)
print("[1] Chargement des outils sauvegardés...")
print("="*70)

model            = joblib.load(os.path.join(MODEL_DIR, BEST_MODEL))
scaler           = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
le_target        = joblib.load(os.path.join(MODEL_DIR, "le_target.pkl"))
label_encoders   = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
FEATURES_FINALES = joblib.load(os.path.join(MODEL_DIR, "features_finales.pkl"))

print(f"    Scaler n_features : {scaler.n_features_in_}")
print(f"    FEATURES_FINALES  : {len(FEATURES_FINALES)}")
print(f"    ✓ Modèle chargé       : {BEST_MODEL}")
print(f"    ✓ Features utilisées  : {FEATURES_FINALES}")
print(f"    ✓ Classes connues     : {list(le_target.classes_)}")

# =============================================================================
# 2. CHARGEMENT DES DONNÉES
# =============================================================================
print("\n" + "="*70)
print("[2] Chargement des données...")
print("="*70)

df = gpd.read_file(INPUT_FILE) # <--- REMPLACER pd.read_csv
print(f"    {len(df):,} tronçons | {len(df.columns)} colonnes (Géométrie incluse)")

# Conversion numérique classe_age
df['classe_age'] = pd.to_numeric(df['classe_age'], errors='coerce')

n_total     = len(df)
# Tronçons avec classe_age valide (2,3,4,5) = labellisés
mask_valide  = df['classe_age'].isin([2, 3, 4, 5])
n_labellise  = mask_valide.sum()
n_manquant   = (~mask_valide).sum()

print(f"\n    Classe âge renseignée (2-5) : {n_labellise:,} ({n_labellise/n_total*100:.1f}%)")
print(f"    Classe âge manquante        : {n_manquant:,}  ({n_manquant/n_total*100:.1f}%)")

# Distribution actuelle
print(f"\n    Distribution actuelle :")
for cls in [2, 3, 4, 5]:
    n = (df['classe_age'] == cls).sum()
    print(f"    Classe {cls} ({LABELS_AGE[cls]}) : {n:,} ({n/n_total*100:.1f}%)")

# =============================================================================
# 3. PRÉTRAITEMENT (identique au script d'entraînement)
# =============================================================================
print("\n" + "="*70)
print("[3] Prétraitement...")
print("="*70)

# log1p
for col in ['Diametre_clean', 'longueur_calc']:
    if col in df.columns:
        df[col] = np.log1p(df[col].clip(lower=0))
        print(f"    ✓ log1p appliqué sur {col}")

# Encodage catégorielles — même LabelEncoders que l'entraînement âge
COLS_CAT = ['Nature_effluent', 'Ville', 'Materiau']
for col in COLS_CAT:
    if col in df.columns and col in label_encoders:
        le            = label_encoders[col]
        known_classes = set(le.classes_)
        fallback      = le.classes_[0]

        col_temp = df[col].fillna(fallback).astype(str)
        col_temp = col_temp.apply(lambda x: x if x in known_classes else fallback)
        df[col + '_enc'] = le.transform(col_temp)

        print(f"    ✓ Encodé : {col} ({df[col].nunique()} modalités)")

# =============================================================================
# 4. SÉPARATION — tronçons à prédire vs tronçons déjà labellisés
# =============================================================================
print("\n" + "="*70)
print("[4] Séparation des tronçons...")
print("="*70)

mask_manquant  = ~mask_valide
mask_labellise = mask_valide

df_a_predire = df[mask_manquant].copy()
df_labellise = df[mask_labellise].copy()

print(f"    Tronçons à prédire    : {len(df_a_predire):,}")
print(f"    Tronçons déjà connus  : {len(df_labellise):,}")

# Vérifier features présentes
cols_manquantes = [c for c in FEATURES_FINALES if c not in df.columns]
if cols_manquantes:
    print(f"\n    ⚠ Features absentes : {cols_manquantes}")
    for col in cols_manquantes:
        df_a_predire[col] = np.nan

# # =============================================================================
# # 4.5 — FILTRE COMPLÉTUDE >70% SUR LES TRONÇONS À PRÉDIRE
# # =============================================================================
# print("\n" + "="*70)
# print("[4.5] Filtre complétude >70% sur les tronçons à prédire...")
# print("="*70)

# COLS_NUM_BRUTES = [c for c in FEATURES_FINALES if '_enc' not in c]
# COLS_CAT_BRUTES = ['Nature_effluent', 'Ville', 'Materiau']
# COLS_FILTRE     = COLS_NUM_BRUTES + [c for c in COLS_CAT_BRUTES if c in df_a_predire.columns]

# n_avant_filtre = len(df_a_predire)
# taux_comp      = df_a_predire[COLS_FILTRE].notna().mean(axis=1)
# mask_complet   = taux_comp >= 0.7

# df_a_predire_filtre = df_a_predire[mask_complet].copy()
# df_a_predire_exclu  = df_a_predire[~mask_complet].copy()

# n_apres_filtre = len(df_a_predire_filtre)
# n_exclu        = n_avant_filtre - n_apres_filtre

# print(f"    Colonnes vérifiées : {COLS_FILTRE}")
# print(f"    Tronçons à prédire avant filtre : {n_avant_filtre:,}")
# print(f"    Tronçons conservés (≥70%)       : {n_apres_filtre:,} ({n_apres_filtre/n_avant_filtre*100:.1f}%)")
# print(f"    Tronçons exclus (<70%)           : {n_exclu:,} ({n_exclu/n_avant_filtre*100:.1f}%) → resteront NaN")

# df_a_predire = df_a_predire_filtre.copy()

# =============================================================================
# 5. IMPUTATION DES NaN
# =============================================================================
print("\n" + "="*70)
print("[5] Imputation KNN des valeurs manquantes...")
print("="*70)

imputer_path = os.path.join(MODEL_DIR, "knn_imputer.pkl")

if os.path.exists(imputer_path):
    print(f"    [CACHE] KNN Imputer chargé depuis {imputer_path}")
    imputer = joblib.load(imputer_path)
    t0 = time.time()
    X_a_predire = df_a_predire[FEATURES_FINALES].values.astype('float32')
    X_a_predire = imputer.transform(X_a_predire)
    print(f"    ✓ Imputation terminée en {time.time()-t0:.1f}s")
else:
    print(f"    ⚠ KNN Imputer non trouvé → fit sur données labellisées")
    t0 = time.time()
    X_a_predire  = df_a_predire[FEATURES_FINALES].values.astype('float32')
    X_labellise  = df_labellise[FEATURES_FINALES].values.astype('float32')
    imputer = KNNImputer(n_neighbors=KNN_K)
    imputer.fit(X_labellise)
    X_a_predire = imputer.transform(X_a_predire)
    print(f"    ✓ Imputation terminée en {time.time()-t0:.1f}s")

# =============================================================================
# 6. NORMALISATION (non utilisée pour RF mais gardée pour cohérence)
# =============================================================================
print("\n" + "="*70)
print("[6] Normalisation...")
print("="*70)

X_a_predire_sc = scaler.transform(X_a_predire)
print(f"    ✓ StandardScaler calculé (non utilisé pour RF — arbres de décision)")

# =============================================================================
# 7. PRÉDICTION
# =============================================================================
print("\n" + "="*70)
print("[7] Prédiction de la classe d'âge...")
print("="*70)

t0 = time.time()

# Prédiction — RF sur données NON scalées
y_pred_encoded = model.predict(X_a_predire)
y_pred_classes = le_target.inverse_transform(y_pred_encoded)  # → 2, 3, 4, 5

# Labels lisibles
y_pred_labels = np.array([LABELS_AGE.get(int(c), str(c)) for c in y_pred_classes])

# Probabilités de confiance
try:
    y_proba   = model.predict_proba(X_a_predire)
    confiance = y_proba.max(axis=1)
    a_confiance = True
    print(f"    ✓ Probabilités de confiance calculées")
except:
    a_confiance = False

print(f"    ✓ Prédiction terminée en {time.time()-t0:.1f}s")
print(f"    ✓ {len(y_pred_classes):,} tronçons prédits")

print(f"\n    Distribution des classes prédites :")
unique, counts = np.unique(y_pred_classes, return_counts=True)
for cls, n in zip(unique, counts):
    label = LABELS_AGE.get(int(cls), str(cls))
    print(f"    Classe {cls} ({label}) : {n:,} ({n/len(y_pred_classes)*100:.1f}%)")

# =============================================================================
# 8. RÉINJECTION DANS LE DATAFRAME
# =============================================================================
print("\n" + "="*70)
print("[8] Réinjection des prédictions...")
print("="*70)

mask_predit = df.index.isin(df_a_predire.index)

df['classe_age_source'] = 'original'
df.loc[mask_labellise, 'classe_age_source'] = 'original'
df.loc[mask_predit,    'classe_age_source'] = f'predit_{BEST_MODEL.replace(".pkl","")}'
df.loc[mask_manquant & ~mask_predit, 'classe_age_source'] = 'non_predit_incomplet'

df.loc[mask_predit, 'classe_age'] = y_pred_classes.astype(float)

if a_confiance:
    df['classe_age_confiance'] = np.nan
    df.loc[mask_predit,    'classe_age_confiance'] = confiance
    df.loc[mask_labellise, 'classe_age_confiance'] = 1.0
    print(f"    ✓ Confiance moyenne des prédictions : {confiance.mean()*100:.1f}%")
    print(f"    ✓ Prédictions avec confiance > 80%  : {(confiance > 0.8).sum():,} ({(confiance > 0.8).mean()*100:.1f}%)")
    print(f"    ✓ Prédictions avec confiance > 70%  : {(confiance > 0.7).sum():,} ({(confiance > 0.7).mean()*100:.1f}%)")
    print(f"    ✓ Prédictions avec confiance < 50%  : {(confiance < 0.5).sum():,} ({(confiance < 0.5).mean()*100:.1f}%) ← à vérifier")

# =============================================================================
# 9. STATISTIQUES FINALES
# =============================================================================
print("\n" + "="*70)
print("[9] Statistiques après complétion...")
print("="*70)

mask_complete = df['classe_age'].isin([2, 3, 4, 5])
n_complete    = mask_complete.sum()

print(f"\n    Avant : {n_labellise:,} tronçons labellisés ({n_labellise/n_total*100:.1f}%)")
print(f"    Après : {n_complete:,} tronçons labellisés ({n_complete/n_total*100:.1f}%)")

print(f"\n    Distribution finale :")
for cls in [2, 3, 4, 5]:
    n = (df['classe_age'] == cls).sum()
    print(f"    Classe {cls} ({LABELS_AGE[cls]}) : {n:,} ({n/n_total*100:.1f}%)")

print(f"\n    Répartition par source :")
src_counts = df['classe_age_source'].value_counts()
for src, n in src_counts.items():
    print(f"    {src:<30} : {n:,} ({n/n_total*100:.1f}%)")

print(f"\n    Tronçons restants sans classe_age : "
      f"{(~df['classe_age'].isin([2,3,4,5])).sum():,} "
      f"({(~df['classe_age'].isin([2,3,4,5])).mean()*100:.1f}%)")

# =============================================================================
# 10. SAUVEGARDE
# =============================================================================
print("\n" + "="*70)
print("[10] Sauvegarde...")
print("="*70)

# Supprimer colonnes encodées temporaires
cols_enc  = [c for c in df.columns if c.endswith('_enc')]
df_output = df.drop(columns=cols_enc)

# Inverser log1p
for col in ['Diametre_clean', 'longueur_calc']:
    if col in df_output.columns:
        df_output[col] = np.expm1(df_output[col])

# --- SAUVEGARDE 1 : GeoPackage (Pour QGIS) ---
print(f"    [INFO] Écriture du GeoPackage...")
df_output.to_file(OUTPUT_GPKG, driver="GPKG")
print(f"    ✓ GeoPackage sauvegardé : {OUTPUT_GPKG}")

# --- SAUVEGARDE 2 : CSV (Pour analyse de données sans géométrie) ---
print(f"    [INFO] Écriture du CSV...")
df_csv = pd.DataFrame(df_output.drop(columns='geometry', errors='ignore'))
df_csv.to_csv(OUTPUT_FILE, sep=';', encoding='utf-8-sig', index=False)
print(f"    ✓ CSV sauvegardé : {OUTPUT_FILE}")
print(f"    ✓ {len(df_output):,} tronçons | {len(df_output.columns)} colonnes")

print("\n" + "="*70)
print("COMPLÉTION CLASSE D'ÂGE TERMINÉE")
print("="*70)