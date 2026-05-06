# -*- coding: utf-8 -*-
"""
Script de production : Étude et optimisation du modèle de prédiction des Matériaux
Modèle : Random Forest Classifier avec recherche d'hyperparamètres (RandomizedSearchCV)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import warnings
import time
from collections import Counter

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)
import joblib
from imblearn.over_sampling import SMOTE
import geopandas as gpd

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

# =============================================================================
# CONFIGURATION DU PROJET
# =============================================================================
# INPUT_FILE   = "BDD_Pretraitee_Finale_plastique_divisé.csv"
# OUTPUT_DIR   = "/home/user-ia/ML2/resultats_comparatif_materiau"

INPUT_FILE = _env_required("INPUT_FILE")
OUTPUT_DIR = _env_required("OUTPUT_DIR")
RANDOM_STATE = 42
TEST_SIZE    = 0.2
KNN_K        = 7

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Variables prédictives
COLS_NUM = [
    'X_centroid', 'Y_centroid',
    'longueur_calc',
    'Pente',
    'Profondeur_finale',
    'Diametre_clean',
    'Age_extracted',
]

COLS_CAT = ['Nature_effluent', 'Ville']
TARGET = 'Materiau'

# =============================================================================
# 3. GESTION DU CACHE
# =============================================================================
imputer_path  = os.path.join(OUTPUT_DIR, "knn_imputer.pkl")
scaler_path   = os.path.join(OUTPUT_DIR, "scaler.pkl")
le_target_path = os.path.join(OUTPUT_DIR, "le_target.pkl")
encoders_path = os.path.join(OUTPUT_DIR, "label_encoders.pkl")
features_path = os.path.join(OUTPUT_DIR, "features_finales.pkl")
model_path    = os.path.join(OUTPUT_DIR, "model_rf.pkl")

cache_complet = all(os.path.exists(p) for p in [
    imputer_path, scaler_path, le_target_path, encoders_path, features_path, model_path
])

if cache_complet:
    print(f"\n[CACHE] Tous les artefacts trouvés dans {OUTPUT_DIR}")
    print(f"        → Pas de réentraînement, fin du script.")
    sys.exit(0)

# =============================================================================
# 1. CHARGEMENT ET NETTOYAGE DES DONNÉES
# =============================================================================
print("\n" + "="*70)
print("[1] Chargement et filtrage initial...")
print("="*70)

import geopandas as gpd

if INPUT_FILE.lower().endswith(".gpkg"):
    print(f"    Lecture GeoPackage : {INPUT_FILE}")
    gdf = gpd.read_file(INPUT_FILE)
    df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
else:
    print(f"    Lecture CSV : {INPUT_FILE}")
    try:
        df = pd.read_csv(INPUT_FILE, sep=';', encoding='utf-8-sig', low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(INPUT_FILE, sep=';', encoding='latin-1', low_memory=False)

print(f"    {len(df):,} tronçons chargés.")

if TARGET in df.columns:
    df = df[df[TARGET].notna()].copy()
    print(f"\nDistribution des classes {TARGET} :")
    counts = df[TARGET].value_counts()
    for cls, n in counts.items():
        print(f"    {cls:<15} : {n:,}")

# Filtrage de complétude (Seuil de 70% de données présentes par ligne)
COLS_TOUTES = [c for c in COLS_NUM if c in df.columns] + [c for c in COLS_CAT if c in df.columns]
n_avant = len(df)
# Note : Le filtrage est commenté dans votre script d'origine, il reste donc désactivé ici.
# taux_remplissage = df[COLS_TOUTES].notna().mean(axis=1)
# df = df[taux_remplissage >= 0.7].copy()
n_apres = len(df)
print(f"    Tronçons conservés : {n_apres:,}")

# =============================================================================
# 2. ENCODAGE ET PRÉPARATION DES FEATURES
# =============================================================================
print("\n[2] Encodage des variables catégorielles...")

label_encoders = {}
cols_cat_present = [c for c in COLS_CAT if c in df.columns]

for col in cols_cat_present:
    df[col] = df[col].fillna('INCONNU').astype(str)
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(df[col])
    label_encoders[col] = le

COLS_CAT_ENC = [c + '_enc' for c in cols_cat_present]
FEATURES_FINALES = [c for c in COLS_NUM if c in df.columns] + COLS_CAT_ENC

# Transformation logarithmique
for col in ['Diametre_clean', 'longueur_calc']:
    if col in df.columns:
        df[col] = np.log1p(df[col].clip(lower=0))

X = df[FEATURES_FINALES].values
le_target = LabelEncoder()
y = le_target.fit_transform(df[TARGET])
CLASS_NAMES = le_target.classes_

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# =============================================================================
# 3. GESTION DE L'IMPUTATION (SYSTÈME DE CACHE)
# =============================================================================
X_train_path = os.path.join(OUTPUT_DIR, "X_train_imputed.npy")
X_test_path  = os.path.join(OUTPUT_DIR, "X_test_imputed.npy")
y_train_path = os.path.join(OUTPUT_DIR, "y_train.npy")
y_test_path  = os.path.join(OUTPUT_DIR, "y_test.npy")

if os.path.exists(X_train_path) and os.path.exists(X_test_path):
    print("\n[CACHE] Chargement des données imputées existantes...")
    X_train = np.load(X_train_path)
    X_test  = np.load(X_test_path)
    y_train = np.load(y_train_path)
    y_test  = np.load(y_test_path)
else:
    print(f"\n[3] Imputation KNN (k={KNN_K})...")
    t0 = time.time()
    imputer = KNNImputer(n_neighbors=KNN_K)
    X_train = imputer.fit_transform(X_train)
    X_test  = imputer.transform(X_test)
    
    np.save(X_train_path, X_train)
    np.save(X_test_path, X_test)
    np.save(y_train_path, y_train)
    np.save(y_test_path, y_test)
    joblib.dump(imputer, os.path.join(OUTPUT_DIR, "knn_imputer.pkl"))
    print(f"    ✓ Imputation terminée en {time.time()-t0:.1f}s")

# =============================================================================
# 4. RÉÉQUILIBRAGE DES CLASSES (SMOTE)
# =============================================================================
print("\n[4] Équilibrage des classes par SMOTE...")
classes_idx = {cls: i for i, cls in enumerate(le_target.classes_)}
counts_avant = Counter(y_train)

smote_strat = {}
cibles_smote = {'PP/PE': 50000, 'FONTE': 80000, 'GRES': 75000, 'AUTRE': 85000}
for cls, cible in cibles_smote.items():
    if cls in classes_idx:
        idx = classes_idx[cls]
        if counts_avant[idx] < cible:
            smote_strat[idx] = cible

smote = SMOTE(sampling_strategy=smote_strat, k_neighbors=5, random_state=RANDOM_STATE)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Split final pour validation interne
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=RANDOM_STATE, stratify=y_train
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)

# Sauvegarde des outils de preprocessing
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))
joblib.dump(le_target, os.path.join(OUTPUT_DIR, "le_target.pkl"))
joblib.dump(label_encoders, os.path.join(OUTPUT_DIR, "label_encoders.pkl"))
joblib.dump(FEATURES_FINALES, os.path.join(OUTPUT_DIR, "features_finales.pkl"))

# =============================================================================
# 5. ENTRAÎNEMENT ET OPTIMISATION (RANDOM FOREST)
# =============================================================================
resultats = {}

print("\n" + "="*70)
print("[5] Optimisation du modèle Random Forest")
print("="*70)

t0 = time.time()

# Grille de recherche pour l'optimisation
param_dist = {
    'n_estimators':      [1400, 1500, 1600],
    'max_depth':         [None, 30, 40],
    'min_samples_leaf':  [1, 2, 3],
    'max_features':      ['sqrt', 0.4],
    'max_samples':       [0.8, 0.9],
    'class_weight':      ['balanced_subsample', 'balanced'],
}

search = RandomizedSearchCV(
    RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE),
    param_distributions=param_dist,
    n_iter=30,
    scoring='f1_macro',
    cv=3,
    n_jobs=1,
    random_state=RANDOM_STATE,
    verbose=3,
)

print(f"[DEBUG] Début RandomizedSearchCV ({30} itérations × 3 folds)...", flush=True)
search.fit(X_train, y_train)
print(f"[DEBUG] RandomizedSearchCV terminé", flush=True)

print(f"\nMeilleurs paramètres trouvés : {search.best_params_}")

rf = search.best_estimator_
y_pred_rf = rf.predict(X_test)
duree_rf  = time.time() - t0

# Sauvegarde du modèle optimisé
joblib.dump(rf, os.path.join(OUTPUT_DIR, "model_rf.pkl"))

acc_rf = accuracy_score(y_test, y_pred_rf)
f1_rf  = f1_score(y_test, y_pred_rf, average='macro')
resultats['Random Forest'] = {
    'acc': acc_rf, 'f1': f1_rf, 'duree': duree_rf,
    'pred': y_pred_rf, 'importance': rf.feature_importances_
}

# =============================================================================
# 6. GÉNÉRATION DES RAPPORTS ET GRAPHIQUES
# =============================================================================

def sauvegarder_visuels(nom, res, output_dir, features, class_names, y_test):
    nom_f = nom.replace(" ", "_")
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, res['pred'])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=False, cmap='Blues', values_format='.2f')
    ax.set_title(f"{nom} - Matrice de Confusion", fontweight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"cm_{nom_f}.png"), dpi=150)
    plt.close()

    # Importance des variables
    imp = res['importance']
    idx = np.argsort(imp)
    plt.figure(figsize=(8, 6))
    plt.barh([features[i] for i in idx], imp[idx], color='steelblue')
    plt.title(f"Importance des variables - {nom}", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"fi_{nom_f}.png"), dpi=150)
    plt.close()

sauvegarder_visuels('Random Forest', resultats['Random Forest'], OUTPUT_DIR, FEATURES_FINALES, CLASS_NAMES, y_test)

# Export CSV des performances
df_res = pd.DataFrame([
    {'Modele': 'Random Forest Optimized', 
     'Accuracy': f"{acc_rf*100:.2f}%",
     'F1_Macro': f"{f1_rf*100:.2f}%", 
     'Duree_s': f"{duree_rf:.1f}"}
])
df_res.to_csv(os.path.join(OUTPUT_DIR, "resultats_comparatif.csv"), index=False, sep=';', encoding='utf-8-sig')

# Affichage du rapport final en console
print("\n" + "="*70)
print("RAPPORT DE PERFORMANCE DÉTAILLÉ")
print("="*70)
print(classification_report(y_test, y_pred_rf, target_names=CLASS_NAMES, digits=4))
print(f"Livrables générés dans : {OUTPUT_DIR}")
print("="*70)