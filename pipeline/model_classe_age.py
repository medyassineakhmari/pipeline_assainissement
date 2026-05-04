# -*- coding: utf-8 -*-
"""
Script de production : Prédiction de la classe d'âge des canalisations
Modèle retenu : Random Forest Classifier avec optimisation d'hyperparamètres
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import warnings
import time
from collections import Counter

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)
import joblib
from imblearn.over_sampling import SMOTE

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
# CONFIGURATION ET PARAMÈTRES
# =============================================================================
# INPUT_FILE = "/home/user-ia/ML3/resultats_comparatif_materiau/BDD_Completee_Materiau_sup_70.csv"
# OUTPUT_DIR = "/home/user-ia/ML3/resultats_comparatif_age"

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
    'Diametre_clean'
]

COLS_CAT = ['Nature_effluent', 'Ville', 'Materiau']
TARGET = 'classe_age'

# =============================================================================
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
# =============================================================================
print("\n" + "="*70)
print("[1] Chargement du dataset...")
print("="*70)

df = pd.read_csv(INPUT_FILE, sep=';', encoding='utf-8-sig', low_memory=False)
print(f"    {len(df):,} tronçons chargés.")

# Filtrage des classes d'âge valides (uniquement 2, 3, 4, 5)
if TARGET in df.columns:
    df[TARGET] = pd.to_numeric(df[TARGET], errors='coerce')
    df = df[df[TARGET].isin([2, 3, 4, 5])].copy()
    df[TARGET] = df[TARGET].astype(int)
    
    print(f"\nDistribution des classes cibles ({TARGET}) :")
    counts = df[TARGET].value_counts().sort_index()
    labels_map = {2: '1950-1970', 3: '1970-1990', 4: '1990-2010', 5: '2010-2025'}
    for cls, n in counts.items():
        print(f"    Classe {cls} ({labels_map[cls]}) : {n:,}")

# =============================================================================
# 1.5 NETTOYAGE ET FILTRAGE DE QUALITÉ
# =============================================================================
print("\n" + "="*70)
print("[1.5] Nettoyage et filtrage de complétude")
print("="*70)

n_avant = len(df)

# Filtrage sur la confiance du matériau (donnée d'entrée fiable requise)
mask_qualite = df['Materiau'].notna() & (df['Materiau_confiance'] >= 0.70)
df = df[mask_qualite].copy()
n_apres_qualite = len(df)

# Seuil de complétude sur les variables physiques (minimum 70% de remplissage)
cols_physiques = [c for c in COLS_NUM if c in df.columns] + ['Nature_effluent', 'Ville']
taux_remplissage = df[cols_physiques].notna().mean(axis=1)
df = df[taux_remplissage >= 0.70].copy()

n_final = len(df)
print(f"    Tronçons conservés : {n_final:,} (Taux de conservation : {n_final/n_avant*100:.1f}%)")

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

# Transformation logarithmique pour réduire la dispersion
for col in ['Diametre_clean', 'longueur_calc']:
    if col in df.columns:
        df[col] = np.log1p(df[col].clip(lower=0))

X = df[FEATURES_FINALES].values
le_target = LabelEncoder()
y = le_target.fit_transform(df[TARGET])

CLASS_NAMES_LABELS = ['1950-1970', '1970-1990', '1990-2010', '2010-2025']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# =============================================================================
# 3. GESTION DE L'IMPUTATION KNN (AVEC SYSTÈME DE CACHE)
# =============================================================================
X_train_path = os.path.join(OUTPUT_DIR, "X_train_imputed.npy")
X_test_path  = os.path.join(OUTPUT_DIR, "X_test_imputed.npy")
y_train_path = os.path.join(OUTPUT_DIR, "y_train.npy")
y_test_path  = os.path.join(OUTPUT_DIR, "y_test.npy")

if os.path.exists(X_train_path) and os.path.exists(X_test_path):
    print("\n[CACHE] Chargement des données imputées...")
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
    
    # Sauvegarde du cache
    np.save(X_train_path, X_train)
    np.save(X_test_path, X_test)
    np.save(y_train_path, y_train)
    np.save(y_test_path, y_test)
    joblib.dump(imputer, os.path.join(OUTPUT_DIR, "knn_imputer.pkl"))
    print(f"    ✓ Imputation terminée en {time.time()-t0:.1f}s")

# =============================================================================
# 4. ÉQUILIBRAGE DES CLASSES (SMOTE)
# =============================================================================
print("\n[4] Rééquilibrage des classes par SMOTE...")
counts_avant = Counter(y_train)

# Stratégie ciblée pour les classes minoritaires
smote_strat = {}
cibles_smote = {0: 100000, 3: 100000}
for idx_classe, cible in cibles_smote.items():
    if counts_avant[idx_classe] < cible:
        smote_strat[idx_classe] = cible

smote = SMOTE(sampling_strategy=smote_strat, k_neighbors=5, random_state=RANDOM_STATE)
X_train, y_train = smote.fit_resample(X_train, y_train)

print(f"    Volume après SMOTE : {len(X_train):,}")

# Mise à l'échelle (Scaling)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Sauvegarde des objets de preprocessing
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))
joblib.dump(le_target, os.path.join(OUTPUT_DIR, "le_target.pkl"))
joblib.dump(label_encoders, os.path.join(OUTPUT_DIR, "label_encoders.pkl"))
joblib.dump(FEATURES_FINALES, os.path.join(OUTPUT_DIR, "features_finales.pkl"))

# =============================================================================
# 5. ENTRAÎNEMENT ET OPTIMISATION DU MODÈLE (RANDOM FOREST)
# =============================================================================
print("\n" + "="*70)
print("[5] Optimisation et Entraînement du Random Forest")
print("="*70)

t0 = time.time()

# Espace de recherche pour l'optimisation
param_dist = {
    'n_estimators':      [500, 1000, 1500],
    'max_depth':         [None, 30, 40],
    'min_samples_leaf':  [1, 2, 5],
    'max_features':      ['sqrt'],
    'max_samples':       [0.8, 0.9],
    'class_weight':      ['balanced', 'balanced_subsample', None],
}

# Recherche aléatoire des meilleurs hyperparamètres
search = RandomizedSearchCV(
    RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE),
    param_distributions=param_dist,
    n_iter=10,
    scoring='f1_macro',
    cv=3,
    n_jobs=1,
    random_state=RANDOM_STATE,
    verbose=3,
)

search.fit(X_train, y_train)
print(f"\nMeilleurs paramètres trouvés : {search.best_params_}")

rf = search.best_estimator_
y_pred_rf = rf.predict(X_test)
duree_rf = time.time() - t0

# Sauvegarde du modèle final
joblib.dump(rf, os.path.join(OUTPUT_DIR, "model_rf.pkl"))

# =============================================================================
# 6. ÉVALUATION ET GÉNÉRATION DES RAPPORTS
# =============================================================================
acc_rf = accuracy_score(y_test, y_pred_rf)
f1_rf  = f1_score(y_test, y_pred_rf, average='macro')

print("\n" + "="*70)
print("PERFORMANCES DU MODÈLE FINAL")
print("="*70)
print(f"Accuracy : {acc_rf*100:.2f}%")
print(f"F1 Macro : {f1_rf*100:.2f}%")
print("\nRapport de classification détaillé :")
print(classification_report(y_test, y_pred_rf, target_names=CLASS_NAMES_LABELS, digits=3))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred_rf)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=CLASS_NAMES_LABELS)
disp.plot(ax=ax, cmap='Blues', values_format='.2f')
ax.set_title(f"Random Forest - Matrice de Confusion\nF1 Macro : {f1_rf*100:.2f}%", fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "matrice_confusion_final.png"), dpi=150)

# Importance des variables
imp = rf.feature_importances_
idx = np.argsort(imp)
plt.figure(figsize=(10, 6))
plt.barh([FEATURES_FINALES[i] for i in idx], imp[idx], color='steelblue')
plt.title("Importance des variables prédictives", fontweight='bold')
plt.xlabel("Coefficient d'importance")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance_final.png"), dpi=150)

# Export des résultats en CSV
res_df = pd.DataFrame([{
    'Modele': 'Random Forest Optimized',
    'Accuracy': f"{acc_rf*100:.2f}%",
    'F1_Macro': f"{f1_rf*100:.2f}%",
    'Duree_s': f"{duree_rf:.1f}"
}])
res_df.to_csv(os.path.join(OUTPUT_DIR, "resultats_final.csv"), index=False, sep=';', encoding='utf-8-sig')

print(f"\n[INFO] Tous les livrables ont été générés dans : {OUTPUT_DIR}")
print("="*70)