# conda activate rapids_gpu
# export LD_LIBRARY_PATH=$HOME/miniconda3/envs/rapids_gpu/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
"""
═══════════════════════════════════════════════════════════════════════════════
COMPARATIF ML — PRÉDICTION ÉTAT DE SANTÉ (eds_e_t)
Multi-targets × Multi-niveaux de classification

TARGETS : 18 variantes EDS selon (pondération × jeu de seuils)
NIVEAUX :
  - niveau_1 : 4 classes {1, 2, 3, 4} (multiclasse)
  - niveau_2 : binaire {1,2,3}=0  vs  {4}=1  (isolation très mauvais état)
  - niveau_3 : binaire {1,2}=0    vs  {3,4}=1 (Bon vs Dégradé)

Activation manuelle via TARGETS_ACTIVES et NIVEAUX_ACTIFS.
═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import warnings
import time
from collections import Counter

from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    AdaBoostClassifier, GradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    precision_score, recall_score, accuracy_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, roc_auc_score,
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import joblib
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


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION DES EXPÉRIENCES — MODIFIE ICI CE QUE TU VEUX LANCER
# ═════════════════════════════════════════════════════════════════════════════

# --- Quelles targets EDS tester ? (False = désactivé) ---
TARGETS_ACTIVES = {
    # Jeu de seuils j1
    'EDS1_p1_j1': True,
    'EDS2_p1_j1': True,
    'EDS3_j1':    True,
    'EDS4_j1':    True,
    'EDS1_p2_j1': True,
    'EDS2_p2_j1': True,
    # Jeu de seuils j2
    'EDS1_p1_j2': False,
    'EDS2_p1_j2': True,
    'EDS3_j2':    True,
    'EDS4_j2':    True,
    'EDS1_p2_j2': True,
    'EDS2_p2_j2': True,
    # Jeu de seuils j3
    'EDS1_p1_j3': False,
    'EDS2_p1_j3': False,
    'EDS3_j3':    True,
    'EDS4_j3':    True,
    'EDS1_p2_j3': False,
    'EDS2_p2_j3': True,
}

# --- Quels niveaux de classification tester pour CHAQUE target active ? ---
NIVEAUX_ACTIFS = {
    'niveau_1_multiclasse':    False,   # 4 classes : 1, 2, 3, 4
    'niveau_2_isolation_4':    True,   # binaire : {1,2,3}=0 vs {4}=1
    'niveau_3_bon_vs_degrade': True,   # binaire : {1,2}=0 vs {3,4}=1
}

# --- Quels modèles activer ? ---
MODELES_ACTIFS = {
    'Random Forest':  True,
    'LightGBM':       True,
    'XGBoost':        True,
    'CatBoost':       False,
    'ExtraTrees':     True,
    'AdaBoost':       False,
    'GTB':            False,
    'CART':           False,
}


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION GÉNÉRALE
# ═════════════════════════════════════════════════════════════════════════════

# INPUT_FILE   = "/run/user/1000/gvfs/smb-share:server=172.16.39.24,share=groupe-merlin/Lyon-Siege/Hors_Affaires/HYDRAU/COMMUN/Stagiaires/M-yassine/test/1m200k/output_gpkg_1m200k.gpkg"
# INPUT_LAYER  = "geom_troncon_final"   # ← nouvelle couche avec les 18 colonnes EDS
# OUTPUT_DIR_BASE = "/home/user-ia/ML2/resultats_multi_eds"

INPUT_FILE      = _env_required("INPUT_FILE")
INPUT_LAYER     = _env_required("INPUT_LAYER")
OUTPUT_DIR_BASE = _env_required("OUTPUT_DIR_BASE")
RANDOM_STATE = 42
TEST_SIZE    = 0.2
KNN_K        = 7

os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# FEATURES
# ═════════════════════════════════════════════════════════════════════════════

COLS_NUM = [
    'X_centroid', 'Y_centroid',
    'prof_moy', 'diametre', 'pente',
    'classe_age', 'taux_arbo', 'taux_imper',
    'length', 'alea_argiles', 'alea_nappes',
]
COLS_CAT = ['materiau_affecte', 'nature_effluent']


# ═════════════════════════════════════════════════════════════════════════════
# FONCTIONS DE RECODAGE DES TARGETS PAR NIVEAU
# ═════════════════════════════════════════════════════════════════════════════

def recoder_target(y_raw, niveau):
    """
    Recode la target selon le niveau demandé.
    y_raw : array avec valeurs 1, 2, 3, 4
    Retourne : (y_recode, class_labels, class_names)
    """
    if niveau == 'niveau_1_multiclasse':
        # 4 classes, on remappe 1-4 → 0-3
        y_new = y_raw - 1
        labels = {0: 'Bon', 1: 'Moyen', 2: 'Mauvais', 3: 'Très mauvais'}
        names = np.array(['Bon', 'Moyen', 'Mauvais', 'Très mauvais'])
        return y_new, labels, names

    elif niveau == 'niveau_2_isolation_4':
        # {1,2,3}=0 vs {4}=1
        y_new = (y_raw == 4).astype(int)
        labels = {0: 'Non critique', 1: 'Très mauvais'}
        names = np.array(['Non critique', 'Très mauvais'])
        return y_new, labels, names

    elif niveau == 'niveau_3_bon_vs_degrade':
        # {1,2}=0 vs {3,4}=1
        y_new = (y_raw >= 3).astype(int)
        labels = {0: 'Bon', 1: 'Dégradé'}
        names = np.array(['Bon', 'Dégradé'])
        return y_new, labels, names

    else:
        raise ValueError(f"Niveau inconnu : {niveau}")


# ═════════════════════════════════════════════════════════════════════════════
# [1] CHARGEMENT (UNE SEULE FOIS)
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 75)
print("[1] Chargement du GeoPackage...")
print("=" * 75)

gdf = gpd.read_file(INPUT_FILE, layer=INPUT_LAYER)
df_full = pd.DataFrame(gdf.drop(columns='geometry', errors='ignore'))
print(f"    {len(df_full):,} tronçons | {len(df_full.columns)} colonnes")

# Centroides
gdf['X_centroid'] = gdf.geometry.centroid.x
gdf['Y_centroid'] = gdf.geometry.centroid.y
df_full['X_centroid'] = gdf['X_centroid'].values
df_full['Y_centroid'] = gdf['Y_centroid'].values
print(f"    ✓ X_centroid et Y_centroid calculés depuis la géométrie")

# COALESCE
if 'profondeur_affectee' in df_full.columns:
    df_full['prof_moy'] = df_full['prof_moy'].fillna(df_full['profondeur_affectee'])
    print(f"    ✓ COALESCE prof_moy ← profondeur_affectee")

if 'diametre_affecte' in df_full.columns:
    df_full['diametre'] = df_full['diametre'].fillna(df_full['diametre_affecte'])
    print(f"    ✓ COALESCE diametre ← diametre_affecte")

# Remplacer INCONNU dans nature_effluent
if 'nature_effluent' in df_full.columns:
    vals_ok = df_full.loc[
        df_full['nature_effluent'].notna() &
        (df_full['nature_effluent'] != 'INCONNU'),
        'nature_effluent'
    ]
    if len(vals_ok) > 0:
        mode_eff = vals_ok.mode()[0]
        df_full['nature_effluent'] = df_full['nature_effluent'].replace('INCONNU', mode_eff)
        print(f"    ✓ nature_effluent : 'INCONNU' → '{mode_eff}'")

# Vérifier la présence des colonnes EDS activées
print(f"\n    Vérification des colonnes EDS activées :")
targets_a_lancer = [t for t, actif in TARGETS_ACTIVES.items() if actif]
targets_manquantes = [t for t in targets_a_lancer if t not in df_full.columns]
if targets_manquantes:
    print(f"    ⚠ Colonnes EDS ABSENTES du GPKG : {targets_manquantes}")
    print(f"    Colonnes présentes similaires : "
          f"{[c for c in df_full.columns if c.startswith('EDS')]}")
    raise ValueError("Certaines colonnes EDS activées sont absentes du GeoPackage.")
else:
    print(f"    ✓ Toutes les targets activées sont présentes ({len(targets_a_lancer)} targets)")

# Niveaux activés
niveaux_a_lancer = [n for n, actif in NIVEAUX_ACTIFS.items() if actif]
modeles_a_lancer = [m for m, actif in MODELES_ACTIFS.items() if actif]

# Récap
n_total_experiences = len(targets_a_lancer) * len(niveaux_a_lancer)
print(f"\n    ━━━ RÉCAP CONFIGURATION ━━━")
print(f"    Targets activées : {len(targets_a_lancer)} → {targets_a_lancer}")
print(f"    Niveaux activés  : {len(niveaux_a_lancer)} → {niveaux_a_lancer}")
print(f"    Modèles activés  : {len(modeles_a_lancer)} → {modeles_a_lancer}")
print(f"    TOTAL EXPÉRIENCES : {n_total_experiences}  "
      f"({n_total_experiences * len(modeles_a_lancer)} entraînements)")


# ═════════════════════════════════════════════════════════════════════════════
# [2] FONCTION DE SAUVEGARDE (un fichier = un modèle)
# ═════════════════════════════════════════════════════════════════════════════

def sauvegarder_modele(nom, res, output_dir, features_finales, class_names,
                       y_test, y_test_proba=None):
    """Sauvegarde graphiques + CSV partiel pour un modèle."""
    nom_fichier = nom.replace(" ", "_").replace("(", "").replace(")", "")

    # Matrice de confusion normalisée
    cm = confusion_matrix(y_test, res['pred'])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=False, cmap='Blues', values_format='.2f')
    ax.set_title(f"{nom}\nAcc={res['acc']*100:.1f}% | F1={res['f1']*100:.1f}%",
                 fontweight='bold', fontsize=10)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"cm_{nom_fichier}.png"),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Feature Importance
    if res['importance'] is not None:
        imp = np.array(res['importance'], dtype=float)
        idx = np.argsort(imp)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh([features_finales[i] for i in idx], imp[idx],
                color='steelblue', alpha=0.85, edgecolor='white')
        ax.set_title(f"Feature Importance — {nom}", fontweight='bold')
        ax.set_xlabel("Importance")
        ax.tick_params(axis='y', labelsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"fi_{nom_fichier}.png"), dpi=150)
        plt.close()

    # Courbe ROC (seulement binaire)
    if y_test_proba is not None and len(np.unique(y_test)) == 2:
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        roc_auc = auc(fpr, tpr)
        res['auc'] = roc_auc
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC = {roc_auc:.3f}')
        ax.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--', label='Aléatoire')
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title(f"Courbe ROC — {nom}", fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"roc_{nom_fichier}.png"), dpi=150)
        plt.close()

    # CSV partiel (ligne par modèle)
    csv_path = os.path.join(output_dir, "resultats_partiel.csv")
    ligne = pd.DataFrame([{
        'Modele': nom,
        'Accuracy':         f"{res['acc']*100:.2f}%",
        'Precision_Macro':  f"{res.get('precision', 0)*100:.2f}%",
        'Recall_Macro':     f"{res.get('recall', 0)*100:.2f}%",
        'F1_Macro':         f"{res['f1']*100:.2f}%",
        'AUC_ROC':          f"{res.get('auc', 0):.3f}",
        'Duree_s':          f"{res['duree']:.1f}",
        'Best_Params':      str(res.get('best_params', ''))
    }])
    if os.path.exists(csv_path):
        ligne.to_csv(csv_path, mode='a', header=False, index=False,
                     sep=';', encoding='utf-8-sig')
    else:
        ligne.to_csv(csv_path, mode='w', header=True, index=False,
                     sep=';', encoding='utf-8-sig')


# ═════════════════════════════════════════════════════════════════════════════
# [3] PIPELINE D'ENTRAÎNEMENT POUR UNE EXPÉRIENCE
# ═════════════════════════════════════════════════════════════════════════════

def run_experience(target_name, niveau, df_source, resultats_globaux):
    """
    Lance le pipeline complet pour une combinaison (target, niveau).
    df_source : df complet avec toutes les colonnes
    resultats_globaux : liste partagée qui accumule toutes les lignes de synthèse
    """
    tag = f"{target_name} × {niveau}"
    print("\n" + "█" * 75)
    print(f"  EXPÉRIENCE : {tag}")
    print("█" * 75)

    output_dir = os.path.join(OUTPUT_DIR_BASE, target_name, niveau)
    os.makedirs(output_dir, exist_ok=True)

    # ── Copie locale du df pour cette expérience ──
    df = df_source.copy()

    # ── Filtrage : garder uniquement les tronçons avec target non-null ──
    n_avant = len(df)
    df = df[df[target_name].notna()].copy()
    df[target_name] = df[target_name].astype(int)
    df = df[df[target_name].isin([1, 2, 3, 4])].copy()
    print(f"    Tronçons avec target valide : {len(df):,} (sur {n_avant:,})")

    # ── Recodage selon le niveau ──
    y_raw = df[target_name].values
    y, class_labels, class_names = recoder_target(y_raw, niveau)
    NUM_CLASSES = len(class_names)
    print(f"    Niveau : {niveau} → {NUM_CLASSES} classes : {list(class_names)}")

    # Distribution
    print(f"    Distribution :")
    for cls_idx, n in Counter(y).items():
        pct = n / len(y) * 100
        print(f"      {class_names[cls_idx]:<15} : {n:>7,}  ({pct:5.1f}%)")

    # ── Suppression tronçons incomplets (<50% features) ──
    COLS_TOUTES = [c for c in COLS_NUM if c in df.columns] + \
                  [c for c in COLS_CAT if c in df.columns]
    taux_remplissage = df[COLS_TOUTES].notna().mean(axis=1)
    mask_ok = taux_remplissage >= 0.5
    df = df[mask_ok].copy()
    y = y[mask_ok.values]
    print(f"    Après filtrage complétude ≥ 50% : {len(df):,} tronçons")

    # ── Encodage catégorielles ──
    label_encoders = {}
    cols_cat_present = [c for c in COLS_CAT if c in df.columns]
    for col in cols_cat_present:
        df[col] = df[col].fillna('INCONNU').astype(str)
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col])
        label_encoders[col] = le
    COLS_CAT_ENC = [c + '_enc' for c in cols_cat_present]

    # ── Log transform ──
    cols_num_present = [c for c in COLS_NUM if c in df.columns]
    for col in ['length', 'diametre']:
        if col in df.columns and col in cols_num_present:
            df[col] = np.log1p(df[col].clip(lower=0))

    FEATURES_FINALES = cols_num_present + COLS_CAT_ENC
    X = df[FEATURES_FINALES].values

    # ── Split train/test stratifié ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"    Train : {len(X_train):,} | Test : {len(X_test):,}")

    # ── KNN Imputation (avec cache) ──
    X_train_path = os.path.join(output_dir, "X_train_imputed.npy")
    X_test_path  = os.path.join(output_dir, "X_test_imputed.npy")
    if os.path.exists(X_train_path) and os.path.exists(X_test_path):
        print(f"    [CACHE] Chargement imputation KNN...")
        X_train = np.load(X_train_path)
        X_test  = np.load(X_test_path)
    else:
        t_imp = time.time()
        imputer = KNNImputer(n_neighbors=KNN_K)
        X_train = imputer.fit_transform(X_train)
        X_test  = imputer.transform(X_test)
        print(f"    ✓ KNN Imputer terminé en {time.time()-t_imp:.1f}s")
        np.save(X_train_path, X_train)
        np.save(X_test_path, X_test)
        joblib.dump(imputer, os.path.join(output_dir, "knn_imputer.pkl"))

    # ── Split validation + scaling ──
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15,
        random_state=RANDOM_STATE, stratify=y_train
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
    joblib.dump(label_encoders, os.path.join(output_dir, "label_encoders.pkl"))
    joblib.dump(FEATURES_FINALES, os.path.join(output_dir, "features_finales.pkl"))

    class_counts = Counter(y_train)
    total_train  = len(y_train)

    # ── Supprimer ancien CSV partiel si existe (évite concat erronée) ──
    csv_part = os.path.join(output_dir, "resultats_partiel.csv")
    if os.path.exists(csv_part):
        os.remove(csv_part)

    resultats = {}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    # ═════════════════════════════════════════════════════════════════════════
    # ENTRAÎNEMENT DES MODÈLES ACTIVÉS
    # ═════════════════════════════════════════════════════════════════════════

    # --- Random Forest ---
    if MODELES_ACTIFS.get('Random Forest', False):
        print(f"\n    ─── Random Forest ───")
        rf_path = os.path.join(output_dir, "model_rf.pkl")
        if os.path.exists(rf_path):
            rf = joblib.load(rf_path)
            y_pred = rf.predict(X_test)
            y_proba = rf.predict_proba(X_test)
            duree = 0.0
            best_params = rf.get_params()
        else:
            t0 = time.time()
            param_dist = {
                'n_estimators':     [200, 400, 800, 1200, 1600],
                'max_depth':        [20, 30, 40, None],
                'min_samples_leaf': [1, 2, 3],
                'max_features':     ['sqrt', 0.3, 0.4],
                'max_samples':      [0.8, 0.9, None],
                'class_weight':     ['balanced_subsample', 'balanced', None],
            }
            search = RandomizedSearchCV(
                RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE),
                param_distributions=param_dist, n_iter=30,
                scoring='f1_macro', cv=cv, n_jobs=1,
                random_state=RANDOM_STATE, verbose=2,
            )
            search.fit(X_train, y_train)
            rf = search.best_estimator_
            best_params = search.best_params_
            y_pred = rf.predict(X_test)
            y_proba = rf.predict_proba(X_test)
            duree = time.time() - t0
            joblib.dump(rf, rf_path)

        y_proba_binary = y_proba[:, 1] if NUM_CLASSES == 2 else None
        res = {
            'acc': accuracy_score(y_test, y_pred),
            'f1':  f1_score(y_test, y_pred, average='macro'),
            'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall':    recall_score(y_test, y_pred, average='macro', zero_division=0),
            'duree': duree, 'pred': y_pred,
            'importance': rf.feature_importances_,
            'best_params': best_params,
        }
        resultats['Random Forest'] = res
        print(f"      Acc={res['acc']*100:.2f}%  F1={res['f1']*100:.2f}%  ({duree:.0f}s)")
        sauvegarder_modele('Random Forest', res, output_dir, FEATURES_FINALES,
                           class_names, y_test, y_proba_binary)

    # --- LightGBM ---
    if MODELES_ACTIFS.get('LightGBM', False):
        print(f"\n    ─── LightGBM ───")
        lgbm_path = os.path.join(output_dir, "model_lgbm.pkl")
        if os.path.exists(lgbm_path):
            lgbm = joblib.load(lgbm_path)
            y_pred = lgbm.predict(X_test)
            y_proba = lgbm.predict_proba(X_test)
            duree = 0.0
            best_params = lgbm.get_params()
        else:
            t0 = time.time()
            param_dist = {
                'n_estimators':     [500, 1000, 2000],
                'learning_rate':    [0.01, 0.03, 0.05, 0.1],
                'max_depth':        [5, 7, 9, -1],
                'num_leaves':       [31, 63, 127],
                'min_child_samples':[20, 50, 100],
                'subsample':        [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'reg_lambda':       [0.0, 0.5, 1.0],
            }
            search = RandomizedSearchCV(
                lgb.LGBMClassifier(device='cpu', class_weight='balanced',
                                    random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
                param_distributions=param_dist, n_iter=20,
                scoring='f1_macro', cv=cv, n_jobs=1,
                random_state=RANDOM_STATE, verbose=2,
            )
            search.fit(X_train, y_train)
            lgbm = search.best_estimator_
            best_params = search.best_params_
            y_pred = lgbm.predict(X_test)
            y_proba = lgbm.predict_proba(X_test)
            duree = time.time() - t0
            joblib.dump(lgbm, lgbm_path)

        y_proba_binary = y_proba[:, 1] if NUM_CLASSES == 2 else None
        res = {
            'acc': accuracy_score(y_test, y_pred),
            'f1':  f1_score(y_test, y_pred, average='macro'),
            'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall':    recall_score(y_test, y_pred, average='macro', zero_division=0),
            'duree': duree, 'pred': y_pred,
            'importance': lgbm.feature_importances_,
            'best_params': best_params,
        }
        resultats['LightGBM'] = res
        print(f"      Acc={res['acc']*100:.2f}%  F1={res['f1']*100:.2f}%  ({duree:.0f}s)")
        sauvegarder_modele('LightGBM', res, output_dir, FEATURES_FINALES,
                           class_names, y_test, y_proba_binary)

    # --- XGBoost ---
    if MODELES_ACTIFS.get('XGBoost', False):
        print(f"\n    ─── XGBoost (CUDA) ───")
        xgb_path = os.path.join(output_dir, "model_xgb.pkl")
        if os.path.exists(xgb_path):
            xgb_model = joblib.load(xgb_path)
            y_pred = xgb_model.predict(X_test)
            y_proba = xgb_model.predict_proba(X_test)
            duree = 0.0
            best_params = xgb_model.get_params()
        else:
            t0 = time.time()
            sample_weights = np.array([
                total_train / (NUM_CLASSES * class_counts[yi]) for yi in y_train
            ])
            param_dist = {
                'n_estimators':     [500, 1000, 2000, 3000],
                'learning_rate':    [0.01, 0.03, 0.05, 0.1],
                'max_depth':        [4, 6, 8, 10],
                'subsample':        [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'reg_lambda':       [0.5, 1.0, 3.0],
                'reg_alpha':        [0.0, 0.1, 0.5],
                'min_child_weight': [1, 5, 10],
            }
            search = RandomizedSearchCV(
                xgb.XGBClassifier(device='cuda', tree_method='hist',
                                  eval_metric='mlogloss',
                                  random_state=RANDOM_STATE, verbosity=0),
                param_distributions=param_dist, n_iter=20,
                scoring='f1_macro', cv=cv, n_jobs=1,
                random_state=RANDOM_STATE, verbose=2,
            )
            search.fit(X_train, y_train, sample_weight=sample_weights)
            xgb_model = search.best_estimator_
            best_params = search.best_params_
            y_pred = xgb_model.predict(X_test)
            y_proba = xgb_model.predict_proba(X_test)
            duree = time.time() - t0
            joblib.dump(xgb_model, xgb_path)

        y_proba_binary = y_proba[:, 1] if NUM_CLASSES == 2 else None
        res = {
            'acc': accuracy_score(y_test, y_pred),
            'f1':  f1_score(y_test, y_pred, average='macro'),
            'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall':    recall_score(y_test, y_pred, average='macro', zero_division=0),
            'duree': duree, 'pred': y_pred,
            'importance': xgb_model.feature_importances_,
            'best_params': best_params,
        }
        resultats['XGBoost'] = res
        print(f"      Acc={res['acc']*100:.2f}%  F1={res['f1']*100:.2f}%  ({duree:.0f}s)")
        sauvegarder_modele('XGBoost', res, output_dir, FEATURES_FINALES,
                           class_names, y_test, y_proba_binary)

    # --- CatBoost ---
    if MODELES_ACTIFS.get('CatBoost', False):
        print(f"\n    ─── CatBoost (GPU) ───")
        cat_path = os.path.join(output_dir, "model_catboost.pkl")
        if os.path.exists(cat_path):
            cat_model = joblib.load(cat_path)
            y_pred = cat_model.predict(X_test).flatten().astype(int)
            y_proba = cat_model.predict_proba(X_test)
            duree = 0.0
        else:
            t0 = time.time()
            cat_model = CatBoostClassifier(
                iterations=15000, learning_rate=0.03,
                depth=8, l2_leaf_reg=3.0, bagging_temperature=0.8,
                random_seed=RANDOM_STATE, task_type='GPU',
                auto_class_weights='Balanced',
                early_stopping_rounds=200, od_type='Iter',
                verbose=200,
                train_dir=f'/tmp/catboost_{target_name}_{niveau}'
            )
            cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
            y_pred = cat_model.predict(X_test).flatten().astype(int)
            y_proba = cat_model.predict_proba(X_test)
            duree = time.time() - t0
            joblib.dump(cat_model, cat_path)

        y_proba_binary = y_proba[:, 1] if NUM_CLASSES == 2 else None
        res = {
            'acc': accuracy_score(y_test, y_pred),
            'f1':  f1_score(y_test, y_pred, average='macro'),
            'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall':    recall_score(y_test, y_pred, average='macro', zero_division=0),
            'duree': duree, 'pred': y_pred,
            'importance': cat_model.get_feature_importance(),
        }
        resultats['CatBoost'] = res
        print(f"      Acc={res['acc']*100:.2f}%  F1={res['f1']*100:.2f}%  ({duree:.0f}s)")
        sauvegarder_modele('CatBoost', res, output_dir, FEATURES_FINALES,
                           class_names, y_test, y_proba_binary)

    # --- ExtraTrees ---
    if MODELES_ACTIFS.get('ExtraTrees', False):
        print(f"\n    ─── ExtraTrees ───")
        et_path = os.path.join(output_dir, "model_extratrees.pkl")
        if os.path.exists(et_path):
            et = joblib.load(et_path)
            y_pred = et.predict(X_test)
            y_proba = et.predict_proba(X_test)
            duree = 0.0
            best_params = et.get_params()
        else:
            t0 = time.time()
            param_dist = {
                'n_estimators':     [800, 1200, 1600, 2000],
                'max_depth':        [20, 30, 40, None],
                'min_samples_leaf': [1, 2, 3],
                'max_features':     ['sqrt', 0.2, 0.3, 0.4],
                'class_weight':     ['balanced_subsample', 'balanced', None],
            }
            search = RandomizedSearchCV(
                ExtraTreesClassifier(n_jobs=-1, random_state=RANDOM_STATE),
                param_distributions=param_dist, n_iter=30,
                scoring='f1_macro', cv=cv, n_jobs=1,
                random_state=RANDOM_STATE, verbose=1,
            )
            search.fit(X_train, y_train)
            et = search.best_estimator_
            best_params = search.best_params_
            y_pred = et.predict(X_test)
            y_proba = et.predict_proba(X_test)
            duree = time.time() - t0
            joblib.dump(et, et_path)

        y_proba_binary = y_proba[:, 1] if NUM_CLASSES == 2 else None
        res = {
            'acc': accuracy_score(y_test, y_pred),
            'f1':  f1_score(y_test, y_pred, average='macro'),
            'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall':    recall_score(y_test, y_pred, average='macro', zero_division=0),
            'duree': duree, 'pred': y_pred,
            'importance': et.feature_importances_,
            'best_params': best_params,
        }
        resultats['ExtraTrees'] = res
        print(f"      Acc={res['acc']*100:.2f}%  F1={res['f1']*100:.2f}%  ({duree:.0f}s)")
        sauvegarder_modele('ExtraTrees', res, output_dir, FEATURES_FINALES,
                           class_names, y_test, y_proba_binary)

    # --- AdaBoost ---
    if MODELES_ACTIFS.get('AdaBoost', False):
        print(f"\n    ─── AdaBoost ───")
        ada_path = os.path.join(output_dir, "model_adaboost.pkl")
        if os.path.exists(ada_path):
            ada = joblib.load(ada_path)
            y_pred = ada.predict(X_test)
            y_proba = ada.predict_proba(X_test)
            duree = 0.0
            best_params = ada.get_params()
        else:
            t0 = time.time()
            param_dist = {
                'n_estimators':  [20, 40, 60, 100, 200, 500, 1000],
                'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0],
                'algorithm':     ['SAMME'],
            }
            search = RandomizedSearchCV(
                AdaBoostClassifier(random_state=RANDOM_STATE),
                param_distributions=param_dist, n_iter=40,
                scoring='f1_macro', cv=cv, n_jobs=-1,
                random_state=RANDOM_STATE, verbose=2,
            )
            search.fit(X_train, y_train)
            ada = search.best_estimator_
            best_params = search.best_params_
            y_pred = ada.predict(X_test)
            y_proba = ada.predict_proba(X_test)
            duree = time.time() - t0
            joblib.dump(ada, ada_path)

        y_proba_binary = y_proba[:, 1] if NUM_CLASSES == 2 else None
        res = {
            'acc': accuracy_score(y_test, y_pred),
            'f1':  f1_score(y_test, y_pred, average='macro'),
            'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall':    recall_score(y_test, y_pred, average='macro', zero_division=0),
            'duree': duree, 'pred': y_pred,
            'importance': ada.feature_importances_,
            'best_params': best_params,
        }
        resultats['AdaBoost'] = res
        print(f"      Acc={res['acc']*100:.2f}%  F1={res['f1']*100:.2f}%  ({duree:.0f}s)")
        sauvegarder_modele('AdaBoost', res, output_dir, FEATURES_FINALES,
                           class_names, y_test, y_proba_binary)

    # --- GTB ---
    if MODELES_ACTIFS.get('GTB', False):
        print(f"\n    ─── Gradient Boosting ───")
        gtb_path = os.path.join(output_dir, "model_gtb.pkl")
        if os.path.exists(gtb_path):
            gtb = joblib.load(gtb_path)
            y_pred = gtb.predict(X_test)
            y_proba = gtb.predict_proba(X_test)
            duree = 0.0
            best_params = gtb.get_params()
        else:
            t0 = time.time()
            param_dist = {
                'n_estimators':     [20, 50, 200, 500, 1000, 2000],
                'learning_rate':    [0.01, 0.03, 0.05, 0.1],
                'max_depth':        [3, 5, 7, 9, 12],
                'subsample':        [0.7, 0.8, 0.9],
                'min_samples_leaf': [5, 10, 20],
            }
            search = RandomizedSearchCV(
                GradientBoostingClassifier(random_state=RANDOM_STATE),
                param_distributions=param_dist, n_iter=20,
                scoring='f1_macro', cv=cv, n_jobs=-1,
                random_state=RANDOM_STATE, verbose=2,
            )
            search.fit(X_train, y_train)
            gtb = search.best_estimator_
            best_params = search.best_params_
            y_pred = gtb.predict(X_test)
            y_proba = gtb.predict_proba(X_test)
            duree = time.time() - t0
            joblib.dump(gtb, gtb_path)

        y_proba_binary = y_proba[:, 1] if NUM_CLASSES == 2 else None
        res = {
            'acc': accuracy_score(y_test, y_pred),
            'f1':  f1_score(y_test, y_pred, average='macro'),
            'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall':    recall_score(y_test, y_pred, average='macro', zero_division=0),
            'duree': duree, 'pred': y_pred,
            'importance': gtb.feature_importances_,
            'best_params': best_params,
        }
        resultats['GTB'] = res
        print(f"      Acc={res['acc']*100:.2f}%  F1={res['f1']*100:.2f}%  ({duree:.0f}s)")
        sauvegarder_modele('GTB', res, output_dir, FEATURES_FINALES,
                           class_names, y_test, y_proba_binary)

    # --- CART ---
    if MODELES_ACTIFS.get('CART', False):
        print(f"\n    ─── CART ───")
        cart_path = os.path.join(output_dir, "model_cart.pkl")
        if os.path.exists(cart_path):
            cart = joblib.load(cart_path)
            y_pred = cart.predict(X_test)
            y_proba = cart.predict_proba(X_test)
            duree = 0.0
            best_params = cart.get_params()
        else:
            t0 = time.time()
            param_dist = {
                'max_depth':         [5, 10, 15, 20, None],
                'min_samples_leaf':  [1, 2, 5, 10, 20],
                'min_samples_split': [2, 5, 10, 20],
                'max_features':      ['sqrt', 0.3, 0.5, None],
                'class_weight':      ['balanced', None],
                'criterion':         ['gini', 'entropy'],
            }
            search = RandomizedSearchCV(
                DecisionTreeClassifier(random_state=RANDOM_STATE),
                param_distributions=param_dist, n_iter=40,
                scoring='f1_macro', cv=cv, n_jobs=-1,
                random_state=RANDOM_STATE, verbose=2,
            )
            search.fit(X_train, y_train)
            cart = search.best_estimator_
            best_params = search.best_params_
            y_pred = cart.predict(X_test)
            y_proba = cart.predict_proba(X_test)
            duree = time.time() - t0
            joblib.dump(cart, cart_path)

        y_proba_binary = y_proba[:, 1] if NUM_CLASSES == 2 else None
        res = {
            'acc': accuracy_score(y_test, y_pred),
            'f1':  f1_score(y_test, y_pred, average='macro'),
            'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall':    recall_score(y_test, y_pred, average='macro', zero_division=0),
            'duree': duree, 'pred': y_pred,
            'importance': cart.feature_importances_,
            'best_params': best_params,
        }
        resultats['CART'] = res
        print(f"      Acc={res['acc']*100:.2f}%  F1={res['f1']*100:.2f}%  ({duree:.0f}s)")
        sauvegarder_modele('CART', res, output_dir, FEATURES_FINALES,
                           class_names, y_test, y_proba_binary)

    # ═════════════════════════════════════════════════════════════════════════
    # SYNTHÈSE DE L'EXPÉRIENCE
    # ═════════════════════════════════════════════════════════════════════════

    print(f"\n    ━━━ RÉCAP {tag} ━━━")
    print(f"    {'Modèle':<20} {'Acc':>8} {'F1':>8} {'AUC':>8}")
    for nom, res in sorted(resultats.items(), key=lambda x: -x[1]['f1']):
        auc_val = res.get('auc', 0)
        print(f"    {nom:<20} {res['acc']*100:>7.2f}% "
              f"{res['f1']*100:>7.2f}% {auc_val:>7.3f}")

    # Graphes comparatifs
    if len(resultats) > 0:
        # Barplot Acc + F1
        noms  = list(resultats.keys())
        accs  = [resultats[n]['acc'] * 100 for n in noms]
        f1s   = [resultats[n]['f1'] * 100 for n in noms]
        x_pos = np.arange(len(noms))
        width = 0.35

        fig, ax = plt.subplots(figsize=(13, 6))
        ax.bar(x_pos - width/2, accs, width, label='Accuracy', color='steelblue', alpha=0.9)
        ax.bar(x_pos + width/2, f1s,  width, label='F1 Macro', color='darkorange', alpha=0.9)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(noms, rotation=15, ha='right')
        ax.set_ylabel("Score (%)")
        ax.set_ylim(0, 110)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter())
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_title(f"{tag}\n({len(FEATURES_FINALES)} features | {len(y_test):,} test)",
                     fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "1_comparatif_accuracy_f1.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # Ajout des résultats à la synthèse globale
    for nom, res in resultats.items():
        resultats_globaux.append({
            'Target':       target_name,
            'Niveau':       niveau,
            'NUM_CLASSES':  NUM_CLASSES,
            'Modele':       nom,
            'Accuracy':     f"{res['acc']*100:.2f}",
            'Precision':    f"{res.get('precision', 0)*100:.2f}",
            'Recall':       f"{res.get('recall', 0)*100:.2f}",
            'F1_Macro':     f"{res['f1']*100:.2f}",
            'AUC_ROC':      f"{res.get('auc', 0):.3f}",
            'Duree_s':      f"{res['duree']:.1f}",
            'N_train':      len(y_train),
            'N_test':       len(y_test),
        })


# ═════════════════════════════════════════════════════════════════════════════
# [4] BOUCLE PRINCIPALE SUR TOUTES LES COMBINAISONS
# ═════════════════════════════════════════════════════════════════════════════

resultats_globaux = []
compteur = 0
t_global = time.time()

for target_name in targets_a_lancer:
    for niveau in niveaux_a_lancer:
        compteur += 1
        print(f"\n\n{'▓'*75}")
        print(f"  EXPÉRIENCE {compteur}/{n_total_experiences}")
        print(f"{'▓'*75}")

        try:
            run_experience(target_name, niveau, df_full, resultats_globaux)
        except Exception as e:
            print(f"\n    ❌ ERREUR sur {target_name} × {niveau} : {e}")
            import traceback
            traceback.print_exc()

duree_total = time.time() - t_global


# ═════════════════════════════════════════════════════════════════════════════
# [5] SYNTHÈSE GLOBALE
# ═════════════════════════════════════════════════════════════════════════════

print("\n\n" + "█" * 75)
print("  SYNTHÈSE GLOBALE DE TOUTES LES EXPÉRIENCES")
print("█" * 75)

if resultats_globaux:
    df_synthese = pd.DataFrame(resultats_globaux)
    synthese_path = os.path.join(OUTPUT_DIR_BASE, "synthese_toutes_experiences.csv")
    df_synthese.to_csv(synthese_path, sep=';', encoding='utf-8-sig', index=False)
    print(f"\n  ✓ Synthèse complète : {synthese_path}")
    print(f"  ✓ {len(df_synthese)} lignes (1 par modèle × target × niveau)")

    # Meilleurs résultats par combinaison (target × niveau)
    print(f"\n  ━━━ MEILLEUR MODÈLE PAR COMBINAISON (F1 Macro) ━━━")
    df_synthese['F1_num'] = df_synthese['F1_Macro'].astype(float)
    idx_max = df_synthese.groupby(['Target', 'Niveau'])['F1_num'].idxmax()
    best = df_synthese.loc[idx_max].sort_values(['Target', 'Niveau'])

    print(f"\n  {'Target':<14} {'Niveau':<28} {'Modèle':<16} {'Acc':>7} {'F1':>7} {'AUC':>7}")
    print(f"  {'-'*85}")
    for _, row in best.iterrows():
        print(f"  {row['Target']:<14} {row['Niveau']:<28} {row['Modele']:<16} "
              f"{row['Accuracy']:>6}% {row['F1_Macro']:>6}% {row['AUC_ROC']:>7}")

    # Export du résumé best-model
    best_path = os.path.join(OUTPUT_DIR_BASE, "synthese_meilleurs_modeles.csv")
    best.drop(columns='F1_num').to_csv(best_path, sep=';',
                                        encoding='utf-8-sig', index=False)
    print(f"\n  ✓ Meilleurs modèles : {best_path}")

print(f"\n  Durée totale : {duree_total/60:.1f} min")
print(f"  Dossier de sortie : {OUTPUT_DIR_BASE}/")
print("█" * 75)