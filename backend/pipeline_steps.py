"""
Définition centralisée des étapes du pipeline.
L'interface a été épurée : les chemins de sortie sont gérés automatiquement par Celery.
"""

PIPELINE_STEPS = [
    # ═════════════════════════════════════════════════════════════════════════
    # PHASE A — Préparation des données
    # ═════════════════════════════════════════════════════════════════════════
    {
        "id":          "update_sig",
        "name":        "Calcul taux arboré / imperméabilisation",
        "description": "Calcule taux_arbo et taux_imper sur un buffer de 5m autour de chaque tronçon.",
        "script":      "update_sig.py",
        "phase":       "A",
        "params": [
            {"env_var": "PATH_BASE",          "label": "Dossier des SIG",
             "type": "directory", "required": True, "default": ""},
            {"env_var": "PATH_EXCEL_MAPPING", "label": "Fichier Excel de mapping",
             "type": "file",      "required": True, "default": "",
             "extensions": [".xlsx"]},
            {"env_var": "FILE_TREE",          "label": "Raster taux d'arborisation",
             "type": "file",      "required": True, "default": "",
             "extensions": [".tif", ".tiff"]},
            {"env_var": "FILE_IMPER",         "label": "Raster imperméabilisation",
             "type": "file",      "required": True, "default": "",
             "extensions": [".tif", ".tiff"]},
        ],
    },
    {
        "id":          "altitudes",
        "name":        "Calcul des altitudes",
        "description": "Enrichit les tronçons avec les cotes TN amont/aval depuis les rasters.",
        "script":      "calcul_spacial_altitude.py",
        "phase":       "A",
        "params": [
            {"env_var": "PATH_SHP",     "label": "Shapefile des tronçons",
             "type": "file",      "required": True, "default": "",
             "extensions": [".shp"]},
            {"env_var": "PATH_DALLES",  "label": "Liste des dalles (txt)",
             "type": "file",      "required": False, "default": "",
             "extensions": [".txt"],
             "help": "Facultatif. Si vide, tous les rasters du dossier seront parcourus.\n"
                    "Ce fichier contient les noms des dalles qui ont une intersection avec au moins un tronçon.\n"
                    "Pour le générer : faire l'intersection dans QGIS et exporter la liste."},
            {"env_var": "PATH_RASTERS", "label": "Dossier des rasters",
             "type": "directory", "required": True, "default": ""},
             {"env_var": "PATH_OUTPUT", "label": "Shapefile de sortie",
                    "type": "string", "required": False, "default": "",
                    "help": "Si vide, écrase le shapefile d'entrée."},
        ],
    },
    {
        "id":          "pretraitement",
        "name":        "Prétraitement & fusion SIG",
        "description": "Fusionne les SIG hétérogènes via mapping Excel, calcule longueurs/centroïdes.",
        "script":      "pretraitement_donnees_latest.py",
        "phase":       "A",
        "params": [
            {"env_var": "PATH_BASE",          "label": "Dossier des SIG",
             "type": "directory", "required": True,  "default": ""},
            {"env_var": "PATH_EXCEL_MAPPING", "label": "Fichier Excel de mapping",
             "type": "file",      "required": True,  "default": "",
             "extensions": [".xlsx", ".xls"]},
        ],
    },
    {
        "id":          "ajout_vars",
        "name":        "Ajout aléas argiles & nappes",
        "description": "Joint spatialement les aléas BRGM (retrait-gonflement, remontée de nappes).",
        "script":      "ajout_autres_vars.py",
        "phase":       "A",
        "params": [
            {"env_var": "INPUT_GPKG",  "label": "GeoPackage d'entrée",
             "type": "file",      "required": True, "default": "",
             "extensions": [".gpkg"]},
            {"env_var": "INPUT_LAYER", "label": "Nom de la couche d'entrée",
             "type": "string",    "required": True, "default": "geom_troncon"},
            {"env_var": "OUTPUT_LAYER","label": "Nom de la couche de sortie",
             "type": "string",    "required": True, "default": "geom_troncon_enrichi"},
            {"env_var": "ARGILES_SHP", "label": "Shapefile aléa argiles",
             "type": "file",      "required": True, "default": "",
             "extensions": [".shp"]},
            {"env_var": "NAPPES_DIR",  "label": "Dossier des nappes (zip dept)",
             "type": "directory", "required": True, "default": ""},
        ],
    },

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE B — Complétion attributs
    # ═════════════════════════════════════════════════════════════════════════
    {
        "id":          "model_materiau",
        "name":        "Entraînement modèle matériau",
        "description": "Entraîne un Random Forest pour prédire le matériau (RF retenu).",
        "script":      "model_materiau.py",
        "phase":       "B",
        "params": [
            {"env_var": "INPUT_FILE", "label": "BDD prétraitée (.gpkg ou .csv)",
             "type": "file",      "required": True, "default": "",
             "extensions": [".gpkg", ".csv"]},
        ],
    },
    {
        "id":          "completion_materiau",
        "name":        "Complétion matériau (≥70%)",
        "description": "Applique le modèle matériau aux tronçons sans étiquette.",
        "script":      "completion_BDD_materiau_sup_70.py",
        "phase":       "B",
        "params": [
            {"env_var": "INPUT_FILE",  "label": "BDD prétraitée",
             "type": "file",   "required": True,  "default": "",
             "extensions": [".gpkg", ".csv"]},
            {"env_var": "MODEL_DIR",   "label": "Dossier du modèle entraîné",
             "type": "directory", "required": True, "default": ""},
            {"env_var": "BEST_MODEL",  "label": "Nom du fichier modèle",
             "type": "string", "required": True,  "default": "model_rf.pkl"},
        ],
    },
    {
        "id":          "model_age",
        "name":        "Entraînement modèle classe d'âge",
        "description": "Entraîne un RF pour prédire la classe d'âge.",
        "script":      "model_classe_age.py",
        "phase":       "B",
        "params": [
            {"env_var": "INPUT_FILE", "label": "BDD avec matériau complété",
             "type": "file",   "required": True, "default": "",
             "extensions": [".gpkg", ".csv"]},
        ],
    },
    {
        "id":          "completion_age",
        "name":        "Complétion classe d'âge",
        "description": "Applique le modèle classe d'âge aux tronçons sans étiquette.",
        "script":      "completion_BDD_age.py",
        "phase":       "B",
        "params": [
            {"env_var": "INPUT_FILE",  "label": "BDD avec matériau complété",
             "type": "file",   "required": True, "default": "",
             "extensions": [".gpkg"]},
            {"env_var": "MODEL_DIR",   "label": "Dossier du modèle âge",
             "type": "directory", "required": True, "default": ""},
        ],
    },

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE C — État de santé
    # ═════════════════════════════════════════════════════════════════════════
    {
        "id":          "comparatif_eds",
        "name":        "Comparatif modèles EDS",
        "description": "Lance le comparatif multi-modèles × multi-niveaux × multi-targets EDS.",
        "script":      "comparatif_modeles_eds.py",
        "phase":       "C",
        "params": [
            {"env_var": "INPUT_FILE",      "label": "GeoPackage final",
             "type": "file",   "required": True,  "default": "",
             "extensions": [".gpkg"]},
            {"env_var": "INPUT_LAYER",     "label": "Couche d'entrée",
             "type": "string", "required": True,  "default": "geom_troncon_final"},
        ],
    },
    {
        "id":          "completion_eds",
        "name":        "Complétion état de santé",
        "description": "Applique le modèle EDS retenu aux tronçons sans inspection.",
        "script":      "completion_BDD_eds.py",
        "phase":       "C",
        "params": [
            {"env_var": "INPUT_FILE",     "label": "GeoPackage source",
             "type": "file",      "required": True, "default": "",
             "extensions": [".gpkg"]},
            {"env_var": "INPUT_LAYER",    "label": "Couche d'entrée",
             "type": "string",    "required": True, "default": "geom_troncon_final"},
            {"env_var": "MODELS_DIR",     "label": "Dossier des modèles EDS entraînés",
             "type": "directory", "required": True, "default": ""},
            {"env_var": "TARGET_NAME",    "label": "Target EDS",
             "type": "string",    "required": True, "default": "EDS4_j1"},
            {"env_var": "MODEL_FILENAME", "label": "Fichier modèle",
             "type": "string",    "required": True, "default": "model_extratrees.pkl"},
        ],
    }
]

def get_step_by_id(step_id: str):
    for step in PIPELINE_STEPS:
        if step["id"] == step_id:
            return step
    return None