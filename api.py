from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import os
import numpy as np
import shap

# =========================
# Config MLflow / modèle
# =========================
# FIX 1: Set the tracking URI to an absolute path for better stability in deployment
# NOTE: You should ideally use an S3/Postgres/etc. URI in production, not a local folder.
# We'll use os.getcwd() to make the local path absolute.
MLFLOW_TRACKING_URI = os.path.join(os.getcwd(), "notebooks", "mlruns")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
print(f"MLflow Tracking URI set to: {MLFLOW_TRACKING_URI}")

EXPERIMENT_ID = "186828650496183123"
RUN_ID = "e2fedff1edd34c0686db8f0c86ae9c49"
MODEL_URI = f"runs:/{RUN_ID}/model"

model = None
run = None
BEST_T = 0.5 # FIX 4: Set a default threshold in case MLflow retrieval fails

# Chargement du modèle MLflow
try:
    model = mlflow.sklearn.load_model(MODEL_URI)
    print("Modèle chargé avec succès.")
except Exception as e:
    # FIX 2: Added a robust error message for debugging
    print(f"Erreur CRITIQUE lors du chargement du modèle à {MODEL_URI} : {e}")

# Récupération du seuil métier dans MLflow
client = MlflowClient()
try:
    run = client.get_run(RUN_ID)
    # Vérifie si la métrique existe avant de tenter la conversion
    if "val_best_threshold" in run.data.metrics:
        BEST_T = float(run.data.metrics["val_best_threshold"])
        print(f"Seuil métier récupéré : {BEST_T}")
    else:
        print(f"AVERTISSEMENT: La métrique 'val_best_threshold' est introuvable. Utilisation du seuil par défaut ({BEST_T}).")
except Exception as e:
    # FIX 3: Catch the MLflow error which likely caused the RunInfo __init__ issue in logs
    print(f"Erreur lors de la récupération du run {RUN_ID} (RunInfo error probable) : {e}. Utilisation du seuil par défaut ({BEST_T}).")


# =========================
# Chargement des données
# =========================
# FIX 5: Use a helper function for path construction (safer cross-OS)
DATA_PATH = os.path.join("notebooks", "data.csv")

if not os.path.exists(DATA_PATH) :
    raise RuntimeError(f"Fichiers data.csv introuvable à : {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
# TRAIN contient TARGET et SK_ID_CURR, et TEST au moins SK_ID_CURR
df_test = df[df['TARGET'].isna()]
# Liste des features utilisées par le modèle (hors ID et cible)
cols_to_exclude = [c for c in ["SK_ID_CURR", "TARGET","Unnamed: 0"] if c in df_test.columns]
FEATURE_COLS = [c for c in df_test.columns if c not in cols_to_exclude]

# Colonnes numériques pour les comparaisons
NUMERIC_FEATURES = df_test[FEATURE_COLS].select_dtypes(include="number").columns.tolist()

# =========================
# SHAP : construction de l'explainer au démarrage
# =========================

# On prend un petit échantillon comme background (pour la régression logistique c'est suffisant)
BACKGROUND_SIZE = 1000
X_bg = df_test[FEATURE_COLS].sample(
    n=min(BACKGROUND_SIZE, len(df_test)),
    random_state=42
)

if hasattr(model, "named_steps"):
    imp = model.named_steps.get("imp", None)
    scal = model.named_steps.get("scal", None)
    clf = model.named_steps.get("clf", model)
else:
    imp = None
    scal = None
    clf = model

X_bg_proc = X_bg.copy()
if imp is not None:
    X_bg_proc = imp.transform(X_bg_proc)
if scal is not None:
    X_bg_proc = scal.transform(X_bg_proc)

# Explainer SHAP (LogisticRegression → LinearExplainer convient bien)
explainer = None # Initialiser l'explainer à None

# FIX 6: Conditionnaly initialize SHAP only if the model is loaded successfully (clf is not None)
if clf is not None:
    try:
        explainer = shap.LinearExplainer(clf, X_bg_proc)
        print("SHAP explainer chargé avec succès.")
    except Exception as e:
        print(f"Erreur lors de la construction du SHAP explainer : {e}")
else:
    print("AVERTISSEMENT: Modèle non chargé. Le SHAP explainer ne sera pas disponible.")

# =========================
# FastAPI app
# =========================
# FIX 7: Add client, imp, scal, and explainer to global scope if they might be used in endpoints
app = FastAPI(title="API Scoring Crédit")


# =========================
# Endpoints
# =========================

# Helper function definition (moved up for scope consistency)
def make_json_safe(value):
    """Convertit une valeur pandas/numpy en valeur JSON-safe."""
    if isinstance(value, (np.floating, float)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value

# 1) etat de sante
@app.get("/health")
def health():
    # FIX 8: Extend health check to model/explainer status
    status = {"status": "ok"}
    if model is None:
        status["model_status"] = "failed to load"
    if explainer is None:
        status["explainer_status"] = "unavailable"
    return status


# 2) Liste des clients (pour le select dans le dashboard) -------------------
@app.get("/clients")
def get_clients(limit: Optional[int] = 1000):
    """
    Retourne une liste de SK_ID_CURR (pour alimenter le select du dashboard).
    """
    if "SK_ID_CURR" not in df_test.columns:
        raise HTTPException(status_code=500, detail="Colonne SK_ID_CURR absente de df_test.")

    client_ids = df_test["SK_ID_CURR"].head(limit).tolist()
    return {"client_ids": client_ids}


# 3) Infos descriptives pour un client donné --------------------------------
@app.get("/client_info")
def client_info(client_id: int):
    """
    Retourne les infos descriptives d'un client (features brutes).
    """
    if "SK_ID_CURR" not in df_test.columns:
        raise HTTPException(status_code=500, detail="Colonne SK_ID_CURR absente de df_test.")

    row = df_test[df_test["SK_ID_CURR"] == client_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Client introuvable dans df_test.")

    row = row.iloc[0]

    # Nettoyage JSON-safe des features
    features = {col: make_json_safe(row[col]) for col in FEATURE_COLS}

    return {
        "client_id": int(client_id),
        "features": features,
        "numeric_features": NUMERIC_FEATURES
    }


# 4) Prédiction pour un client à partir de son SK_ID_CURR -------------------
@app.get("/predict_client")
def predict_client(client_id: int):
    """
    Prédit le risque de défaut pour un client à partir de son SK_ID_CURR.
    Utilisé par le dashboard.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Le modèle ML n'a pas pu être chargé. Prédiction impossible.")

    if "SK_ID_CURR" not in df_test.columns:
        raise HTTPException(status_code=500, detail="Colonne SK_ID_CURR absente de df_test.")

    row = df_test[df_test["SK_ID_CURR"] == client_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Client introuvable dans df_test.")

    row = row.iloc[0]
    X_row = pd.DataFrame([row[FEATURE_COLS].to_dict()])

    # NOTE: The model variable is used here, not clf. This is correct if you loaded a Pipeline.
    proba = model.predict_proba(X_row)[:, 1][0]
    y_pred = int(proba >= BEST_T)

    response = {
        "client_id": int(client_id),
        "probability_default": proba,
        "prediction": y_pred,
        "threshold_used": BEST_T,
        "top_features": []
    }

    # FIX 9: Only run SHAP logic if the explainer is available
    if explainer is not None:
        # 2) Préparation des données comme vues par le classifieur pour SHAP
        X_shap = X_row.copy()
        if imp is not None:
            X_shap = imp.transform(X_shap)
        if scal is not None:
            X_shap = scal.transform(X_shap)

        # Assurez-vous que l'explainer a les bonnes méthodes avant d'appeler
        try:
            shap_vals = explainer(X_shap)
            shap_row = shap_vals.values[0]

            N_TOP = 10
            abs_contrib = np.abs(shap_row)
            top_idx = abs_contrib.argsort()[::-1][:N_TOP]

            top_features = []
            for i in top_idx:
                fname = FEATURE_COLS[i]
                top_features.append({
                    "feature": fname,
                    "value": make_json_safe(row[fname]),
                    "impact": float(shap_row[i])  # >0 = augmente le risque, <0 = diminue
                })
            
            response["top_features"] = top_features
        except Exception as e:
            # Handle SHAP error specifically if it occurs during prediction
            print(f"Erreur SHAP lors de la prédiction du client {client_id}: {e}")
            response["shap_error"] = f"Error generating SHAP explanation: {e}"
    else:
        response["shap_info"] = "SHAP explanation unavailable due to model/explainer loading failure."

    return response


# 5) Distribution globale d'une variable pour comparaison -------------------
@app.get("/global_distribution")
def global_distribution(feature: str, client_id: int):
    """
    Retourne la distribution d'une feature pour :
    - l'ensemble des clients (train ou test)
    - un groupe "similaire" (ici, on simplifie : même dataset)
    - la valeur du client sélectionné
    """
    if feature not in df_test.columns:
        raise HTTPException(status_code=400, detail=f"Feature {feature} absente de df_test.")

    # Valeur client
    row = df_test[df_test["SK_ID_CURR"] == client_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Client introuvable.")
    # FIX 10: Ensure client_value is JSON-safe
    client_value = row.iloc[0][feature]

    # Distribution globale (on peut prendre train ou test, ici train pour l'historique)
    # FIX 10: Ensure distribution values are JSON-safe
    all_vals = [make_json_safe(v) for v in df_test[feature].dropna().tolist()]

    # Groupe similaire : pour simplifier, on prend test. Tu peux filtrer par critère si tu veux.
    similar_vals = [make_json_safe(v) for v in df_test[feature].dropna().tolist()]

    return {
        "feature": feature,
        "client_id": int(client_id),
        "client_value": make_json_safe(client_value),
        "all_clients": all_vals,
        "similar_clients": similar_vals
    }