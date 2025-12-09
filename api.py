# ===============================================================
#  API FastAPI – Version PRODUCTION
#  Chargement du modèle depuis notebooks/models/model.pkl
# ===============================================================

from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
import joblib
import json
import os

# ===============================================================
# 1) Chargement du modèle
# ===============================================================

MODEL_PATH = os.path.join("notebooks", "models", "credit_risk_model.pkl")

model = None
try:
    model = joblib.load(MODEL_PATH)
    print(f"Modèle chargé avec succès depuis : {MODEL_PATH}")
except Exception as e:
    print(f"ERREUR : Impossible de charger le modèle : {e}")

# ===============================================================
# 2) Chargement du seuil BEST_T
# ===============================================================

BEST_T = 0.5218934148703536   # Valeur trouvé par la simulation

# ===============================================================
# 3) Chargement des données (prod clients)
# ===============================================================

DATA_PATH = os.path.join("notebooks", "data.csv")

if not os.path.exists(DATA_PATH):
    raise RuntimeError(f"data.csv introuvable à {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# Séparation train/test
df_test = df[df["TARGET"].isna()]

# Features utilisées
cols_to_exclude = ["SK_ID_CURR", "TARGET", "Unnamed: 0"]
FEATURE_COLS = [c for c in df_test.columns if c not in cols_to_exclude]

NUMERIC_FEATURES = df_test[FEATURE_COLS].select_dtypes(include="number").columns.tolist()

# ===============================================================
# 4) App FASTAPI
# ===============================================================

app = FastAPI(title="API Crédit - Production")

# ===============================================================
# 5) Fonctions utilitaires
# ===============================================================

def make_json_safe(value):
    """Convertit une valeur numpy/pandas en type JSON compatible."""
    if isinstance(value, (np.floating, float)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value

# ===============================================================
# 6) Endpoints
# ===============================================================

@app.get("/health")
def health():
    """Vérifie que l'API tourne et que le modèle est chargé."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "threshold": BEST_T
    }

@app.get("/clients")
def get_clients(limit: int = 1000):
    if "SK_ID_CURR" not in df_test.columns:
        raise HTTPException(500, "Colonne SK_ID_CURR absente")
    
    ids = df_test["SK_ID_CURR"].head(limit).tolist()
    return {"client_ids": ids}

@app.get("/client_info")
def client_info(client_id: int):
    row = df_test[df_test["SK_ID_CURR"] == client_id]
    if row.empty:
        raise HTTPException(404, "Client introuvable")

    row = row.iloc[0]

    features = {col: make_json_safe(row[col]) for col in FEATURE_COLS}

    return {
        "client_id": client_id,
        "features": features,
        "numeric_features": NUMERIC_FEATURES
    }

@app.get("/predict_client")
def predict_client(client_id: int):
    if model is None:
        raise HTTPException(503, "Modèle non chargé")

    row = df_test[df_test["SK_ID_CURR"] == client_id]
    if row.empty:
        raise HTTPException(404, "Client introuvable")

    row = row.iloc[0]
    X_row = pd.DataFrame([row[FEATURE_COLS].to_dict()])

    proba = model.predict_proba(X_row)[:, 1][0]
    y_pred = int(proba >= BEST_T)

    return {
        "client_id": client_id,
        "probability_default": proba,
        "prediction": y_pred,
        "threshold_used": BEST_T,
    }

@app.get("/global_distribution")
def global_distribution(feature: str, client_id: int):
    if feature not in df_test.columns:
        raise HTTPException(400, f"Feature {feature} absente")

    row = df_test[df_test["SK_ID_CURR"] == client_id]
    if row.empty:
        raise HTTPException(404, "Client introuvable")

    client_value = make_json_safe(row.iloc[0][feature])
    all_vals = [make_json_safe(v) for v in df_test[feature].dropna().tolist()]

    return {
        "feature": feature,
        "client_id": client_id,
        "client_value": client_value,
        "all_clients": all_vals,
        "similar_clients": all_vals
    }