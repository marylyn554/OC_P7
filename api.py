# api.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException

# =========================
# Configuration & chemins
# =========================
BASE_DIR = Path(__file__).resolve().parent

# Paramétrable via variables d'environnement en prod (EB)
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    str(BASE_DIR / "notebooks" / "mlruns")  # fallback local
)
EXPERIMENT_ID = os.getenv("MLFLOW_EXPERIMENT_ID", "186828650496183123")
RUN_ID = os.getenv("MLFLOW_RUN_ID", "abe2fac0542147baa4246c06d0d72762")

# Données (test/inférence)
DATA_PATH = Path(os.getenv("DATA_PATH", str(BASE_DIR / "notebooks" / "data.csv")))

# Seuil par défaut si on ne retrouve pas la métrique dans MLflow
DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.5"))

# Appliquer la config MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# =========================
# Chargements globaux
# =========================
model = None
BEST_T = DEFAULT_THRESHOLD
imp = None
scal = None
_explainer = None  # SHAP, initialisé à la demande

def _safe_print(msg: str) -> None:
    try:
        print(msg, flush=True)
    except Exception:
        pass

# ========== Chargement modèle MLflow ==========
MODEL_URI = f"runs:/{RUN_ID}/sklearn_model"
try:
    # Peut être un Pipeline (scikit-learn)
    model = mlflow.sklearn.load_model(MODEL_URI)
    _safe_print(f"[BOOT] Modèle chargé depuis {MODEL_URI}")
except Exception as e:
    _safe_print(f"[BOOT][ERREUR] Impossible de charger le modèle {MODEL_URI} : {e}")
    model = None

# ========== Récupération seuil BEST_T ==========
try:
    client = MlflowClient()
    run = client.get_run(RUN_ID)
    if "val_best_threshold" in run.data.metrics:
        BEST_T = float(run.data.metrics["val_best_threshold"])
        _safe_print(f"[BOOT] Seuil métier récupéré : {BEST_T}")
    else:
        _safe_print(f"[BOOT][AVERTISSEMENT] 'val_best_threshold' absent. Seuil par défaut {BEST_T} utilisé.")
except Exception as e:
    _safe_print(f"[BOOT][ERREUR] Récupération du run {RUN_ID} impossible : {e}. Seuil par défaut {BEST_T} utilisé.")

# ========== Chargement données ==========
if not DATA_PATH.exists():
    raise RuntimeError(f"[BOOT] Fichier de données introuvable : {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# Dataset d'inférence (ex : test = TARGET manquante)
if "TARGET" in df.columns:
    df_test = df[df["TARGET"].isna()].copy()
else:
    df_test = df.copy()

# Définition des colonnes features (exclusion des colonnes non-modélisables)
_cols_to_exclude = [c for c in ["SK_ID_CURR", "TARGET", "Unnamed: 0"] if c in df_test.columns]
FEATURE_COLS: List[str] = [c for c in df_test.columns if c not in _cols_to_exclude]

# Numériques pour comparaisons / distributions
NUMERIC_FEATURES: List[str] = df_test[FEATURE_COLS].select_dtypes(include="number").columns.tolist()

# Récupération éventuelle de steps si le modèle est un Pipeline
try:
    named_steps = getattr(model, "named_steps", {})
    imp = named_steps.get("imp") if named_steps else None
    scal = named_steps.get("scal") if named_steps else None
    # clf = named_steps.get("clf", model)  # utile si on sépare l'estimateur
except Exception:
    imp, scal = None, None


# =========================
# Utilitaires
# =========================
def make_json_safe(value: Any) -> Any:
    """Convertit une valeur pandas/numpy en valeur JSON-safe."""
    if isinstance(value, (np.floating, float)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    # Dates/TS
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _preprocess(X: pd.DataFrame) -> Any:
    """Applique les steps (imputer/scaler) si présents, sinon renvoie X."""
    Xp = X
    if imp is not None:
        Xp = imp.transform(Xp)
    if scal is not None:
        Xp = scal.transform(Xp)
    return Xp


# =========================
# SHAP : initialisation lazy
# =========================
def get_explainer():
    """Construit une seule fois l'explainer SHAP (lourd) au premier appel."""
    global _explainer
    if _explainer is not None:
        return _explainer

    if model is None:
        _safe_print("[SHAP] Modèle non chargé : explainer indisponible.")
        return None

    try:
        import shap  # import ici pour éviter coût si jamais non utilisé
    except Exception as e:
        _safe_print(f"[SHAP][ERREUR] shap non disponible : {e}")
        return None

    try:
        # Background sample
        BACKGROUND_SIZE = 1000
        if len(df_test) == 0:
            _safe_print("[SHAP][AVERTISSEMENT] df_test vide : explainer indisponible.")
            return None

        X_bg = df_test[FEATURE_COLS].sample(
            n=min(BACKGROUND_SIZE, len(df_test)),
            random_state=42
        )
        X_bg_proc = _preprocess(X_bg.copy())

        # Pour une régression logistique / modèles linéaires, LinearExplainer marche bien.
        # Si modèle non-linéaire, TreeExplainer ou Explainer générique peuvent être préférables.
        try:
            _explainer = shap.LinearExplainer(model if hasattr(model, "predict_proba") else model, X_bg_proc)
        except Exception:
            _explainer = shap.Explainer(model, X_bg_proc)

        _safe_print("[SHAP] Explainer initialisé.")
    except Exception as e:
        _safe_print(f"[SHAP][ERREUR] Construction échouée : {e}")
        _explainer = None

    return _explainer


# =========================
# FastAPI app
# =========================
app = FastAPI(title="API Scoring Crédit", version="1.0.0")


@app.get("/health")
def health():
    status = {"status": "ok"}
    status["model_loaded"] = model is not None
    status["mlflow_tracking_uri"] = MLFLOW_TRACKING_URI
    status["run_id"] = RUN_ID
    status["threshold"] = BEST_T
    status["data_path_exists"] = DATA_PATH.exists()
    status["explainer_ready"] = get_explainer() is not None
    return status


@app.get("/clients")
def get_clients(limit: Optional[int] = 1000):
    if "SK_ID_CURR" not in df_test.columns:
        raise HTTPException(status_code=500, detail="Colonne SK_ID_CURR absente de df_test.")
    client_ids = df_test["SK_ID_CURR"].head(limit).astype(int).tolist()
    return {"client_ids": client_ids}


@app.get("/client_info")
def client_info(client_id: int):
    if "SK_ID_CURR" not in df_test.columns:
        raise HTTPException(status_code=500, detail="Colonne SK_ID_CURR absente de df_test.")

    row = df_test[df_test["SK_ID_CURR"] == client_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Client introuvable dans df_test.")
    row = row.iloc[0]

    features: Dict[str, Any] = {col: make_json_safe(row[col]) for col in FEATURE_COLS}
    return {
        "client_id": int(client_id),
        "features": features,
        "numeric_features": NUMERIC_FEATURES,
    }


@app.get("/predict_client")
def predict_client(client_id: int):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle ML indisponible (chargement échoué).")

    if "SK_ID_CURR" not in df_test.columns:
        raise HTTPException(status_code=500, detail="Colonne SK_ID_CURR absente de df_test.")

    row = df_test[df_test["SK_ID_CURR"] == client_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Client introuvable dans df_test.")
    row = row.iloc[0]

    X_row = pd.DataFrame([row[FEATURE_COLS].to_dict()])
    # Si 'model' est un Pipeline, predict_proba inclut déjà les steps ; sinon on préprocess.
    try:
        proba = model.predict_proba(X_row)[:, 1][0]
    except Exception:
        Xp = _preprocess(X_row.copy())
        proba = model.predict_proba(Xp)[:, 1][0]

    y_pred = int(proba >= BEST_T)
    response: Dict[str, Any] = {
        "client_id": int(client_id),
        "probability_default": float(proba),
        "prediction": y_pred,
        "threshold_used": BEST_T,
        "top_features": [],
    }

    # ====== SHAP si disponible ======
    explainer = get_explainer()
    if explainer is not None:
        try:
            X_shap = X_row.copy()
            X_shap_proc = _preprocess(X_shap)
            shap_vals = explainer(X_shap_proc)
            values = getattr(shap_vals, "values", None)
            if values is None:
                # shap.Explainer récent renvoie un array directement
                values = np.array(shap_vals)[0]
            else:
                values = values[0]

            N_TOP = 10
            top_idx = np.argsort(np.abs(values))[::-1][:N_TOP]
            top_features = []
            for i in top_idx:
                fname = FEATURE_COLS[i]
                top_features.append({
                    "feature": fname,
                    "value": make_json_safe(row[fname]),
                    "impact": float(values[i]),  # >0 augmente le risque, <0 le diminue
                })
            response["top_features"] = top_features
        except Exception as e:
            response["shap_error"] = f"SHAP indisponible pour ce client: {e}"
    else:
        response["shap_info"] = "Explainer non initialisé."

    return response


@app.get("/global_distribution")
def global_distribution(feature: str, client_id: int):
    if feature not in df_test.columns:
        raise HTTPException(status_code=400, detail=f"Feature '{feature}' absente de df_test.")

    row = df_test[df_test["SK_ID_CURR"] == client_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Client introuvable.")
    client_value = row.iloc[0][feature]

    all_vals = [make_json_safe(v) for v in df_test[feature].dropna().tolist()]
    similar_vals = [make_json_safe(v) for v in df_test[feature].dropna().tolist()]  # simplification

    return {
        "feature": feature,
        "client_id": int(client_id),
        "client_value": make_json_safe(client_value),
        "all_clients": all_vals,
        "similar_clients": similar_vals,
    }


# ========= Démarrage local (optionnel) =========
# Permet de lancer: python api.py (dev uniquement)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)