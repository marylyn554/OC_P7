# test_api.py
# ------------------------------------------------------------
# Tests unitaires de l'API FastAPI décrite dans ton message.
# Hypothèse : ton fichier s'appelle "api.py".
# Exécuter:  pytest -q
# ------------------------------------------------------------
import sys
import types
import json
import numpy as np
import pandas as pd
from pathlib import Path
import importlib

import pytest
from fastapi.testclient import TestClient
from api import app

# ---------- Fixtures utilitaires ----------

@pytest.fixture(scope="session")
def fake_mlflow_module():
    """
    Construit un faux module mlflow (et sous-modules) avec
    juste ce qu'il faut pour que api.py puisse s'importer.
    """
    mlflow = types.ModuleType("mlflow")

    # mlflow.set_tracking_uri(...)
    def set_tracking_uri(uri: str):
        return None
    mlflow.set_tracking_uri = set_tracking_uri

    # mlflow.sklearn.load_model(...)
    class FakeModel:
        """Modèle minimal: predict_proba retourne proba classe 1."""
        def __init__(self):
            # pipeline-like attrs non utilisés ici
            self.named_steps = None

        def predict_proba(self, X):
            # Retourne une proba fixe en fonction d'une feature
            # (pour rendre le test déterministe).
            # shape = (n_samples, 2)
            n = len(X)
            p1 = np.full(n, 0.7)  # proba défaut
            p0 = 1.0 - p1
            return np.c_[p0, p1]

    sklearn = types.ModuleType("mlflow.sklearn")
    def load_model(uri: str):
        return FakeModel()
    sklearn.load_model = load_model
    mlflow.sklearn = sklearn

    # mlflow.tracking.MlflowClient().get_run(...).data.metrics["val_best_threshold"]
    tracking = types.ModuleType("mlflow.tracking")

    class FakeRunData:
        def __init__(self):
            self.metrics = {"val_best_threshold": 0.5}

    class FakeRun:
        def __init__(self):
            self.data = FakeRunData()

    class MlflowClient:
        def get_run(self, run_id: str):
            return FakeRun()

    tracking.MlflowClient = MlflowClient
    mlflow.tracking = tracking

    # Injecte dans sys.modules (pour que "from mlflow.tracking import MlflowClient" fonctionne)
    sys.modules["mlflow"] = mlflow
    sys.modules["mlflow.sklearn"] = sklearn
    sys.modules["mlflow.tracking"] = tracking

    return mlflow


@pytest.fixture(scope="session")
def fake_shap_module():
    """
    Faux module shap. LinearExplainer renvoie un callable qui
    renvoie un objet avec .values de la bonne forme.
    """
    shap = types.ModuleType("shap")

    class FakeShapValues:
        def __init__(self, n_samples, n_features):
            # valeurs shap arbitraires mais déterministes
            self.values = np.array([[0.1] * n_features for _ in range(n_samples)])

    class FakeExplainer:
        def __init__(self, model, background):
            # ignore les entrées; on ne s'en sert pas vraiment
            self.n_features_ = background.shape[1] if hasattr(background, "shape") else 2

        def __call__(self, X):
            n_samples = len(X)
            n_features = X.shape[1]
            return FakeShapValues(n_samples, n_features)

    shap.LinearExplainer = FakeExplainer

    # Injecte dans sys.modules
    sys.modules["shap"] = shap
    return shap

@pytest.fixture()
def client():
    return TestClient(app)


# ---------- TESTS ----------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_get_clients(client):
    r = client.get("/clients", params={"limit": 10})
    assert r.status_code == 200
    data = r.json()
    # On a bien les 2 clients issus de df_test (TARGET NaN)
    assert "client_ids" in data
    assert isinstance(data["client_ids"], list)
    assert data["client_ids"] == [100001, 100002]


def test_client_info_ok(client):
    r = client.get("/client_info", params={"client_id": 100001})
    assert r.status_code == 200
    payload = r.json()
    assert payload["client_id"] == 100001
    assert "features" in payload
    # Les features doivent contenir feat1 et feat2 (pas SK_ID_CURR ni TARGET)
    assert "feat1" in payload["features"]
    assert "feat2" in payload["features"]
    assert "numeric_features" in payload
    assert set(payload["numeric_features"]) >= {"feat1", "feat2"}


def test_client_info_not_found(client):
    r = client.get("/client_info", params={"client_id": 123456})
    assert r.status_code == 404
    assert "Client introuvable" in r.text


def test_predict_client_ok(client):
    r = client.get("/predict_client", params={"client_id": 100001})
    assert r.status_code == 200
    payload = r.json()
    # Clés attendues
    for key in ["client_id", "probability_default", "prediction", "threshold_used", "top_features"]:
        assert key in payload
    # Notre fake modèle renvoie p(default)=0.7 et seuil=0.5 -> prediction 1
    assert payload["client_id"] == 100001
    assert pytest.approx(payload["probability_default"], rel=1e-6) == 0.7
    assert payload["prediction"] == 1
    assert payload["threshold_used"] == 0.5
    # SHAP factice : top_features non vide
    assert isinstance(payload["top_features"], list)
    assert len(payload["top_features"]) >= 1
    assert {"feature", "value", "impact"} <= set(payload["top_features"][0].keys())


def test_predict_client_not_found(client):
    r = client.get("/predict_client", params={"client_id": 424242})
    assert r.status_code == 404
    assert "Client introuvable" in r.text


def test_global_distribution_ok(client):
    r = client.get("/global_distribution", params={"feature": "feat1", "client_id": 100001})
    assert r.status_code == 200
    data = r.json()
    assert data["feature"] == "feat1"
    assert data["client_id"] == 100001
    # valeur client = 1.0 dans le CSV de test
    assert data["client_value"] == 1.0
    assert isinstance(data["all_clients"], list)
    assert isinstance(data["similar_clients"], list)
    assert len(data["all_clients"]) >= 2


def test_global_distribution_bad_feature(client):
    r = client.get("/global_distribution", params={"feature": "unknown_col", "client_id": 100001})
    assert r.status_code == 400
    assert "absente de df_test" in r.text