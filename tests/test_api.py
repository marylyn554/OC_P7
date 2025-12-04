# test_api.py
# ------------------------------------------------------------
# Tests unitaires de l'API FastAPI (api.py).
# Exécuter:  pytest
# ------------------------------------------------------------
import sys
import types
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

# Le code de l'API sera importé via la fixture 'app' après le mocking

# --- Fixtures pour le Mocking (Isolation des Dépendances) ---

@pytest.fixture(scope="session")
def fake_df_test():
    """
    Crée un DataFrame minimal et déterministe pour simuler les données
    chargées par pd.read_csv().
    """
    data = {
        'SK_ID_CURR': [100001, 100002],
        'TARGET': [np.nan, np.nan],
        'feat1': [1.0, 5.0],  # Valeurs utilisées dans les assertions de distribution
        'feat2': [0.1, 0.5],
    }
    # Important: indexer sur SK_ID_CURR pour que la recherche par ID fonctionne
    df = pd.DataFrame(data)
    return df


@pytest.fixture(scope="session")
def fake_mlflow_module():
    """
    Construit et injecte un faux module mlflow pour simuler le chargement
    du modèle et du seuil (BEST_T).
    """
    mlflow = types.ModuleType("mlflow")

    # 1. FakeModel pour predict_proba
    class FakeModel:
        def predict_proba(self, X):
            n = len(X)
            p1 = np.full(n, 0.7)  # proba défaut fixe de 0.7
            p0 = 1.0 - p1
            return np.c_[p0, p1]

    # 2. mlflow.sklearn
    sklearn = types.ModuleType("mlflow.sklearn")
    # Simule mlflow.sklearn.load_model(...)
    sklearn.load_model = lambda uri: FakeModel()
    mlflow.sklearn = sklearn
    mlflow.set_tracking_uri = lambda uri: ()

    # 3. mlflow.tracking (pour le seuil BEST_T = 0.5)
    class FakeRunData:
        def __init__(self):
            self.metrics = {"val_best_threshold": 0.5}

    class MlflowClient:
        def get_run(self, run_id: str):
            return types.SimpleNamespace(data=FakeRunData())

    tracking = types.ModuleType("mlflow.tracking")
    tracking.MlflowClient = MlflowClient
    mlflow.tracking = tracking

    # Injection dans sys.modules
    sys.modules["mlflow"] = mlflow
    sys.modules["mlflow.sklearn"] = sklearn
    sys.modules["mlflow.tracking"] = tracking

    return mlflow


@pytest.fixture(scope="session")
def fake_shap_module():
    """
    Construit et injecte un faux module shap (LinearExplainer).
    """
    shap = types.ModuleType("shap")

    class FakeShapValues:
        # Valeurs SHAP factices
        def __init__(self, n_samples):
            # 2 features dans le df_test, donc 2 colonnes dans values
            self.values = np.array([[0.1, -0.1] for _ in range(n_samples)])

    class FakeExplainer:
        def __init__(self, model, background):
            pass

        def __call__(self, X):
            n_samples = len(X)
            return FakeShapValues(n_samples)

    shap.LinearExplainer = FakeExplainer

    # Injection dans sys.modules
    sys.modules["shap"] = shap
    return shap


@pytest.fixture(scope="session")
def app(fake_mlflow_module, fake_shap_module, fake_df_test):
    """
    Patch des dépendances AVANT l'import puis import de api.py
    situé au niveau parent du dossier des tests.
    """
    import importlib.util
    from unittest.mock import patch
    from pathlib import Path
    import sys
    import pandas as pd
    from fastapi import FastAPI, HTTPException
    from fastapi.testclient import TestClient
    
    tests_dir = Path(__file__).resolve().parent
    api_path = tests_dir.parent / "api.py"   # <-- ICI la différence clé

    # S'assurer que le parent (qui contient api.py) est dans sys.path
    parent_dir = str(tests_dir.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    with patch("os.path.exists", return_value=True), \
         patch("pathlib.Path.exists", return_value=True), \
         patch("pandas.read_csv", return_value=fake_df_test):

        # Import robuste via chemin absolu
        assert api_path.exists(), f"api.py introuvable à {api_path}"
        spec = importlib.util.spec_from_file_location("api", str(api_path))
        module = importlib.util.module_from_spec(spec)
        sys.modules["api"] = module
        spec.loader.exec_module(module)
        app = module.app

        yield app


@pytest.fixture()
def client(app):
    """Client de test FastAPI."""
    return TestClient(app)


# --- TESTS des Endpoints ---

def test_health(client):
    """Vérifie le point de terminaison de santé."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_get_clients(client):
    """Vérifie la liste des clients (issue du df factice)."""
    r = client.get("/clients", params={"limit": 10})
    assert r.status_code == 200
    data = r.json()
    assert "client_ids" in data
    assert data["client_ids"] == [100001, 100002]


def test_client_info_ok(client):
    """Vérifie les informations d'un client existant."""
    r = client.get("/client_info", params={"client_id": 100001})
    assert r.status_code == 200
    payload = r.json()

    assert payload["client_id"] == 100001
    assert "features" in payload
    # Vérifie que les features du df factice sont bien présentes
    assert all(feat in payload["features"] for feat in ["feat1", "feat2"])


def test_predict_client_ok(client):
    """Vérifie la prédiction et l'explication SHAP (issue des mocks)."""
    r = client.get("/predict_client", params={"client_id": 100001})
    assert r.status_code == 200
    payload = r.json()

    # Notre modèle factice renvoie p(default)=0.7 et seuil=0.5 -> prediction 1
    assert payload["client_id"] == 100001
    assert pytest.approx(payload["probability_default"], rel=1e-6) == 0.7
    assert payload["prediction"] == 1
    assert payload["threshold_used"] == 0.5
    
    # SHAP factice : vérifie la structure
    assert isinstance(payload["top_features"], list)
    assert len(payload["top_features"]) >= 1
    assert {"feature", "value", "impact"} <= set(payload["top_features"][0].keys())


def test_global_distribution_ok(client):
    """Vérifie la distribution d'une feature (issue du df factice)."""
    r = client.get("/global_distribution", params={"feature": "feat1", "client_id": 100001})
    assert r.status_code == 200
    data = r.json()

    assert data["feature"] == "feat1"
    assert data["client_id"] == 100001
    # client_value = 1.0 car c'est la valeur de 'feat1' pour 100001 dans fake_df_test
    assert data["client_value"] == 1.0
    assert isinstance(data["all_clients"], list)
    assert len(data["all_clients"]) == 2 # 2 lignes dans le df factice


@pytest.mark.parametrize("endpoint, client_id, expected_status, error_text", [
    ("/client_info", 123456, 404, "Client introuvable"),
    ("/predict_client", 424242, 404, "Client introuvable"),
])
def test_client_endpoints_not_found(client, endpoint, client_id, expected_status, error_text):
    """Regroupe les tests pour client_info et predict_client avec ID non trouvé."""
    r = client.get(endpoint, params={"client_id": client_id})
    assert r.status_code == expected_status
    assert error_text in r.text


def test_global_distribution_bad_feature(client):
    """Vérifie le cas d'échec pour une feature inconnue."""
    r = client.get("/global_distribution", params={"feature": "unknown_col", "client_id": 100001})
    assert r.status_code == 400
    assert "absente de df_test" in r.text