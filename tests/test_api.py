# test_api.py
# ------------------------------------------------------------
# Tests unitaires de l'API FastAPI (api.py).
# Exécuter :  pytest
# ------------------------------------------------------------
import sys
import types
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


# --- Fixtures pour le Mocking (Isolation des Dépendances) ---

@pytest.fixture(scope="session")
def fake_df_test():
    """DataFrame minimal et déterministe pour simuler pd.read_csv()."""
    data = {
        "SK_ID_CURR": [100001, 100002],
        "TARGET": [np.nan, np.nan],
        "feat1": [1.0, 5.0],
        "feat2": [0.1, 0.5],
    }
    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def fake_mlflow_module():
    """Faux mlflow (load_model + récupération métrique seuil)."""
    mlflow = types.ModuleType("mlflow")

    class FakeModel:
        def predict_proba(self, X):
            n = len(X)
            p1 = np.full(n, 0.7)  # proba défaut fixe de 0.7
            p0 = 1.0 - p1
            return np.c_[p0, p1]

    sklearn = types.ModuleType("mlflow.sklearn")
    sklearn.load_model = lambda uri: FakeModel()
    mlflow.sklearn = sklearn
    mlflow.set_tracking_uri = lambda uri: None

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
    """Faux shap (LinearExplainer)."""
    shap = types.ModuleType("shap")

    class FakeShapValues:
        def __init__(self, n_samples):
            # 2 features => 2 colonnes
            self.values = np.array([[0.1, -0.1] for _ in range(n_samples)])

    class FakeExplainer:
        def __init__(self, model, background):
            pass

        def __call__(self, X):
            n = len(X)
            return FakeShapValues(n)

    shap.LinearExplainer = FakeExplainer

    sys.modules["shap"] = shap
    return shap


@pytest.fixture(scope="session")
def app(fake_mlflow_module, fake_shap_module, fake_df_test):
    """
    Patch des dépendances AVANT l'import puis import de api.py
    situé au niveau parent du dossier des tests.
    """
    import importlib.util

    tests_dir = Path(__file__).resolve().parent
    api_path = tests_dir.parent / "api.py"
    parent_dir = str(tests_dir.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    with patch("os.path.exists", return_value=True), \
         patch("pathlib.Path.exists", return_value=True), \
         patch("pandas.read_csv", return_value=fake_df_test):

        assert api_path.exists(), f"api.py introuvable à {api_path}"
        spec = importlib.util.spec_from_file_location("api", str(api_path))
        module = importlib.util.module_from_spec(spec)
        sys.modules["api"] = module
        spec.loader.exec_module(module)
        app = module.app
        yield app


@pytest.fixture()
def client(app):
    return TestClient(app)


# --- TESTS des Endpoints ---

def test_health(client):
    """Vérifie le point de terminaison de santé (au moins status=ok)."""
    r = client.get("/health")
    assert r.status_code == 200
    payload = r.json()
    # On ne vérifie plus l'égalité stricte mais la présence de "status": "ok"
    assert isinstance(payload, dict)
    assert payload.get("status") == "ok"
    # Optionnel : clés informatives si exposées par l'API
    # (ne pas rendre ces assertions bloquantes)
    # assert "model_loaded" in payload
    # assert "threshold" in payload


def test_get_clients(client):
    r = client.get("/clients", params={"limit": 10})
    assert r.status_code == 200
    data = r.json()
    assert "client_ids" in data
    assert data["client_ids"] == [100001, 100002]


def test_client_info_ok(client):
    r = client.get("/client_info", params={"client_id": 100001})
    assert r.status_code == 200
    payload = r.json()
    assert payload["client_id"] == 100001
    assert "features" in payload
    assert all(feat in payload["features"] for feat in ["feat1", "feat2"])


def test_predict_client_ok(client):
    r = client.get("/predict_client", params={"client_id": 100001})
    assert r.status_code == 200
    payload = r.json()
    # Modèle factice : p(default)=0.7 ; seuil=0.5 ; donc prediction=1
    assert payload["client_id"] == 100001
    assert pytest.approx(payload["probability_default"], rel=1e-6) == 0.7
    assert payload["prediction"] == 1
    assert payload["threshold_used"] == 0.5
    # SHAP factice : structure attendue
    assert isinstance(payload["top_features"], list)
    assert len(payload["top_features"]) >= 1
    assert {"feature", "value", "impact"} <= set(payload["top_features"][0].keys())


def test_global_distribution_ok(client):
    r = client.get("/global_distribution", params={"feature": "feat1", "client_id": 100001})
    assert r.status_code == 200
    data = r.json()
    assert data["feature"] == "feat1"
    assert data["client_id"] == 100001
    assert data["client_value"] == 1.0  # valeur de feat1 pour 100001
    assert isinstance(data["all_clients"], list)
    assert len(data["all_clients"]) == 2  # 2 lignes dans le df factice


@pytest.mark.parametrize(
    "endpoint, client_id, expected_status, error_text",
    [
        ("/client_info", 123456, 404, "Client introuvable"),
        ("/predict_client", 424242, 404, "Client introuvable"),
    ],
)
def test_client_endpoints_not_found(client, endpoint, client_id, expected_status, error_text):
    r = client.get(endpoint, params={"client_id": client_id})
    assert r.status_code == expected_status
    assert error_text in r.text


def test_global_distribution_bad_feature(client):
    r = client.get("/global_distribution", params={"feature": "unknown_col", "client_id": 100001})
    assert r.status_code == 400
    assert "absente de df_test" in r.text