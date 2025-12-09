# ===============================================================
# Tests unitaires pour l'API FastAPI (nouvelle version sans MLflow)
# ===============================================================

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------

@pytest.fixture(scope="session")
def fake_df_test():
    """DataFrame minimal simulant data.csv."""
    data = {
        "SK_ID_CURR": [100001, 100002],
        "TARGET": [np.nan, np.nan],
        "feat1": [1.0, 5.0],
        "feat2": [0.1, 0.5],
    }
    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def fake_model():
    """Faux modèle simulant model.pkl."""
    class FakeModel:
        def predict_proba(self, X):
            # Pour tester : p(default) = 0.7 → prédiction = 1 si BEST_T=0.5
            n = len(X)
            p1 = np.full(n, 0.7)
            p0 = 1.0 - p1
            return np.c_[p0, p1]
    return FakeModel()


@pytest.fixture(scope="session")
def app(fake_df_test, fake_model):
    """
    Import de api.py avec :
    - mock de pandas.read_csv
    - mock de joblib.load
    - mock des checks d'existence
    """
    import importlib.util

    tests_dir = Path(__file__).resolve().parent
    api_path = tests_dir.parent / "api.py"

    # Hack sys.path
    parent_dir = str(tests_dir.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    with patch("os.path.exists", return_value=True), \
         patch("pandas.read_csv", return_value=fake_df_test), \
         patch("joblib.load", return_value=fake_model):

        spec = importlib.util.spec_from_file_location("api", str(api_path))
        module = importlib.util.module_from_spec(spec)
        sys.modules["api"] = module
        spec.loader.exec_module(module)
        app = module.app
        yield app


@pytest.fixture()
def client(app):
    return TestClient(app)


# ---------------------------------------------------------------
# Tests des endpoints
# ---------------------------------------------------------------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "threshold" in data


def test_get_clients(client):
    r = client.get("/clients", params={"limit": 10})
    assert r.status_code == 200
    assert r.json()["client_ids"] == [100001, 100002]


def test_client_info_ok(client):
    r = client.get("/client_info", params={"client_id": 100001})
    assert r.status_code == 200
    data = r.json()
    assert data["client_id"] == 100001
    assert "features" in data
    assert "feat1" in data["features"]
    assert "feat2" in data["features"]


def test_predict_client_ok(client):
    r = client.get("/predict_client", params={"client_id": 100001})
    assert r.status_code == 200
    payload = r.json()

    assert payload["client_id"] == 100001
    assert pytest.approx(payload["probability_default"], rel=1e-6) == 0.7
    assert payload["prediction"] == 1
    assert "threshold_used" in payload


def test_global_distribution_ok(client):
    r = client.get(
        "/global_distribution",
        params={"feature": "feat1", "client_id": 100001}
    )
    assert r.status_code == 200
    data = r.json()

    assert data["feature"] == "feat1"
    assert data["client_value"] == 1.0
    assert len(data["all_clients"]) == 2


@pytest.mark.parametrize(
    "endpoint, cid",
    [
        ("/client_info", 999999),
        ("/predict_client", 888888),
    ]
)
def test_client_not_found(client, endpoint, cid):
    r = client.get(endpoint, params={"client_id": cid})
    assert r.status_code == 404
    assert "Client introuvable" in r.text


def test_global_distribution_bad_feature(client):
    r = client.get(
        "/global_distribution",
        params={"feature": "wrong_feature", "client_id": 100001}
    )
    assert r.status_code == 400
    assert "absente" in r.text