from streamlit.testing.v1 import AppTest
from unittest.mock import patch
from pathlib import Path
from unittest.mock import patch, Mock
from dashboard import get_client_ids, API_URL

def test_title():
    at = AppTest.from_file("dashboard.py").run()
    assert at.title[0].value == "Dashboard – Scoring Crédit Client"

def test_sidebar_header():
    at = AppTest.from_file("dashboard.py").run()
    assert at.sidebar.header[0].value == "Sélection du client"

def test_get_client_ids_calls_clients_endpoint():
    fake_resp = Mock()
    fake_resp.raise_for_status.return_value = None
    fake_resp.json.return_value = {"client_ids": [1, 2, 3]}
    
    with patch("dashboard.requests.get", return_value=fake_resp) as mock_get:
        result = get_client_ids()

    # 1️⃣ vérifier l'appel HTTP
    mock_get.assert_called_once_with(f"{API_URL}/clients")