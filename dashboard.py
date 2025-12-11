import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from pathlib import Path
import os

# ==========================
# Configuration
# ==========================
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Dashboard scoring crédit",
    layout="wide"
)

st.title("Dashboard – Scoring Crédit Client")

# ==========================
# Helpers pour appeler l'API
# ==========================
@st.cache_data
def get_client_ids():
    resp = requests.get(f"{API_URL}/clients")
    resp.raise_for_status()
    data = resp.json()
    return data["client_ids"]


@st.cache_data
def get_client_info(client_id: int):
    resp = requests.get(f"{API_URL}/client_info", params={"client_id": client_id})
    resp.raise_for_status()
    return resp.json()


@st.cache_data
def get_prediction_client(client_id: int):
    resp = requests.get(f"{API_URL}/predict_client", params={"client_id": client_id})
    resp.raise_for_status()
    return resp.json()


@st.cache_data
def get_global_distribution(feature: str, client_id: int):
    resp = requests.get(
        f"{API_URL}/global_distribution",
        params={"feature": feature, "client_id": client_id}
    )
    resp.raise_for_status()
    return resp.json()


# ==========================
# Barre latérale : choix client
# ==========================
st.sidebar.header("Sélection du client")

try:
    client_ids = get_client_ids()
except Exception as e:
    st.sidebar.error(f"Impossible de récupérer les clients depuis l'API : {e}")
    st.stop()

selected_client_id = st.sidebar.selectbox(
    "Client (SK_ID_CURR)",
    client_ids
)


# ==========================
# Récupération des infos API
# ==========================
try:
    client_info = get_client_info(selected_client_id)
    pred_info = get_prediction_client(selected_client_id)
except Exception as e:
    st.error(f"Erreur lors de l'appel à l'API : {e}")
    st.stop()


# ==========================
# Bloc 1 : Score + interprétation
# ==========================
col_score, col_profile = st.columns([1, 2])

with col_score:
    st.subheader("Score de risque")

    score = pred_info["probability_default"]
    decision = pred_info["prediction"]
    threshold = pred_info.get("threshold_used", 0.5)

    st.metric(
        "Probabilité de défaut estimée",
        f"{score:.1%}",
        help="Score de probabilité que ce client fasse défaut sur son crédit."
    )

    if decision == 1:
        st.error(f"Décision : REFUS (seuil = {threshold:.2f})")
    else:
        st.success(f"Décision : ACCEPTATION (seuil = {threshold:.2f})")

    st.caption(
        "La décision est prise en comparant le score au seuil métier optimisé "
        "(biz_threshold) afin de minimiser le coût des faux positifs et des faux négatifs."
    )

    # Explications (top features) si dispo
    st.markdown("### Principales variables explicatives")

    top_features = pred_info.get("top_features", [])

    if top_features:
        exp_df = pd.DataFrame(top_features)

        # colonne direction + formatage
        exp_df["impact_direction"] = exp_df["impact"].apply(
            lambda x: "↑ augmente le risque" if x > 0 else "↓ diminue le risque"
        )

        # Option : ordonner par importance absolue
        exp_df["importance_abs"] = exp_df["impact"].abs()
        exp_df = exp_df.sort_values("importance_abs", ascending=False)

        st.dataframe(
            exp_df[["feature", "value", "impact", "impact_direction"]],
            use_container_width=True
        )
    else:
        st.info("Aucune explication SHAP renvoyée par l'API pour ce client.")

# ==========================
# Bloc 2 : Profil descriptif du client
# ==========================
with col_profile:
    st.subheader("Profil du client")

    features = client_info.get("features", {})
    if features:
        descr_df = pd.DataFrame(
            list(features.items()),
            columns=["Variable", "Valeur"]
        )
        st.dataframe(descr_df, use_container_width=True, height=400)
    else:
        st.write("Aucune information descriptive reçue pour ce client.")


# ==========================
# Bloc 3 : Comparaison à la population
# ==========================
st.markdown("---")
st.subheader("Comparaison du client à la population")

numeric_features = client_info.get("numeric_features", [])
if not numeric_features and features:
    # fallback : on prend les colonnes numériques si l'API ne renvoie pas numeric_features
    tmp_df = pd.DataFrame([features])
    numeric_features = tmp_df.select_dtypes(include="number").columns.tolist()

if numeric_features:
    feature_to_compare = st.selectbox(
        "Choisissez une variable numérique pour la comparaison :",
        numeric_features
    )

    try:
        dist = get_global_distribution(feature_to_compare, selected_client_id)
        all_vals = dist["all_clients"]
        similar_vals = dist["similar_clients"]
        client_value = dist["client_value"]

        fig, ax = plt.subplots()
        ax.hist(all_vals, bins=30, alpha=0.5, label="Tous les clients")
        ax.hist(similar_vals, bins=30, alpha=0.5, label="Groupe similaire")

        ax.axvline(client_value, linestyle="--", linewidth=2, label="Client sélectionné")
        ax.set_xlabel(feature_to_compare)
        ax.set_ylabel("Effectif")
        ax.legend()

        st.pyplot(fig)

        st.caption(
            "Le groupe 'similaire' peut être défini dans l'API (par exemple même type de contrat, "
            "même tranche de revenus, etc.). Ici, il peut être égal au dataset test pour simplifier."
        )

    except Exception as e:
        st.error(f"Erreur lors de la récupération de la distribution globale : {e}")
else:
    st.info("Aucune variable numérique disponible pour la comparaison.")


# ==========================
# Bloc 4 : Rapport Evidently (Data Drift)
# ==========================
st.markdown("---")
st.subheader("Surveillance du Data Drift (Evidently)")

report_path = Path('notebooks/artifacts/data_drift_report.html')

try:
    html = report_path.read_text(encoding="utf-8")
    st.components.v1.html(html, height=800, scrolling=True)
except Exception as e:
    st.info(
        "Rapport Evidently non trouvé. "
        "`notebooks/artifacts/data_drift_report.html`."
    )
    st.caption(f"Erreur : {e!r}")