import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from pathlib import Path
import os
import numpy as np

# ==========================
# Configuration
# ==========================
API_URL = os.getenv("API_URL", "http://localhost:8000")
client_ids=[]
pred_info=None
client_info=None


st.set_page_config(
    page_title="Dashboard scoring crédit (WCAG 2.1)",
    layout="wide"
)

# ==========================
# Styles : lisibilité + contrastes (sans casser Streamlit)
# ==========================
st.markdown(
    """
    <style>
      /* Améliore la lisibilité générale */
      html, body, [class*="css"]  {
        font-size: 16px;
        line-height: 1.5;
        color: #fff;
      }

      /* Focus clavier plus visible */
      :focus {
        outline: 3px solid #111 !important;
        outline-offset: 2px !important;
      }

      /* Cartes "info" (ne pas dépendre que de la couleur) */
      .wcag-card {
        border: 2px solid #222;
        border-radius: 10px;
        padding: 12px 14px;
        background: #fff;
      }
      .wcag-card h3 {
        margin: 0 0 6px 0;
        font-size: 18px;
      }
      .wcag-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 999px;
        border: 2px solid #222;
        font-weight: 700;
        margin-right: 8px;
      }
      .badge-ok { background: #e8f5e9; }      /* clair mais lisible car bordure + texte */
      .badge-bad { background: #ffebee; }
      .badge-mid { background: #fff8e1; }

      /* Table: éviter le gris trop clair */
      .stDataFrame, .stTable {
        color: #111;
      }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Dashboard – Scoring Crédit Client")
st.caption(
    "Accessibilité (WCAG 2.1) : la décision et le niveau de risque sont explicités par du texte + icônes, "
    "et les graphiques ne reposent pas uniquement sur la couleur."
)

# ==========================
# Helpers pour appeler l'API
# ==========================
@st.cache_data
def get_client_ids():
    resp = requests.get(f"{API_URL}/clients", timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["client_ids"]

@st.cache_data
def get_client_info(client_id: int):
    resp = requests.get(f"{API_URL}/client_info", params={"client_id": client_id}, timeout=30)
    resp.raise_for_status()
    return resp.json()

@st.cache_data
def get_prediction_client(client_id: int):
    resp = requests.get(f"{API_URL}/predict_client", params={"client_id": client_id}, timeout=30)
    resp.raise_for_status()
    return resp.json()

@st.cache_data
def get_global_distribution(feature: str, client_id: int):
    resp = requests.get(
        f"{API_URL}/global_distribution",
        params={"feature": feature, "client_id": client_id},
        timeout=30
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
    "Client (SK_ID_CURR) — choisissez un identifiant",
    client_ids
)

# Aide explicite (non dépendante d’un tooltip uniquement)
st.sidebar.caption(
    "Astuce accessibilité : vous pouvez naviguer au clavier (Tab/Shift+Tab) et valider avec Entrée."
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
# Utilitaires WCAG : risque en texte
# ==========================
def risk_level_from_score(score: float) -> str:
    # Exemple simple (à adapter à ton métier)
    if score < 0.20:
        return "Faible"
    if score < 0.50:
        return "Modéré"
    return "Élevé"

def decision_text(pred: int) -> str:
    return "REFUS" if pred == 1 else "ACCEPTATION"

def decision_icon(pred: int) -> str:
    return "❌" if pred == 1 else "✅"

def decision_badge_class(pred: int) -> str:
    return "badge-bad" if pred == 1 else "badge-ok"

# ==========================
# Bloc 1 : Score + interprétation
# ==========================
col_score, col_profile = st.columns([1, 2])

with col_score:
    st.subheader("Score de risque")
    
    if pred_info :
        score = float(pred_info["probability_default"])
        decision = int(pred_info["prediction"])
        threshold = float(pred_info.get("threshold_used", 0.5))

        level = risk_level_from_score(score)
        decision_lbl = decision_text(decision)
        icon = decision_icon(decision)

        # 1) Information explicite (pas seulement couleur)
        st.markdown(
            f"""
            <div class="wcag-card" role="region" aria-label="Résumé du score et de la décision">
            <h3>Résumé</h3>
            <p><strong>Probabilité de défaut estimée :</strong> {score:.1%}</p>
            <p><strong>Niveau de risque :</strong> {level}</p>
            <p>
                <span class="wcag-badge {decision_badge_class(decision)}">{icon} {decision_lbl}</span>
                <strong>Décision</strong> (seuil métier = {threshold:.2f})
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 2) Garder metric mais ne pas dépendre du help uniquement
        st.metric(
            "Probabilité de défaut estimée (en %)",
            f"{score:.1%}",
            help="Probabilité estimée que ce client fasse défaut. La décision ci-dessous est déterminée par comparaison au seuil métier."
        )

        # 3) Message décision : couleur + texte + icône (OK WCAG)
        if decision == 1:
            st.error(f"{icon} Décision : {decision_lbl} (seuil = {threshold:.2f}) — le score dépasse le seuil.")
        else:
            st.success(f"{icon} Décision : {decision_lbl} (seuil = {threshold:.2f}) — le score est sous le seuil.")

        st.caption(
            "La décision est prise en comparant le score au seuil métier optimisé (biz_threshold) afin de minimiser "
            "le coût des faux positifs et des faux négatifs. La décision est toujours explicitée en texte (pas uniquement par couleur)."
        )

        # Explications (top features)
        st.markdown("### Principales variables explicatives")
        top_features = pred_info.get("top_features", [])

        if top_features:
            exp_df = pd.DataFrame(top_features)

            exp_df["impact_direction"] = exp_df["impact"].apply(
                lambda x: "↑ augmente le risque" if x > 0 else "↓ diminue le risque"
            )
            exp_df["importance_abs"] = exp_df["impact"].abs()
            exp_df = exp_df.sort_values("importance_abs", ascending=False)

            # Renommer colonnes (plus parlant)
            exp_df = exp_df.rename(columns={
                "feature": "Variable",
                "value": "Valeur",
                "impact": "Impact (SHAP)"
            })

            st.dataframe(
                exp_df[["Variable", "Valeur", "Impact (SHAP)", "impact_direction"]],
                use_container_width=True
            )

            # Alternative textuelle : résumé
            top3 = exp_df.head(3)[["Variable", "impact_direction"]].values.tolist()
            summary = "; ".join([f"{v} : {d}" for v, d in top3])
            st.caption(f"Résumé (top 3) : {summary}.")
        else:
            st.info("Aucune explication renvoyée par l'API pour ce client.")

# ==========================
# Bloc 2 : Profil descriptif du client
# ==========================
with col_profile:
    st.subheader("Profil du client")

    if client_info:
        features = client_info.get("features", {})
        if features:
            descr_df = pd.DataFrame(list(features.items()), columns=["Variable", "Valeur"])
            st.dataframe(descr_df, use_container_width=True, height=400)

            # Export utile (accessibilité : permet lecture hors UI)
            csv = descr_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Télécharger le profil client (CSV)",
                data=csv,
                file_name=f"profil_client_{selected_client_id}.csv",
                mime="text/csv"
            )
        else:
            st.write("Aucune information descriptive reçue pour ce client.")
    else:
        st.write("Aucune information reçue pour ce client.")
# ==========================
# Bloc 3 : Comparaison à la population
# ==========================
st.markdown("---")
st.subheader("Comparaison du client à la population")

if client_info:
    numeric_features = client_info.get("numeric_features", [])
    if not numeric_features and features:
        tmp_df = pd.DataFrame([features])
        numeric_features = tmp_df.select_dtypes(include="number").columns.tolist()

    if numeric_features:
        feature_to_compare = st.selectbox(
            "Choisissez une variable numérique pour la comparaison (histogrammes + repère client)",
            numeric_features
        )

        try:
            dist = get_global_distribution(feature_to_compare, selected_client_id)
            all_vals = np.asarray(dist["all_clients"])
            similar_vals = np.asarray(dist["similar_clients"])
            client_value = float(dist["client_value"])

            # Graphique accessible : ne pas dépendre de la couleur -> hachures + styles + labels
            fig, ax = plt.subplots()
            ax.hist(all_vals, bins=30, alpha=0.6, label="Tous les clients", hatch="//")
            ax.hist(similar_vals, bins=30, alpha=0.6, label="Groupe similaire", hatch="..")

            ax.axvline(client_value, linestyle="--", linewidth=2, label="Client sélectionné (repère)")
            ax.set_title(f"Distribution de {feature_to_compare}")
            ax.set_xlabel(feature_to_compare)
            ax.set_ylabel("Effectif")
            ax.legend()

            st.pyplot(fig)

            # Alternative textuelle au graphique
            st.markdown(
                f"""
                **Description du graphique (texte alternatif)** :  
                Ce graphique compare la distribution de **{feature_to_compare}** pour tous les clients et pour un groupe similaire.
                La valeur du client sélectionné est **{client_value:.3g}** et est matérialisée par une ligne verticale en pointillés.
                """
            )

            st.caption(
                "Le groupe 'similaire' peut être défini dans l'API (ex : même type de contrat, même tranche de revenus, etc.). "
                "Les deux histogrammes sont distingués par des hachures en plus de la couleur."
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

report_path = Path("notebooks/artifacts/data_drift_report.html")

# Donner une alternative textuelle : le rapport est un iframe HTML (souvent peu accessible)
st.caption(
    "Note accessibilité : le rapport Evidently est intégré sous forme HTML. "
    "Si l’affichage est difficile, utilisez le téléchargement ci-dessous."
)

try:
    html = report_path.read_text(encoding="utf-8")

    # Téléchargement (meilleure accessibilité)
    st.download_button(
        label="Télécharger le rapport Data Drift (HTML)",
        data=html.encode("utf-8"),
        file_name="data_drift_report.html",
        mime="text/html"
    )

    # Affichage intégré
    st.components.v1.html(html, height=800, scrolling=True)

except Exception as e:
    st.info("Rapport Evidently non trouvé : `notebooks/artifacts/data_drift_report.html`.")
    st.caption(f"Détail : {e!r}")