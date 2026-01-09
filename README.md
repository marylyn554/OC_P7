# 📦 Structure du projet

Ce dépôt contient l'ensemble des éléments nécessaires au développement, au test et au déploiement d'une **API Flask** et d'un **dashboard Streamlit**, accompagnés de notebooks de data science et d'un pipeline CI/CD.

---

## 🔁 CI/CD

### 📁 `.github/workflows/cicd.yml`
Pipeline **GitHub Actions** permettant :
- ✅ **Intégration continue**
  - Exécution des tests unitaires
  - Génération des artifacts
- 🚀 **Déploiement continu**
  - Déploiement de l'API
  - Déploiement du dashboard  
  sur **AWS Elastic Beanstalk**

---

## 📦 Dépendances globales

### 📄 `requirements-all.txt`
Liste complète des packages Python nécessaires à l'ensemble du projet.

---

## 🔌 API (Flask)

### Fichiers principaux
- **`api.py`**  
  Code source de l'API Flask
- **`Procfile-api.txt`**  
  Script de démarrage de l'API sur AWS Elastic Beanstalk
- **`requirements-api.txt`**  
  Dépendances spécifiques à l'API
- **`test_api.py`**  
  Tests unitaires de l'API

---

## 📊 Dashboard (Streamlit)

### Fichiers principaux
- **`dashboard.py`**  
  Code source du dashboard Streamlit
- **`Procfile-dashboard.txt`**  
  Script de démarrage du dashboard sur AWS Elastic Beanstalk
- **`requirements-dashboard.txt`**  
  Dépendances spécifiques au dashboard
- **`test_dashboard.py`**  
  Tests unitaires du dashboard

---

## 📓 Notebooks & Artifacts Data Science

### 📁 `notebooks/artifacts/`
- **`data_drift_report.html`**  
  Rapport HTML d'analyse du *data drift* généré avec **Evidently**
- **`mlruns/`**  
  Dossier de tracking **MLflow** contenant :
  - expérimentations
  - métriques
  - paramètres
  - artifacts
- **`models/`**  
  Modèle final enregistré avec MLflow et utilisé par l'API

### 📒 Notebooks Jupyter
- **`Analyse Exploratoire.ipynb`**  
  Analyse exploratoire des données
- **`projet7_Modelisation.ipynb`**  
  Modélisation et entraînement du modèle

### 📄 Données
- **`data.csv`**  
  Jeu de données de production déployé avec l'API

---

## 🧠 Technologies utilisées
- Python
- Flask
- Streamlit
- MLflow
- Evidently
- GitHub Actions
- AWS Elastic Beanstalk

---

## ✅ Objectif du projet

Fournir une solution complète de **machine learning industrialisé**, incluant :
- expérimentation et traçabilité des modèles
- surveillance de la dérive des données
- exposition du modèle via une API
- visualisation via un dashboard
- déploiement automatisé CI/CD
