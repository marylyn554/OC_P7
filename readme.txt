Le dossier contient les fichiers suivants :


- .github/workflows/cicd.yml : pipeline git actions pour :
							- l'intégration continue (Execution des tests, generation des artifacts)
							- Le deploiement continue (Deploiement de l'api et du dashbord sur Aws ElasticBeansTalk)
- requirements-all.txt : La liste des packages pour tout le projet
							
*************** API ********************							
							
- api.py : Code source de l'api (Api Flask)
- Procfile-api.txt : Code de demarrage de l'api sur la plateforme de deploiement (AWS)
- requirements-api.txt : La liste des packages pour l'api
- test_api.py : test unitaires de l'api

****************** Dashboard *****************

- dashboard.py : Code source du dashboard (StreamLit)
- Procfile-dashboard.txt : Code de demarrage du dashboard sur la plateforme de deploiement (AWS)
- requirements-dashboard.txt : La liste des packages pour le dashboard
- test_dashboard.py : test unitaires du dashboard

****************** Notebooks *******************

- notebooks/artifacts/data_drift_report.html : Le tableau HTML d’analyse de data drift réalisé à partir d’evidently
- notebooks/artifacts/mlruns : dossier de tracking mlflow contenant les differentes experimentations et artifacts
- notebooks/artifacts/models : Contient le model final obtenu avec Mlflow et utilisé par l'api
- notebooks/Analyse Exploratoire.ipynb : le notebook jupyter pour l'analyse exploratoire
- notebooks/projet7_Modelisation.ipynb : le notebook jupyter pour la modelisation
- notebooks/data.csv : Le fichier data de prod deployé avec l'api
