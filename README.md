# Système Intelligent de Détection de Fraude à la Carte Bancaire

> Projet académique du cours de Concept Statistiques pour la science de données (INF 5103) à Université du Québec en Outaouais - UQO.

## Description du Projet
Ce projet a pour objectif de concevoir, d'entraîner et de déployer un modèle de Machine Learning capable de détecter les transactions frauduleuses à partir d'un jeu de données moderne de **20 000 transactions explicites** (comportant des variables comportementales, contextuelles et de risque lisibles, sans compression ACP). 

L'application intègre 4 onglets :

-Un onglet Accueil pour la présentation du contexte de l'étude

-Un onglet Analyse de données qui les différentes distributions des données utilisées par rapport à la variable d'interête

-Un onglet Modélisation qui presente les indicateurs de performance pour l'architecture retenue

-Un onglet Simulation pour permettre aux analystes des risques de tester des scénarios en temps réel et de visualiser les scores de probabilité de fraude.

---

## Structure du Projet

```text
TraceFraud/
|-- app/
|   |-- app.py
|   |-- pages/
|       |-- 1_accueil.py
|       |-- 2_analyse_donnees.py
|       |-- 3_modelisation.py
|       |-- 4_simulation.py
|-- data/
|   |-- processed/
|   |   |-- clean_transactions.csv
|   |-- raw/
|       |-- credit_card_fraud.csv
|-- models/
|   |-- modele_detect_fraude.pkl
|-- utils/
|   |-- __init__.py
|   |-- data_loader.py
|   |-- explore_data.py
|   |-- features.py
|   |-- modelling.py
|   |-- image_analyse/
|   |-- image_modele/
|--- README.md              
|--- requirements.txt         
|--- README.md                
|---.gitignore