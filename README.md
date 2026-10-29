# Système Intelligent de Détection de Fraude à la Carte Bancaire

> Projet académique de Machine Learning appliqué à la gestion des risques financiers (Université du Québec en Outaouais - UQO).

## Description du Projet
Ce projet a pour objectif de concevoir, d'entraîner et de déployer un modèle de Machine Learning capable de détecter les transactions frauduleuses à partir d'un jeu de données moderne de **20 000 transactions explicites** (comportant des variables comportementales, contextuelles et de risque lisibles, sans compression ACP). 

L'application intègre également un tableau de bord interactif sous **Streamlit** pour permettre aux analystes des risques de tester des scénarios en temps réel et de visualiser les scores de probabilité de fraude.

---

## Structure du Projet

```text
projet_fraude/
│
├── data/
│   ├── raw/                 # Données brutes immuables (transactions_2026.csv)
│   └── processed/           # Données nettoyées et transformées
│
├── src/                     # Code source modulaire
│   ├── __init__.py
│   ├── data_loader.py       # Chargement et validation des données
│   ├── features.py          # Feature engineering et encodage
│   └── train.py             # Entraînement et évaluation du modèle XGBoost
│
├── app/
│   └── dashboard.py         # Interface utilisateur interactive (Streamlit)
│
├── models/                  # Modèles sérialisés (.joblib)
├── requirements.txt         # Dépendances du projet
├── README.md                # Documentation du projet
└── .gitignore