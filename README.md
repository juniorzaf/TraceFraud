# Système Intelligent de Détection de Fraude à la Carte Bancaire

> Projet académique du cours de Concept Statistiques pour la science de données (INF 5103) à Université du Québec en Outaouais - UQO.

## Description du Projet
Ce projet a pour objectif de concevoir, d'entraîner et de déployer un modèle de Machine Learning capable de détecter les transactions frauduleuses à partir d'un jeu de données moderne de **20 000 transactions explicites** (comportant des variables comportementales, contextuelles et de risque lisibles, sans compression ACP). 

L'application intègre également un tableau de bord interactif sous **Streamlit** pour permettre aux analystes des risques de tester des scénarios en temps réel et de visualiser les scores de probabilité de fraude.

---

## Structure du Projet

```text
projet_fraude/
│
├── data/
│   ├── raw/                 
│   └── processed/           
│
├── src/                     
│   ├── data_loader.py       
│   ├── features.py          
│   └── train.py             
│
├── app/
│   └── dashbord.py         
│
├── models/                  
├── requirements.txt         
├── README.md                
└── .gitignore