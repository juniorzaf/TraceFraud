# accueil
import streamlit as st

st.title("🛡 TraceFraud : Détection Intelligente de Fraude")
st.markdown("### Projet Académique du cours de Concepts Statistique pour la Science de données à UQO (INF 5103)")

st.info(
    "**Objectif du projet :** Concevoir et évaluer un pipeline de machine learning "
    "capable de détecter les transactions bancaires frauduleuses dans un contexte de données fortement déséquilibrées."
)

st.header("Contexte du Problème")
st.write(
    "La fraude à la carte bancaire représente un défi majeur pour les institutions financières. "
    "Le problème principal réside dans l'extrême rareté des cas de fraude (souvent moins de 1 % des transactions totales), "
    "ce qui rend les modèles traditionnels inefficaces sans une approche rigoureuse de traitement des données et d'évaluation."
)

col1, col2, col3 = st.columns(3)
col1.metric("Volume analysé", "20 000", "Transactions")
col2.metric("Taux de fraude", "1,7%", "Très déséquilibré")
col3.metric("Modèle Utilisé", "XGBoost", "Optimisés")

st.header("Navigation dans l'application")
st.write(
    "Utilisez le menu latéral pour explorer les différentes sections :"
)
st.markdown(
    """
    - **📊 Analyse des Données (EDA) :** Visualisation de la distribution des transactions et du déséquilibre des classes.
    - **🤖 Modélisation & ML :** Comparaison des performances des algorithmes et choix des métriques (ROC-AUC, PR-AUC).
    - **⚡ Simulateur en direct :** Testez une transaction en temps réel pour évaluer la décision du modèle.
    """
)