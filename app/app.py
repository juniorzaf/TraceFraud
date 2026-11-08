import streamlit as st

# Configuration de la page Streamlit
st.set_page_config(
    page_title="TraceFraud",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Définition des pages de l'application
pg = st.navigation([
    st.Page("pages/accueil.py", title="Accueil et Contexte ", icon="🏠"),
    st.Page("pages/analyse_donnees.py", title="Analyse des données", icon="📊"),
    st.Page("pages/modelisation.py", title="Modélisation", icon="🤖"),
    st.Page("pages/simulation.py", title="Simulateur de Fraude", icon="⚡"),
])

# Lancement de la navigation
pg.run()