import streamlit as st

st.set_page_config(page_title="Mon Application", layout="wide")

# Définition des pages de l'application
pg = st.navigation([
    st.Page("pages/accueil.py", title="Accueil", icon="🏠"),
    st.Page("pages/analyse_descriptif.py", title="Analyse Descriptif", icon=""),
    st.Page("pages/simulation.py", title="Simulateur de Fraude", icon=""),
])

# Lancement de la navigation
pg.run()