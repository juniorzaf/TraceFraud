#Modélisation
import streamlit as st
import pandas as pd

import streamlit as st
import os

st.title("Analyse du Modèle")

#Tableau des métriques
st.subheader("Tableau synthétique des performances du modèle")
metrics_data = {
    "Seuil de Décision": [0.5],
    "ROC-AUC": [0.9373],
    "PR-AUC (Précision-Rappel)": [0.3658],
    "Rappel (Recall)": [0.85],
    "F1-Score": [0.17]
}
df_metrics = pd.DataFrame(metrics_data)
st.dataframe(df_metrics.set_index("Seuil de Décision"), use_container_width=True)
st.markdown("----")

st.subheader("Analyse des choix techniques")
st.markdown(
    """
    - **Pourquoi le PR-AUC plutôt que le ROC-AUC ?** Sur des données hautement déséquilibrées, le score PR-AUC est beaucoup plus strict et reflète fidèlement la capacité du modèle à détecter la fraude sans générer trop de fausses alertes.
    - **Optimisation des seuils :** Plutôt que de fixer le seuil de décision à 0.5 par défaut, un ajustement du seuil (ex: 0.3) permet d'optimiser le compromis entre le coût bloquant des faux positifs et le risque financier des faux négatifs.
    """
)
st.markdown("----")

st.subheader("Visualisations graphiques des indicateurs de performance du modèle.")
#On remonte de 2 niveaux puis on entre dans 'utils/image_modele'
dossier_images = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "utils", "image_modele"))

if os.path.exists(dossier_images):
    fichiers_images = [f for f in os.listdir(dossier_images) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if fichiers_images:
        fichiers_images.sort()

        for fichier in fichiers_images:
            chemin_complet = os.path.join(dossier_images, fichier)
            nom_propre = fichier.split('.')[0].replace('_', ' ').capitalize()

            st.subheader(nom_propre)
            st.image(chemin_complet, use_container_width=True)
            st.markdown("---")
    else:
        st.warning(f"Aucune image trouvée dans `{dossier_images}`.")
else:
    st.error(f"Le dossier `{dossier_images}` est introuvable. Vérifiez son emplacement.")




