import streamlit as st
import os

st.title("Analyse Exploratoire des Données")
st.write("Visualisations graphiques chargées depuis le dossier utils à la racine.")

#On remonte de 2 niveaux puis entre dans 'utils/image_analyse'
dossier_images = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "utils", "image_analyse"))

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