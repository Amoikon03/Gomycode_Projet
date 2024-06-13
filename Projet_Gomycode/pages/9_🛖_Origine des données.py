import streamlit as st

# Titre de l'application
st.title("Origine Des Données")

# Description de l'application
st.subheader("Veuillez cliquer sur ce lien pour consulter l'origine du fichier.")

# Ajout d'un lien vers la source de données
st.markdown("[Incidences de maladies sur la population de 2012 à 2015](https://data.gouv.ci/datasets/incidence-de-maladies-sur-la-population-de-2012-a-2015)")

# Lien vers les autres pages ou sections
st.subheader("Veuillez cliquer sur ces hyperliens pour être dirigé vers d'autres pages.")

st.write("""
- [Informations](http://localhost:8503/Informations)
- [Exploration des données](http://localhost:8503/Exploration_des_donn%C3%A9es)
- [Manipulation des données](http://localhost:8503/Manipulation_des_donn%C3%A9es)
- [Visualisation des données](http://localhost:8503/Visualisation_des_donn%C3%A9es)
- [Plus de visualisation](http://localhost:8503/Plus_de_visualisation)
- [Modélisation des données](http://localhost:8503/Mod%C3%A9lisation_des_donn%C3%A9es)
- [Modèle de prédiction](http://localhost:8503/Mod%C3%A8le_de_pr%C3%A9diction)

""")
