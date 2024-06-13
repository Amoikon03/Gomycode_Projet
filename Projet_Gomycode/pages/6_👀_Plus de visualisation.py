import pygwalker as pyg
import pickle as pk
import streamlit.components.v1 as components
import streamlit as st
import pandas as pd

def afficher_make_visualisation():
    st.set_page_config(layout="wide")

    # Chargement des données prétraitées dans data manipulation
    # Chargement du fichier avec read_csv
    df = pd.read_csv("Projet_Gomycode/Incidence_Maladie.csv")


    # Ajouter un titre
    st.title("Voici l'Endroit Pour Créer Vos Visualisations")
    st.info("Déplacez les noms des colonnes de la 'Field List' vers les axes X et Y en les glissant et déposant.")


    # Générer le HTML en utilisant Pygwalker
    pyg_html = pyg.to_html(df)

    # Intégrer le HTML dans l'application Streamlit app
    components.html(pyg_html, height=1000, scrolling=True)


afficher_make_visualisation()

# Lien vers les autres pages ou sections
st.subheader("Veuillez cliquer sur ces hyperliens pour être dirigé vers d'autres pages.")

st.write("""
- [Acceuil](http://localhost:8503/)
- [Informations](http://localhost:8503/Informations)
- [Exploration des données](http://localhost:8503/Exploration_des_donn%C3%A9es)
- [Manipulation des données](http://localhost:8503/Manipulation_des_donn%C3%A9es)
- [Visualisation des données](http://localhost:8503/Visualisation_des_donn%C3%A9es)
- [Modélisation des données](http://localhost:8503/Mod%C3%A9lisation_des_donn%C3%A9es)
- [Modèle de prédiction](http://localhost:8503/Mod%C3%A8le_de_pr%C3%A9diction)
- [Origine des données](http://localhost:8501/Origine_des_donn%C3%A9es)

""")
