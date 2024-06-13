# Importater les librairies
import streamlit as st
import pandas as pd

# Titre de la page
st.title("Les Informations Et Leurs Spécificités")


# Chargement de données du fichier csv
df = pd.read_csv("Projet_Gomycode/Incidence.csv")


# Définir les styles CSS pour le conteneur
style = """
    <style>
        .custom-text {
            color: white;
            background-color: #0D1D2C;
            padding: 10px;
            font-size: 18px;
            font-family: Arial, sans-serif;
        }
    </style>
"""
st.markdown(style, unsafe_allow_html=True)
st.markdown('<div class="custom-text">Voici les données requises pour cette application </div>', unsafe_allow_html=True)

st.write(" ")

# Afficher les noms des données avec le style spécifié
st.write(df.head())

# Afficher les noms des colonnes
st.subheader("**Les libellés des colonnes**")

# Utilisation de st.write() pour afficher chaque ligne de texte avec les classes CSS appliquées
st.write("<span class='uppercase-bold'>**1 - ANNEE**</span> : Indique l'année à laquelle les données se rapportent. Chaque entrée dans cette colonne représente une année spécifique, par exemple, 2012, 2013, 2014 ou 2015.</div>", unsafe_allow_html=True)
st.write("<span class='uppercase-bold'>**2 - REGIONS/DISTRICTS**</span> : Fournit les noms des régions ou des districts géographiques auxquels les données sont associées. Les données peuvent être regroupées par région ou district pour analyser les variations géographiques dans l'incidence des maladies.</div>", unsafe_allow_html=True)
st.write("<span class='uppercase-bold'>**3 - VILLES/COMMUNES**</span> : Noms des villes ou communes spécifiques qui sont concernées par les données. Cela permet une analyse plus granulaire au niveau local, en examinant comment l'incidence des maladies varie d'une ville à l'autre.</div>", unsafe_allow_html=True)
st.write("<span class='uppercase-bold'>**4 - MALADIE**</span> : Répertorie le type de maladie enregistré dans les données. Chaque entrée correspond à une maladie spécifique, par exemple, paludisme, choléra, grippe, etc.</div>", unsafe_allow_html=True)
st.write("<span class='uppercase-bold'>**5 - INCIDENCE_SUR_LA_POPULATION_GENERALE_(%)**</span> : Cette colonne représente l'incidence de la maladie sur la population générale, exprimée en pourcentage. Elle indique la proportion de la population qui a été affectée par la maladie pendant une année donnée et dans une région ou une ville spécifique.</div>", unsafe_allow_html=True)

# Utilisation de HTML pour styliser le texte avec un fond jaune
st.write("**NB** : Ces informations donnent un aperçu des données contenues dans chaque colonne de lensemble de données sur l'Incidences de maladies sur la population de 2012 à 2015")

# Lien vers les autres pages ou sections
st.subheader("Veuillez cliquer sur ces hyperliens pour être dirigé vers d'autres pages.")

st.write("""
- [Acceuil](http://localhost:8503/)
- [Exploration des données](http://localhost:8503/Exploration_des_donn%C3%A9es)
- [Manipulation des données](http://localhost:8503/Manipulation_des_donn%C3%A9es)
- [Visualisation des données](http://localhost:8503/Visualisation_des_donn%C3%A9es)
- [Plus de visualisation](http://localhost:8503/Plus_de_visualisation)
- [Modélisation des données](http://localhost:8503/Mod%C3%A9lisation_des_donn%C3%A9es)
- [Modèle de prédiction](http://localhost:8503/Mod%C3%A8le_de_pr%C3%A9diction)
- [Origine des données](http://localhost:8501/Origine_des_donn%C3%A9es)

""")
