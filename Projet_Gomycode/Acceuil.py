# Import des librairies
import streamlit as st
from PIL import Image

# Configuration de la page
st.set_page_config(
    page_title="Incidence De Maladies Sur La Population De 2012 à 2015",
    page_icon="☏"
)

# Fonction pour afficher une section avec titre et contenu
def section(titre, contenu):
    st.header(titre)
    st.write(contenu)

# Fonction pour afficher une image avec un titre en dessous
def image_with_caption(image_path, caption):
    img = Image.open(image_path)
    st.image(img, caption=caption, use_column_width=True)

# Fonction pour afficher un paragraphe justifié
def paragraphe(texte):
    st.write(f"<div style='text-align: justify'>{texte}</div>", unsafe_allow_html=True)

# Titre de page
st.title("Incidence De Maladies Sur La Population De 2012 à 2015")

# Image illustrative de l'application
image_with_caption("moustique_tigre.jpg", "Moustique-Tigre.jpg")

# Description de l'application
paragraphe("""

Le projet vise à analyser l'incidence des maladies en Côte d'Ivoire entre 2012 et 2015 pour identifier les tendances et variations. Il inclut le nettoyage des données, une analyse exploratoire détaillée, et l'application de modèles de machine learning comme Random Forests et Gradient Boosting pour prédire les incidences futures. Une application interactive sera développée pour visualiser ces données et prédictions, aidant ainsi les décideurs à améliorer les politiques de santé publique. L'objectif final est de fournir des outils pratiques pour mieux comprendre et gérer les incidences des maladies.
""")


# Définition de la section "Fonctionnalités de l'application"
def fonctionnalites_application():
    st.header("Fonctionnalités de l'application")

    # Texte justifié en HTML
    justification_texte = """
    <div style="text-align:justify">
   L'application devrait permettre aux utilisateurs d'explorer les données à travers des graphiques interactifs, offrant une visualisation claire des tendances d'incidence des maladies. 
   Les utilisateurs devraient également pouvoir effectuer des recherches spécifiques par région ou par type de maladie pour obtenir des informations ciblées. 
   De plus, la comparaison des tendances au fil du temps devrait être facilitée, permettant aux utilisateurs de voir comment l'incidence des maladies évolue dans différentes régions ou pour différents types de maladies.
    </div>
    """
    st.markdown(justification_texte, unsafe_allow_html=True)

# Affichage de la section
fonctionnalites_application()

# Définition de la section avec justification CSS
def section(titre, texte):
    st.header(titre)
    st.markdown(f'<div style="text-align: justify;">{texte}</div>', unsafe_allow_html=True)

# Affichage de chaque section avec justification
section("Informations sur les données", "Cette étape consiste à communiquer les résultats de l'analyse des données de manière claire et compréhensible. Cela peut inclure la création de rapports, de visualisations ou de tableaux de bord pour présenter les conclusions de l'analyse. L'objectif est de fournir des informations exploitables aux parties prenantes et de soutenir la prise de décision basée sur les données.")

section("Exploration des données", "Cette étape consiste à explorer et à comprendre les données brutes. Elle inclut l'examen initial des données pour identifier les tendances, les modèles, les valeurs aberrantes et les relations entre les différentes variables. L'objectif principal de cette étape est de générer des hypothèses et des questions pour guider les analyses ultérieures.")

section("Manipulation des données", "Une fois que les données ont été explorées, la manipulation des données intervient pour nettoyer, transformer et préparer les données en vue de l'analyse. Cela peut inclure le traitement des valeurs manquantes, la normalisation des données, la création de nouvelles variables dérivées, etc. L'objectif est de rendre les données prêtes à être utilisées dans les modèles d'analyse ou les visualisations.")

section("Modélisation", "Cette étape implique la construction de modèles statistiques ou d'algorithmes d'apprentissage automatique pour répondre à des questions spécifiques ou résoudre des problèmes. Cela peut inclure l'utilisation de techniques telles que la régression, la classification, le clustering, etc. L'objectif est de créer des modèles prédictifs ou des représentations des données qui peuvent être utilisés pour prendre des décisions ou générer des insights.")

section("Contactez-Nous", "Prendre contact pour plus d'information")

section("À Propos De Nous", "Qui sommes nous et comment nous rejoindre")

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
- [Origine des données](http://localhost:8501/Origine_des_donn%C3%A9es)

""")
