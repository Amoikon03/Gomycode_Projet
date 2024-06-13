import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def main():
    st.title("Manipulation Des Données")

    # Charger le fichier CSV
    df = pd.read_csv("Incidence.csv")

    afficher_informations_generales(df)
    supprimer_doublons(df)
    traiter_valeurs_manquantes(df)
    enregistrer_donnees_pretraitees(df)


def afficher_informations_generales(df):
    st.write("### **1 - Présenter un aperçu complet du jeu de données** ")
    st.write(df.head())
    st.write(df.info())

def supprimer_doublons(df):
    # Trouver les lignes dupliquées
    duplique = df.duplicated()
    duplicates = df[duplique]

    # Afficher les lignes dupliquées
    st.write("### **2 - Lignes dupliquées** ")
    st.write(duplicates)

    st.write("### **3 - Supprimer les valeurs dupliquées, s'ils existent** ")
    df.drop_duplicates(inplace=True)
    st.write("**Les lignes dupliquées ont été supprimées avec succès.**")


def traiter_valeurs_manquantes(df):

    st.write("### **3 - Traiter les valeurs manquantes, si elles existent** ")
    st.write("#### **Données manquantes dans le DataFrame**")
    st.write(df.isna().sum().T)

    st.write("### **4 - Remplacer les valeurs manquantes par le mode pour les colonnes catégorielles et par zéro pour les colonnes numériques** ")

    # Remplacer les valeurs manquantes par le mode pour les colonnes catégorielles et par zéro pour les colonnes numériques
    fill_values = {col: df[col].mode()[0] if df[col].dtype == 'object' else 0 for col in df.columns}
    df.fillna(value=fill_values, inplace=True)

    st.write("### **5 - Afficher le DataFrame après le traitement des valeurs manquantes**")
    st.write(df.head())

    st.write("### **6 - Encoder les variables catégorielles** ")

    # Fonction pour encoder les variables catégorielles
    def preprocess_data(df):
        colonnes_categorielles = df.select_dtypes(include=['object']).columns
        label_encoder = LabelEncoder()
        for colonne in colonnes_categorielles:
            df[colonne] = label_encoder.fit_transform(df[colonne])
        return df

     # Encodage des variables catégorielles
    df_encoded = preprocess_data(df)

    # Affichage du DataFrame encodé
    #st.title("DataFrame après encodage")
    st.dataframe(df_encoded)

    # Affichage des types de données
    st.write("### **7 - Types de données dans le DataFrame**")
    st.write(df.dtypes.transpose())

    st.write("### **8 - Information**")

def enregistrer_donnees_pretraitees(df):
    df.to_pickle("Incidence_Maladie.pkl")
    # Charger une image depuis votre système de fichiers
    image_path = "paludisme.jpg"

    # Afficher l'image en arrière-plan
    # Redimensionner et afficher l'image en arrière-plan
    st.image(image_path, width=150, caption=' ', use_column_width=False)

    # Ajouter du texte au-dessus de l'image
    st.write(
    "<div style='color: #FFFFFF; background-color: #042B29; padding: 10px;'>"
    "<b>Les données prétraitées sont sauvegardées dans <span style='color: red;'>Incidence_Maladie</span> au format pickle pour une utilisation ultérieure.</b>"
    "</div>",
    unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

st.write(' ')
st.write(' ')

# Lien vers les autres pages ou sections
st.subheader("Veuillez cliquer sur ces hyperliens pour être dirigé vers d'autres pages.")

st.write("""
- [Acceuil](http://localhost:8503/)
- [Informations](http://localhost:8503/Informations)
- [Exploration des données](http://localhost:8503/Exploration_des_donn%C3%A9es)
- [Visualisation des données](http://localhost:8503/Visualisation_des_donn%C3%A9es)
- [Plus de visualisation](http://localhost:8503/Plus_de_visualisation)
- [Modélisation des données](http://localhost:8503/Mod%C3%A9lisation_des_donn%C3%A9es)
- [Modèle de prédiction](http://localhost:8503/Mod%C3%A8le_de_pr%C3%A9diction)
- [Origine des données](http://localhost:8501/Origine_des_donn%C3%A9es)

""")
