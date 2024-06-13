# Import des librairies
import streamlit as st
import pandas as pd


# Personnalisation de l'affichage
st.title("Exploration Des Données")

# Charger le fichier CSV
@st.cache_data
def load_data():
    return pd.read_csv("Incidence.csv")

df = load_data()

# Afficher la forme du DataFrame (nombre de lignes et de colonnes)
st.write("### **Dimensions du DataFrame** ")
st.write(f"Le DataFrame a une quantité de lignes égal à : {df.shape[0]}")
st.write(f"Le DataFrame a une quantité de colonnes égal à : {df.shape[1]}")

# Afficher les premières lignes du DataFrame
st.write("### **Les premières entrées du DataFrame** ")
st.write(df.head())

# Afficher les informations sur les (type de données, valeurs manquantes, etc.)
st.write(df.info())

# Vérification des valeurs manquantes
st.write("### **Nombre de valeurs manquantes par colonne**")
missing_values = df.isnull().sum()
st.write(missing_values[missing_values > 0])

# Vérification des valeurs aberrantes
st.write("### **Les valeurs anormales dans la colonne**")
for column in df.select_dtypes(include=['int64', 'float64']).columns:
    outliers = df[column][((df[column] - df[column].mean()) / df[column].std()).abs() > 3]
    if not outliers.empty:
        st.write(f" '{column}':")
        st.write(outliers)

# Afficher les statistiques descriptives pour les colonnes numériques
st.write("### **Statistiques descriptives pour les variables numériques :**")
st.write(df.describe())

# Lien vers les autres pages ou sections
st.subheader("Veuillez cliquer sur ces hyperliens pour être dirigé vers d'autres pages.")

st.write("""
- [Acceuil](http://localhost:8503/)
- [Informations](http://localhost:8503/Informations)
- [Manipulation des données](http://localhost:8503/Manipulation_des_donn%C3%A9es)
- [Visualisation des données](http://localhost:8503/Visualisation_des_donn%C3%A9es)
- [Plus de visualisation](http://localhost:8503/Plus_de_visualisation)
- [Modélisation des données](http://localhost:8503/Mod%C3%A9lisation_des_donn%C3%A9es)
- [Modèle de prédiction](http://localhost:8503/Mod%C3%A8le_de_pr%C3%A9diction)
- [Origine des données](http://localhost:8501/Origine_des_donn%C3%A9es)

""")
