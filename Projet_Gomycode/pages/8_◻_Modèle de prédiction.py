import streamlit as st
import pickle
import numpy as np
import pandas as pd


from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor

# Charger le DataFrame à partir d'un fichier pickle
def load_dataframe(filepath):
    with open(filepath, 'rb') as file:
        dataframe = pickle.load(file)
    return dataframe

# Fonction pour entraîner les modèles
def train_models(data):
    # Sélectionner les caractéristiques et la cible
    X = data[['ANNEE', 'REGIONS / DISTRICTS', 'VILLES / COMMUNES', 'MALADIE']]  # Caractéristiques
    y = data['INCIDENCE SUR LA POPULATION GENERALE (%)']  # Variable cible

    # Encodage des variables catégorielles
    le = LabelEncoder()
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = le.fit_transform(X[col])

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraîner le modèle Random Forest Regressor
    model_rf = RandomForestRegressor()
    model_rf.fit(X_train, y_train)

    # Entraîner le modèle de régression linéaire
    model_lin_reg = LinearRegression()
    model_lin_reg.fit(X_train, y_train)

    # Entraîner le modèle de réseau de neurones artificiels (ANN)
    model_ann = MLPRegressor(max_iter=1000)
    model_ann.fit(X_train, y_train)

    # Entraîner le modèle KNN Regressor
    model_knn = KNeighborsRegressor()
    model_knn.fit(X_train, y_train)

    return model_rf, model_lin_reg, model_ann, model_knn, le, X_train.columns

# Fonction pour préparer les caractéristiques
def prepare_features(features, train_columns, le):
    # Séparer les colonnes numériques et catégorielles
    numeric_cols = features.select_dtypes(include=['int', 'float']).columns
    categorical_cols = features.select_dtypes(include=['object']).columns

    # Imputer les données numériques avec la moyenne
    if not numeric_cols.empty:
        imputer_numeric = SimpleImputer(strategy='mean')
        filled_numeric = imputer_numeric.fit_transform(features[numeric_cols])
        features[numeric_cols] = filled_numeric

    # Imputer les données catégorielles avec la valeur la plus fréquente
    if not categorical_cols.empty:
        imputer_categorical = SimpleImputer(strategy='most_frequent')
        filled_categorical = imputer_categorical.fit_transform(features[categorical_cols])
        features[categorical_cols] = filled_categorical

        # Encoder les variables catégorielles
        for col in categorical_cols:
            features[col] = le.fit_transform(features[col].astype(str))

    # Ajouter les colonnes manquantes avec des valeurs par défaut
    for col in train_columns:
        if col not in features.columns:
            features[col] = 0  # Ou une autre valeur par défaut appropriée

    # Assurez-vous que les colonnes sont dans le même ordre que lors de l'entraînement
    features = features[train_columns]
    return features

# Fonction pour prédire avec les modèles
def predict_with_model(model, features):
    prediction = model.predict(features)
    return prediction

# Définir le titre de l'application avec des options de personnalisation
titre = "Prédiction de l'Incidence Des Maladies En Côte d'Ivoire (2012-2015)"

# Définir les options de personnalisation
couleur_texte = "#0C0F0A"
couleur_fond = "#FFFFFF"

# Définir le style CSS pour le titre
style_titre = f"color: {couleur_texte}; background-color: {couleur_fond}; padding: 10px;"

# Afficher le titre personnalisé
st.markdown(f'<h1 style="{style_titre}">{titre}</h1>', unsafe_allow_html=True)

st.write(" ")
st.write(" ")

# Charger le DataFrame à partir du fichier pickle
data = load_dataframe("Projet_Gomycode/Incidence_Maladie.pkl")

# Créer une colonne latérale pour les paramètres
with st.sidebar:
    st.header("Paramètres")

    year = st.selectbox("Année", [2012, 2013, 2014, 2015])

    age_group = st.selectbox("Groupe d'âge", ["0-5", "6-12", "13-18", "19-35", "36-60", "60+"])

    gender = st.selectbox("Sexe", ["Homme", "Femme", "Enfant"])

    region = st.selectbox("Région", ['ABIDJAN 2', 'ABIDJAN 1-GRANDS PONTS', 'AGNEBY-TIASSA-ME', 'BELIER',
                                     'BOUNKANI-GONTOUGO', 'CAVALLY-GUEMON', 'GBEKE', 'GBOKLE-NAWA-SAN-PEDRO',
                                     'GÔH', 'HAMBOL', 'HAUT SASSANDRA', 'INDENIE DUABLIN',
                                     'PORO-TCHOLOGO-BAGOUE', 'SUD-COMOE', 'TONKPI', 'WORODOUGOU-BERE'])

    villes_communes = st.selectbox("Villes/Communes", ['ABOBO EST', 'ABOBO OUEST', 'ANYAMA', 'COCODY-BINGERVILLE',
                                                       'KOUMASSI-PORT-BOUET-VRIDI', 'MARCORY-TREICHVILLE',
                                                       'ADJAME-PLATEAU-ATTECOUBE', 'DABOU', 'GRAND LAHOU', 'JACQUEVILLE',
                                                       'YOPOUGON EST', 'YOPOUGON OUEST', 'ADZOPE', 'AGBOVILLE', 'AKOUPE', 'ALEPE',
                                                       'SIKENSI', 'TIASSALE', 'DIDIEVI', 'TIEBISSOU', 'TOUMODI', 'YAMOUSSOUKRO',
                                                       'BONDOUKOU', 'BOUNA', 'NASSIAN', 'TANDA', 'BANGOLO', 'BLOLEQUIN', 'DUEKOUE',
                                                       'GUIGLO', 'KOUIBLY', 'TOULEPLEU', 'BEOUMI', 'BOUAKÉ NORD- EST',
                                                       'BOUAKÉ NORD- OUEST', 'BOUAKÉ SUD', 'SAKASSOU', 'GUEYO', 'SAN-PEDRO',
                                                       'SASSANDRA', 'SOUBRE', 'TABOU', 'GAGNOA', 'OUME', 'DABAKALA', 'KATIOLA',
                                                       'NIAKARAMADOUGOU', 'DALOA', 'ISSIA', 'VAVOUA', 'ABENGOUROU', 'AGNIBILEKROU',
                                                       'BETTIE', 'MINIGNAN', 'ODIENNE', 'TOUBA', 'DIVO', 'FRESCO', 'LAKOTA', 'BOUAFLE',
                                                       'SINFRA', 'ZUENOULA', 'BOCANDA', 'BONGOUANOU', 'DAOUKRO', 'DIMBOKRO',
                                                       "M'BAHIAKRO", 'PRIKRO', 'BOUNDIALI', 'FERKE', 'KORHOGO', 'OUANGOLODOUGOU',
                                                       'TENGRELA', 'ABOISSO', 'ADIAKE', 'GRAND-BASSAM', 'BIANKOUMA', 'DANANE', 'MAN',
                                                       'ZOUAN HOUNIEN', 'MANKONO', 'SEGUELA', 'HAMBOL'])

    disease = st.selectbox("Maladie", ['PALUDISME', 'BILHARZIOZE URINAIRE', 'CONJONCTIVITE', 'DIARRHEE',
                                       'MALNUTRITION (0 - 4 ANS)'])

    # Créer le bouton de prédiction
    bouton_predire = st.button("Prédire")

# Création d'un dictionnaire avec les paramètres
params = {
    'year': year,
    'age_group': age_group,
    'gender': gender,
    'region': region,
    'villes_communes': villes_communes,
    'disease': disease,
}

# Convertir le dictionnaire en DataFrame
features = pd.DataFrame([params])

# Charger les modèles, l'encodeur et les colonnes d'entraînement
model_rf, model_lin_reg, model_ann, model_knn, le, train_columns = train_models(data)

# Préparer les caractéristiques
features = prepare_features(features, train_columns, le)

# Prédire avec Random Forest
prediction_rf = predict_with_model(model_rf, features)

# Prédire avec la régression linéaire
prediction_lin_reg = predict_with_model(model_lin_reg, features)

# Prédire avec le réseau de neurones artificiels (ANN)
prediction_ann = predict_with_model(model_ann, features)

# Prédire avec KNN
prediction_knn = predict_with_model(model_knn, features)

# Affichage des résultats des prédictions
prediction_rf_value = prediction_rf[0]
prediction_lin_reg_value = prediction_lin_reg[0]
prediction_ann_value = prediction_ann[0]
prediction_knn_value = prediction_knn[0]

# Formater les prédictions avec deux décimales
format_prediction = lambda pred: f"{pred:.2f}"

# Affichage des paramètres d'entrée
st.subheader("Paramètres sélectionnés")
st.write(f"Année : {year}")
st.write(f"Groupe d'âge : {age_group}")
st.write(f"Sexe : {gender}")
st.write(f"Région : {region}")
st.write(f"Villes/Communes : {villes_communes}")
st.write(f"Maladie : {disease}")

# Explication des modèles
st.subheader("Modèles utilisés")
st.write("""
- **Random Forest** : Une collection d'arbres de décision où chaque arbre contribue à la prédiction finale par un vote majoritaire.
- **Régression Linéaire** : Un modèle qui détermine la ligne droite la mieux adaptée aux données pour faire des prédictions.
- **Régression Polynomial** : Un modèle qui généralise la régression linéaire en utilisant des courbes au lieu de lignes droites pour s'adapter aux données.
- **Réseau de Neurones Artificiels (ANN)** : Un modèle sophistiqué qui s'inspire du fonctionnement du cerveau humain pour traiter et analyser les données.
- **KNN (K-Nearest Neighbors)** : Un modèle qui effectue des prédictions en utilisant les points de données les plus proches dans l'espace de caractéristiques.
""")

# Affichage des prédictions pour chaque modèle
st.subheader("Prédictions")
st.write(f"Incidence prévue (Random Forest) : {format_prediction(prediction_rf_value)}%")
st.write(f"Incidence prévue (Régression Linéaire) : {format_prediction(prediction_lin_reg_value)}%")
st.write(f"Prédiction Régression Polynomial : {prediction_poly_reg_value}%")
st.write(f"Incidence prévue (Réseau de Neurones Artificiels) : {format_prediction(prediction_ann_value)}%")
st.write(f"Incidence prévue (KNN) : {format_prediction(prediction_knn_value)}%")

# Interprétation des résultats
st.subheader("Interprétation des résultats")

st.markdown(
    """
    <div style='text-align: justify;'>
    Ces prédictions représentent le pourcentage de la population générale qui pourrait être affecté par la maladie
    sélectionnée dans la région et la ville choisies pour l'année spécifiée. Les différents modèles peuvent fournir des
    prédictions légèrement différentes en raison de la manière dont ils traitent et analysent les données.
    </div>
    """, unsafe_allow_html=True)


# Lien vers les autres pages ou sections
st.subheader("Veuillez cliquer sur ces hyperliens pour être dirigé vers d'autres pages.")

st.write("""
- [Acceuil](http://localhost:8503/)
- [Informations](http://localhost:8503/Informations)
- [Exploration des données](http://localhost:8503/Exploration_des_donn%C3%A9es)
- [Manipulation des données](http://localhost:8503/Manipulation_des_donn%C3%A9es)
- [Visualisation des données](http://localhost:8503/Visualisation_des_donn%C3%A9es)
- [Plus de visualisation](http://localhost:8503/Plus_de_visualisation)
- [Modélisation des données](http://localhost:8503/Mod%C3%A9lisation_des_donn%C3%A9es)
- [Origine des données](http://localhost:8501/Origine_des_donn%C3%A9es)

""")
