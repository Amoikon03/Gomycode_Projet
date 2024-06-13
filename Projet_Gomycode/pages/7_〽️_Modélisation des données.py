import streamlit as st
import pickle as pk
import numpy as np
import warnings
import kerastuner as kt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def afficher_data_modelisation():
    # Ignorer spécifiquement les avertissements de dépréciation
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Titre de la page
    st.title("Modélisation Des Données")

    # Chargement des données
    df = pk.load(open("Projet_Gomycode/Incidence_Maladie.pkl", "rb"))

    # Créer un bouton pour afficher un message d'information
    show_info = st.checkbox("**Cliquez ici pour cacher l'information**", value=True)

    if show_info:
        # Ajout du CSS personnalisé
        st.markdown(
            """
            <style>
            .custom-info {
                color: #1A2B3C; /* Couleur du texte */
                background-color: #00FFFF; /* Couleur du fond */
                font-size: 20px; /* Taille de la police */
                padding: 10px;
                border-radius: 5px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Affichage du message avec la classe CSS personnalisée
        st.markdown(
            '<div class="custom-info">5 Algorithmes de Machine Learning sont disponibles sur cette page</div>',
            unsafe_allow_html=True)

        # Ajout du CSS personnalisé
        st.markdown(
            """
            <style>
            .custom-success {
                color: #FBFBFB; /* Couleur du texte */
                background-color: #00353F; /* Couleur du fond */
                font-size: 18px; /* Taille de la police */
                padding: 10px;
                border-radius: 5px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Affichage du message avec la classe CSS personnalisée
        st.markdown(
            '''
            <div class="custom-success">
                <strong>Les résultats du modèle que vous avez sélectionné seront affichés en bas de ces informations, vous permettant ainsi d'effectuer des prédictions.</strong>
            </div>
            ''',
            unsafe_allow_html=True
        )

        st.write(' ')
        st.write(' ')

        st.markdown("""
            <div style="text-align: justify;">
            <strong>1 - Régression linéaire simple :</strong> Ce modèle pourrait être utilisé pour modéliser la relation linéaire entre une variable explicative (par exemple, le nombre de cas de maladies dans une région) et une variable cible (par exemple, l'année). Cela permettrait de visualiser la tendance générale de l'incidence des maladies au fil des années et de prédire cette incidence pour les années futures.
            <br><br><strong>2 - Régression polynomiale :</strong> Si la relation entre les variables explicatives et la variable cible n'est pas strictement linéaire, une régression polynomiale pourrait être utilisée pour capturer des tendances non linéaires dans les données. Cela pourrait être utile si, par exemple, l'incidence des maladies augmente de manière non linéaire au fil des années.
            <br><br><strong>3 - KNN (k-nearest neighbors) :</strong> Ce modèle pourrait être utilisé pour prédire l'incidence future des maladies en se basant sur les données historiques des régions, districts ou villes similaires. KNN fonctionne en trouvant les k points les plus proches dans l'espace des caractéristiques et en prédisant la classe (ou la valeur dans le cas d'une régression) en fonction de ces voisins.
            <br><br><strong>4 et 5 - Random Forest (RF) et Réseaux de Neurones Artificiels (ANN) :</strong> Prédiction de l'incidence future des maladies, tant les random forests que les réseaux de neurones peuvent être utilisés pour prédire l'incidence future des maladies en fonction des données historiques.
            </div>
            """, unsafe_allow_html=True)

    # Les modèles disponibles
    model = st.sidebar.selectbox("Choisissez un modèle",
                ["Regression Linéaire Simple", "Regression Polynomiale", "RandomForest", "K-Nearest Neighbors (KNN)","Réseau de Neurones Artificiels (ANN)"])


    # ✂️ Selection et découpage des données
    seed = 123
    def select_split(dataframe):
        x = dataframe[['ANNEE', 'REGIONS / DISTRICTS', 'VILLES / COMMUNES', 'MALADIE']]  # Caractéristiques
        y = dataframe['INCIDENCE SUR LA POPULATION GENERALE (%)']  # Variable cible

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
        return x_train, x_test, y_train, y_test

    # Création des variables d'entrainement et test
    x_train, x_test, y_train, y_test = select_split(dataframe=df)

    # Réglage du paramètres de chaque modèle

    # 1️⃣ Regression Linéaire simple
    if model == "Regression Linéaire Simple":
        if st.sidebar.button("Prédire", key="regression"):
            st.subheader("Résultat de la Regression Linéaire Simple")

            # Initialiser le modèle
            reg_model = LinearRegression()

            # Entrainer le modèle
            reg_model.fit(x_train, y_train)

            # Prédiction du modèle
            y_pred = reg_model.predict(x_test)

            # Calcul des métriques d'évaluation
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.write("L'erreur quadratique moyenne (MSE) du modèle est :", mse)
            st.write("Le R carré du modèle est :", r2)
            st.write("L'erreur absolue moyenne (MAE) sur l'ensemble de test:", mae)

            # Affichage des résultats
            if r2 < 0.8:
                st.info(
                    f"**Le R² du modèle est de {r2:.3f}, ce qui est inférieur à 0.8. Il est généralement recommandé d'améliorer le modèle avant de l'utiliser pour faire des prédictions.**")
                st.write(
                    "[Prêt à utiliser ce modèle pour essayer de prédire des résultats](http://localhost:8501/Mod%C3%A8le_de_pr%C3%A9diction)")
            elif r2 >= 0.8 and r2 < 1:
                st.write(
                    "[Prêt à tenter le coup des prédictions avec ce modèle](http://localhost:8501/Mod%C3%A8le_de_pr%C3%A9diction)")
                st.info(f"Le R² du modèle est de {r2:.3f}, ce qui indique une bonne qualité des prédictions.")

            else:
                st.success(
                    "Excellent ! Un R carré supérieur ou égal à 0,8 et proche de 1 indique une excellente performance du modèle en matière de prédiction.")
                st.info("Ce modèle permet de prédire la note de data_exploration.")
                st.write(
                    "[Prêt à utiliser ce modèle pour essayer de prédire des résultats](http://localhost:8501/Mod%C3%A8le_de_pr%C3%A9diction)")

    # 2️⃣ Regression polynomiale
    elif model == "Regression Polynomiale":
        if st.sidebar.button("Prédire", key="regression"):
            st.subheader("Résultat de la Regression Polynomiale")

            # Création d'un pipeline avec mise à l'échelle des caractéristiques, création de caractéristiques polynomiales et régularisation Ridge
            pipeline = Pipeline([
                ('poly', PolynomialFeatures(degree=2)),
                ('scaler', StandardScaler()),
                ('ridge', Ridge(alpha=0.1))
            ])

            # Entraînement du modèle avec validation croisée
            scores = cross_val_score(pipeline, x_train, y_train, cv=5, scoring='neg_mean_squared_error')
            mse_cv = -scores.mean()

            # Entraînement du modèle final sur l'ensemble d'entraînement
            pipeline.fit(x_train, y_train)

            # Prédiction du modèle
            y_pred = pipeline.predict(x_test)

            # Calcul des métriques d'évaluation
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            st.write("L'erreur quadratique moyenne (MSE) du modèle est :", mse)
            st.write("Le R carré du modèle est :", r2)
            st.write("L'erreur absolue moyenne (MAE) sur l'ensemble de test:", mae)
            st.write("Erreur quadratique moyenne (MSE) en validation croisée:", mse_cv)

            # Affichage des résultats
            if r2 < 0.8:
                st.info(
                    f"**Le R² du modèle est de {r2:.3f}, ce qui est inférieur à 0.8. Il est généralement recommandé d'améliorer le modèle avant de l'utiliser pour faire des prédictions.**")
                st.write(
                    "[Prêt à utiliser ce modèle pour essayer de prédire des résultats](http://localhost:8501/Mod%C3%A8le_de_pr%C3%A9diction)")
            elif r2 >= 0.8 and r2 < 1:
                st.write(
                    "[Prêt à utiliser ce modèle pour essayer de prédire des résultats](http://localhost:8501/Mod%C3%A8le_de_pr%C3%A9diction)")
                st.info(f"Le R² du modèle est de {r2:.3f}, ce qui indique une bonne qualité des prédictions.")

            else:
                st.success(
                    "Excellent ! Un R carré supérieur ou égal à 0,8 et proche de 1 indique une excellente performance du modèle en matière de prédiction.")
                #st.info("Ce modèle permet de prédire...")
                st.write(
                    "[Prêt à utiliser ce modèle pour essayer de prédire des résultats](http://localhost:8501/Mod%C3%A8le_de_pr%C3%A9diction)")

    # 3️⃣ KNN
    elif model == "K-Nearest Neighbors (KNN)":
        if st.sidebar.button("Prédire", key="K-Nearest Neighbors (KNN)"):
            st.subheader("Résultat de K-Nearest Neighbors (KNN)")

            # Entraînez votre modèle de KNN sur l'ensemble d'entraînement
            knn = KNeighborsRegressor()
            knn.fit(x_train, y_train)

            # Prédisez les étiquettes de classe sur l'ensemble de test
            y_pred = knn.predict(x_test)

            # Calcul des métriques d'évaluation
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            st.write("L'erreur quadratique moyenne (MSE) du modèle est :", mse)
            st.write("Le R carré du modèle est :", r2)
            st.write("L'erreur absolue moyenne (MAE) sur l'ensemble de test:", mae)

            # Affichage des résultats
            if r2 < 0.8:
                st.info(
                    f"**Le R² du modèle est de {r2:.3f}, ce qui est inférieur à 0.8. Il est généralement recommandé d'améliorer le modèle avant de l'utiliser pour faire des prédictions.**")
                st.write(
                    "[Prêt à utiliser ce modèle pour essayer de prédire des résultats](http://localhost:8501/Mod%C3%A8le_de_pr%C3%A9diction)")
            elif r2 >= 0.8 and r2 < 1:
                st.write(
                    "[Prêt à utiliser ce modèle pour essayer de prédire des résultats](http://localhost:8501/Mod%C3%A8le_de_pr%C3%A9diction)")
                st.info(f"Le R² du modèle est de {r2:.3f}, ce qui indique une bonne qualité des prédictions.")

            else:
                st.success(
                    "Excellent ! Un R carré supérieur ou égal à 0,8 et proche de 1 indique une excellente performance du modèle en matière de prédiction.")
                #st.info("Ce modèle permet de prédire...")
                st.write(
                    "[Prêt à utiliser ce modèle pour essayer de prédire des résultats](http://localhost:8501/Mod%C3%A8le_de_pr%C3%A9diction)")

    # 4️ RandomForest
    elif model == "RandomForest":
        if st.sidebar.button("Prédire", key="RandomForest"):
            st.subheader("Résultat de RandomForest")

            # Création d'un pipeline avec mise à l'échelle des caractéristiques
            pipeline = Pipeline([
                ('scaler', StandardScaler()),  # Mise à l'échelle des caractéristiques
                ('random_forest', RandomForestRegressor(random_state=42))
                # Modèle de forêt aléatoire pour la régression
            ])

            # Entraînement du modèle
            pipeline.fit(x_train, y_train)

            # Prédiction sur l'ensemble de test
            y_pred = pipeline.predict(x_test)

            # Calcul des métriques de performance
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Affichage des métriques de performance
            st.write("Mean Squared Error (MSE):", mse)
            st.write("R² Score:", r2)

            # Validation croisée pour évaluer la performance du modèle
            cv_scores = cross_val_score(pipeline, x_train, y_train, cv=5, scoring='neg_mean_squared_error')
            st.write("Cross-validation Mean Squared Error mean:", -np.mean(cv_scores))

            # Affichage des résultats basés sur le R²
            if r2 < 0.8:
                st.info(
                    f"**Le R² du modèle est de {r2:.3f}, ce qui est inférieur à 0.8. Il est généralement recommandé d'améliorer le modèle avant de l'utiliser pour faire des prédictions.**")
                st.write(
                    "[Prêt à utiliser ce modèle pour essayer de prédire des résultats](http://localhost:8501/Mod%C3%A8le_de_pr%C3%A9diction)")

            elif r2 >= 0.8 and r2 < 1:
                st.info(f"Le R² du modèle est de {r2:.3f}, ce qui indique une bonne qualité des prédictions.")
                st.write(
                    "[Prêt à utiliser ce modèle pour essayer de prédire des résultats](http://localhost:8501/Mod%C3%A8le_de_pr%C3%A9diction)")

            else:
                st.success(
                    f"Excellent ! Un R carré supérieur ou égal à 0,8 et proche de 1 indique une excellente performance du modèle en matière de prédiction.")
                #st.info("Ce modèle permet de prédire...")
                st.write(
                    "[Prêt à utiliser ce modèle pour essayer de prédire des résultats](http://localhost:8501/Mod%C3%A8le_de_pr%C3%A9diction)")

    # 5️⃣ ANN
    elif model == "Réseau de Neurones Artificiels (ANN)":
        if st.sidebar.button("Prédire", key="Réseau de Neurones Artificiels (ANN)"):
            st.subheader("Résultat du Réseau de Neurones Artificiels (ANN)")

            # Fonction pour construire le modèle avec KerasTuner
            def build_model(hp):
                model = Sequential()

                # Première couche Dense
                model.add(Dense(
                    units=hp.Int('units1', min_value=32, max_value=512, step=32),
                    activation='relu',
                    input_shape=(X_train_scaled.shape[1],)
                ))

                # Deuxième couche Dense
                model.add(Dense(
                    units=hp.Int('units2', min_value=32, max_value=512, step=32),
                    activation='relu'
                ))

                # Couche de sortie
                model.add(Dense(1, activation='linear'))

                # Compilation du modèle
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(
                        hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
                    ),
                    loss='mean_squared_error',
                    metrics=['mean_absolute_error']
                )

                return model


            # Séparation des caractéristiques et de la variable cible
            X = df[['ANNEE', 'REGIONS / DISTRICTS', 'VILLES / COMMUNES', 'MALADIE']]
            y = df['INCIDENCE SUR LA POPULATION GENERALE (%)']

            # Transformation des caractéristiques catégorielles
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), ['ANNEE']),
                    ('cat', OneHotEncoder(sparse_output=False), ['REGIONS / DISTRICTS', 'VILLES / COMMUNES', 'MALADIE'])
                ])

            # Division des données en ensembles d'entraînement et de test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Application de la transformation sur les ensembles d'entraînement et de test
            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)

            # Configuration du tuner
            tuner = kt.RandomSearch(
                build_model,
                objective='val_mean_absolute_error',
                max_trials=10,
                executions_per_trial=3,
                directory='my_dir',
                project_name='helloworld'
            )

            # Division des données en ensembles d'entraînement et de validation
            X_train_split, X_val, y_train_split, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2,
                                                                          random_state=42)

            # Lancer la recherche d'hyperparamètres
            tuner.search(X_train_split, y_train_split, epochs=10, validation_data=(X_val, y_val))

            # Obtenir les meilleurs hyperparamètres
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

            # Afficher les meilleurs hyperparamètres
            st.write("**Meilleurs hyperparamètres trouvés :**")
            st.write(f"Nombre de neurones dans la première couche : {best_hps.get('units1')}")
            st.write(f"Nombre de neurones dans la deuxième couche : {best_hps.get('units2')}")
            st.write(f"Taux d'apprentissage : {best_hps.get('learning_rate')}")

            # Construire le meilleur modèle
            model = tuner.hypermodel.build(best_hps)

            # Entraîner le modèle avec les meilleurs hyperparamètres
            history = model.fit(X_train_scaled, y_train, epochs=50, validation_data=(X_val, y_val))

            # Afficher les performances du modèle sur l'ensemble de test
            st.write("**Performances du modèle sur l'ensemble de test :**")
            loss, mae = model.evaluate(X_test_scaled, y_test)
            st.write(f'Loss: {loss}')
            st.write(f'Mean Absolute Error : {mae}')

            # Prédiction sur les données de test
            y_pred = model.predict(X_test_scaled)

            # Aplatir les prédictions
            y_pred = y_pred.flatten()

            # Calcul du R-squared
            r_squared = r2_score(y_test, y_pred)
            st.write(f'R-squared : {r_squared}')

            # Affichage des résultats basés sur le R²
            if r_squared < 0.8:
                st.info(
                    f"**Le R² du modèle est de {r_squared:.3f}, ce qui est inférieur à 0.8. Il est généralement recommandé d'améliorer le modèle avant de l'utiliser pour faire des prédictions.**")
                st.write(
                    "[Prêt à utiliser ce modèle pour essayer de prédire des résultats](http://localhost:8501/Mod%C3%A8le_de_pr%C3%A9diction)")

            elif r_squared >= 0.8 and r_squared < 1:
                st.info(f"Le R² du modèle est de {r_squared:.3f}, ce qui indique une bonne qualité des prédictions.")
                st.write(
                    "[Prêt à utiliser ce modèle pour essayer de prédire des résultats](http://localhost:8501/Mod%C3%A8le_de_pr%C3%A9diction)")

            else:
                st.success(
                    f"Excellent ! Un R carré supérieur ou égal à 0,8 et proche de 1 indique une excellente performance du modèle en matière de prédiction.")
                #st.info("Ce modèle permet de prédire...")
                st.write(
                    "[Prêt à utiliser ce modèle pour essayer de prédire des résultats](http://localhost:8501/Mod%C3%A8le_de_pr%C3%A9diction)")


afficher_data_modelisation()

st.write(' ')

# Lien vers les autres pages ou sections
st.subheader("Veuillez cliquer sur ces hyperliens pour être dirigé vers d'autres pages.")

st.write("""
- [Acceuil](http://localhost:8503/)
- [Informations](http://localhost:8503/Informations)
- [Exploration des données](http://localhost:8503/Exploration_des_donn%C3%A9es)
- [Manipulation des données](http://localhost:8503/Manipulation_des_donn%C3%A9es)
- [Visualisation des données](http://localhost:8503/Visualisation_des_donn%C3%A9es)
- [Plus de visualisation](http://localhost:8503/Plus_de_visualisation)
- [Modèle de prédiction](http://localhost:8503/Mod%C3%A8le_de_pr%C3%A9diction)
- [Origine des données](http://localhost:8501/Origine_des_donn%C3%A9es)

""")
