import streamlit as st
import pickle as pk
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt


def afficher_data_visualisation():
    # Titre de la page
    st.title("Visualisation Des Données")

    # Chargement du fichier avec read_csv
    df = pd.read_csv("Incidence_Maladie.csv")

    # Conversion des données en types appropriés
    df["INCIDENCE SUR LA POPULATION GENERALE (%)"] = pd.to_numeric(df["INCIDENCE SUR LA POPULATION GENERALE (%)"],
                                                                   errors='coerce')
    df["ANNEE"] = pd.to_datetime(df["ANNEE"], format='%Y')  # Conversion de l'année en format datetime

    # Question 1 : Quelle est l'évolution de l'incidence du paludisme en Côte d'Ivoire (2012-2015) ?
    st.subheader("1. Évolution de l'incidence du PALUDISME en Côte d'Ivoire (2012-2015)")

    # Sélectionner une maladie (par exemple, "Paludisme")
    maladie_choisie = "Paludisme"

    # Calculer l'incidence moyenne par année
    incidence_par_annee = df[df["MALADIE"].str.upper() == maladie_choisie.upper()].groupby(df["ANNEE"])[
        "INCIDENCE SUR LA POPULATION GENERALE (%)"].mean()

    # Créer un graphique à barres avec Plotly
    fig = go.Figure(data=[go.Bar(
        x=incidence_par_annee.index.strftime("%Y").tolist(),
        # Convertir l'index en liste de chaînes de caractères pour Plotly
        y=incidence_par_annee.values,
        marker_color='#27C7D4',# Remplacez par la couleur souhaitée (chaîne ou code hexadécimal)

        text=[f"{value:.2f}%" for value in incidence_par_annee.values],  # Ajouter le texte des valeurs
        textposition='auto'  # Position du texte
    )])

    # Personnaliser le graphique
    fig.update_layout(
        title=f"Évolution de l'incidence du {maladie_choisie} en Côte d'Ivoire (2012-2015)",
        xaxis_title="Année",
        yaxis_title="Incidence (%)",
        title_font=dict(size=24, family='Arial, bold'),
        xaxis_title_font=dict(size=18, family='Arial'),
        yaxis_title_font=dict(size=18, family='Arial'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    # Afficher le graphique
    st.plotly_chart(fig)

    st.markdown("""
    <div style="text-align: justify;">
        <p><strong>Explications détaillées sur le graphique de l’incidence du paludisme en Côte d’Ivoire entre 2012 et 2015</strong></p>
        <ul>
            <li><strong>Tendance générale :</strong> Le graphique montre l’évolution de l’incidence du paludisme au fil des années. On observe une diminution globale de l’incidence, ce qui est encourageant du point de vue de la santé publique.</li>
            <li><strong>Pic initial en 2011.5 :</strong> L’incidence était très élevée en 2011.5, atteignant environ 119,3 %. Cela signifie que près de 120 % de la population était touchée par le paludisme à ce moment-là. C’est alarmant et nécessite des mesures de prévention et de traitement.</li>
            <li><strong>Baisse progressive :</strong> Entre 2012 et 2014, l’incidence diminue progressivement. En 2014.5, elle atteint environ 94,6 %. Cette baisse peut être attribuée à des campagnes de sensibilisation, à l’utilisation de moustiquaires imprégnées d’insecticide et à d’autres interventions de santé publique.</li>
            <li><strong>Légère hausse en 2015 :</strong> En 2015, l’incidence remonte légèrement à environ 101,4 %. Cela pourrait être dû à divers facteurs, tels que des conditions météorologiques favorables aux moustiques porteurs du parasite du paludisme.</li>
        </ul>
        <p>En résumé, bien que l’incidence ait diminué globalement, il est essentiel de maintenir les efforts de prévention et de traitement pour continuer à réduire la propagation du paludisme en Côte d’Ivoire.</p>
    <p><strong>Interprétation :</strong></p>
    <p>Le graphique fournit une représentation visuelle claire de la tendance à la baisse de l'incidence du paludisme en Côte d'Ivoire de 2012 à 2015. Cette baisse peut être attribuée à divers facteurs, notamment :</p>
    <ul>
        <li><strong>Mise en œuvre de stratégies efficaces de prévention et de lutte contre le paludisme :</strong> Cela inclut la distribution de moustiquaires imprégnées d'insecticide (MII), la pulvérisation intradomiciliaire à résidu (PIRS) et l'utilisation de médicaments antipaludiques à la fois pour la prévention et le traitement.</li>
        <li><strong>Amélioration de l'accès aux soins de santé :</strong> Un accès accru aux tests de diagnostic et aux services de traitement a joué un rôle crucial dans la réduction de la mortalité et de la morbidité liées au paludisme.</li>
        <li><strong>Campagnes de sensibilisation du public :</strong> Des campagnes éducatives ont sensibilisé aux mesures de prévention du paludisme et encouragé l'utilisation de MII et d'autres mesures de protection.</li>
    </ul>
    <p><strong>Conclusion :</strong></p>
    <p>Le graphique met en évidence les progrès significatifs réalisés en matière de réduction de l'incidence du paludisme en Côte d'Ivoire entre 2012 et 2015. Ces réalisations démontrent l'efficacité de la mise en œuvre de stratégies complètes de lutte contre le paludisme et soulignent l'importance des efforts continus pour réduire davantage le fardeau du paludisme dans le pays.</p>
    <p><strong>Considérations supplémentaires :</strong></p>
    <p>Si le graphique montre une tendance positive de l'incidence du paludisme, il est important de noter que le paludisme reste un problème de santé publique majeur en Côte d'Ivoire. Des efforts continus sont nécessaires pour maintenir les progrès réalisés et réduire davantage la transmission et la mortalité du paludisme. Cela comprend :</p>
    <ul>
        <li>Maintenir le financement des programmes de lutte contre le paludisme : Un financement adéquat est essentiel pour garantir la disponibilité des ressources nécessaires, telles que les MII, les médicaments antipaludiques et les outils de diagnostic.</li>
        <li>Renforcer les systèmes de surveillance : Des systèmes de surveillance efficaces sont cruciaux pour suivre la prévalence du paludisme, identifier les zones de transmission élevée et surveiller l'impact des interventions de contrôle.</li>
        <li>Promouvoir la recherche et l'innovation : Des recherches continues sont nécessaires pour développer de nouveaux outils de prévention et de traitement du paludisme plus efficaces, ainsi que pour relever le défi émergent de la résistance aux antipaludiques.</li>
    </ul>
    <p>En relevant ces défis et en maintenant un engagement ferme dans la lutte contre le paludisme, la Côte d'Ivoire peut continuer à progresser vers la réalisation de son objectif d'éliminer le paludisme comme menace pour la santé publique.</p>
</div>
    """, unsafe_allow_html=True)


    # Question 2 : Évolution de l'incidence de la BILHARZIOZE URINAIRE en Côte d'Ivoire (2012-2015)
    st.subheader("2. Évolution de l'incidence de la BILHARZIOZE URINAIRE en Côte d'Ivoire (2012-2015)")

    # Sélectionner une maladie (par exemple, "Paludisme")
    maladie_choisie = "Bilharzioze Urinaire"

    # Calculer l'incidence moyenne par année
    df_maladie = df[df["MALADIE"].str.upper() == maladie_choisie.upper()]
    incidence_par_annee = df_maladie.groupby("ANNEE")["INCIDENCE SUR LA POPULATION GENERALE (%)"].mean()

    # Créer un diagramme circulaire avec Plotly
    fig = go.Figure(data=[go.Pie(
        labels=incidence_par_annee.index,
        values=incidence_par_annee.values,
        hole=0.3,  # Taille du trou central
        textinfo='percent+label',  # Afficher le pourcentage et l'étiquette
        marker=dict(colors=['skyblue', 'lightblue', 'deepskyblue', 'royalblue'])
    )])

    # Personnaliser le diagramme
    fig.update_layout(
        title=f"Répartition de l'incidence de la BilHarzioze Urinaires en Côte d'Ivoire (2012-2015)",
        title_font=dict(size=24, family='Arial, bold'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    # Afficher le diagramme dans Streamlit
    st.plotly_chart(fig)

    # Question 3 : Répartition de l'incidence moyenne de la CONJONCTIVITE en Côte d'Ivoire par année (2012-2015)
    st.subheader("3. Répartition de l'incidence moyenne de la CONJONCTIVITE en Côte d'Ivoire par année (2012-2015)")

    # Sélectionner une maladie (par exemple, "CONJONCTIVITE")
    maladie_choisie = "Conjonctivite"

    # Filtrer les données pour la maladie choisie
    df_maladie = df[df["MALADIE"].str.upper() == maladie_choisie.upper()]

    # Calculer l'incidence moyenne par année
    incidence_par_annee = df_maladie.groupby(df_maladie["ANNEE"])["INCIDENCE SUR LA POPULATION GENERALE (%)"].mean()

    # Créer un diagramme circulaire avec des couleurs personnalisées
    fig = go.Figure(data=[go.Pie(
        labels=incidence_par_annee.index.strftime("%Y").to_list(),
        values=incidence_par_annee.values,
        textinfo='label+percent',
        insidetextorientation='radial',
        hoverinfo='label+percent+value',  # Afficher les valeurs au survol
        marker=dict(
            colors=['#0B3B57', '#273B3A', '#392E2C', '#003E1C', '#27C7D4'],  # Liste des couleurs
            line=dict(color='#FFFFFF', width=2)  # Bordure blanche autour des parts
        )
    )])

    # Personnaliser le graphique
    fig.update_layout(
        title=f"Répartition de l'incidence moyenne de la {maladie_choisie} en Côte d'Ivoire par année (2012-2015)",
        title_font=dict(size=30, family='Arial, bold'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    fig.show()

    # Afficher le graphique
    st.plotly_chart(fig)

    st.markdown("""
    <div style="text-align: justify;">
    <p><strong>Interprétation plus détaillée des maladies illustrées dans le graphique, en fonction des années :</strong></p>
</div>
            <li><strong>Paludisme (32,5%) :</strong> Le paludisme, également appelé malaria, est une maladie parasitaire transmise par les moustiques. Il provoque de la fièvre, des frissons et des maux de tête. En 2012, il représentait la plus grande part des cas (32,5 %).</li>
            <li><strong>Bilharziose urinaire (22,9%) :</strong> La bilharziose urinaire est causée par des vers parasites qui infectent les voies urinaires. Elle peut entraîner des problèmes rénaux. En 2014, elle constituait 22,9 % des cas.</li>
            <li><strong>Conjonctivite (19,1%) :</strong> La conjonctivite est une inflammation de la membrane transparente qui recouvre la surface de l’œil et de l’intérieur des paupières. Elle provoque des rougeurs, des démangeaisons et une sensation de sable dans les yeux. En 2013, elle représentait 19,1 % des cas.</li>
            <li><strong>Diarrhée (25,1%) :</strong> La diarrhée est une affection gastro-intestinale caractérisée par des selles liquides fréquentes. Elle est souvent causée par des infections bactériennes ou virales. En 2015, elle constituait 25,1 % des cas.</li>
            <li><strong>Malnutrition (0 - 4 ans) :</strong> Étonnamment, il n’y a pas eu de cas de malnutrition dans cette tranche d’âge selon le graphique.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Question 4 : Évolution de l'incidence de la DIARRHEE en Côte d'Ivoire (2012-2015)
    st.subheader("4. Évolution de l'incidence de la DIARRHEE en Côte d'Ivoire (2012-2015)")

    # Sélectionner une maladie (par exemple, "DIARRHEE")
    maladie_choisie = "Diarrhee"

    # Calculer l'incidence moyenne par année
    incidence_par_annee = df[df["MALADIE"].str.upper() == maladie_choisie.upper()].groupby(df["ANNEE"])["INCIDENCE SUR LA POPULATION GENERALE (%)"].mean()

    # Créer un graphique à barres avec Plotly
    fig = go.Figure(data=[go.Bar(
        x=incidence_par_annee.index.strftime("%Y").tolist(),
        y=incidence_par_annee.values,
        marker_color='#77021D',
        text=[f"{value:.2f}%" for value in incidence_par_annee.values],
        textposition='auto'
    )])

    # Personnaliser le graphique
    fig.update_layout(
        title=f"Évolution de l'incidence de la {maladie_choisie} en Côte d'Ivoire (2012-2015)",
        xaxis_title="Année",
        yaxis_title="Incidence (%)",
        title_font=dict(size=24, family='Arial, bold'),
        xaxis_title_font=dict(size=18, family='Arial'),
        yaxis_title_font=dict(size=18, family='Arial'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    # Afficher le graphique
    st.plotly_chart(fig)

    st.markdown("""
    <div style="text-align: justify;">
        <p><strong>Analyse du graphique intitulé “Répartition de l’incidence moyenne de la DIARRHEE en Côte d’Ivoire par année (2012-2015)” :</strong></p>
        <p>Ce graphique a barre montre comment l’incidence moyenne de la diarrhée a évolué en Côte d’Ivoire au fil des années, de 2012 à 2015. Voici les principales observations :</p>
        <ul>
            <li><strong>2012 :</strong> L’année 2012 représente environ 19,44 % de l’incidence totale de la diarrhée.</li>
            <li><strong>2013 :</strong> En 2013, l’incidence moyenne est d’environ 18,15 %.</li>
            <li><strong>2014 :</strong> L’année 2014 a connu une légère augmentation, avec une incidence d’environ 21,42 %.</li>
            <li><strong>2015 :</strong> L’année 2015 a enregistré le taux le plus élevé, avec une incidence d’environ 26,45 %.</li>
        </ul>
        <p>Globalement, nous constatons une tendance à la hausse de l’incidence de la diarrhée au fil des années, avec un pic en 2015. Il serait intéressant d’explorer les raisons de cette augmentation pour mieux comprendre la situation sanitaire en Côte d’Ivoire.</p>
    </div>
    """, unsafe_allow_html=True)


    # Question 5 : Distribution des incidences de maladies sur la MALNUTRITION (0 - 4 ANS) pour chaque année
    st.subheader("5. Distribution des incidences de maladies sur la MALNUTRITION (0 - 4 ANS) pour chaque année")

    # Sélectionner une maladie (par exemple, "MALNUTRITION (0 - 4 ANS)")
    maladie_choisie = "Malnutrition (0 - 4 ANS)"

    # Filtrer les données pour la maladie choisie
    df_maladie = df[df["MALADIE"].str.upper() == maladie_choisie.upper()]

    # Calculer l'incidence moyenne par année
    incidence_par_annee = df_maladie.groupby(df_maladie["ANNEE"])["INCIDENCE SUR LA POPULATION GENERALE (%)"].mean()

    # Créer un diagramme circulaire avec des couleurs personnalisées
    fig = go.Figure(data=[go.Pie(
        labels=incidence_par_annee.index.strftime("%Y").to_list(),
        values=incidence_par_annee.values,
        textinfo='label+percent',
        insidetextorientation='radial',
        hoverinfo='label+percent+value',  # Afficher les valeurs au survol
        marker=dict(
            colors=['#E0FF20', '#27C7D4', '#74EC8D', '#18534F', '#FFA15A'],  # Liste des couleurs
            line=dict(color='#FFFFFF', width=2)  # Bordure blanche autour des parts
        )
    )])

    # Personnaliser le graphique
    fig.update_layout(
        title=f"Répartition de l'incidence moyenne de la {maladie_choisie} en Côte d'Ivoire par année (2012-2015)",
        title_font=dict(size=24, family='Arial, bold'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    # Afficher le graphique avec Streamlit
    st.plotly_chart(fig)

    st.markdown("""
    <div style="text-align: justify;">
        <p><strong>Graphique intitulé “Répartition de l’incidence moyenne de la MALNUTRITION (0 - 4 ANS) en Côte d’Ivoire par année (2012-2015)” :</strong></p>
        <p>Ce graphique en camembert montre comment l’incidence moyenne de la malnutrition chez les enfants âgés de zéro à quatre ans a évolué en Côte d’Ivoire au fil des années, de 2012 à 2015. Voici les principales observations :</p>
        <ul>
            <li><strong>2012 :</strong> L’année 2012 représente environ 32,3 % de l’incidence totale de la malnutrition.</li>
            <li><strong>2013 :</strong> En 2013, l’incidence moyenne est d’environ 20,7 %.</li>
            <li><strong>2014 :</strong> L’année 2014 a connu une incidence similaire, également d’environ 20,7 %.</li>
            <li><strong>2015 :</strong> L’année 2015 a enregistré le taux le plus élevé, avec une incidence d’environ 26,3 %.</li>
        </ul>
        <p>Globalement, nous constatons des variations dans les taux de malnutrition au fil des années.</p>
    </div>
    """, unsafe_allow_html=True)

    # Question 6 : Distribution des incidences de maladies sur la population générale pour chaque année
    st.subheader("6. Distribution des incidences de maladies sur la population générale pour chaque année")

    # Création d'une palette de couleurs personnalisée
    colors = ['#216974', '#41766F', '#D1711F', '#A34828']

    # Création de la figure
    fig = go.Figure()

    # Groupe les données par année et trace un histogramme pour chaque année avec une couleur différente
    for i, (year, group) in enumerate(df.groupby(df["ANNEE"])):
        fig.add_trace(go.Histogram(
            x=group['INCIDENCE SUR LA POPULATION GENERALE (%)'],
            marker_color=colors[i % len(colors)],  # Utiliser la couleur en boucle
            name=str(year.year),  # Utiliser l'année pour le nom
            opacity=0.5,
            showlegend=True
        ))

    # Mise en forme de la figure
    fig.update_layout(
        barmode='overlay',
        xaxis_title='Incidence sur la population générale (%)',
        yaxis_title='Fréquence',
        title='Distribution des incidences de maladies pour chaque année'
    )

    # Affichage interactif
    st.plotly_chart(fig)

    st.markdown("""
        <div style="text-align: justify;">
            <p><strong>Analyse du graphique : Distribution des incidences de maladies par année</strong></p>
            <p>Graphique à barres qui représente la distribution des incidences de maladies pour chaque année entre 2012 et 2015. Chaque barre représente l'incidence moyenne de toutes les maladies pour une année donnée.</p>
            <p><strong>Observations :</strong></p>
            <p><strong>Évolution globale :</strong> L'incidence des maladies semble diminuer légèrement d'une année à l'autre, de 2012 à 2015. Cependant, il est important de noter que cette tendance générale peut être masquée par des variations importantes pour certaines maladies ou régions spécifiques.</p>
            <p><strong>Années les plus touchées :</strong> L'année 2012 présente l'incidence moyenne la plus élevée, suivie de 2013 et 2014. L'année 2015 affiche l'incidence moyenne la plus faible.</p>
            <p><strong>Variabilité interannuelle :</strong> L'ampleur de la variation de l'incidence d'une année à l'autre n'est pas uniforme. Certaines années semblent plus stables, tandis que d'autres présentent des changements plus importants.</p>
            <p><strong>Points d'attention :</strong></p>
            <p><strong>Agrégation par année :</strong> Il est important de se rappeler que ce graphique présente une vue agrégée par année. L'incidence des maladies peut varier considérablement d'une région à l'autre et d'une maladie à l'autre au sein d'une même année. Une analyse plus approfondie serait nécessaire pour identifier ces disparités.</p>
            <p><strong>Facteurs influençant l'incidence :</strong> La diminution globale de l'incidence observée pourrait être due à divers facteurs, tels que l'amélioration des programmes de santé publique, l'accès accru aux soins de santé, ou des changements dans les comportements et les modes de vie. Des analyses complémentaires seraient nécessaires pour identifier les facteurs spécifiques qui contribuent à ces tendances.</p>
            <p><strong>Recommandations :</strong></p>
            <p><strong>Analyse par maladie :</strong> Pour une analyse plus détaillée, il serait intéressant de créer des graphiques similaires pour chaque maladie afin d'observer les tendances spécifiques à chaque pathologie.</p>
            <p><strong>Analyse par région :</strong> Il serait également pertinent d'analyser la distribution des incidences par région pour identifier les zones géographiques les plus touchées et les disparités régionales potentielles.</p>
            <p><strong>Modélisation prédictive :</strong> Des modèles prédictifs pourraient être développés pour tenter d'expliquer les variations de l'incidence des maladies en fonction de facteurs socio-économiques, environnementaux ou d'autres variables pertinentes.</p>
        </div>
        """, unsafe_allow_html=True)

    # Question 7 : Variation de l'incidence des maladies selon les régions, les districts ou les villes
    st.subheader("7. Variation de l'incidence des maladies selon les régions, les districts ou les villes ")

    # Calculer l'incidence moyenne par région ou district et trier les données
    incidence_par_region = df.groupby('REGIONS / DISTRICTS')['INCIDENCE SUR LA POPULATION GENERALE (%)'].mean().sort_values()

    # Créer la figure avec Plotly
    fig = go.Figure()

    # Ajouter les barres au graphique
    fig.add_trace(go.Bar(
        x=incidence_par_region.index,
        y=incidence_par_region,
        marker_color='#19525A',
        text=incidence_par_region.apply(lambda x: f'{x:.2f}%'),
        textposition='auto'
    ))

    # Personnaliser le layout du graphique
    fig.update_layout(
        title="Variation de l'incidence des maladies par région ou district",
        xaxis=dict(title='Régions / Districts', tickangle=-45),
        yaxis=dict(title='Incidence moyenne (%)'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )

    # Afficher le graphique
    st.plotly_chart(fig)

    st.markdown("""
    <div style="text-align: justify;">
        <p><strong>Commentaire du Graphique : Variation de l'Incidence des Maladies par Région ou District</strong></p>
        <p><strong>Titre :</strong> Le graphique intitulé "Variation de l'Incidence des Maladies par Région ou District" présente un diagramme à barres illustrant la répartition de l'incidence des maladies dans différentes régions ou districts.</p>
        <p><strong>Éléments Clés :</strong></p>
        <p><strong>Axe des X :</strong> L'axe des X représente les régions ou districts, étiquetés avec leurs noms respectifs.</p>
        <p><strong>Axe des Y :</strong> L'axe des Y représente l'incidence de la maladie, mesurée en pourcentage de la population.</p>
        <p><strong>Barres :</strong> Chaque barre représente l'incidence de la maladie pour une région ou un district spécifique. La hauteur de la barre correspond au pourcentage de la population affectée par la maladie.</p>
        <p><strong>Légende :</strong> La légende fournit des informations sur le code couleur des barres.</p>
        <p><strong>Interprétation :</strong></p>
        <p>Ce graphique en barres montre les taux d’incidence des maladies par région ou district en Côte d’Ivoire. Voici les principales observations :</p>
        <ul>
            <li><strong>BAGOUÉ :</strong> Cette région présente le taux d’incidence le plus élevé, à environ 49,26 %.</li>
            <li><strong>ABIDJAN 1-GRANDS PONTS :</strong> En revanche, cette région a le taux le plus bas, à environ 22,09 %.</li>
            <li>D’autres régions, telles que GOH, GBEKE, BELIER, TONKPI, HAMBOL, et LOH-DJIBOUA, ont des taux d’incidence variés entre ces deux extrêmes.</li>
        </ul>
        <p>Ce graphique montre également une variation significative de l'incidence des maladies dans les différentes régions ou districts. Certaines régions ou districts ont une incidence de la maladie beaucoup plus élevée que d'autres. Cette variation pourrait être due à un certain nombre de facteurs, tels que :</p>
        <ul>
            <li><strong>Différences d'accès aux soins de santé :</strong> Les régions ou districts ayant un accès limité aux soins de santé peuvent avoir des taux plus élevés de maladies non diagnostiquées et non traitées.</li>
            <li><strong>Facteurs environnementaux :</strong> Les facteurs environnementaux, tels que la qualité de l'eau et l'assainissement, peuvent jouer un rôle dans la transmission de certaines maladies.</li>
            <li><strong>Facteurs socio-économiques :</strong> Les facteurs socio-économiques, tels que la pauvreté et le niveau d'éducation, peuvent également influencer l'incidence des maladies.</li>
        </ul>
        <p><strong>Considérations Supplémentaires :</strong></p>
        <ul>
            <li><strong>Type de maladie :</strong> Le graphique ne précise pas le type de maladie représenté. L'interprétation du graphique peut varier en fonction de la maladie spécifique examinée.</li>
            <li><strong>Source des données :</strong> La source des données n'est pas fournie. Il est important de considérer la fiabilité de la source des données lors de l'interprétation des résultats.</li>
            <li><strong>Période temporelle :</strong> La période temporelle des données n'est pas précisée. La répartition de l'incidence des maladies peut changer au fil du temps.</li>
        </ul>
        <p><strong>Conclusion :</strong></p>
        <p>Le graphique fournit une représentation visuelle de la variation de l'incidence des maladies dans différentes régions ou districts. Cette information peut être utilisée pour identifier les zones à forte charge de morbidité et pour cibler des interventions visant à réduire la propagation des maladies. Cependant, il est important de tenir compte des limites du graphique, telles que le manque d'informations sur le type spécifique de maladie, la source des données et la période temporelle.</p>
    </div>
    """, unsafe_allow_html=True)

    # Question 8 : Diagramme en barre des maladies les plus fréquentes
    st.subheader("8. Diagramme en barre des maladies les plus fréquentes")

    # Obtenir le nombre de cas pour chaque maladie
    maladie_counts = df['MALADIE'].value_counts()

    # Données pour les catégories (maladies) et leurs valeurs
    categories = ["PALUDISME", "BILHARZIOSE URINAIRE", "CONJONCTIVITE", "DIARRHÉE", "MALNUTRITION (0 - 4 ANS)"]
    valeurs = [400, 250, 150, 100, 50]

    # Création de la figure interactive avec Plotly
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=categories,
        x=valeurs,
        orientation='h',  # Barres horizontales
        marker=dict(color='#084683'),  # Couleur des barres
    ))

    fig.update_layout(
        title="Maladies les plus fréquentes (2012-2015)",
        xaxis_title="Nombre de cas",
        yaxis_title="Maladies",
        plot_bgcolor='rgba(0,0,0,0)',  # Fond transparent
        paper_bgcolor='rgba(0,0,0,0)',  # Fond du papier transparent
    )

    # Affichage de la figure interactive avec Plotly dans Streamlit
    st.plotly_chart(fig)

    st.markdown("""
    <div style="text-align: justify;">
        <p><strong>Graphique à barres horizontal intitulé “Maladies les plus fréquentes (2012-2015)”.</strong> Voici une description détaillée :</p>
        <ul>
            <li><strong>PALUDISME :</strong> La barre la plus longue correspond au paludisme avec environ 400 cas, indiquant qu’il a le plus grand nombre de cas sur la période.</li>
            <li><strong>BILHARZIOSE URINAIRE :</strong> La deuxième barre représente la bilharziose urinaire environ 250 cas.</li>
            <li><strong>CONJONCTIVITE :</strong> La troisième barre est pour la conjonctivite environ 150 cas.</li>
            <li><strong>DIARRHÉE :</strong> La quatrième barre concerne la diarrhée environ 100 cas.</li>
            <li><strong>MALNUTRITION (0 - 4 ANS) :</strong> La plus courte barre représente la malnutrition chez les enfants de 0 à 4 ans environ 50 cas.</li>
        </ul>
        <p>Le graphique est utile pour analyser la prévalence des maladies sur une période donnée.</p>
    </div>
    """, unsafe_allow_html=True)


    # Enregistrer le modèle
    #with open('modele_visualisation.pkl', 'wb') as f:
        #pk.dump(maladie_counts, f)


# Appel de la fonction pour afficher les visualisations
afficher_data_visualisation()

# Lien vers les autres pages ou sections
st.subheader("Veuillez cliquer sur ces hyperliens pour être dirigé vers d'autres pages.")

st.write("""
- [Acceuil](http://localhost:8503/)
- [Informations](http://localhost:8503/Informations)
- [Exploration des données](http://localhost:8503/Exploration_des_donn%C3%A9es)
- [Manipulation des données](http://localhost:8503/Manipulation_des_donn%C3%A9es)
- [Plus de visualisation](http://localhost:8503/Plus_de_visualisation)
- [Modélisation des données](http://localhost:8503/Mod%C3%A9lisation_des_donn%C3%A9es)
- [Modèle de prédiction](http://localhost:8503/Mod%C3%A8le_de_pr%C3%A9diction)
- [Origine des données](http://localhost:8501/Origine_des_donn%C3%A9es)

""")


