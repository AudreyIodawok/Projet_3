import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from scipy.stats import kruskal
from statannot import add_stat_annotation
import streamlit as st
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def set_bg_hack_url():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://t4.ftcdn.net/jpg/02/58/46/45/360_F_258464541_GP8tzclZhURDrz1bvX41PGYR50Siayfg.jpg");
             background-size: cover;
             background-repeat: repeat;
             background-color: rgba(255, 255, 255, 0.2)
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

logo_url = "https://images.teamtailor-cdn.com/images/s3/teamtailor-production/logotype-v3/image_uploads/4b3f0830-1d6e-43d9-87e2-b6011a00cedb/original.png"
logo_url2 = "https://upload.wikimedia.org/wikipedia/fr/thumb/2/2a/Logo-INRAE_Transparent.svg/1200px-Logo-INRAE_Transparent.svg.png"
st.sidebar.image(logo_url2, width=200, use_column_width=False)
st.sidebar.image(logo_url, width=200, use_column_width=False)

# Charger le fichier CSV avec st.cache
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path, usecols= lambda x: 'Unnamed' not in x)

# Affichage dynamique en fonction de l'onglet sélectionné
def display_thematique(selected_disease):
    st.subheader(f"_Informations sur : {selected_disease}_")
    # Afficher les informations sur la maladie sélectionnée

def display_prediction(selected_disease):
    st.subheader(f"_Prédiction de : {selected_disease}_")
    # Afficher le formulaire de prédiction pour la maladie sélectionnée

def display_modele(selected_disease):
    st.subheader(f"_Modèles de prédiction de : {selected_disease}_")
    # Afficher le formulaire de prédiction pour la maladie sélectionnée

def main():
    set_bg_hack_url()
    st.title("Prédiction des maladies chroniques")

    # Affichage des onglets pour les sections
    st.sidebar.markdown("<h3 style='font-size: 18px;'>Sélectionnez une section</h3>", unsafe_allow_html=True)
    selected_section = st.sidebar.radio("section", ["Accueil", "Prédiction", "Informations maladies", "Performances des modèles"], index=0, label_visibility="collapsed")
    
    ## ACCUEIL ##
    if selected_section == "Accueil":
        st.markdown("<h3 style='text-align: center;'><em>Bienvenue sur l'application de prédiction des maladies chroniques</em></h3>", unsafe_allow_html=True)
        st.image("images_presentation.png", use_column_width=True)
        st.write("""**Cet outil constitue une aide à l'interprétation des résultats d'analyses médicales afin de proposer un diagnostic sur les maladies chroniques suivantes :**
                \n - Diabète\n - Maladies cardiaques\n - Maladies rénales\n - Cancer du sein\n - Maladies du foie
                 """)
        st.write("""**L'application est divisée en plusieurs sections :**\n - La section **Prédiction** offre une interface permettant
                à l'utilisateur de renseigner les valeurs des variables de chaque maladie et d'obtenir un diagnostic,\n- La section 
                **Informations maladies** détaille des informations sur les variables ainsi que des statistiques descriptives pour chaque 
                maladie,\n - La section **Performances des modèles** permet de visualiser la précision des modèles de prédiction et 
                d'appuyer le choix du modèle retenu pour l'application de prédiction.
                """)


    ## SECTION INFORMATIONS MALADIES ##
    if selected_section == "Informations maladies":
        st.sidebar.subheader("Sélectionnez une maladie")
        selected_disease = st.sidebar.selectbox("info", ["Diabète", "Maladies cardiaques", "Maladies rénales", "Cancer du sein", "Maladies du foie"], label_visibility="collapsed")
        display_thematique(selected_disease)
    

        ## INFORMATIONS SUR DIABETE ##
        if selected_disease == "Diabète":

        # Charger les données
            file_path = "df_diabete_fin (1).csv"
            diabete = load_data(file_path)

            # Afficher image
            st.image("Presentation_diabete.png", use_column_width=True)

            # Afficher la définition des variables
            st.subheader('Définition des variables :')
            st.write("""
                    - **Pregnancies (Grossesses)**: Le nombre de fois que la personne a été enceinte. Il n’y a pas de preuve concluante que le nombre de grossesses influence le diabète gestationnel. Les femmes **_ayant présenté un diabète gestationnel ont un risque augmenté de développer ultérieurement un diabète de type 2_**.
                 
                    - **Glucose (Glucose)**: La concentration de **_glucose plasmatique en mg/dl_** à 2 heures lors d'un test de tolérance au glucose oral. Cela mesure la réponse du corps au glucose après avoir consommé une quantité définie de sucre.
                    Selon la Société Française d’Endocrinologie, on parle de **_diabète sucré si la glycémie à jeun est ≥ 126 mg/dl à deux reprises_**.

                    - **Blood Pressure (Pression artérielle)**: La **_pression artérielle diastolique en millimètres de mercure (mm Hg)_**. Pour une personne non diabétique, une pression artérielle normale est de 120/80 mmHg.

                    - **Skin Thickness (Épaisseur de la peau)**: L'épaisseur du **_pli cutané du triceps en millimètres_**. Cette mesure peut être utilisée comme **_indicateur de la masse grasse_** (pour estimer la quantité de graisse corporelle).

                    - **Insulin (Insuline)**: La **_concentration d'insuline sérique à 2 heures, mesurée en milli-unités par millilitre (mu U/ml)_**. Cela indique la réponse de l'organisme à l'insuline après une charge de glucose. Le dosage de l’insuline dans le sang (insulinémie) n’est pas utilisé pour le diagnostic ni pour le suivi du diabète (qui reposent sur l’analyse de la glycémie et de l’hémoglobine glyquée). Cependant, il peut être utile de doser l’insuline dans le sang pour connaître la capacité du pancréas à la sécréter. Cela peut être utile au médecin à certaines phases de la maladie diabétique. 
                    A titre indicatif, à jeun, l’insulinémie est **_normalement inférieure à 25 mIU / L (µUI / mL)_**. Elle est comprise **_entre 30 et 230 mIU / L environ 30 minutes après l’administration de glucose_**.

                    - **BMI (Indice de masse corporelle)**: L'indice de masse corporelle est calculé en divisant le poids en kilogrammes par le carré de la taille en mètres. Il est utilisé comme indicateur du statut pondéral d'une personne.

                        Selon la Fédération Française des Diabétiques (FFD), l’apparition du diabète de type 2 est **_favorisée par un IMC supérieur à 30 kg/m²_**, limite caractérisant l’obésité.
                        Les personnes ayant un IMC élevé ont un risque accru de développer un diabète de type 2, bien plus qu’un facteur familial soit une prédisposition génétique.

                    - **Diabetes Pedigree Function (Fonction de lignée diabétique)**: Une fonction qui évalue la probabilité de diabète en **_fonction des antécédents familiaux_**. Elle prend en compte les antécédents familiaux de diabète pour estimer le risque génétique.
                    La fonction de lignée diabétique est utilisée pour évaluer le risque de diabète chez les personnes ayant des antécédents familiaux de diabète. Elle est basée sur des données telles que l’âge, le nombre de parents atteints de diabète et le degré de parenté.

                    - **Age (age)**: L'âge de la personne en années. L'âge peut être un facteur important dans la prédiction du diabète, car **_le risque augmente généralement avec l'âge_**.
                    En France, la moyenne d’âge des diabétiques est de **_65 ans_**. Cependant, il est important de noter que le diabète peut également affecter les jeunes adultes et les enfants."""
                    )


            # Afficher une série de graphiques
            st.subheader("Graphiques :")

            selected_graph_type = st.selectbox("Sélectionnez le type de graphique", ["Boxplot", "Heatmap", "Analyse en Composantes Principales"])

            # Boxplots Diabète
            if selected_graph_type == "Boxplot":
                # Afficher les données
                st.write("""Les boîtes à moustache ci-dessous représentent la distribution
                        des patients malades et sains en fonction des variables étudiées. 
                        Si la valeur p du test de Kruskal Wallis est inférieure au seuil de signification, le plus souvent 5 %, l'hypothèse nulle est rejetée et on admet qu'il y a des différences significatives entre les deux échantillons.
                        """)

                # Définir la fonction pour créer le graphique
                def plot_boxplots(diabete):
                    mapping_diabete = {0 : 'sain', 1 : 'malade'}
                    diabete['Outcome'] = diabete['Outcome'].map(mapping_diabete)
                    diabete['Outcome'] = diabete['Outcome'].astype(str)

                    liste_variables = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
                    col = 2
                    rows = 4

                    fig, axes = plt.subplots(rows, col, figsize=(18, 28))
                    axes = axes.flatten()

                    fig.suptitle('Boxplots des variables sur le diagnostic du diabète - Test de Kruskal-Wallis', fontsize=30)

                    for i, variable in enumerate(liste_variables):
                        groups = [diabete[variable][diabete['Outcome'] == level] for level in diabete['Outcome']]

                        # Effectuer le test de Kruskal-Wallis
                        h_stat, p_value = kruskal(*groups)

                        # Tracer le boxplot
                        ax = sns.boxplot(data=diabete, x='Outcome', y=variable, ax=axes[i])
                        axes[i].set_xlabel('Outcome', fontsize = 25)
                        axes[i].set_ylabel(variable, fontsize=25, color = 'blue')

                        # Augmenter la taille des ticks sur l'axe des x
                        ax.tick_params(axis='x', labelsize=20)  # Définir la taille des ticks sur l'axe des x
                        ax.tick_params(axis='y', labelsize=20)

                        # Ajouter l'annotation pour la p-valeur avec statannot
                        add_stat_annotation(ax, data=diabete, x='Outcome', y=variable,
                                            box_pairs=[('sain', 'malade')],  # Comparaison entre les deux niveaux d'Outcome
                                            test='Kruskal', text_format='full', loc='outside',
                                            verbose=2, fontsize= 20)

                    plt.tight_layout(rect=[0, 0, 1, 0.95])

                    # Afficher le graphique dans Streamlit
                    st.pyplot(fig)

                # Appel de la fonction pour créer et afficher le graphique
                plot_boxplots(diabete)

            # Heatmap Diabète
            elif selected_graph_type == "Heatmap":
                # Définir la fonction pour créer le graphique
                def plot_heatmap(diabete):
                    fig, ax = plt.subplots(figsize = (15,13))
                    sns.heatmap(diabete.corr(), center=0, cmap="coolwarm")
                    ax.set_title("Heatmap de la corrélation entre les variables du diabète", fontsize = 27)
                    ax.tick_params(axis='x', labelsize=20, rotation=90)  # Taille des ticks sur l'axe des x
                    ax.tick_params(axis='y', labelsize=20, rotation=0)  # Taille des ticks sur l'axe des y
                    ax.legend(fontsize = 22)
                    st.pyplot(fig)  # Afficher le graphique dans Streamlit

                # Appel de la fonction pour créer et afficher le graphique
                plot_heatmap(diabete)

                st.write("""Les corrélations les plus visibles sont entre le taux d'insuline et
                        le taux de glucose, l'indice de masse corporelle et le skin thickness.
                        Le diagnostic du malade du diabète est moyennement corrélé avec les variables
                        glucose, insulin et à moindre mesure les variables BMI et skin thickness.
                        """)

            # ACP Diabète
            elif selected_graph_type == "Analyse en Composantes Principales":
                st.image("ACP diabèe.png", use_column_width=True)

        ## INFORMATIONS SUR MALADIES CARDIAQUES ##
        # Charger les données
        if selected_disease == "Maladies cardiaques":
            file_path = "dfheartML.csv"
            coeur = load_data(file_path)

            # Afficher image
            st.image("Presentation_maladies_cardiaques.png", use_column_width=True)

            # Afficher la définition des variables
            st.subheader('Définition des variables :')
            st.write("""
                    - **Age (age du patient)** : La probabilité d'avoir un accident cardiovasculaire ou cardiaque 
                    **_augmente nettement après 50 ans chez l'homme_** et après **_60 ans chez la femme_**.
                     
                     \n- **Sex 
                    (Genre du patient)** : **_Les femmes sont plus vulnérables que les hommes aux maladies cardiovasculaires_**. 
                    56 % en meurent contre 46 % des hommes.
                     
                     \n- **Cp (Chest Pain Type) / Type de douleur thoracique ressentie** : 
                     **_0 - pas de douleur, 1 - faible, 2 - modérée, 3 - intense,
                    4 - extrêmement intense_**. L'anamnèse de la maladie actuelle doit rechercher la 
                    localisation, la durée, le caractère et la qualité de la douleur. Il faut interroger le patient 
                    sur tout événement déclenchant (p. ex., surmenage des muscles thoraciques), ainsi que tout 
                    facteur calmant. D'importants symptômes associés à la recherche comprennent une 
                    **_dyspnée_**, des **_palpitations_**, des **_syncopes_**, une **_transpiration_**, des **_nausées ou des 
                    vomissements_**, une **_toux_**, une **_fièvre ou des frissons_**.
                     
                     \n- **Trestbps (Resting Blood Pressure) / Pression artérielle au repos** : La valeur limite au-delà de laquelle on parle **_d’hypertension artérielle 
                     est de 140/90 lorsque la mesure est faite au cabinet médical_** et **_135/85 lors d’une automesure_**. Plus 
                     la tension est élevée, plus le risque de maladie cardiovasculaire est important.\n- **Chol (Serum Cholestrol) 
                     / Cholestérol sérique en mg/dl** : Le cholestérol total
                    Sous le terme de cholestérol total, on inclut les taux de cholestérol HDL et LDL, ainsi qu’un cinquième 
                     du taux de triglycérides. **_Ce taux est habituellement inférieur à 200 mg/dl._**
                     """)
            col1, col2 = st.columns([1, 1])  # Rapport de largeur 1:1   
            with col1:
                st.write("""**_Cholestérol LDL_** : Egalement appelé mauvais cholestérol. En l’absence de facteur de risque, un taux de cholestérol LDL est considéré comme normal 
                     lorsqu’il est inférieur à 1,6 g/l. Si le patient présente un ou plusieurs facteurs de risque (par exemple, 
                     un homme de plus de 50 ans), cette valeur limite est de 1,3 g/l.""") 
            with col2:
                 st.write("""**_Cholestérol HDL_** : Également appelé bon cholestérol, son rôle est de capter le cholestérol en excès dans 
                     le sang et de le conduire au foie pour qu’il soit éliminé avec la bile. Le taux de cholestérol HDL est considéré trop 
                     faible lorsqu’il est inférieur à 0,35 g/l. Un taux élevé de cholestérol HDL (plus de 0,60 g/l) protège des maladies 
                     cardiovasculaires et annule un facteur de risque cardiovasculaire. 
                          """)
            st.write("""**_NB : Dans le modèle de prédiction proposé, les valeurs de HDL et LDL ne sont pas connues et donc pas prises en compte,
                     ce qui peut consister un biais dans le résultat obtenu._**
                     """)
            st.write("""
                     - **Fbs (Fasting Blood Sugar) / Sucre dans le sang à jeun > 120 mg/dl (Oui - Non)** : 
                     La glycémie à jeun se situe entre 70 et 110 mg par décilitre de sang à jeun. 
                     Lorsque la glycémie est basse, celle-ci est inférieure à 60 mg par décilitre. Le patient est alors en état 
                     d’hypoglycémie.A contrario, quand la glycémie est élevée, le taux de sucre est supérieur à 110 mg par décilitre, à jeun. 
                     Le patient est ainsi en état d’hyperglycémie modérée.
⚠️                  La glycémie est à un taux trop élevé quand elle est **_supérieure à 120 mg par décilitre_**.
                     
                     \n- **Restecg (Resting Electrocardiographic Results) / Résultats électrocardiographiques au repos** : **_0 - aucune anomalie, 1 - légère anomalie, 
                     2 - anomalie modérée, 3 - anomalie grave_**. Un électrocardiogramme, ou ECG, est un examen 
                     qui mesure l’activité électrique du cœur. Ces informations peuvent être utilisées pour évaluer la fréquence et le rythme 
                     cardiaques, ainsi que la fonction cardiaque globale.
                     La fourchette normale de la fréquence cardiaque est de **_60 à 100 battements par minute_**.\n- **Thalach (Maximum Heart 
                     Rate Achieved)** : Fréquence cardiaque maximale atteinte. 
                     
                     \n- **Exang (Exercise Induced Angina) / Angine induite par l'exercice** : 
                     **_0 - absence, 1 - presence_**. L’angine est un type de malaise ou de douleur à la poitrine qui survient quand le cœur ne reçoit 
                     pas tout l’oxygène dont il a besoin pour faire son travail. On décrit souvent l’angine comme une douleur ou une pression au milieu de la poitrine 
                     qui peut se propager aux bras, au cou ou à la mâchoire. Ces symptômes sont parfois accompagnés d’un essoufflement, de transpiration ou de 
                     nausées.
                     
                     \n- **Oldpeak (ST Depression Induced by Exercise) / Dépression du segment ST induite par l'exercice par rapport 
                     au repos en mm** : Le segment ST, en conditions normales, est plat ou isoélectrique, bien qu’il puisse présenter de petites 
                     variations mineures de 0.5 mm. **_Une variation du segment ST de plus de 2 mm peut être le signe d'une maladie coronarienne._**
                     
                     \n- **Slope (Slope of the Peak Exercise ST Segment) / Pente du segment ST à l'effort maximal en mm** : 
                     on en voit souvent durant l’effort physique ; elles présentent habituellement une élévation rapide au moment où elles croisent 
                     la ligne isoélectrique rapidement (pente ascendante). Le critère classique est un **_ST additionnel supérieur à 1 mm en cas de pente 
                     horizontale ou descendante et de 1,5 mm en cas pente ascendante._**
                     
                     \n- **Ca (Number of Major Vessels Colored by Fluoroscopy) / Nombre de 
                     vaisseaux majeurs colorés par fluoroscopie** : **_0 - aucun vaisseau coloré (absence de blocage), 1 - un seul vaisseau coloré (blocage 
                     partiel ou modéré),  2 - deux vaisseaux colorés (blocages multiples),  3 - trois vaisseaux colorés (maladie cardiovasculaire avancée 
                     avec blocages importants ou critiques)._**
                     
                     \n- **Thal (Thalassemia) / Type de thalassémie** : **_0 - absence de caractéristiques associées à la 
                     thalassémie, 1 - présence de caractéristiques mineures (forme bénigne de la maladie), 2 - présence de caractéristiques
                     modérées (maladie plus prononcée ou risque accru de complications), 3 - présence de caractéristiques sévères de la thalassémie (maladie grave
                     ou prononcée avec un risque de complications potentiellement graves)_**. Les thalassémies sont un groupe de maladies héréditaires dues à une 
                     perturbation de la fabrication de l’une des quatre chaînes d’acides aminés qui constituent l’hémoglobine (la protéine présente dans les 
                     globules rouges qui transporte l’oxygène).
                     """
                    )

            # Afficher une série de graphiques
            st.subheader("Graphiques :")

            selected_graph_type = st.selectbox("Sélectionnez le type de graphique", ["Heatmap", "Boxplot", "Analyse en Composantes Principales"])

            # Heatmap Maladies cardiaques
            if selected_graph_type == "Heatmap":
                # Définir la fonction pour créer le graphique
                def plot_heatmap(coeur):
                    fig, ax = plt.subplots(figsize = (15,13))
                    sns.heatmap(coeur.corr(), center=0, cmap="coolwarm")
                    ax.set_title("Heatmap de la corrélation entre les variables du coeur", fontsize = 27)
                    ax.tick_params(axis='x', labelsize=20, rotation=90)  # Taille des ticks sur l'axe des x
                    ax.tick_params(axis='y', labelsize=20, rotation=0)  # Taille des ticks sur l'axe des y
                    ax.legend(fontsize = 22)
                    st.pyplot(fig)  # Afficher le graphique dans Streamlit

                # Appel de la fonction pour créer et afficher le graphique
                plot_heatmap(coeur)

                st.write("""Il existe des corrélations entre "trestbps" (resting blood pressure), "oldpeak" (ST depression induced by exercice) et "ca"
                        (number of vessels colored by fluoroscopy).
                        Le diagnostic de maladie cardiaque semble fortement corrélé avec "cp" (chest pain type), "thalach" (maximum heart rate achieved) et "slope" (slope of the peak exercise ST segment).
                        """
                        )

            # Boxplot Maladies cardiaques
            elif selected_graph_type == "Boxplot":
                st.write("""Les boîtes à moustache ci-dessous représentent la distribution
                        des patients malades et sains en fonction des variables étudiées. 
                        Si la valeur p du test de Kruskal Wallis est inférieure au seuil de signification, le plus souvent 5 %, l'hypothèse nulle est rejetée et on admet qu'il y a des différences significatives entre les deux échantillons.
                        """)
                def plot_boxplots(coeur):
                    mapping_coeur = {0 : 'sain', 1 : 'malade'}
                    coeur['target'] = coeur['target'].map(mapping_coeur)
                    coeur['target'] = coeur['target'].astype(str)

                    liste_variables_coeur = coeur.columns[:-1].tolist()
                    col = 2
                    rows = 8

                    fig, axes = plt.subplots(rows, col, figsize=(18, 56))
                    axes = axes.flatten()

                    fig.suptitle('Boxplots des variables sur le diagnostic des maladies cardiaques - Test de Kruskal-Wallis', fontsize=30)

                    for i, variable in enumerate(liste_variables_coeur):
                        groups = [coeur[variable][coeur['target'] == level] for level in coeur['target']]

                        # Effectuer le test de Kruskal-Wallis
                        h_stat, p_value = kruskal(*groups)

                        # Tracer le boxplot
                        ax = sns.boxplot(data=coeur, x='target', y=variable, ax=axes[i])
                        axes[i].set_xlabel('target', fontsize = 25)
                        axes[i].set_ylabel(variable, fontsize=25, color = 'blue')
            
                        # Augmenter la taille des ticks sur l'axe des x
                        ax.tick_params(axis='x', labelsize=20)  # Définir la taille des ticks sur l'axe des x
                        ax.tick_params(axis='y', labelsize=20)

                        # Ajouter l'annotation pour la p-valeur avec statannot
                        add_stat_annotation(ax, data=coeur, x='target', y=variable,
                                            box_pairs=[('malade', 'sain')],  # Comparaison entre les deux niveaux d'Outcome
                                            test='Kruskal', text_format='full', loc='outside',
                                            verbose=2, fontsize= 20)
        
                    for j in range(rows * col - 2, rows * col):
                        fig.delaxes(axes[j])

                    plt.tight_layout(rect=[0, 0, 1, 0.95])
        
                    # Afficher le graphique dans Streamlit
                    st.pyplot(fig)

                # Appel de la fonction pour créer et afficher le graphique
                plot_boxplots(coeur)

            # ACP Maladies cardiaques
            elif selected_graph_type == "Analyse en Composantes Principales":
                st.image("ACP heart.png", use_column_width=True)

        ## INFORMATIONS SUR MALADIES RENALES ##
        # Charger les données
        if selected_disease == "Maladies rénales":
            file_path = "dfreinclean.csv"
            rein = load_data(file_path)

            # Afficher image
            st.image("Presentation_maladies_renales.png", use_column_width=True)

            # Afficher la définition des variables
            st.subheader('Définition des variables :')
            st.write("""
                    - **Age (age)** : L'âge du patient, un facteur important dans l'évaluation du risque de maladie rénale chronique, car la **_fonction rénale a tendance à diminuer avec l'âge_**.
                    
                    - **Blood Pressure (Pression artérielle) - bp** : La pression artérielle du patient, **_mesurée en millimètres de mercure (mmHg)_**. La **_pression artérielle élevée est un facteur de risque bien connu pour la maladie rénale chronique_**.
                    
                    - **Specific Gravity (Gravité spécifique de l'urine) - sg** : La concentration de l'urine, mesurée par la gravité spécifique. Des changements dans la gravité spécifique peuvent indiquer des problèmes rénaux. 
                    Une **_densité inférieure à 1,003 peut indiquer un problème rénal_**, mais également une consommation d'eau excessive.
                    
                    - **Albumin (Taux d'albumine) - al en g/L** : Le taux d'albumine dans le sang. **_Un taux d'albumine normal se situe généralement entre 3,5 et 5 g/L de sang._**
                    
                    - **Sugar (Taux de sucre) - su en g/L** : Le taux de sucre dans le sang. Les valeurs normales de **_glycémie à jeun se situent entre 0,7 et 1 g/L de sang_**. 
                     **_Des valeurs supérieures à 1,2 g/L peuvent indiquer un diabète_**. Le diabète, s'il n'est pas contrôlé, peut contribuer au développement de la maladie rénale chronique.
                    
                    - **Red Blood Cells (Nombre de globules rouges) - rbc** **_(0 - normal, 1 - anormal)_** et **rc** **_en millions de cellules par microlitre de sang_** : Le nombre de globules rouges dans le sang, qui peut être affecté 
                    par des problèmes rénaux. Elles se situent **_entre 4,5 et 5,5 millions de globules rouges par microlitre de sang_**, mais diminuent
                    dans le cas d'une affection rénale.
                    
                    - **Pus Cell (Présence de cellules de pus) - pc** : **_0 - normal, 1 - anormal_**. La présence de cellules de pus dans l'urine, qui peut être un signe d'infection rénale.
                    
                    - **Pus Cell Clumps (Agrégats de cellules de pus) - pcc** : **_0 - normal, 1 - anormal_**. Agrégats de cellules de pus dans l'urine, un autre indicateur potentiel d'infection.
                    
                    - **Bacteria (Présence de bactéries) - ba** : **_0 - normal, 1 - anormal_**. La présence de bactéries dans l'urine, indiquant une possible infection.
                    
                    - **Blood Glucose Random (Taux de sucre sanguin aléatoire) - bgr en mg/dl** : Le taux de sucre dans le sang à un moment donné, ce qui peut être lié au contrôle du diabète.
                    
                    - **Blood Urea (Taux d'urée sanguine) - bu en mg/dl** : Le taux d'urée dans le sang, qui peut être affecté par la fonction rénale.
                    Les valeurs normales de l’urée sont comprises entre : **_2,5 et 7,6 mmol / L (ou 10 à 55 mg / dl) dans le sang_**.                          
                    Un **_taux élevé d'urée peut indiquer un dysfonctionnement rénal ou hépatique_**, qui peut être aigu ou chronique et affecter les deux reins.
                    
                    - **Serum Creatinine (Taux de créatinine sérique) en mg/dl - sc** : Le taux de créatinine dans le sang, une mesure courante de la fonction rénale.
                    En général, pour les adultes, les **_valeurs normales se situent autour de 0,5 et 1,2 mg/dl_**. Elles peuvent augmenter rapidement en cas d'insuffisance rénale.
                    
                    - **Sodium (Taux de sodium) - sod en mEq/L ou mmol/L** : Le taux de sodium dans le sang, qui peut être affecté par les problèmes rénaux.
                    Le **_taux normal de sodium dans le sang se situe entre 136 et 145 milliéquivalents par litre_** (mEq/L). Les médecins parlent d’hyponatrémie lorsque le taux de sodium dans le sang (natrémie) se situe en dessous de 135 mEq/L. 
                    Un faible taux de sodium a plusieurs causes, notamment une consommation excessive de liquides, l'insuffisance rénale, l'insuffisance cardiaque, la cirrhose et l'utilisation de diurétiques.
                    
                    - **Potassium (Taux de potassium) - pot en mEq/L ou mmol/L** : Le taux de potassium dans le sang, une électrolyte important régulée par les reins.
                    **_En temps normal, le taux de potassium dans le sang se situe entre 3,6 et 5 mmol/l_**. 
                    Les deux principales causes d’un taux de potassium élevé sont la prise de médicaments (certains diurétiques contre l’hypertension artérielle, les digitaliques dans les troubles du rythme cardiaque…) 
                    et **_l’insuffisance rénale_**. Les autres causes sont le diabète, l’exercice physique intense ou encore les causes de lyses cellulaires (destruction des cellules), comme les brûlures étendues, les infarctus.
                    C’est lorsque le taux de potassium dans le sang atteint ou dépasse **_6,5 mmol/l que la présence de ce minéral dans le corps est vraiment dangereuse_**.
                    
                    - **Haemoglobin (Taux d'hémoglobine) - hemo en g/dl** : La concentration d'hémoglobine dans le sang.
                    En général, chez les adultes, les valeurs normales se situent entre **_12 et 16 g/dL pour les femmes et entre 13 et 17 g/dL pour les hommes_**. 
                    Un **_taux d’hémoglobine faible est synonyme d’anémie, et peut être la conséquence d’une insuffisance rénale_** (cf globules rouges et anémie).
                    
                    - **Packed Cell Volume (Volume de globules rouges tassés - hématocrite) - pcv en %** : La proportion du volume sanguin occupé par les globules rouges.
                    **_Les valeurs normales sont situées entre 40 et 52 %._** Un taux d'hématocrite trop faible ou trop élevé est signe d'un dysfonctionnement des reins.
                    
                    - **White Blood Cell Count (Nombre de globules blancs) - wc en cellules / microlitre** : Le nombre de globules blancs dans le sang, **_indiquant une possible réponse immunitaire ou une infection_**.
                    Le nombre normal total se situe entre **_4 000 et 11 000 cellules par microlitre_**.
                    
                    - **Hypertension (Présence d'hypertension) - htn** : **_0 - absence, 1 - presence_**. La présence ou non d'une pression artérielle élevée.
                    
                    - **Diabetes Mellitus - dm** : **_0 - absence, 1 - presence_**. Le diabète est une des principales causes de maladie rénale.
                      
                    - **Coronary Artery Disease (Présence de maladie coronarienne) cad** : **_0 - absence, 1 - presence_**. La présence de maladie coronarienne, qui peut affecter la circulation sanguine vers les reins.
                    
                    - **Appetite (Niveaux d'appétit)- appet** : **_0 - poor, 1 - good_**. Les niveaux d'appétit du patient.
                    A mesure que la fonction rénale s’aggrave et que de plus en plus de déchets métaboliques s’accumulent dans le sang, la personne peut ressentir une asthénie, une faiblesse généralisée et des difficultés de concentration intellectuelle. Elle peut présenter une perte d’appétit et un essoufflement.
                    
                    - **Pedal Edema (Présence d'œdème des pieds) - pe** : **_0 - absence, 1 - presence_**. La présence d'œdème des pieds, pouvant être associée à une rétention d'eau liée à la maladie rénale.
                    
                    - **Anemia (Présence d'anémie) - ane** : **_0 - absence, 1 - presence_**. La présence d'anémie, souvent liée à des problèmes rénaux.
                    """
                )

            # Afficher une série de graphiques
            st.subheader("**Graphiques :**")

            selected_graph_type = st.selectbox("Sélectionnez le type de graphique", ["Boxplot", "Heatmap", "Analyse en Composantes Principales"])
            
            # Boxplots Maladies rénales
            if selected_graph_type == "Boxplot":

                # Définir la fonction pour créer le graphique
                st.write("""Les boîtes à moustache ci-dessous représentent la distribution
                        des patients malades et sains en fonction des variables étudiées. 
                        Si la valeur p du test de Kruskal Wallis est inférieure au seuil de signification, le plus souvent 5 %, l'hypothèse nulle est rejetée et on admet qu'il y a des différences significatives entre les deux échantillons.
                        """)
                def plot_boxplots(rein):

                    liste_variables_rein = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'wc', 'rc']
                    col = 2
                    rows = 7

                    fig, axes = plt.subplots(rows, col, figsize=(18, 52))
                    axes = axes.flatten()

                    rein['classification'] = rein['classification'].astype('str')

                    fig.suptitle('Boxplots des variables sur le diagnostic des maladies rénales - Test de Kruskal-Wallis', fontsize=30)

                    for i, variable in enumerate(liste_variables_rein):
                        groups = [rein[variable][rein['classification'] == level] for level in rein['classification']]

                        # Effectuer le test de Kruskal-Wallis
                        h_stat, p_value = kruskal(*groups)

                        # Tracer le boxplot
                        ax = sns.boxplot(data=rein, x='classification', y=variable, ax=axes[i])
                        axes[i].set_xlabel('classification', fontsize = 25)
                        axes[i].set_ylabel(variable, fontsize=25, color = 'blue')

                        # Augmenter la taille des ticks sur l'axe des x
                        ax.tick_params(axis='x', labelsize=20)  # Définir la taille des ticks sur l'axe des x
                        ax.tick_params(axis='y', labelsize=20)

                        # Ajouter l'annotation pour la p-valeur avec statannot
                        add_stat_annotation(ax, data=rein, x='classification', y=variable,
                                            box_pairs=[('ckd', 'notckd')],  # Comparaison entre les deux niveaux d'Outcome
                                            test='Kruskal', text_format='full', loc='outside',
                                            verbose=2, fontsize= 20)
            
                    for j in range(rows * col - 1, rows * col):
                        fig.delaxes(axes[j])

                    plt.tight_layout(rect=[0, 0, 1, 0.95])

                    # Afficher le graphique dans Streamlit
                    st.pyplot(fig)

                # Appel de la fonction pour créer et afficher le graphique
                plot_boxplots(rein)

            # Heatmap Maladies rénales
            elif selected_graph_type == "Heatmap":
                # Définir la fonction pour créer le graphique
                def plot_heatmap(rein):
                    fig, ax = plt.subplots(figsize = (15,13))
                    sns.heatmap(rein.select_dtypes('number').corr(), center=0, cmap="coolwarm")
                    ax.set_title("Heatmap de la corrélation entre les variables du rein", fontsize = 27)
                    ax.tick_params(axis='x', labelsize=20, rotation=90)  # Taille des ticks sur l'axe des x
                    ax.tick_params(axis='y', labelsize=20, rotation=0)  # Taille des ticks sur l'axe des y
                    ax.legend(fontsize = 22)
                    st.pyplot(fig)  # Afficher le graphique dans Streamlit

                # Appel de la fonction pour créer et afficher le graphique
                plot_heatmap(rein)

                st.write("""Les corrélations les plus visibles sont entre les variables "sg" (specific gravity), "hemo" (taux d'hémoglobine), "bu" (blood urea) et "sod" (sodium).
                        Il existe également des corrélations entre "bu" (blood urea) et "sc" (serum creatinine).
                        """)
            
            # ACP maladies rénales
            elif selected_graph_type == "Analyse en Composantes Principales":
                st.image("ACP rein.png", use_column_width=True)
        
        ## INFORMATIONS SUR CANCER DU SEIN ##
        # Charger les données
        if selected_disease == "Cancer du sein":
            file_path = "dfcancersein.csv"
            sein = load_data(file_path)

            # Afficher image
            st.image("Presentation_cancer_sein.png", use_column_width=True)

            # Afficher la définition des variables
            st.subheader('Définition des variables :')
            st.write("""
                    - **Radius Mean en Um** : Moyenne des distances du centre aux points sur le périmètre de la cellule.
                    
                    - **Texture Mean** : Valeur adimensionnelle qui exprime la variation de l'intensité des niveaux de gris dans une région donnée de l'image médicale.
                    
                    - **Perimeter Mean en mm** : Moyenne des périmètres des contours des masses ou des tumeurs présentes dans une image médicale.
                    
                    - **Area Mean en mm^2** : Moyenne de la surface de la tumeur, indicative de la taille de la tumeur.
                    
                    - **Smoothness Mean (sans unité ou normalisée)** : Mesure de la variation locale des longueurs de rayon dans le contour d'une masse ou d'une tumeur dans une image médicale.
                    
                    - **Compactness Mean (sans unité ou normalisée)** : Compacité des cellules observées dans des images médicales, telles que des images de tumeurs ou de masses.
                    
                    - **Concavity Mean (sans unité ou normalisée)** : Mesure la gravité des parties concaves de la tumeur, donnant une indication sur la forme des contours.
                    
                    - **Concave Points Mean (sans dimension)** : Moyenne du nombre de parties concaves du contour, aide à caractériser davantage la forme de la tumeur.
                    
                    - **Symmetry Mean (sans dimension)** : Moyenne de la symétrie des cellules observées dans des images médicales.
                    
                    - **Fractal Dimension Mean (sans dimension)** : Complexité de la structure géométrique des contours des cellules observées dans des images médicales.
                    
                    - **Radius Se en Um** : Fournit une indication de la variabilité ou de la dispersion des mesures de rayon des noyaux cellulaires dans un échantillon.
                    
                    - **Texture Se (adimensionnelle)** : Fournit une indication de la variabilité ou de la dispersion des mesures de texture dans un échantillon.
                    
                    - **Perimeter Se en mm** : Fournit une indication de la variabilité ou de la dispersion des mesures de périmètre dans un échantillon.
                    
                    - **Area Se en mm^2** : Représente l'erreur standard de la surface de la tumeur.
                    
                    - **Smoothness Se (sans unité ou normalisée)** : Fournit une indication de la variabilité ou de la dispersion des mesures de la régularité des contours dans un échantillon.
                    
                    - **Compactness Se (sans unité ou normalisée)** : Fournit une indication de la variabilité ou de la dispersion des mesures de la compacité dans un échantillon.
                    
                    - **Concavity Se (sans unité ou normalisée)** : Fournit une indication de la variabilité ou de la dispersion des mesures de la concavité dans un échantillon.
                    
                    - **Concave Points Se (sans dimension)** : Fournit une indication de la variabilité ou de la dispersion des mesures des points concaves dans un échantillon.
                    
                    - **Symmetry Se (sans dimension)** : Fournit une indication de la variabilité ou de la dispersion des mesures de symétrie dans un échantillon.
                    
                    - **Fractal Dimension Se (sans dimension)** : Fournit une indication de la variabilité ou de la dispersion des mesures de la dimension fractale dans un échantillon.
                    
                    - **Radius Worst en Um** : Fait référence à la plus grande distance entre le centre d'une masse tumorale et le point le plus éloigné de ses limites.
                    
                    - **Texture Worst (adimensionnelle)** : Fait référence à la variation locale de la luminosité des pixels dans une image numérique.
                    
                    - **Perimeter Worst en mm** : Fait référence à la longueur du contour de la lésion la plus étendue identifiée dans une image médicale.
                    
                    - **Area Worst en mm^2** : Fait référence à la moyenne des trois plus grandes valeurs de la surface de la tumeur. Elle donne une indication de la taille maximale probable de la tumeur.
                    
                    - **Smoothness Worst (sans unité ou normalisée)** : Fait référence à la mesure de la variation locale de la longueur du rayon entre les points du contour d'une tumeur identifiée dans une image médicale.
                    
                    - **Compactness Worst (sans unité ou normalisée)** : Fait référence à la mesure de la compacité maximale probable de la tumeur, basée sur la moyenne des trois plus grandes valeurs de la caractéristique de compacité.
                    
                    - **Concavity Worst (sans unité ou normalisée)** : Fait référence aux trois plus grandes valeurs de la concavité, donnant une indication sur la gravité des parties concaves dans la tumeur.
                    
                    - **Concave Points Worst (sans dimension)** : Fait référence à une mesure des points concaves les plus proéminents le long de la frontière d'une masse tumorale identifiée dans une image médicale.
                    
                    - **Symmetry Worst (sans dimension)** : Fait référence à la moyenne des trois plus grandes valeurs de la symétrie de la tumeur.
                    
                    - **Fractal Dimension Worst (sans dimension)** : C'est la moyenne des trois plus grandes valeurs de la dimension fractale de la tumeur.
                    """
            )
                 
            # Afficher une série de graphiques
            st.subheader("**Graphiques :**")

            selected_graph_type = st.selectbox("Sélectionnez le type de graphique", ["Heatmaps", "Boxplots", "Analyse en Composantes Principales"])

            sein_mean = sein[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean','smoothness_mean',
                            'compactness_mean', 'concavity_mean','concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',]]
            sein_se = sein[['radius_se', 'texture_se', 'perimeter_se', 'area_se','smoothness_se',
                            'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',]]
            sein_worst = sein[['radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
                            'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']]

            # Heatmaps Cancer du sein
            if selected_graph_type == "Heatmaps":
                # Définir la fonction pour créer le graphique
                def plot_heatmap(sein, title):
                    fig, ax = plt.subplots(figsize=(15, 13))
                    sns.heatmap(sein.corr(), center=0, cmap="coolwarm", ax=ax)
                    ax.set_title(title, fontsize=27)
                    ax.tick_params(axis='both', labelsize=18)
                    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)  # rotation des yticks
                    plt.tight_layout()
                    st.pyplot(fig)

                # Appel de la fonction pour créer et afficher les graphiques
                plot_heatmap(sein_mean, "Heatmap de corrélation entre les variables 'mean'")
                plot_heatmap(sein_se, "Heatmap de corrélation entre les variables 'se'")
                plot_heatmap(sein_worst, "Heatmap de corrélation entre les variables 'worst'")

                st.write("""Les corrélations positives fortes sont entre :
                        radius avec perimeter,
                        radius avec area.
                        Les 3 variables sont essentiellement toutes liées à la taille de la cellule et contiennent la même information mais dans des échelles différentes.
                        Les corrélations moyennes :
                        concavity avec compactness,
                        concavity avec concave points.
                        """)
        
            # Boxplots Cancer du sein
            elif selected_graph_type == "Boxplots":

                # Afficher les données
                st.write("""Les boîtes à moustache ci-dessous représentent la distribution
                        des patients malades et sains en fonction des variables étudiées. 
                        Si la valeur p du test de Kruskal Wallis est inférieure au seuil de signification, le plus souvent 5 %, l'hypothèse nulle est rejetée et on admet qu'il y a des différences significatives entre les deux échantillons.
                        """)
                def plot_boxplots(sein):
                    mapping_sein = {0 : 'malignes', 1 : 'bénignes'}
                    sein['diagnosis'] = sein['diagnosis'].map(mapping_sein)
                    sein['diagnosis'] = sein['diagnosis'].astype(str)

                    liste_variables_sein = sein.columns[1:].tolist()
                    col = 2
                    rows = 15

                    fig, axes = plt.subplots(rows, col, figsize=(18, 105))
                    axes = axes.flatten()

                    fig.suptitle('Boxplots des variables sur le diagnostic du cancer du sein - Test de Kruskal-Wallis', fontsize=30)

                    for i, variable in enumerate(liste_variables_sein):
                        groups = [sein[variable][sein['diagnosis'] == level] for level in sein['diagnosis']]

                        # Effectuer le test de Kruskal-Wallis
                        h_stat, p_value = kruskal(*groups)

                        # Tracer le boxplot
                        ax = sns.boxplot(data=sein, x='diagnosis', y=variable, ax=axes[i])
                        axes[i].set_xlabel('diagnosis', fontsize = 25)
                        axes[i].set_ylabel(variable, fontsize=25, color = 'blue')

                        # Augmenter la taille des ticks sur l'axe des x
                        ax.tick_params(axis='x', labelsize=20)  # Définir la taille des ticks sur l'axe des x
                        ax.tick_params(axis='y', labelsize=20)

                        # Ajouter l'annotation pour la p-valeur avec statannot
                        add_stat_annotation(ax, data=sein, x='diagnosis', y=variable,
                                            box_pairs=[('bénignes', 'malignes')],  # Comparaison entre les deux niveaux d'Outcome
                                                test='Kruskal', text_format='full', loc='outside',
                                            verbose=2, fontsize= 20)

                    plt.tight_layout(rect=[0, 0, 1, 0.95])

                    # Afficher le graphique dans Streamlit
                    st.pyplot(fig)

                # Appel de la fonction pour créer et afficher le graphique
                plot_boxplots(sein)

            # ACP Cancer du sein
            elif selected_graph_type == "Analyse en Composantes Principales":
                st.image("ACP cancer sein.png", use_column_width=True)


        ## INFORMATIONS SUR MALADIES DU FOIE ##
        # Charger les données
        if selected_disease == "Maladies du foie":
            file_path = "df_liver_M0F1 .csv"
            foie = load_data(file_path)

            # Afficher image
            st.image("Presentation_maladie_foie.png", use_column_width=True)
            
            # Afficher la définition des variables
            st.subheader('Définition des variables :')
            st.write("""
                    - **Age (age du patient)** : L'age moyen des personnes développant une maladie hépatique se situe **_autour de 50-55 ans_**.
                    
                    - **Gender (Genre du patient)** : Depuis 20 ans, le nombre de **_cirrhose du foie augmente chez la femme_**. Elle aurait plus de risque de développer une maladie hépatique liée à l'alcool ou un "foie gras". 
                    
                    - **Total Bilirubin (Bilirubine totale dans le sang) an mg/dl** : La bilirubine est produite par la dégradation des globules rouges, en particulier de l’hémoglobine. 
                    La quantité de bilirubine totale dans le sang est **_normalement comprise entre 0,3 et 1,9 mg / dl_**. Des valeurs supérieures peuvent être
                    associées à des problèmes hépatiques.
                    
                    - **Alkaline Phosphotase (Taux de phosphatase alcaline dans le sang) en UI/L** : 
                    Les phosphatases alcalines (PAL) sont des enzymes fabriquées par plusieurs tissus de l’organisme et plus particulièrement par le foie, les os, l’intestin et le placenta lors de la grossesse. Parfois bénin, un **_taux de phosphatases alcalines élevé sert également au diagnostic de maladies du foie et des os_**.
                    Un **_taux normal de phosphatases alcalines se situe entre : 
                    30 et 125 UI/L chez l’adulte,
                    70 et 450 UI/L chez l’enfant et l’adolescent_**.
                    Chez la femme enceinte, le taux de PAL tend à augmenter.
                    
                    - **Alamine Aminotransferase (Taux d'alanine aminotransférase dans le sang - ALT) en U/L** : 
                    L'alamine aminotransférase est un enzyme nécessaire au bon fonctionnement de l'organisme. On le retrouve à divers endroits dans le corps, dont les muscles, le cœur, les reins et le foie. C'est au niveau du foie qu'on en retrouve les plus grandes quantités. **_L'augmentation de cette enzyme dans le sang est presque toujours associée à une atteinte hépatique_**.
                    Les valeurs cibles sont les suivantes : 
                    **_chez l'homme : 10 à 32 U/L_**
                    **_chez la femme : 9 à 24 U/L_**
                    **_Son augmentation dans le plasma sanguin signe une cytolyse hépatique_**.
                    
                    - **Total protiens en g/dl** : Les protéines totales dans le sang comprennent deux types principaux de protéines : l'albumine et les globulines.
                    **_Les valeurs normales varient généralement entre 6,0 et 8,3 g/dL._** 
                    
                    - **Albumin and Globulin Ratio (Rapport de l'albumine à la globuline dans le sang)** : 
                    **_L'Albumine_** aide à empêcher le sang de s'échapper des vaisseaux sanguins. Elle aide également à déplacer les hormones, les médicaments, les vitamines et d’autres substances importantes dans tout le corps.
                    **_Les Globulines_** aident à combattre les infections et à déplacer les nutriments dans tout le corps.
                    Dans des conditions normales, le rapport albumine/globuline (A/G) est généralement supérieur à 1, c'est-à-dire que la concentration d'albumine est plus élevée que celle des globulines. **_Les valeurs typiques se situent généralement entre 1,2 et 2,1._**
                    **_Une diminution de la production d'albumine par le foie peut entraîner une diminution du rapport A/G._**
                    """
                )
    
            # Afficher une série de graphiques
            st.subheader("**Graphiques :**")

            selected_graph_type = st.selectbox("Sélectionnez le type de graphique", ["Boxplot", "Heatmap", "Analyse en Composantes Principales"])

            # Boxplots Maladies du foie
            if selected_graph_type == "Boxplot":

                st.write("""Les boîtes à moustache ci-dessous représentent la distribution
                            des patients malades et sains en fonction des variables étudiées. 
                            Si la valeur p du test de Kruskal Wallis est inférieure au seuil de signification, le plus souvent 5 %, l'hypothèse nulle est rejetée et on admet qu'il y a des différences significatives entre les deux échantillons.
                            """)
    
                def plot_boxplots(foie):
                    mapping_foie = {0 : 'sain', 1 : 'malade'}
                    foie['Dataset'] = foie['Dataset'].map(mapping_foie)
                    foie['Dataset'] = foie['Dataset'].astype(str)

                    liste_variables_foie = foie.columns[:-1].tolist()
                    col = 2
                    rows = 5

                    fig, axes = plt.subplots(rows, col, figsize=(18, 35))
                    axes = axes.flatten()

                    fig.suptitle('Boxplots des variables sur le diagnostic des maladies du foie - Test de Kruskal-Wallis', fontsize=30)

                    for i, variable in enumerate(liste_variables_foie):
                        groups = [foie[variable][foie['Dataset'] == level] for level in foie['Dataset']]

                        # Effectuer le test de Kruskal-Wallis
                        h_stat, p_value = kruskal(*groups)

                        # Tracer le boxplot
                        ax = sns.boxplot(data=foie, x='Dataset', y=variable, ax=axes[i])
                        axes[i].set_xlabel('Dataset', fontsize = 25)
                        axes[i].set_ylabel(variable, fontsize=25, color = 'blue')

                        # Augmenter la taille des ticks sur l'axe des x
                        ax.tick_params(axis='x', labelsize=20)  # Définir la taille des ticks sur l'axe des x
                        ax.tick_params(axis='y', labelsize=20)

                        # Ajouter l'annotation pour la p-valeur avec statannot
                        add_stat_annotation(ax, data=foie, x='Dataset', y=variable,
                                            box_pairs=[('sain', 'malade')],  # Comparaison entre les deux niveaux de Dataset
                                            test='Kruskal', text_format='full', loc='outside',
                                            verbose=2, fontsize= 20)

                    plt.tight_layout(rect=[0, 0, 1, 0.95])

                    # Afficher le graphique dans Streamlit
                    st.pyplot(fig)

                # Appel de la fonction pour créer et afficher le graphique
                plot_boxplots(foie)

            # Heatmap Maladies du foie
            elif selected_graph_type == "Heatmap":
                # Définir la fonction pour créer le graphique
                def plot_heatmap(foie):
                    fig, ax = plt.subplots(figsize = (15,13))
                    sns.heatmap(foie.select_dtypes('number').corr(), center=0, cmap="coolwarm")
                    ax.set_title("Heatmap de la corrélation entre les variables du foie", fontsize = 27)
                    ax.tick_params(axis='x', labelsize=20, rotation=90)  # Taille des ticks sur l'axe des x
                    ax.tick_params(axis='y', labelsize=20, rotation=0)  # Taille des ticks sur l'axe des y
                    ax.legend(fontsize = 22)
                    st.pyplot(fig)  # Afficher le graphique dans Streamlit

                # Appel de la fonction pour créer et afficher le graphique
                plot_heatmap(foie)

                st.write("""Les corrélations les plus visibles sont entre les variables "Alamine aminotransférase" et "aspartate aminotransférase".
                        Il existe également des corrélations entre "Albumin", "Total Protiens" et "Albumin & Globulin Ratio".
                        """)
                
            # ACP Maladies du foie
            elif selected_graph_type == "Analyse en Composantes Principales":
                st.image("ACP liver.png", use_column_width=True)


    ## SECTION PREDICTION MALADIES ##    
    elif selected_section == "Prédiction":
        st.sidebar.subheader("Sélectionnez une maladie")
        selected_disease = st.sidebar.selectbox("info", ["Diabète", "Maladies cardiaques", "Maladies rénales", "Cancer du sein", "Maladies du foie"], label_visibility="collapsed")
        display_prediction(selected_disease)
    
        ## PREDICTION DIABETE ## 
        if selected_disease == "Diabète":
    
            # Charger les données
            file_path = "df_diabete_fin (1).csv"
            diabete = load_data(file_path)

            # Features and target variable
            feature_columns1 = ['Pregnancies', 'Glucose', 'BloodPressure', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            predicted_class1 = 'Outcome'

            # Split the data
            X = diabete[feature_columns1].values
            y = diabete[predicted_class1].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=60)

            # Train your model
            random_forest_model = RandomForestClassifier(min_weight_fraction_leaf=0.05, random_state = 60)
            random_forest_model.fit(X_train, y_train)

            # Input fields
            st.subheader("_Entrez les informations du patient_ :")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>Pregnancies :</h6>", unsafe_allow_html=True)
            Pregnancies = st.slider("Pregnancies", min_value=0.0, max_value=17.0, step=1.0, value=4.0, label_visibility="collapsed")
            st.markdown("<h6 style='font-size: 18px;'>Glucose :</h6>", unsafe_allow_html=True)
            Glucose = st.slider("Glucose", min_value=40.0, max_value=200.0, step=1.0, value=120.0, label_visibility="collapsed")
            st.markdown("<h6 style='font-size: 18px;'>Blood pressure :</h6>", unsafe_allow_html=True)
            BloodPressure = st.slider("Blood pressure", min_value=20.0, max_value=130.0, step=1.0, value=72.0, label_visibility="collapsed")
            st.markdown("<h6 style='font-size: 18px;'>Insulin :</h6>", unsafe_allow_html=True)
            Insulin = st.number_input("Insulin", min_value=10.0, max_value=1000.0, step=0.01, value=157.0, label_visibility="collapsed")
            st.markdown("<h6 style='font-size: 18px;'>BMI :</h6>", unsafe_allow_html=True)
            BMI = st.number_input("BMI", min_value=10.0, max_value=100.0, step=0.01, value=32.0, label_visibility="collapsed")
            st.markdown("<h6 style='font-size: 18px;'>DiabetesPedigreeFonction :</h6>", unsafe_allow_html=True)
            DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFonction", min_value=0.0, max_value=3.0, step=0.01, value=0.50, label_visibility="collapsed")
            st.markdown("<h6 style='font-size: 18px;'>Age :</h6>", unsafe_allow_html=True)
            Age = st.slider("Age", min_value=5, max_value=100, step=1, value=33, label_visibility="collapsed")

            # Make prediction
            input_data = [[Pregnancies, Glucose, BloodPressure, Insulin, BMI, DiabetesPedigreeFunction, Age]]
            prediction = random_forest_model.predict(input_data)
            predicted_probabilities = random_forest_model.predict_proba(input_data)[0]
            
            # Display result
            st.subheader("_Prédiction_ :")
            if prediction[0] == 1:
                st.markdown("<h3 style='font-size: 25px;'>Avec les informations renseignées, le modèle prédit que le patient est susceptible d'être atteint d'un diabète.</h3>", unsafe_allow_html=True)
                st.image("https://www.arsenre.com/wp-content/uploads/2022/07/85477-580-doctor-6810751_640.png", use_column_width=True)   
            else:
                st.markdown("<h3 style='font-size: 25px;'>Avec les informations renseignées, le modèle prédit que le patient n'est pas atteint d'un diabète.</h3>", unsafe_allow_html=True)
                st.image("diabète.png", use_column_width=True)

            # Afficher la fiabilité de la prédiction
            st.subheader("_Fiabilité de la prédiction_ :")
            if prediction[0] == 1:
                st.markdown(f"<h3 style='font-size: 25px;'>Probabilité que le patient soit atteint de diabète : {round(predicted_probabilities[1]*100,2)} %</h3>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h3 style='font-size: 25px;'>Probabilité que le patient ne soit pas atteint de diabète : {round(predicted_probabilities[0]*100,2)} %</h3>", unsafe_allow_html=True)
            st.write("*La prédiction est basée sur le modèle Random Forest Classifier (cf Performances des modèles).*")
                
        ## PREDICTION MALADIES CARDIAQUES ##
        if selected_disease == "Maladies cardiaques":
            file_path = "dfheartML.csv"
            coeur = load_data(file_path)

            # Features and target variable
            X = coeur[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca' , 'thal']]
            y = coeur['target']

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 42)

            # Génération d'un scaler et application sur X
            scaler_lr = StandardScaler()
            X_train_scaled = scaler_lr.fit_transform(X_train)
            X_test_scaled = scaler_lr.transform(X_test)

            # Train your model
            logistic_regression_model_lrcr = LogisticRegression()
            logistic_regression_model_lrcr.fit(X_train_scaled, y_train)

            # Input fields
            st.subheader("_Entrez les informations du patient_ :")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>Age :</h6>", unsafe_allow_html=True)
            age = st.slider("Age", min_value=0.0, max_value=100.0, step=1.0, value=54.0, label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>Sex (0 : male, 1 : female):</h6>", unsafe_allow_html=True)
            sex = st.selectbox("Sex", [0.0, 1.0], label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>cp :</h6>", unsafe_allow_html=True)
            cp = st.slider("cp", min_value=0.0, max_value=5.0, step=1.0, value=1.0, label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>trestbps :</h6>", unsafe_allow_html=True)
            trestbps = st.slider("trestbps", min_value=50.0, max_value=250.0, step=1.0, value=131.0, label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>chol :</h6>", unsafe_allow_html=True)
            chol = st.slider("chol", min_value=100.0, max_value=750.0, step=1.0, value=246.0, label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>fbs (0 : No, 1 : Yes) :</h6>", unsafe_allow_html=True)
            fbs = st.selectbox("fbs", [0.0, 1.0], label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>restecg :</h6>", unsafe_allow_html=True)
            restecg = st.slider("restecg", min_value=0.0, max_value=3.0, step=1.0, value=1.0, label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>thalach :</h6>", unsafe_allow_html=True)
            thalach = st.slider("thalach", min_value=50.0, max_value=250.0, step=1.0, value=149.0, label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>exang (0 : absence, 1 : presence) :</h6>", unsafe_allow_html=True)
            exang = st.selectbox("exang", [0.0, 1.0], label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>oldpeak :</h6>", unsafe_allow_html=True)
            oldpeak = st.number_input("oldpeak", min_value=0.0, max_value=10.0, step=0.1, value=1.0, label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>slope :</h6>", unsafe_allow_html=True)
            slope = st.slider("slope", min_value=0.0, max_value=3.0, step=1.0, value=0.0, label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>ca :</h6>", unsafe_allow_html=True)
            ca = st.slider("ca", min_value=0.0, max_value=5.0, step=1.0, value=0.0, label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>thal :</h6>", unsafe_allow_html=True)
            thal = st.slider("thal", min_value=0.0, max_value=5.0, step=1.0, value=0.0, label_visibility="collapsed")

            # Make prediction
            input_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca , thal]]
            input_data_scaled = scaler_lr.transform(input_data)
            prediction = logistic_regression_model_lrcr.predict(input_data_scaled)
            predicted_probabilities = logistic_regression_model_lrcr.predict_proba(input_data_scaled)[0]

            # Display result
            st.subheader("_Prédiction_ :")
            if prediction[0] == 1:
                st.markdown("<h3 style='font-size: 25px;'>Avec les informations renseignées, le modèle prédit que le patient est susceptible d'être atteint d'une maladie cardiaque.</h3>", unsafe_allow_html=True)
                st.image("https://www.arsenre.com/wp-content/uploads/2022/07/85477-580-doctor-6810751_640.png", use_column_width=True)
            else:
                st.markdown("<h3 style='font-size: 25px;'>Avec les informations renseignées, le modèle prédit que le patient n'est pas atteint d'une maladie cardiaque.</h3>", unsafe_allow_html=True)
                st.image("https://haltemis.fr/wp-content/uploads/2022/01/healthy-lifestyle.png", use_column_width=True)

            # Afficher la fiabilité de la prédiction
            st.subheader("_Fiabilité de la prédiction_ :")
            if prediction[0] == 1:
                st.markdown(f"<h3 style='font-size: 25px;'>Probabilité que le patient soit atteint d'une maladie cardiaque : {round(predicted_probabilities[1]*100,2)} %</h3>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h3 style='font-size: 25px;'>Probabilité que le patient ne soit pas atteint d'une maladie cardiaque : {round(predicted_probabilities[0]*100,2)} %</h3>", unsafe_allow_html=True)
            st.write("*La prédiction est basée sur le modèle Logistic Regression (cf Performances des modèles).*")

        ## PREDICTION CANCER DU SEIN ##
        if selected_disease == "Cancer du sein":
            file_path = "dfcancersein.csv"
            sein = load_data(file_path)

            # Features and target variable
            feature_columns1 = ['radius_mean','texture_mean','smoothness_mean','compactness_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se','smoothness_se','compactness_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','smoothness_worst','compactness_worst','symmetry_worst','fractal_dimension_worst']

            # Split the data
            X = sein[feature_columns1].values
            y = sein.iloc[:, 0].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)

            # Application d'un scaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train your model
            log_reg_model = LogisticRegression()
            log_reg_model.fit(X_train_scaled, y_train)

            # Input fields
            st.subheader("_Entrez les informations du patient_ :")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>radius_mean :</h6>", unsafe_allow_html=True)
            radius_mean = st.number_input("radius_mean", min_value=7.0000, max_value=30.0000, step=0.0100, value=14.0000, format="%.4f", label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>texture_mean :</h6>", unsafe_allow_html=True)
            texture_mean = st.number_input("texture_mean", min_value=8.0000, max_value=50.0000, step=0.0100, value=19.0000, format="%.4f", label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>smoothness_mean :</h6>", unsafe_allow_html=True)
            smoothness_mean = st.number_input("smoothness_mean", min_value=0.0000, max_value=1.0000, step=0.0010, value=0.1000, format="%.4f", label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>compactness_mean :</h6>", unsafe_allow_html=True)
            compactness_mean = st.number_input("compactness_mean", min_value=0.0000, max_value=1.0000, step=0.0010, value=0.1000, format="%.4f", label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>symmetry_mean :</h6>", unsafe_allow_html=True)
            symmetry_mean = st.number_input("symmetry_mean", min_value=0.0000, max_value=0.5000, step=0.0010, value=0.2000, format="%.4f", label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>fractal_dimension_mean :</h6>", unsafe_allow_html=True)
            fractal_dimension_mean = st.number_input("fractal_dimension_mean", min_value=0.0000, max_value=0.1000, step=0.0010, value=0.0600, format="%.4f", label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>radius_se :</h6>", unsafe_allow_html=True)
            radius_se = st.number_input("radius_se", min_value=0.0000, max_value=3.0000, step=0.0100, value=0.4000, format="%.4f", label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>texture_se :</h6>", unsafe_allow_html=True)
            texture_se = st.number_input("texture_se", min_value=0.0000, max_value=4.0000, step=0.0100, value=1.2000, format="%.4f", label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>smoothness_se :</h6>", unsafe_allow_html=True)
            smoothness_se = st.number_input("smoothness_se", min_value=0.0000, max_value=0.1000, step=0.0001, value=0.0100, format="%.4f", label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>compactness_se :</h6>", unsafe_allow_html=True)
            compactness_se = st.number_input("compactness_se", min_value=0.0000, max_value=0.2000, step=0.0010, value=0.0300, format="%.4f", label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>symmetry_se :</h6>", unsafe_allow_html=True)
            symmetry_se = st.number_input("symmetry_se", min_value=0.0000, max_value=0.1000, step=0.0010, value=0.0200, format="%.4f", label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>fractal_dimension_se :</h6>", unsafe_allow_html=True)
            fractal_dimension_se = st.number_input("fractal_dimension_se", min_value=0.0000, max_value=0.0500, step=0.0001, value=0.0030, format="%.4f", label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>radius_worst :</h6>", unsafe_allow_html=True)
            radius_worst = st.number_input("radius_worst", min_value=7.0000, max_value=40.0000, step=0.0100, value=16.3000, format="%.4f", label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>texture_worst :</h6>", unsafe_allow_html=True)
            texture_worst = st.number_input("texture_worst", min_value=10.0000, max_value=50.0000, step=0.0100, value=25.6800, format="%.4f", label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>smoothness_worst :</h6>", unsafe_allow_html=True)
            smoothness_worst = st.number_input("smoothness_worst", min_value=0.0000, max_value=0.3000, step=0.0010, value=0.1300, format="%.4f", label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>compactness_worst :</h6>", unsafe_allow_html=True)
            compactness_worst = st.number_input("compactness_worst", min_value=0.0000, max_value=1.5000, step=0.0100, value=0.2500, format="%.4f", label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>symmetry_worst :</h6>", unsafe_allow_html=True)
            symmetry_worst = st.number_input("symmetry_worst", min_value=0.0000, max_value=0.7000, step=0.0100, value=0.2900, format="%.4f", label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>fractal_dimension_worst :</h6>", unsafe_allow_html=True)
            fractal_dimension_worst = st.number_input("fractal_dimension_worst", min_value=0.0000, max_value=1.0000, step=0.0100, value=0.0800, format="%.4f", label_visibility="collapsed")
    
            # Make prediction
            input_data = [[radius_mean, texture_mean, smoothness_mean, compactness_mean, symmetry_mean, fractal_dimension_mean, radius_se,texture_se, smoothness_se, compactness_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst, smoothness_worst, compactness_worst, symmetry_worst, fractal_dimension_worst]]
            input_data_scaled = scaler.transform(input_data)
            prediction = log_reg_model.predict(input_data_scaled)
            predicted_probabilities = log_reg_model.predict_proba(input_data_scaled)[0]

            # Display result
            st.subheader("_Prédiction_ :")
            if prediction[0] == 0:
                st.markdown("<h3 style='font-size: 25px;'>Avec les informations renseignées, le modèle prédit que le patient est susceptible d'être atteint d'un cancer du sein.</h3>", unsafe_allow_html=True)
                st.image("https://www.arsenre.com/wp-content/uploads/2022/07/85477-580-doctor-6810751_640.png", use_column_width=True)
            else:
                st.markdown("<h3 style='font-size: 25px;'>Avec les informations renseignées, le modèle prédit que le patient n'est pas atteint d'un cancer du sein.</h3>", unsafe_allow_html=True)
                st.image("https://onconormandie.fr/wp-content/uploads/2023/09/Design-sans-titre-7.png", use_column_width=True)

            # Afficher la fiabilité de la prédiction
            st.subheader("_Fiabilité de la prédiction_ :")
            if prediction[0] == 0:
                st.markdown(f"<h3 style='font-size: 25px;'>Probabilité que le patient soit atteint d'un cancer du sein : {round(predicted_probabilities[0]*100,2)} %</h3>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h3 style='font-size: 25px;'>Probabilité que le patient ne soit pas atteint d'un cancer du sein : {round(predicted_probabilities[1]*100,2)} %</h3>", unsafe_allow_html=True)
            st.write("*La prédiction est basée sur le modèle Logistic Regression (cf Performances des modèles).*")

        ## PREDICTION MALADIE REIN ##
        if selected_disease == "Maladies rénales":
            file_path = "dfrein_ML.csv"
            rein = load_data(file_path)

            # Features and target variable
            X = rein[['age', 'bp', 'sg', 'al', 'su', 'rc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rbc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']]
            y = rein.iloc[:, -1]

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.38, random_state=42)

            # Train your model
            Logistic_regression = LogisticRegression(solver = 'liblinear')
            Logistic_regression.fit(X_train, y_train)

            # Input fields
            st.subheader("_Entrez les informations du patient_ :")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>Age (age) :</h6>", unsafe_allow_html=True)
            age = st.slider("Age", min_value=1.0, max_value=110.0, step=1.0, value=51.0, label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>Blood pressure (bp) :</h6>", unsafe_allow_html=True)
            bp = st.slider("Blood pressure", min_value=30.0, max_value=130.0, step=1.0, value=75.0, label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>Specific Gravity (sg) :</h6>", unsafe_allow_html=True)
            sg = st.number_input("Specific Gravity", min_value=1.0, max_value=2.0, step=0.01, value=1.0, label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>Albumin (al) :</h6>", unsafe_allow_html=True)
            al = st.slider("Albumin", min_value=0.0, max_value=6.0, step=0.5, value=1.0, label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>Sugar (su) :</h6>", unsafe_allow_html=True)
            su = st.slider("Sugar", min_value=0.0, max_value=6.0, step=1.0, value=0.0, label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>Red Blood Cells (rbc) - (0 : normal, 1 : abnormal) :</h6>", unsafe_allow_html=True)
            rbc = st.selectbox("Red cells", [0.0, 1.0], label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style ='font-size: 18px;'>Pus Cells (pc) - (0 : normal, 1 : abnormal) :</h6>", unsafe_allow_html=True)
            pc= st.selectbox("Pus cells", [0.0, 1.0], label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style ='font-size: 18px;'>Pus Cells Clumps (pcc) - (0 : normal, 1 : abnormal) :</h6>", unsafe_allow_html=True)
            pcc= st.selectbox("Pus cells clumps", [0.0, 1.0], label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style ='font-size: 18px;'>Bacteria (ba) - (0 : normal, 1 : abnormal) :</h6>", unsafe_allow_html=True)
            ba= st.selectbox("Bacteria", [0.0, 1.0], label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>Blood Glucose Random (bgr) :</h6>", unsafe_allow_html=True)
            bgr = st.slider("Blood Glucose Random", min_value=15.0, max_value=520.0, step=1.0, value=143.0, label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>Blood Urea (bu) :</h6>", unsafe_allow_html=True)
            bu = st.number_input("Blood Urea", min_value=0.4, max_value=350.0, step=0.01, value=55.0, label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>Serum Creatinine (sc) :</h6>", unsafe_allow_html=True)
            sc = st.number_input("Serum creatinine", min_value=0.2, max_value=40.0, step=0.1, value=2.5, label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>Sodium (sod) :</h6>", unsafe_allow_html=True)
            sod = st.slider("Sodium", min_value=90.0, max_value=180.0, step=1.0, value=138.0, label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>Potassium (pot) :</h6>", unsafe_allow_html=True)
            pot = st.number_input("Potassium", min_value=1.0, max_value=55.0, step=0.1, value=5.0, label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>Hemoglobin (hemo) :</h6>", unsafe_allow_html=True)
            hemo = st.number_input("Hémoglobine", min_value=1.0, max_value=30.0, step=0.1, value=13.0, label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>Packed Cell Volume (pcv) :</h6>", unsafe_allow_html=True)
            pcv = st.number_input("Packed Cell Volume", min_value=2.0, max_value=70.0, step=0.1, value=39.0, label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>White Blood Cell Count (wc) :</h6>", unsafe_allow_html=True)
            wc = st.number_input('White cell count', min_value=2000.0, max_value=30000.0, value=8449.0, step=1.0, label_visibility = 'collapsed')
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>Red Blood Cell Count (rc) :</h6>", unsafe_allow_html=True)
            rc = st.number_input('White cell count', min_value=1.50, max_value=10.00, value=4.68, step=0.01, label_visibility = 'collapsed')
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>Hypertension (htn) - (0 : absence, 1 : presence) :</h6>", unsafe_allow_html=True)
            htn = st.selectbox("Hypertension", [0.0, 1.0], label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>Diabetes Mellitus (dm) - (0 : absence, 1 : presence) :</h6>", unsafe_allow_html=True)
            dm = st.selectbox("Diabetes Mellitus", [0.0, 1.0], label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>Coronary Artery Disease (cad) - (0 : absence, 1 : presence) :</h6>", unsafe_allow_html=True)
            cad = st.selectbox("Coronary Artery Disease", [0.0, 1.0], label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>Appetite (appet) - (0 : poor, 1 : good) :</h6>", unsafe_allow_html=True)
            appet = st.selectbox("Appetite", [0.0, 1.0], label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>Pedal Edema (pe) - (0 : absence, 1 : presence) :</h6>", unsafe_allow_html=True)
            pe = st.selectbox("Pedal Edema", [0.0, 1.0], label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>Anemia (ane) - (0 : absence, 1 : presence) :</h6>", unsafe_allow_html=True)
            ane = st.selectbox("Anemia", [0.0, 1.0], label_visibility="collapsed")
            st.write("")

            # Make prediction
            input_data = [[age, bp, sg, al, su, rc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rbc, htn, dm, cad, appet, pe, ane]]
            prediction = Logistic_regression.predict(input_data)
            predicted_probabilities = Logistic_regression.predict_proba(input_data)[0]

            # Display result
            st.subheader("_Prédiction_ :")
            if prediction[0] == 1:
                st.markdown("<h3 style='font-size: 25px;'>Avec les informations renseignées, le modèle prédit que le patient est susceptible d'être atteint d'une maladie rénale.</h3>", unsafe_allow_html=True)
                st.image("https://www.arsenre.com/wp-content/uploads/2022/07/85477-580-doctor-6810751_640.png", use_column_width=True)
            else:         
                st.markdown("<h3 style='font-size: 25px;'>Avec les informations renseignées, le modèle prédit que le patient n'est pas atteint d'une maladie rénale.</h3>", unsafe_allow_html=True)
                st.image("https://us.123rf.com/450wm/kahovsky/kahovsky1801/kahovsky180100062/92916263-%E3%81%8B%E3%82%8F%E3%81%84%E3%81%84%E9%9D%A2%E7%99%BD%E3%81%84%E3%80%81%E7%AC%91%E9%A1%94%E3%81%AE%E5%8C%BB%E8%80%85%E3%81%A8%E5%81%A5%E5%BA%B7%E5%B9%B8%E3%81%9B%E3%81%AA%E8%82%BA%E3%81%AE%E3%82%A4%E3%83%A9%E3%82%B9%E3%83%88%E3%80%82.jpg", use_column_width=True)
    
            # Afficher la fiabilité de la prédiction
            st.subheader("_Fiabilité de la prédiction_ :")
            if prediction[0] == 1:
                st.markdown(f"<h3 style='font-size: 25px;'>Probabilité que le patient soit atteint d'une maladie rénale : {round(predicted_probabilities[1]*100,2)} %</h3>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h3 style='font-size: 25px;'>Probabilité que le patient ne soit pas atteint d'une maladie rénale : {round(predicted_probabilities[0]*100,2)} %</h3>", unsafe_allow_html=True)
            st.write("*La prédiction est basée sur le modèle Logistic Regression (cf Performances des modèles).*")

        ## PREDICTION MALADIES DU FOIE ##
        if selected_disease == "Maladies du foie":
            file_path = "df_liver_M0F1 .csv"
            foie = load_data(file_path)

            # Features and target variable
            X = foie[['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio']]
            y = foie.iloc[:, -1]

            # random_state
            random_state = 42

            # Split the data
            test_size=0.25
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            # Sampling strategy
            sampling_strategy = {0 : 400, 1 : 400}
            X_resampled, y_resampled = SMOTE(sampling_strategy = sampling_strategy, random_state = random_state).fit_resample(X_train, y_train)

            # Train your model
            RandomForest = RandomForestClassifier(random_state = random_state)
            RandomForest.fit(X_resampled, y_resampled)

            # Input fields
            st.subheader("_Entrez les informations du patient_ :")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>Age :</h6>", unsafe_allow_html=True)
            Age = st.slider("Age", min_value=0.0, max_value=125.0, step=1.0, value=43.0, label_visibility = 'collapsed')
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>Gender (0 : male, 1 : female) :</h6>", unsafe_allow_html=True)
            Gender = st.selectbox("Gender", [0.0, 1.0], label_visibility="collapsed")
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>Total bilirubin :</h6>", unsafe_allow_html=True)
            Total_Bilirubin = st.number_input('Total bilirubin', min_value=0.0, max_value=40.0, value=3.12, step=0.01, label_visibility = 'collapsed')
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>Direct bilirubin :</h6>", unsafe_allow_html=True)
            Direct_Bilirubin = st.number_input('Direct bilirubin', min_value=0.0, max_value=40.0, value=1.49, step=0.01, label_visibility = 'collapsed')
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>Alkaline phosphotase :</h6>", unsafe_allow_html=True)
            Alkaline_Phosphotase = st.number_input("Alkaline phosphotase", min_value=0.0, max_value=2200.0, step=0.01, value=291.29, label_visibility = 'collapsed')
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>Alamine aminotransferase :</h6>", unsafe_allow_html=True)
            Alamine_Aminotransferase = st.number_input("Alamine aminotransferase", min_value=0.0, max_value=2000.0, step=0.01, value=74.66, label_visibility = 'collapsed')
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>Aspartate aminotransférase :</h6>", unsafe_allow_html=True)
            Aspartate_Aminotransferase = st.number_input("Aspartate aminotransferase", min_value=0.0, max_value=2000.0, step=0.01, value=96.32, label_visibility = 'collapsed')
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>Total Protiens :</h6>", unsafe_allow_html=True)
            Total_Protiens = st.number_input("Total Protiens", min_value=0.0, max_value=15.0, step=0.01, value=6.50, label_visibility = 'collapsed')
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>Albumin :</h6>", unsafe_allow_html=True)
            Albumin = st.number_input("Albumin", min_value=0.0, max_value=10.0, step=0.01, value=3.15, label_visibility = 'collapsed')
            st.write("")
            st.markdown("<h6 style='font-size: 18px;'>Albumin and globulin ratio :</h6>", unsafe_allow_html=True)
            Albumin_and_Globulin_Ratio = st.number_input("Albumin and globulin ratio", min_value=0.0, max_value=10.0, step=0.01, value=0.95, label_visibility = 'collapsed')

            # Make prediction
            input_data = [[Age, Gender, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase, Alamine_Aminotransferase, Aspartate_Aminotransferase, Total_Protiens, Albumin, Albumin_and_Globulin_Ratio]]
            prediction = RandomForest.predict(input_data)
            predicted_probabilities = RandomForest.predict_proba(input_data)[0]

            # Display result
            st.subheader("_Prédiction_ :")
            if prediction[0] == 1:
                st.markdown("<h3 style='font-size: 25px;'>Avec les informations renseignées, le modèle prédit que le patient est susceptible d'être atteint d'une maladie hépatique.</h3>", unsafe_allow_html=True)
                st.image("https://www.arsenre.com/wp-content/uploads/2022/07/85477-580-doctor-6810751_640.png", use_column_width=True)
            else:
                st.markdown("<h3 style='font-size: 25px;'>Avec les informations renseignées, le modèle prédit que le patient n'est pas atteint d'une maladie hépatique.</h3>", unsafe_allow_html=True)
                st.image("foie sain.png", use_column_width=True)

            # Afficher la fiabilité de la prédiction
            st.subheader("_Fiabilité de la prédiction_ :")
            if prediction[0] == 1:
                st.markdown(f"<h3 style='font-size: 25px;'>Probabilité que le patient soit atteint d'une maladie hépatique : {round(predicted_probabilities[1]*100,2)} %</h3>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h3 style='font-size: 25px;'>Probabilité que le patient ne soit pas atteint d'une maladie hépatique : {round(predicted_probabilities[0]*100,2)} %</h3>", unsafe_allow_html=True)
            st.write("*La prédiction est basée sur le modèle Random Forest Classifier (cf Performances des modèles).*")

    ## SECTION PERFORMANCES MODELES ##
    
    elif selected_section == "Performances des modèles":
        st.sidebar.subheader("Sélectionnez une maladie")
        selected_disease = st.sidebar.selectbox("info", ["Diabète", "Maladies cardiaques", "Maladies rénales", "Cancer du sein", "Maladies du foie"], label_visibility="collapsed")
        display_modele(selected_disease)

        ## PERFORMANCES MODELES DIABETE ##
        #Charger les données
        file_path = "df_diabete_fin (1).csv"
        diabete = load_data(file_path)

        feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        predicted_class = 'Outcome'

        X_d = diabete[feature_columns].values
        y_d = diabete[predicted_class].values
    
        if selected_disease == "Diabète":
    
            # Choix d'un modèle
            selected_model= st.selectbox("**Sélectionnez le modèle :**", ["RandomForestClassifier", "LogisticRegression"])

            # randomforestclassifier Diabète
            if selected_model == "RandomForestClassifier":

                st.subheader("_Choix des variables pour l'entrainement du modèle RandomForestClassifier_ :")

                st.markdown("""
                            **_Les variables retenues pour le modèle de prédiction sont_** :\n- Pregnancies\n- Glucose\n - Blood Pressure\n - Insulin\n- BMI\n- Diabetes Pedigree Function\n- Age
                            """)
                st.write("""
                        Les données d'entrainement et de test sont séparées telles que : **jeu de test = 30 % et jeu d'entrainement = 70 %.**
                        """)
                X_rfd_train, X_rfd_test, y_rfd_train, y_rfd_test = train_test_split(X_d, y_d, test_size = 0.30, random_state=60)
                random_forest_model_rfd = RandomForestClassifier(min_weight_fraction_leaf = 0.05, random_state=60)
                random_forest_model_rfd.fit(X_rfd_train, y_rfd_train)

                st.subheader("_Scores du modèle_ :")
                st.write("**_Les performances du modèle sont les suivantes :_**")
                train_score_rfd = random_forest_model_rfd.score(X_rfd_train, y_rfd_train)
                test_score_rfd = random_forest_model_rfd.score(X_rfd_test, y_rfd_test)
                st.write(f"- Le score pour le jeu d'entrainement est : {train_score_rfd*100:.2f}%")
                st.write(f"- Le score pour le jeu de test est : {test_score_rfd*100:.2f}%")

                st.subheader("_Rapport de classification du modèle_ :")
                predict_train_data_rfd = random_forest_model_rfd.predict(X_rfd_test)
                report_rfd = classification_report(y_rfd_test, predict_train_data_rfd, output_dict = True)
                df_report_rfd = pd.DataFrame(report_rfd).transpose()
                st.write(df_report_rfd)
                st.write("_*0 : patient sain, 1 : patient malade_")

                st.subheader("_Matrices de confusion_ :")
                # Créer deux colonnes
                col1, col2 = st.columns([1, 1])  # Rapport de largeur 1:1

                # Calculer la matrice de confusion
                cm_rfd = confusion_matrix(y_rfd_test, predict_train_data_rfd)
                df_confusion_rfd = pd.DataFrame(cm_rfd, index=['Réel Sain', 'Réel Malade'], columns=['Prédit Sain', 'Prédit Malade'])
                with col1:
                    st.write("**_Matrice de confusion en valeur absolue :_**")
                    st.write(df_confusion_rfd)

                # Calculer total malade / total sain
                total_malade_rfd = np.sum(y_rfd_test == 1)
                total_sain_rfd = np.sum(y_rfd_test == 0)
            
                # Convertir la matrice de confusion en pourcentage par rapport au total de chaque classe
                cm_percent_rfd = cm_rfd.copy().astype(float)
                cm_percent_rfd[1, :] /= total_malade_rfd
                cm_percent_rfd[0, :] /= total_sain_rfd
                cm_percent_rfd *= 100

                # Créer un DataFrame à partir de la matrice de confusion en pourcentage
                df_confusion_percent_rfd = pd.DataFrame(cm_percent_rfd, index=['Réel Sain %', 'Réel Malade %'], columns=['Prédit Sain %', 'Prédit Malade %'])

                # Afficher le DataFrame dans Streamlit
                with col2:
                    st.write("**_Matrice de confusion en pourcentage :_**")
                    st.write(df_confusion_percent_rfd)

                st.subheader("_Conclusion_ :")
                st.write("Random Forest sera le modèle retenu pour la prédiction.")

            # Logistic Regression Diabète
            elif selected_model == "LogisticRegression":

                st.subheader("_Choix des variables pour l'entrainement du modèle LogisticRegression_ :")

                st.markdown("""
                            **_Les variables retenues pour le modèle de prédiction sont_** :\n- Pregnancies\n- Glucose\n - Blood Pressure\n - Insulin\n- BMI\n- Diabetes Pedigree Function\n- Age
                            """)

                st.write("""
                        Les données d'entrainement et de test sont séparées telles que : **jeu de test = 20 % et jeu d'entrainement = 80 %, avec répartition des poids des classes selon 'poids sain' = 0.77 et 'poids malade' = 1,43.
                        Les variables prédictives sont standardisées**.
                        """)
                X_lrd_train, X_lrd_test, y_lrd_train, y_lrd_test = train_test_split(X_d, y_d, test_size = 0.2, random_state=60)
                
                # Calculer le poids pour la classe 0
                weight_0 = len(y_lrd_train) / (2 * np.bincount(y_lrd_train)[0])
                # Calculer le poids pour la classe 1
                weight_1 = len(y_lrd_train) / (2 * np.bincount(y_lrd_train)[1])

                # Créer un dictionnaire de poids de classe
                class_weights = {0: weight_0, 1: weight_1}

                # Créer un standard scaler pour les variables X_lrd
                scaler_LR = StandardScaler()

                # Fit the scaler to training data and transform training data
                X_lrd_train_scaled = scaler_LR.fit_transform(X_lrd_train)

                # Transform test data using the fitted scaler
                X_lrd_test_scaled = scaler_LR.transform(X_lrd_test) 
                
                LogisticRegression_model_lrd = LogisticRegression(class_weight = class_weights)
                LogisticRegression_model_lrd.fit(X_lrd_train_scaled, y_lrd_train)

                st.subheader("_Scores du modèle_ :")
                st.write("**_Les performances du modèle sont les suivantes :_**")
                train_score_lrd = LogisticRegression_model_lrd.score(X_lrd_train_scaled, y_lrd_train)
                test_score_lrd = LogisticRegression_model_lrd.score(X_lrd_test_scaled, y_lrd_test)
                st.write(f"- Le score pour le jeu d'entrainement est : {train_score_lrd*100:.2f}%")
                st.write(f"- Le score pour le jeu de test est : {test_score_lrd*100:.2f}%")

                st.subheader("_Rapport de classification du modèle_ :")
                predict_train_data_lrd = LogisticRegression_model_lrd.predict(X_lrd_test_scaled)
                report_lrd = classification_report(y_lrd_test, predict_train_data_lrd, output_dict = True)
                df_report_lrd = pd.DataFrame(report_lrd).transpose()
                st.write(df_report_lrd)
                st.write("_*0 : patient sain, 1 : patient malade_")

                st.subheader("_Matrices de confusion_ :")
                # Créer deux colonnes
                col1, col2 = st.columns([1, 1])  # Rapport de largeur 1:1

                # Calculer la matrice de confusion
                cm_lrd = confusion_matrix(y_lrd_test, predict_train_data_lrd)
                df_confusion_lrd = pd.DataFrame(cm_lrd, index=['Réel Sain', 'Réel Malade'], columns=['Prédit Sain', 'Prédit Malade'])
                with col1:
                    st.write("**_Matrice de confusion en valeur absolue :_**")
                    st.write(df_confusion_lrd)

                # Calculer total malade / total sain
                total_malade_lrd = np.sum(y_lrd_test == 1)
                total_sain_lrd = np.sum(y_lrd_test == 0)
            
                # Convertir la matrice de confusion en pourcentage par rapport au total de chaque classe
                cm_percent_lrd = cm_lrd.copy().astype(float)
                cm_percent_lrd[1, :] /= total_malade_lrd
                cm_percent_lrd[0, :] /= total_sain_lrd
                cm_percent_lrd *= 100

                # Créer un DataFrame à partir de la matrice de confusion en pourcentage
                df_confusion_percent_lrd = pd.DataFrame(cm_percent_lrd, index=['Réel Sain %', 'Réel Malade %'], columns=['Prédit Sain %', 'Prédit Malade %'])

                # Afficher le DataFrame dans Streamlit
                with col2:
                    st.write("**_Matrice de confusion en pourcentage :_**")
                    st.write(df_confusion_percent_lrd)

                st.subheader("_Conclusion_ :")
                st.write("La régression logistique est moins performante que Random Forest dans la prédiction du diabète.")


        ## PERFORMANCES MODELES MALADIES CARDIAQUES ##
        #Charger les données
        file_path = "dfheartML.csv"
        coeur = load_data(file_path)

        feature_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca' , 'thal']
        predicted_class = 'target'

        X_cr = coeur[feature_columns].values
        y_cr = coeur[predicted_class].values
    
        if selected_disease == "Maladies cardiaques":
    
            # Choix d'un modèle
            selected_model= st.selectbox("**Sélectionnez le modèle :**", ["RandomForestClassifier", "GradientBoostingClassifier", "LogisticRegression"])

            # randomforestclassifier maladies cardiaques
            if selected_model == "RandomForestClassifier":

                st.subheader("_Choix des variables pour l'entrainement du modèle RandomForestClassifier_ :")

                st.markdown("""
                            **_Les variables retenues pour le modèle de prédiction sont_** :\n- Age\n- Sex\n - Chest Pain Type\n - Resting Blood Pressure\n- Chlolestérol\n- Fasting Blood Sugar\n- Resting Electrocardiographic Results\n - Thalach (Maximum Heart Rate Achieved)\n - Exercice Induced Angina\n - Oldpeak\n - Slope\n - Ca (Number of Major Vessels Colored by Fluoroscopie)\n - Thalassemia
                            """)
                st.write("""
                        Les données d'entrainement et de test sont séparées telles que : **jeu de test = 25 % et jeu d'entrainement = 75 %.**
                        """)
                X_rfcr_train, X_rfcr_test, y_rfcr_train, y_rfcr_test = train_test_split(X_cr, y_cr, test_size = 0.25, random_state=0)
                random_forest_model_rfcr = RandomForestClassifier()
                random_forest_model_rfcr.fit(X_rfcr_train, y_rfcr_train)

                st.subheader("_Scores du modèle_ :")
                st.write("**_Les performances du modèle sont les suivantes :_**")
                train_score_rfcr = random_forest_model_rfcr.score(X_rfcr_train, y_rfcr_train)
                test_score_rfcr = random_forest_model_rfcr.score(X_rfcr_test, y_rfcr_test)
                st.write(f"- Le score pour le jeu d'entrainement est : {train_score_rfcr*100:.2f}%")
                st.write(f"- Le score pour le jeu de test est : {test_score_rfcr*100:.2f}%")

                st.subheader("_Rapport de classification du modèle_ :")
                predict_train_data_rfcr = random_forest_model_rfcr.predict(X_rfcr_test)
                report_rfcr = classification_report(y_rfcr_test, predict_train_data_rfcr, output_dict = True)
                df_report_rfcr = pd.DataFrame(report_rfcr).transpose()
                st.write(df_report_rfcr)
                st.write("_*0 : patient sain, 1 : patient malade_")

                st.subheader("_Matrices de confusion_ :")
                # Créer deux colonnes
                col1, col2 = st.columns([1, 1])  # Rapport de largeur 1:1

                # Calculer la matrice de confusion
                cm_rfcr = confusion_matrix(y_rfcr_test, predict_train_data_rfcr)
                df_confusion_rfcr = pd.DataFrame(cm_rfcr, index=['Réel Sain', 'Réel Malade'], columns=['Prédit Sain', 'Prédit Malade'])
                with col1:
                    st.write("**_Matrice de confusion en valeur absolue :_**")
                    st.write(df_confusion_rfcr)

                # Calculer total malade / total sain
                total_malade_rfcr = np.sum(y_rfcr_test == 1)
                total_sain_rfcr = np.sum(y_rfcr_test == 0)
            
                # Convertir la matrice de confusion en pourcentage par rapport au total de chaque classe
                cm_percent_rfcr = cm_rfcr.copy().astype(float)
                cm_percent_rfcr[1, :] /= total_malade_rfcr
                cm_percent_rfcr[0, :] /= total_sain_rfcr
                cm_percent_rfcr *= 100

                # Créer un DataFrame à partir de la matrice de confusion en pourcentage
                df_confusion_percent_rfcr = pd.DataFrame(cm_percent_rfcr, index=['Réel Sain %', 'Réel Malade %'], columns=['Prédit Sain %', 'Prédit Malade %'])

                # Afficher le DataFrame dans Streamlit
                with col2:
                    st.write("**_Matrice de confusion en pourcentage :_**")
                    st.write(df_confusion_percent_rfcr)

                st.subheader("_Conclusion_ :")
                st.write("Random Forest présente un surentrainement.")


            # Gradient Boosting Classifier maladies cardiaques
            elif selected_model == "GradientBoostingClassifier":

                st.subheader("_Choix des variables pour l'entrainement du modèle GradientBoostClassifier_ :")

                st.markdown("""
                            **_Les variables retenues pour le modèle de prédiction sont_** :\n- Age\n- Sex\n - Chest Pain Type\n - Resting Blood Pressure\n- Chlolestérol\n- Fasting Blood Sugar\n- Resting Electrocardiographic Results\n - Thalach (Maximum Heart Rate Achieved)\n - Exercice Induced Angina\n - Oldpeak\n - Slope\n - Ca (Number of Major Vessels Colored by Fluoroscopie)\n - Thalassemia
                            """)
                st.write("""
                        Les données d'entrainement et de test sont séparées telles que : **jeu de test = 30 % et jeu d'entrainement = 70 %.**
                        """)
                X_gbcr_train, X_gbcr_test, y_gbcr_train, y_gbcr_test = train_test_split(X_cr, y_cr, test_size = 0.30, random_state=10)
                GradientBoost_model_gbcr = GradientBoostingClassifier()
                GradientBoost_model_gbcr.fit(X_gbcr_train, y_gbcr_train)

                st.subheader("_Scores du modèle_ :")
                st.write("**_Les performances du modèle sont les suivantes :_**")
                train_score_gbcr = GradientBoost_model_gbcr.score(X_gbcr_train, y_gbcr_train)
                test_score_gbcr = GradientBoost_model_gbcr.score(X_gbcr_test, y_gbcr_test)
                st.write(f"- Le score pour le jeu d'entrainement est : {train_score_gbcr*100:.2f}%")
                st.write(f"- Le score pour le jeu de test est : {test_score_gbcr*100:.2f}%")

                st.subheader("_Rapport de classification du modèle_ :")
                predict_train_data_gbcr = GradientBoost_model_gbcr.predict(X_gbcr_test)
                report_gbcr = classification_report(y_gbcr_test, predict_train_data_gbcr, output_dict = True)
                df_report_gbcr = pd.DataFrame(report_gbcr).transpose()
                st.write(df_report_gbcr)
                st.write("_*0 : patient sain, 1 : patient malade_")

                st.subheader("_Matrices de confusion_ :")
                # Créer deux colonnes
                col1, col2 = st.columns([1, 1])  # Rapport de largeur 1:1
                
                # Calculer la matrice de confusion
                cm_gbcr = confusion_matrix(y_gbcr_test, predict_train_data_gbcr)
                df_confusion_gbcr = pd.DataFrame(cm_gbcr, index=['Réel Sain', 'Réel Malade'], columns=['Prédit Sain', 'Prédit Malade'])
                with col1:
                    st.write("**_Matrice de confusion en valeur absolue :_**")
                    st.write(df_confusion_gbcr)

                # Calculer total malade / total sain
                total_malade_gbcr= np.sum(y_gbcr_test == 1)
                total_sain_gbcr = np.sum(y_gbcr_test == 0)
            
                # Convertir la matrice de confusion en pourcentage par rapport au total de chaque classe
                cm_percent_gbcr = cm_gbcr.copy().astype(float)
                cm_percent_gbcr[1, :] /= total_malade_gbcr
                cm_percent_gbcr[0, :] /= total_sain_gbcr
                cm_percent_gbcr *= 100

                # Créer un DataFrame à partir de la matrice de confusion en pourcentage
                df_confusion_percent_gbcr = pd.DataFrame(cm_percent_gbcr, index=['Réel Sain %', 'Réel Malade %'], columns=['Prédit Sain %', 'Prédit Malade %'])

                # Afficher le DataFrame dans Streamlit
                with col2:
                    st.write("**_Matrice de confusion en pourcentage :_**")
                    st.write(df_confusion_percent_gbcr)

                st.subheader("_Conclusion_ :")
                st.write("Gradient Boosting Classifier présente un surentrainement et il est moins performant que Random Forest.")

            # Logistic Regression maladies cardiaques
            elif selected_model == "LogisticRegression":

                st.subheader("_Choix des variables pour l'entrainement du modèle LogisticRegression_ :")

                st.markdown("""
                            **_Les variables retenues pour le modèle de prédiction sont_** :\n- Age\n- Sex\n - Chest Pain Type\n - Resting Blood Pressure\n- Chlolestérol\n- Fasting Blood Sugar\n- Resting Electrocardiographic Results\n - Thalach (Maximum Heart Rate Achieved)\n - Exercice Induced Angina\n - Oldpeak\n - Slope\n - Ca (Number of Major Vessels Colored by Fluoroscopie)\n - Thalassemia
                            """)
                st.write("""
                        Les données d'entrainement et de test sont séparées telles que : **jeu de test = 25 % et jeu d'entrainement = 75 %.
                        Un standard scaler est utilisé sur les variables prédictives.**
                        """)
                X_lrcr_train, X_lrcr_test, y_lrcr_train, y_lrcr_test = train_test_split(X_cr, y_cr, test_size = 0.25, random_state=42)
                
                # Initialisation d'un scaler et utilisation sur X
                scaler_lr = StandardScaler()
                X_lrcr_train_scaled = scaler_lr.fit_transform(X_lrcr_train)
                X_lrcr_test_scaled = scaler_lr.transform(X_lrcr_test)
                
                logisticRegression_lrcr = LogisticRegression()
                logisticRegression_lrcr.fit(X_lrcr_train_scaled, y_lrcr_train)

                st.subheader("_Scores du modèle_ :")
                st.write("**_Les performances du modèle sont les suivantes :_**")
                train_score_lrcr = logisticRegression_lrcr.score(X_lrcr_train_scaled, y_lrcr_train)
                test_score_lrcr = logisticRegression_lrcr.score(X_lrcr_test_scaled, y_lrcr_test)
                st.write(f"- Le score pour le jeu d'entrainement est : {train_score_lrcr*100:.2f}%")
                st.write(f"- Le score pour le jeu de test est : {test_score_lrcr*100:.2f}%")

                st.subheader("_Rapport de classification du modèle_ :")
                predict_train_data_lrcr = logisticRegression_lrcr.predict(X_lrcr_test_scaled)
                report_lrcr = classification_report(y_lrcr_test, predict_train_data_lrcr, output_dict = True)
                df_report_lrcr = pd.DataFrame(report_lrcr).transpose()
                st.write(df_report_lrcr)
                st.write("_*0 : patient sain, 1 : patient malade_")

                st.subheader("_Matrices de confusion_ :")
                # Créer deux colonnes
                col1, col2 = st.columns([1, 1])  # Rapport de largeur 1:1
                
                # Calculer la matrice de confusion
                cm_lrcr = confusion_matrix(y_lrcr_test, predict_train_data_lrcr)
                df_confusion_lrcr = pd.DataFrame(cm_lrcr, index=['Réel Sain', 'Réel Malade'], columns=['Prédit Sain', 'Prédit Malade'])
                with col1:
                    st.write("**_Matrice de confusion en valeur absolue :_**")
                    st.write(df_confusion_lrcr)

                # Calculer total malade / total sain
                total_malade_lrcr = np.sum(y_lrcr_test == 1)
                total_sain_lrcr = np.sum(y_lrcr_test == 0)
            
                # Convertir la matrice de confusion en pourcentage par rapport au total de chaque classe
                cm_percent_lrcr = cm_lrcr.copy().astype(float)
                cm_percent_lrcr[1, :] /= total_malade_lrcr
                cm_percent_lrcr[0, :] /= total_sain_lrcr
                cm_percent_lrcr *= 100

                # Créer un DataFrame à partir de la matrice de confusion en pourcentage
                df_confusion_percent_lrcr = pd.DataFrame(cm_percent_lrcr, index=['Réel Sain %', 'Réel Malade %'], columns=['Prédit Sain %', 'Prédit Malade %'])

                # Afficher le DataFrame dans Streamlit
                with col2:
                    st.write("**_Matrice de confusion en pourcentage :_**")
                    st.write(df_confusion_percent_lrcr)

                st.subheader("_Conclusion_ :")
                st.write("La régression logistique est plus adapté et offre des scores plus intéressants, ce sera le modèle retenu pour notre prédiction.")

        ## PERFORMANCES MODELES CANCER DU SEIN ##
        #Charger les données
        file_path = "dfcancersein.csv"
        sein = load_data(file_path)

        feature_columns = feature_columns1 = ['radius_mean','texture_mean','smoothness_mean','compactness_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se','smoothness_se','compactness_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','smoothness_worst','compactness_worst','symmetry_worst','fractal_dimension_worst']
        predicted_class = 'diagnosis'

        X_cs = sein[feature_columns].values
        y_cs = sein[predicted_class].values
    
        if selected_disease == "Cancer du sein":
    
            # Choix d'un modèle
            selected_model= st.selectbox("**Sélectionnez le modèle :**", ["RandomForestClassifier", "LogisticRegression"])

            # randomforestclassifier cancer du sein
            if selected_model == "RandomForestClassifier":

                st.subheader("_Choix des variables pour l'entrainement du modèle RandomForestClassifier_ :")

                st.markdown("""
                            **_Les variables retenues pour le modèle de prédiction sont_** :\n- Radius Mean\n- Texture Mean\n - Smoothness Mean\n - Compactness Mean\n- Symmetry Mean\n- Fractal Dimension Mean\n- Radius Se\n - Texture Se\n - Smoothness Se\n - Compactness Se\n - Symmetry Se\n - Fractal Dimension Se\n - Radius Worst\n - Texture Worst\n - Smoothness Worst\n - Compactness Worst\n - Symmetry Worst\n - Fractal Dimension Worst
                            """)
                st.write("""
                        Les données d'entrainement et de test sont séparées telles que : **jeu de test = 30 % et jeu d'entrainement = 75 %.**
                        """)
                X_rfcs_train, X_rfcs_test, y_rfcs_train, y_rfcs_test = train_test_split(X_cs, y_cs, test_size = 0.3, random_state=42)
                random_forest_model_rfcs = RandomForestClassifier()
                random_forest_model_rfcs.fit(X_rfcs_train, y_rfcs_train)
            
                st.subheader("_Scores du modèle_ :")
                st.write("**_Les performances du modèle sont les suivantes :_**")
                train_score_rfcs = random_forest_model_rfcs.score(X_rfcs_train, y_rfcs_train)
                test_score_rfcs = random_forest_model_rfcs.score(X_rfcs_test, y_rfcs_test)
                st.write(f"- Le score pour le jeu d'entrainement est : {train_score_rfcs*100:.2f}%")
                st.write(f"- Le score pour le jeu de test est : {test_score_rfcs*100:.2f}%")

                st.subheader("_Rapport de classification du modèle_ :")
                predict_train_data_rfcs = random_forest_model_rfcs.predict(X_rfcs_test)
                report_rfcs = classification_report(y_rfcs_test, predict_train_data_rfcs, output_dict = True)
                df_report_rfcs = pd.DataFrame(report_rfcs).transpose()
                st.write(df_report_rfcs)
                st.write("_*1 : patient sain, 0 : patient malade_")

                st.subheader("_Matrices de confusion_ :")
                # Créer deux colonnes
                col1, col2 = st.columns([1, 1])  # Rapport de largeur 1:1

                # Calculer la matrice de confusion
                cm_rfcs = confusion_matrix(y_rfcs_test, predict_train_data_rfcs)
                df_confusion_rfcs = pd.DataFrame(cm_rfcs, index=['Réel Malade', 'Réel Sain'], columns=['Prédit Malade', 'Prédit Sain'])
                with col1:
                    st.write("**_Matrice de confusion en valeur absolue :_**")
                    st.write(df_confusion_rfcs)

                # Calculer total malade / total sain
                total_malade_rfcs = np.sum(y_rfcs_test == 0)
                total_sain_rfcs = np.sum(y_rfcs_test == 1)
            
                # Convertir la matrice de confusion en pourcentage par rapport au total de chaque classe
                cm_percent_rfcs = cm_rfcs.copy().astype(float)
                cm_percent_rfcs[0, :] /= total_malade_rfcs
                cm_percent_rfcs[1, :] /= total_sain_rfcs
                cm_percent_rfcs *= 100

                # Créer un DataFrame à partir de la matrice de confusion en pourcentage
                df_confusion_percent_rfcs = pd.DataFrame(cm_percent_rfcs, index=['Réel Malade %', 'Réel Sain %'], columns=['Prédit Malade %', 'Prédit Sain %'])

                # Afficher le DataFrame dans Streamlit
                with col2:
                    st.write("**_Matrice de confusion en pourcentage :_**")
                    st.write(df_confusion_percent_rfcs)
                st.write("Le modèle Random Forest présente un léger surentrainement et n'est pas aussi performant que la régression logistique.")

            # Logistic Regression cancer du sein
            elif selected_model == "LogisticRegression":

                st.subheader("_Choix des variables pour l'entrainement du modèle LogisticRegression_ :")

                st.markdown("""
                            **_Les variables retenues pour le modèle de prédiction sont_** :\n- Radius Mean\n- Texture Mean\n - Smoothness Mean\n - Compactness Mean\n- Symmetry Mean\n- Fractal Dimension Mean\n- Radius Se\n - Texture Se\n - Smoothness Se\n - Compactness Se\n - Symmetry Se\n - Fractal Dimension Se\n - Radius Worst\n - Texture Worst\n - Smoothness Worst\n - Compactness Worst\n - Symmetry Worst\n - Fractal Dimension Worst
                            """)
                st.write("""
                        Les données d'entrainement et de test sont séparées telles que : **jeu de test = 25 % et jeu d'entrainement = 75 %.
                        Les variables prédictives sont standardisées.**
                        """)
                X_lrcs_train, X_lrcs_test, y_lrcs_train, y_lrcs_test = train_test_split(X_cs, y_cs, test_size = 0.25, random_state=0)
                
                # Génération d'un standard scaler et application sur X
                scaler_lrcs = StandardScaler()
                X_lrcs_train_scaled = scaler_lrcs.fit_transform(X_lrcs_train)
                X_lrcs_test_scaled = scaler_lrcs.transform(X_lrcs_test)
                
                logisticRegression_lrcs = LogisticRegression()
                logisticRegression_lrcs.fit(X_lrcs_train_scaled, y_lrcs_train)

                st.subheader("_Scores du modèle_ :")
                st.write("**_Les performances du modèle sont les suivantes :_**")
                train_score_lrcs = logisticRegression_lrcs.score(X_lrcs_train_scaled, y_lrcs_train)
                test_score_lrcs = logisticRegression_lrcs.score(X_lrcs_test_scaled, y_lrcs_test)
                st.write(f"- Le score pour le jeu d'entrainement est : {train_score_lrcs*100:.2f}%")
                st.write(f"- Le score pour le jeu de test est : {test_score_lrcs*100:.2f}%")

                st.subheader("_Rapport de classification du modèle_ :")
                predict_train_data_lrcs = logisticRegression_lrcs.predict(X_lrcs_test_scaled)
                report_lrcs = classification_report(y_lrcs_test, predict_train_data_lrcs, output_dict = True)
                df_report_lrcs = pd.DataFrame(report_lrcs).transpose()
                st.write(df_report_lrcs)
                st.write("_*1 : patient sain, 0 : patient malade_")

                st.subheader("_Matrices de confusion_ :")
                # Créer deux colonnes
                col1, col2 = st.columns([1, 1])  # Rapport de largeur 1:1

                # Calculer la matrice de confusion
                cm_lrcs = confusion_matrix(y_lrcs_test, predict_train_data_lrcs)
                df_confusion_lrcs = pd.DataFrame(cm_lrcs, index=['Réel Malade', 'Réel Sain'], columns=['Prédit Malade', 'Prédit Sain'])
                with col1:
                    st.write("**_Matrice de confusion en valeur absolue :_**")
                    st.write(df_confusion_lrcs)

                # Calculer total malade / total sain
                total_malade_lrcs = np.sum(y_lrcs_test == 0)
                total_sain_lrcs = np.sum(y_lrcs_test == 1)
            
                # Convertir la matrice de confusion en pourcentage par rapport au total de chaque classe
                cm_percent_lrcs = cm_lrcs.copy().astype(float)
                cm_percent_lrcs[0, :] /= total_malade_lrcs
                cm_percent_lrcs[1, :] /= total_sain_lrcs
                cm_percent_lrcs *= 100

                # Créer un DataFrame à partir de la matrice de confusion en pourcentage
                df_confusion_percent_lrcs = pd.DataFrame(cm_percent_lrcs, index=['Réel Malade %', 'Réel Sain %'], columns=['Prédit Malade %', 'Prédit Sain %'])

                # Afficher le DataFrame dans Streamlit
                with col2:
                    st.write("**_Matrice de confusion en pourcentage :_**")
                    st.write(df_confusion_percent_lrcs)
                st.write("Les scores obtenus pour le modèle de régression logistique sont très bons, il sera le modèle retenu pour la prédiction.")

        ## PERFORMANCES MODELES MALADIES RENALES ##
        #Charger les données
        file_path = "dfrein_ML.csv"
        rein = load_data(file_path)

        feature_columns = ['age', 'bp', 'sg', 'al', 'su', 'rc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rbc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
        predicted_class = 'classification'

        X_r = rein[feature_columns].values
        y_r = rein[predicted_class].values
    
        if selected_disease == "Maladies rénales":
    
            # Choix d'un modèle
            selected_model= st.selectbox("**Sélectionnez le modèle :**", ["GradientBoostingClassifier", "LogisticRegression"])

            # GraidientBoostingClassifier maladies rénales
            if selected_model == "GradientBoostingClassifier":

                st.subheader("_Choix des variables pour l'entrainement du modèle GradientBoostingClassifier_ :")

                st.markdown("""
                            **_Les variables retenues pour le modèle de prédiction sont_** : \n- Age\n- Blood pressure (bp)\n- Specific gravity (sg)\n- Albumin (al)\n- Sugar (su)\n- Red cells (rc)\n- Pus cells (pc)\n- Pus cells clumps (pcc)\n- Bacteria (ba)\n- Blood glucose random (bgr)\n- Blood urea (bu)\n- Serum creatinine (sc)\n- Sodium (sod)\n- Potassium (pot)\n- Hemoglobin (hemo)\n- Packed cell volume (pcv)\n- White cells (wc)\n- Red blood cells (rbc)\n- Hypertension (htn)\n- Diabete Mellitus (dm)\n- Coronary Artery Disease (cad)\n- Appetite (appet)\n- Pedal Edema (pe)\n- Anemia (ane)
                            """)

                st.write("""
                        Les données d'entrainement et de test sont séparées telles que : **jeu de test = 38 % et jeu d'entrainement = 62 %.**
                        """)
                X_rfr_train, X_rfr_test, y_rfr_train, y_rfr_test = train_test_split(X_r, y_r, test_size = 0.38, random_state=33)
                gradient_boost_rfr = GradientBoostingClassifier(random_state = 33)
                gradient_boost_rfr.fit(X_rfr_train, y_rfr_train)

                st.subheader("_Scores du modèle_ :")
                st.write("**_Les performances du modèle sont les suivantes :_**")
                train_score_rfr = gradient_boost_rfr.score(X_rfr_train, y_rfr_train)
                test_score_rfr = gradient_boost_rfr.score(X_rfr_test, y_rfr_test)
                st.write(f"- Le score pour le jeu d'entrainement est : {train_score_rfr*100:.2f}%")
                st.write(f"- Le score pour le jeu de test est : {test_score_rfr*100:.2f}%")

                st.subheader("_Rapport de classification du modèle_ :")
                predict_train_data_rfr = gradient_boost_rfr.predict(X_rfr_test)
                report_rfr = classification_report(y_rfr_test, predict_train_data_rfr, output_dict = True)
                df_report_rfr = pd.DataFrame(report_rfr).transpose()
                st.write(df_report_rfr)
                st.write("_*0 : patient sain, 1 : patient malade_")

                st.subheader("_Matrices de confusion_ :")
                # Créer deux colonnes
                col1, col2 = st.columns([1, 1])  # Rapport de largeur 1:1
                
                # Calculer la matrice de confusion
                cm_rfr = confusion_matrix(y_rfr_test, predict_train_data_rfr)
                df_confusion_rfr = pd.DataFrame(cm_rfr, index=['Réel Sain', 'Réel Malade'], columns=['Prédit Sain', 'Prédit Malade'])
                with col1:
                    st.write("**_Matrice de confusion en valeur absolue :_**")
                    st.write(df_confusion_rfr)

                # Calculer total malade / total sain
                total_malade_rfr = np.sum(y_rfr_test == 1)
                total_sain_rfr = np.sum(y_rfr_test == 0)
            
                # Convertir la matrice de confusion en pourcentage par rapport au total de chaque classe
                cm_percent_rfr = cm_rfr.copy().astype(float)
                cm_percent_rfr[1, :] /= total_malade_rfr
                cm_percent_rfr[0, :] /= total_sain_rfr
                cm_percent_rfr *= 100

                # Créer un DataFrame à partir de la matrice de confusion en pourcentage
                df_confusion_percent_rfr = pd.DataFrame(cm_percent_rfr, index=['Réel Sain %', 'Réel Malade %'], columns=['Prédit Sain %', 'Prédit Malade %'])

                # Afficher le DataFrame dans Streamlit
                with col2:
                    st.write("**_Matrice de confusion en pourcentage :_**")
                    st.write(df_confusion_percent_rfr)
                st.write("Le modèle Gradient boosting classifier présente un score d'entrainement de 100 %, il y a un risque de surapprentissage.")

            # Logistic Regression maladies rénales
            elif selected_model == "LogisticRegression":

                st.subheader("_Choix des variables pour l'entrainement du modèle LogisticRegression_ :")

                st.markdown("""
                            **_Les variables retenues pour le modèle de prédiction sont_** :\n- Age\n- Blood pressure (bp)\n- Specific gravity (sg)\n- Albumin (al)\n- Sugar (su)\n- Red cells (rc)\n- Pus cells (pc)\n- Pus cells clumps (pcc)\n- Bacteria (ba)\n- Blood glucose random (bgr)\n- Blood urea (bu)\n- Serum creatinine (sc)\n- Sodium (sod)\n- Potassium (pot)\n- Hemoglobin (hemo)\n- Packed cell volume (pcv)\n- White cells (wc)\n- Red blood cells (rbc)\n- Hypertension (htn)\n- Diabete Mellitus (dm)\n- Coronary Artery Disease (cad)\n- Appetite (appet)\n- Pedal Edema (pe)\n- Anemia (ane)
                            """)
                st.write("""
                        Les données d'entrainement et de test sont séparées telles que : **jeu de test = 38 % et jeu d'entrainement = 62 %.**
                        """)
                X_lrr_train, X_lrr_test, y_lrr_train, y_lrr_test = train_test_split(X_r, y_r, test_size = 0.38, random_state=42)
                logisticRegression_lrr = LogisticRegression(solver = 'liblinear')
                logisticRegression_lrr.fit(X_lrr_train, y_lrr_train)

                st.subheader("_Scores du modèle_ :")
                st.write("**_Les performances du modèle sont les suivantes :_**")
                train_score_lrr = logisticRegression_lrr.score(X_lrr_train, y_lrr_train)
                test_score_lrr = logisticRegression_lrr.score(X_lrr_test, y_lrr_test)
                st.write(f"- Le score pour le jeu d'entrainement est : {train_score_lrr*100:.2f}%")
                st.write(f"- Le score pour le jeu de test est : {test_score_lrr*100:.2f}%")

                st.subheader("_Rapport de classification du modèle_ :")
                predict_train_data_lrr = logisticRegression_lrr.predict(X_lrr_test)
                report_lrr = classification_report(y_lrr_test, predict_train_data_lrr, output_dict = True)
                df_report_lrr = pd.DataFrame(report_lrr).transpose()
                st.write(df_report_lrr)
                st.write("_*0 : patient sain, 1 : patient malade_")

                st.subheader("_Matrices de confusion_ :")
                # Créer deux colonnes
                col1, col2 = st.columns([1, 1])  # Rapport de largeur 1:1
                
                # Calculer la matrice de confusion
                cm_lrr = confusion_matrix(y_lrr_test, predict_train_data_lrr)
                df_confusion_lrr = pd.DataFrame(cm_lrr, index=['Réel Sain', 'Réel Malade'], columns=['Prédit Sain', 'Prédit Malade'])
                with col1:
                    st.write("**_Matrice de confusion en valeur absolue :_**")
                    st.write(df_confusion_lrr)

                # Calculer total malade / total sain
                total_malade_lrr = np.sum(y_lrr_test == 1)
                total_sain_lrr = np.sum(y_lrr_test == 0)
            
                # Convertir la matrice de confusion en pourcentage par rapport au total de chaque classe
                cm_percent_lrr = cm_lrr.copy().astype(float)
                cm_percent_lrr[1, :] /= total_malade_lrr
                cm_percent_lrr[0, :] /= total_sain_lrr
                cm_percent_lrr *= 100

                # Créer un DataFrame à partir de la matrice de confusion en pourcentage
                df_confusion_percent_lrr = pd.DataFrame(cm_percent_lrr, index=['Réel Sain %', 'Réel Malade %'], columns=['Prédit Sain %', 'Prédit Malade %'])

                # Afficher le DataFrame dans Streamlit
                with col2:
                    st.write("**_Matrice de confusion en pourcentage :_**")
                    st.write(df_confusion_percent_lrr)
                st.write("Les scores obtenus pour le modèle de régression logistique équilibrés et précis, il sera le modèle retenu pour la prédiction.")

        ## PERFORMANCES MODELES MALADIES DU FOIE ##
        #Charger les données
        file_path = "df_liver_M0F1 .csv"
        foie = load_data(file_path)

        feature_columns = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio']
        predicted_class = 'Dataset'

        X_f = foie[feature_columns].values
        y_f = foie[predicted_class].values

        if selected_disease == "Maladies du foie":
    
            # Choix d'un modèle
            selected_model= st.selectbox("**Sélectionnez le modèle :**", ["RandomForestClassifier", "GradientBoostingClassifier", "LogisticRegression"])

            # randomforestclassifier maladies du foie
            if selected_model == "RandomForestClassifier":

                st.subheader("_Choix des variables pour l'entrainement du modèle RandomForestClassifier_ :")

                st.markdown("""
                            **_Les variables retenues pour le modèle de prédiction sont_** :\n- Age\n- Gender\n - Total Bilirubin\n - Direct Bilirubin\n- Alkaline Phosphatase\n- Alamine Aminotransférase\n- Aspartate Aminostransférase\n- Total Protiens\n- Albumin\n- Albumin and Globulin Ratio
                            """)
                st.write("""
                        Les données d'entrainement et de test sont séparées telles que : **jeu de test = 25 % et jeu d'entrainement = 75 %. Les données
                        des classes sont équilibrées telles que données 'sain' = 400 et données 'malade' = 400.**
                        """)
                random_state = 42
                
                X_rff_train, X_rff_test, y_rff_train, y_rff_test = train_test_split(X_f, y_f, test_size = 0.25, random_state=random_state)
                
                # Sampling strategy
                sampling_strategy = {0: 402, 1: 402}
                X_rff_resampled, y_rff_resampled = SMOTE(sampling_strategy = sampling_strategy).fit_resample(X_rff_train, y_rff_train)

                random_forest_model_rff = RandomForestClassifier(random_state = random_state)
                random_forest_model_rff.fit(X_rff_resampled, y_rff_resampled)

                st.subheader("_Scores du modèle_ :")
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
                cv_scores_rff = cross_val_score(random_forest_model_rff, X_rff_resampled, y_rff_resampled, cv=cv)
                st.write("Cross-validation scores:", cv_scores_rff)
                st.write("Mean CV score:", cv_scores_rff.mean())
                
                st.subheader("_Rapport de classification du modèle_ :")
                predict_train_data_rff_cv = cross_val_predict(random_forest_model_rff, X_rff_resampled, y_rff_resampled, cv = cv)
                report_rff = classification_report(y_rff_resampled, predict_train_data_rff_cv, output_dict = True)
                df_report_rff = pd.DataFrame(report_rff).transpose()
                st.write(df_report_rff)
                st.write("_*0 : patient sain, 1 : patient malade_")

                st.subheader("_Matrices de confusion_ :")
                # Créer deux colonnes
                col1, col2 = st.columns([1, 1])  # Rapport de largeur 1:1
                
                # Calculer la matrice de confusion
                cm_rff = confusion_matrix(y_rff_resampled, predict_train_data_rff_cv)
                df_confusion_rff = pd.DataFrame(cm_rff, index=['Réel Sain', 'Réel Malade'], columns=['Prédit Sain', 'Prédit Malade'])
                with col1:
                    st.write("**_Matrice de confusion en valeur absolue :_**")
                    st.write(df_confusion_rff)

                # Calculer total malade / total sain
                total_malade_rff = np.sum(y_rff_resampled == 1)
                total_sain_rff = np.sum(y_rff_resampled == 0)
            
                # Convertir la matrice de confusion en pourcentage par rapport au total de chaque classe
                cm_percent_rff = cm_rff.copy().astype(float)
                cm_percent_rff[1, :] /= total_malade_rff
                cm_percent_rff[0, :] /= total_sain_rff
                cm_percent_rff *= 100

                # Créer un DataFrame à partir de la matrice de confusion en pourcentage
                df_confusion_percent_rff = pd.DataFrame(cm_percent_rff, index=['Réel Sain %', 'Réel Malade %'], columns=['Prédit Sain %', 'Prédit Malade %'])

                # Afficher le DataFrame dans Streamlit
                with col2:
                    st.write("**_Matrice de confusion en pourcentage :_**")
                    st.write(df_confusion_percent_rff)
                st.write("Le modèle Random Forest offre les meilleurs scores, il sera le modèle retenu pour notre prédiction.")

            # Gradient Boosting Classifier maladies du foie
            elif selected_model == "GradientBoostingClassifier":

                st.subheader("_Choix des variables pour l'entrainement du modèle GradientBoostClassifier_ :")

                st.markdown("""
                            **_Les variables retenues pour le modèle de prédiction sont_** :\n- Age\n- Gender\n - Total Bilirubin\n - Direct Bilirubin\n- Alkaline Phosphatase\n- Alamine Aminotransférase\n- Aspartate Aminostransférase\n- Total Protiens\n- Albumin\n- Albumin and Globulin Ratio
                            """)
                st.write("""
                        Les données d'entrainement et de test sont séparées telles que : **jeu de test = 30 % et jeu d'entrainement = 70 %. Les données
                        des classes sont équilibrées telles que données 'sain' = 400 et données 'malade' = 400.**
                        """)
                test_size = 0.30
                random_state = 42
                X_gbf_train, X_gbf_test, y_gbf_train, y_gbf_test = train_test_split(X_f, y_f, test_size = test_size, random_state=random_state)
                
                sampling_strategy = {0: 400, 1: 400}
                X_gbf_resampled, y_gbf_resampled = SMOTE(sampling_strategy=sampling_strategy).fit_resample(X_gbf_train, y_gbf_train)

                GradientBoost_model_gbf = GradientBoostingClassifier(random_state = random_state)
                GradientBoost_model_gbf.fit(X_gbf_resampled, y_gbf_resampled)

                st.subheader("_Scores du modèle_ :")
                st.write("**_Les performances du modèle sont les suivantes :_**")
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
                cv_scores_gbf = cross_val_score(GradientBoost_model_gbf, X_gbf_resampled, y_gbf_resampled, cv=cv)
                st.write("Cross-validation scores:", cv_scores_gbf)
                st.write("Mean CV score:", cv_scores_gbf.mean())

                st.subheader("_Rapport de classification du modèle_ :")
                predict_train_data_gbf_cv = cross_val_predict(GradientBoost_model_gbf, X_gbf_resampled, y_gbf_resampled, cv = cv)
                report_gbf = classification_report(y_gbf_resampled, predict_train_data_gbf_cv, output_dict = True)
                df_report_gbf = pd.DataFrame(report_gbf).transpose()
                st.write(df_report_gbf)
                st.write("_*0 : patient sain, 1 : patient malade_")

                st.subheader("_Matrices de confusion_ :")
                # Créer deux colonnes
                col1, col2 = st.columns([1, 1])  # Rapport de largeur 1:1
                
                # Calculer la matrice de confusion
                cm_gbf = confusion_matrix(y_gbf_resampled, predict_train_data_gbf_cv)
                df_confusion_gbf = pd.DataFrame(cm_gbf, index=['Réel Sain', 'Réel Malade'], columns=['Prédit Sain', 'Prédit Malade'])
                with col1:
                    st.write("**_Matrice de confusion en valeur absolue :_**")
                    st.write(df_confusion_gbf)

                # Calculer total malade / total sain
                total_malade_gbf= np.sum(y_gbf_resampled == 1)
                total_sain_gbf = np.sum(y_gbf_resampled == 0)
            
                # Convertir la matrice de confusion en pourcentage par rapport au total de chaque classe
                cm_percent_gbf = cm_gbf.copy().astype(float)
                cm_percent_gbf[1, :] /= total_malade_gbf
                cm_percent_gbf[0, :] /= total_sain_gbf
                cm_percent_gbf *= 100

                # Créer un DataFrame à partir de la matrice de confusion en pourcentage
                df_confusion_percent_gbf = pd.DataFrame(cm_percent_gbf, index=['Réel Sain %', 'Réel Malade %'], columns=['Prédit Sain %', 'Prédit Malade %'])

                # Afficher le DataFrame dans Streamlit
                with col2:
                    st.write("**_Matrice de confusion en pourcentage :_**")
                    st.write(df_confusion_percent_gbf)
                st.write("Le modèle Gradient Boosting est un peu moins performant que le modèle Random Forest.")

            # Logistic Regression maladies du foie
            elif selected_model == "LogisticRegression":

                st.subheader("_Choix des variables pour l'entrainement du modèle LogisticRegression_ :")

                st.markdown("""
                            **_Les variables retenues pour le modèle de prédiction sont_** :\n- Age\n- Gender\n - Total Bilirubin\n - Direct Bilirubin\n- Alkaline Phosphatase\n- Alamine Aminotransférase\n- Aspartate Aminostransférase\n- Total Protiens\n- Albumin\n- Albumin and Globulin Ratio
                            """)
                st.write("""
                        Les données d'entrainement et de test sont séparées telles que : **jeu de test = 25 % et jeu d'entrainement = 75 %. Les données
                        des classes sont équilibrées telles que données 'sain' = 400 et données 'malade' = 400.**
                        """)
                test_size = 0.25
                random_state = 42
                X_lrf_train, X_lrf_test, y_lrf_train, y_lrf_test = train_test_split(X_f, y_f, test_size = test_size, random_state=random_state)
                
                sampling_strategy = {0: 400, 1: 400}
                X_lrf_resampled, y_lrf_resampled = SMOTE(sampling_strategy=sampling_strategy).fit_resample(X_lrf_train, y_lrf_train)

                logisticRegression_lrf = LogisticRegression(max_iter = 1500, random_state = random_state)
                logisticRegression_lrf.fit(X_lrf_resampled, y_lrf_resampled)

                st.subheader("_Scores du modèle_ :")
                st.write("**_Les performances du modèle sont les suivantes :_**")
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
                cv_scores_lrf = cross_val_score(logisticRegression_lrf, X_lrf_resampled, y_lrf_resampled, cv=cv)
                st.write("Cross-validation scores:", cv_scores_lrf)
                st.write("Mean CV score:", cv_scores_lrf.mean())

                st.subheader("_Rapport de classification du modèle_ :")
                predict_train_data_lrf_cv = cross_val_predict(logisticRegression_lrf, X_lrf_resampled, y_lrf_resampled, cv=cv)
                report_lrf = classification_report(y_lrf_resampled, predict_train_data_lrf_cv, output_dict = True)
                df_report_lrf = pd.DataFrame(report_lrf).transpose()
                st.write(df_report_lrf)
                st.write("_*0 : patient sain, 1 : patient malade_")

                st.subheader("_Matrices de confusion_ :")
                # Créer deux colonnes
                col1, col2 = st.columns([1, 1])  # Rapport de largeur 1:1
                
                # Calculer la matrice de confusion
                cm_lrf = confusion_matrix(y_lrf_resampled, predict_train_data_lrf_cv)
                df_confusion_lrf = pd.DataFrame(cm_lrf, index=['Réel Sain', 'Réel Malade'], columns=['Prédit Sain', 'Prédit Malade'])
                with col1:
                    st.write("**_Matrice de confusion en valeur absolue :_**")
                    st.write(df_confusion_lrf)

                # Calculer total malade / total sain
                total_malade_lrf = np.sum(y_lrf_resampled == 1)
                total_sain_lrf = np.sum(y_lrf_resampled == 0)
            
                # Convertir la matrice de confusion en pourcentage par rapport au total de chaque classe
                cm_percent_lrf = cm_lrf.copy().astype(float)
                cm_percent_lrf[1, :] /= total_malade_lrf
                cm_percent_lrf[0, :] /= total_sain_lrf
                cm_percent_lrf *= 100

                # Créer un DataFrame à partir de la matrice de confusion en pourcentage
                df_confusion_percent_lrf = pd.DataFrame(cm_percent_lrf, index=['Réel Sain %', 'Réel Malade %'], columns=['Prédit Sain %', 'Prédit Malade %'])

                # Afficher le DataFrame dans Streamlit
                with col2:
                    st.write("**_Matrice de confusion en pourcentage :_**")
                    st.write(df_confusion_percent_lrf)
                st.write("""Le modèle Logistic Regression offre des performances discutables, surtout sur la classe 'malade', il ne sera donc pas retenu pour notre prédiction.
                         Il faut noter qu'il y a un déséquilibre important entre les classes 'malade' et 'sain' et que malgré le rééquilibrage effectué, il
                         serait intéressant d'obtenir davantage de données de la classe minoritaire pour affiner le modèle.
                         """)

if __name__ == "__main__":
    main()






        







    