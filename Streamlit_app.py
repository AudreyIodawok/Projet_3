import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from scipy.stats import kruskal
from statannot import add_stat_annotation
import streamlit as st 

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#from xgboost import XGBClassifier

# Charger les logos
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
    st.subheader(f"Informations sur : {selected_disease}")
    # Afficher les informations sur la maladie sélectionnée

def display_prediction(selected_disease):
    st.subheader(f"Prédiction de : {selected_disease}")
    # Afficher le formulaire de prédiction pour la maladie sélectionnée

def main():
    st.title("Prédiction des maladies chroniques")

    # Affichage des onglets pour les deux sections
    st.sidebar.markdown("<h3 style='font-size: 18px;'>Sélectionnez une section</h3>", unsafe_allow_html=True)
    selected_section = st.sidebar.radio("section", ["Informations maladies", "Prédiction"], index=0, label_visibility="collapsed")

    if selected_section == "Informations maladies":
        st.sidebar.subheader("Sélectionnez une maladie")
        selected_disease = st.sidebar.selectbox("info", ["Diabète", "Maladies cardiaques", "Maladies rénales", "Cancer du sein", "Maladies du foie"], label_visibility="collapsed")
        display_thematique(selected_disease)
    elif selected_section == "Prédiction":
        st.sidebar.subheader("Sélectionnez une maladie")
        selected_disease = st.sidebar.selectbox("info", ["Diabète", "Maladies cardiaques", "Maladies rénales", "Cancer du sein", "Maladies du foie"], label_visibility="collapsed")
        display_prediction(selected_disease)
    
    ## DIABETE ##
    if selected_section == "Informations maladies" and selected_disease == "Diabète":

    # Charger les données
        file_path = "df_diabete_fin (1).csv"
        diabete = load_data(file_path)

    # Afficher la définition des variables
        st.subheader('Définition des variables :')
        st.write("""
                - **Pregnancies (Grossesses)**: Le nombre de fois que la personne a été enceinte. Il n’y a pas de preuve concluante que le nombre de grossesses influence le diabète gestationnel. Les femmes **_ayant présenté un diabète gestationnel ont un risque augmenté de développer ultérieurement un diabète de type 2_**.
                 
                - **Glucose (Glucose)**: La concentration de glucose plasmatique à 2 heures lors d'un test de tolérance au glucose oral. Cela mesure la réponse du corps au glucose après avoir consommé une quantité définie de sucre.
                Selon la Société Française d’Endocrinologie, on parle de **_diabète sucré si la glycémie à jeun est ≥ 1,26 g/l à deux reprises_**.

                - **Blood Pressure (Pression artérielle)**: La pression artérielle diastolique en millimètres de mercure (mm Hg). La pression diastolique est la pression exercée sur les parois des artères lorsque le cœur est au repos entre deux battements.
                Chez les personnes vivant avec le diabète, la tension artérielle devrait être **_inférieure à 130/80 mmHg_**. Pour une personne non diabétique, une pression artérielle normale est de 120/80 mmHg.

                - **Skin Thickness (Épaisseur de la peau)**: L'épaisseur du pli cutané du triceps en millimètres. Cette mesure peut être utilisée comme **_indicateur de la masse grasse_** (pour estimer la quantité de graisse corporelle).

                - **Insulin (Insuline)**: La concentration d'insuline sérique à 2 heures, mesurée en milli-unités par millilitre (mu U/ml). Cela indique la réponse de l'organisme à l'insuline après une charge de glucose. Le dosage de l’insuline dans le sang (insulinémie) n’est pas utilisé pour le diagnostic ni pour le suivi du diabète (qui reposent sur l’analyse de la glycémie et de l’hémoglobine glyquée). Cependant, il peut être utile de doser l’insuline dans le sang pour connaître la capacité du pancréas à la sécréter. Cela peut être utile au médecin à certaines phases de la maladie diabétique. 
                A titre indicatif, à jeun, l’insulinémie est **_normalement inférieure à 25 mIU / L (µUI / mL)_**. Elle est comprise **_entre 30 et 230 mIU / L environ 30 minutes après l’administration de glucose_**.

                - **BMI (Indice de masse corporelle)**: L'indice de masse corporelle est calculé en divisant le poids en kilogrammes par le carré de la taille en mètres. Il est utilisé comme indicateur du statut pondéral d'une personne.
                Il est calculé en divisant le poids par la taille au carré (taille en mètre). 

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

        elif selected_graph_type == "Analyse en Composantes Principales":
            st.image("ACP diabèe.png", use_column_width=True)

    ## MALADIES CARDIAQUES ##
    # Charger les données
    if selected_section == "Informations maladies" and selected_disease == "Maladies cardiaques":
        file_path = "dfheartML.csv"
        coeur = load_data(file_path)

    # Afficher la définition des variables
        st.subheader('Définition des variables :')
        st.write("""
                - **Age (age du patient)** : La probabilité d'avoir un accident cardiovasculaire ou cardiaque **_augmente nettement après 50 ans chez l'homme_** et après **_60 ans chez la femme_**.\n- **Sex (Genre du patient)** : Parmi les maladies jugées « masculines », les maladies cardiovasculaires sont un exemple typique de la façon dont les représentations sociales du féminin et du masculin influencent les pratiques médicales et l’attitude des patient.es. **_Les femmes sont plus vulnérables que les hommes aux maladies cardiovasculaires_** : 56 % en meurent contre 46 % des hommes. Or l’infarctus du myocarde est encore sous diagnostiqué chez les femmes car considéré à tort comme une maladie d’hommes stressés au travail. Le retard de diagnostic et de prise en charge reste fréquent.\n- **Cp (Chest Pain Type) / Type de douleur thoracique ressentie** : L'anamnèse de la maladie actuelle doit rechercher la localisation, la durée, le caractère et la qualité de la douleur. Il faut interroger le patient sur tout événement déclenchant (p. ex., surmenage des muscles thoraciques), ainsi que tout facteur déclenchant et calmant. Les facteurs spécifiques à rechercher comprennent le fait que la **_douleur soit présente à l'effort ou au repos_**, l'existence d'un **_stress psychologique_**, le fait que la douleur se produise **_lors de la respiration ou de la toux_**, qu'elle s'accompagne de **_difficultés à avaler_**, qu'elle soit en **_relation avec les repas_**, qu'elle soit soulagée ou aggravée par certaines positions (p. ex., couché à plat, penché en avant). Des antécédents d'épisodes semblables et leurs circonstances de survenue doivent être notés avec une attention particulière accordée à la similitude ou à l'absence de similitude avec les épisodes actuels et au fait que la fréquence et/ou la durée des épisodes augmentent. D'importants symptômes associés à la recherche comprennent une **_dyspnée_**, des **_palpitations_**, des **_syncopes_**, une **_transpiration_**, des **_nausées ou des vomissements_**, une **_toux_**, une **_fièvre ou des frissons_**.\n- **Trestbps (Resting Blood Pressure) / Pression artérielle au repos** : La valeur normale de la pression artérielle est de 120/80*. Le chiffre le plus élevé est la pression maximale, lorsque le cœur se contracte pour se vider. C’est la pression systolique. Le chiffre le moins élevé est la pression minimale, lorsque le cœur se relâche pour se remplir. C’est la pression diastolique.\nLa valeur limite au-delà de laquelle on parle **_d’hypertension artérielle est de 140/90_**, lorsque la mesure est faite au cabinet médical et 135/85 lors d’une automesure. Plus la tension est élevée, plus le risque de maladie cardiovasculaire est important.\n- **Chol (Serum Cholestrol) / Cholestérol sérique en mg/dl** : Le cholestérol total
                Sous le terme de cholestérol total, on inclut les taux de cholestérol HDL et LDL, ainsi qu’un cinquième du taux de triglycérides. Ce taux est habituellement inférieur à 2 g/l.
                    \n**_Le cholestérol LDL_** : Egalement appelé mauvais cholestérol. Dans le sang, la grande majorité du cholestérol total est composée de cholestérol LDL. Chez un patient, le taux de cholestérol LDL souhaitable est déterminé par le médecin en fonction de la présence de facteurs de risque cardiovasculaire. En l’absence de facteur de risque, un taux de cholestérol LDL est considéré comme normal lorsqu’il est inférieur à 1,6 g/l. Si le patient présente un ou plusieurs facteurs de risque (par exemple, un homme de plus de 50 ans), cette valeur limite est de 1,3 g/l. Au-delà, des mesures thérapeutiques doivent être prises.
                    \n**_Le cholestérol HDL_** : Également appelé bon cholestérol, son rôle est de capter le cholestérol en excès dans le sang et de le conduire au foie pour qu’il soit éliminé avec la bile. Le taux de cholestérol HDL est considéré trop faible lorsqu’il est inférieur à 0,35 g/l. Un taux élevé de cholestérol HDL (plus de 0,60 g/l) protège des maladies cardiovasculaires et annule un facteur de risque cardiovasculaire. Ainsi, un homme de 50 ans (un facteur de risque) qui présente un taux de cholestérol LDL de 1,5 g/l et un taux de cholestérol HDL de 0,65 g/l ne sera pas considéré comme nécessitant une prise en charge médicale.\n- **Fbs (Fasting Blood Sugar) / Sucre dans le sang à jeun > 120 mg/dl** : La glycémie, c’est la teneur en sucre ou taux de glucose dans le sang. Ce sucre a une importance pour le bon fonctionnement de l’organisme, puisque les cellules du corps humain en ont besoin pour produire de l’énergie par l’effet de transformation du glucose en glycogène.
                Les personnes qui ne sont pas sujettes au diabète ou tout autre type d’affection chronique ont une glycémie normale. Celle-ci se situe entre 0,70 et 1,10 g par litre de sang à jeun. 
                Lorsque la glycémie est basse, celle-ci est inférieure à 0,60 g par litre. Le patient est alors en état d’hypoglycémie.A contrario, quand la glycémie est élevée, le taux de sucre est supérieur à 1,10 g par litre, à jeun. Le patient est ainsi en état d’hyperglycémie modérée.
⚠️              La glycémie est à un taux trop élevé quand elle est **_supérieure à 1,26 g par litre_**.\n- **Restecg (Resting Electrocardiographic Results) / Résultats électrocardiographiques au repos** : Un électrocardiogramme, ou ECG, est un examen qui mesure l’activité électrique du cœur. Ces informations peuvent être utilisées pour évaluer la fréquence et le rythme cardiaques, ainsi que la fonction cardiaque globale.
                La fourchette normale de la fréquence cardiaque est de **_60 à 100 battements par minute_**.
                Une fréquence cardiaque trop rapide ou trop lente peut être le signe d’un problème. Un rythme cardiaque irrégulier peut également être le signe d’une affection sous-jacente.\n- **Thalach (Maximum Heart Rate Achieved)** : Fréquence cardiaque maximale atteinte. \n- **Exang (Exercise Induced Angina) / Angine induite par l'exercice** : L’angine est un type de malaise ou de douleur à la poitrine qui survient quand le cœur ne reçoit pas tout l’oxygène dont il a besoin pour faire son travail. On décrit souvent l’angine comme une douleur ou une pression au milieu de la poitrine qui peut se propager aux bras, au cou ou à la mâchoire. Ces symptômes sont parfois accompagnés d’un essoufflement, de transpiration ou de nausées.\n- **Oldpeak (ST Depression Induced by Exercise) / Dépression du segment ST induite par l'exercice par rapport au repos** : Le segment ST, en conditions normales, est plat ou isoélectrique, bien qu’il puisse présenter de petites variations mineures de 0.5 mm.\n- **Slope (Slope of the Peak Exercise ST Segment) / Pente du segment ST à l'effort maximal** : on en voit souvent durant l’effort physique ; elles présentent habituellement une élévation rapide au moment où elles croisent la ligne isoélectrique rapidement (pente ascendante).\n- **Ca (Number of Major Vessels Colored by Fluoroscopy) / Nombre de vaisseaux majeurs colorés par fluoroscopie** \n- **Thal (Thalassemia) / Type de thalassémie** : Les thalassémies sont un groupe de maladies héréditaires dues à une perturbation de la fabrication de l’une des quatre chaînes d’acides aminés qui constituent l’hémoglobine (la protéine présente dans les globules rouges qui transporte l’oxygène)."""
                )

    # Afficher une série de graphiques
        st.subheader("Graphiques :")

        selected_graph_type = st.selectbox("Sélectionnez le type de graphique", ["Heatmap", "Boxplot", "Analyse en Composantes Principales"])

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

        elif selected_graph_type == "Analyse en Composantes Principales":
            st.image("ACP heart.png", use_column_width=True)

    ## MALADIES RENALES ##
    # Charger les données
    if selected_section == "Informations maladies" and selected_disease == "Maladies rénales":
        file_path = "dfreinclean.csv"
        rein = load_data(file_path)

    # Afficher la définition des variables
        st.subheader('Définition des variables :')
        st.write("""
                - **Age (age)** : L'âge du patient, un facteur important dans l'évaluation du risque de maladie rénale chronique, car la **_fonction rénale a tendance à diminuer avec l'âge_**.
                - **Blood Pressure (Pression artérielle) - bp** : La pression artérielle du patient, mesurée en millimètres de mercure (mmHg). La **_pression artérielle élevée est un facteur de risque bien connu pour la maladie rénale chronique_**.
                - **Specific Gravity (Gravité spécifique de l'urine) - sg** : La concentration de l'urine, mesurée par la gravité spécifique. Des changements dans la gravité spécifique peuvent indiquer des problèmes rénaux. 
                Une **_densité inférieure à 1,003 peut indiquer un problème rénal_**, mais également une consommation d'eau excessive.
                - **Albumin (Taux d'albumine) - al** : Le taux d'albumine dans le sang. **_Une augmentation de l'albumine dans l'urine peut être un signe précoce de dysfonctionnement rénal_**.
                - **Sugar (Taux de sucre) - su** : Le taux de sucre dans le sang. **_Le diabète, s'il n'est pas contrôlé, peut contribuer au développement de la maladie rénale chronique_**.
                - **Red Blood Cells (Nombre de globules rouges) - rc** : Le nombre de globules rouges dans le sang, qui peut être affecté par des problèmes rénaux.
                - **Pus Cell (Présence de cellules de pus)** : La présence de cellules de pus dans l'urine, qui peut être un signe d'infection rénale.
                - **Pus Cell Clumps (Agrégats de cellules de pus)** : Agrégats de cellules de pus dans l'urine, un autre indicateur potentiel d'infection.
                - **Bacteria (Présence de bactéries)** : La présence de bactéries dans l'urine, indiquant une possible infection.
                - **Blood Glucose Random (Taux de sucre sanguin aléatoire) - bgr** : Le taux de sucre dans le sang à un moment donné, ce qui peut être lié au contrôle du diabète.
                - **Blood Urea (Taux d'urée sanguine) - bu** : Le taux d'urée dans le sang, qui peut être affecté par la fonction rénale.
                Les valeurs normales de l’urée sont comprises entre :\n
                    _2,5 et 7,6 mmol / L (ou 0,10 à 0,55 g / L), dans le sang_                          
                    _300 et 500 mmol / 24 heures, dans les urines._\n
                    Chez les enfants et les femmes enceintes, ces valeurs sont variables.
                Un **_taux élevé d'urée peut indiquer un dysfonctionnement rénal ou hépatique_**, qui peut être aigu ou chronique et affecter les deux reins.
                - **Serum Creatinine (Taux de créatinine sérique)** : Le taux de créatinine dans le sang, une mesure courante de la fonction rénale.\n
                    _créatinine sanguine chez la femme_ : 53 à 115 µmol/L\n
                    _créatinine sanguine chez l'homme_ : 88 à 150 µmol/L\n
                    _créatinine urinaire chez la femme_ : 6 à 13 mmol/24 heures\n
                    _créatinine urinaire chez l'homme_ : 7 à 14 mmol/24 heures\n
                Un **_taux de créatinine par prise de sang élevé peut signifier que vos reins ne fonctionnent pas correctement_**. C'est souvent le signe d'une maladie rénale, mais cela peut aussi être dû à d'autres conditions fréquentes, comme l'hypertension artérielle, le diabète, ou certaines maladies musculaires.
                - **Sodium (Taux de sodium) - sod** : Le taux de sodium dans le sang, qui peut être affecté par les problèmes rénaux.
                Le **_taux normal de sodium dans le sang se situe entre 136 et 145 milliéquivalents par litre_** (mEq/L). Les médecins parlent d’hyponatrémie lorsque le taux de sodium dans le sang (natrémie) se situe en dessous de 135 mEq/L. 
                Un faible taux de sodium a plusieurs causes, notamment une consommation excessive de liquides, l'insuffisance rénale, l'insuffisance cardiaque, la cirrhose et l'utilisation de diurétiques.
                - **Potassium (Taux de potassium) - pot** : Le taux de potassium dans le sang, une électrolyte important régulée par les reins.
                En temps normal, le taux de potassium dans le sang se situe entre 3,6 et 5 mmol/l (ou 130 à 200 mg/l). 
                Les deux principales causes d’un taux de potassium élevé sont la prise de médicaments (certains diurétiques contre l’hypertension artérielle, les digitaliques dans les troubles du rythme cardiaque…) et l’insuffisance rénale. Les autres causes sont le diabète, l’exercice physique intense ou encore les causes de lyses cellulaires (destruction des cellules), comme les brûlures étendues, les infarctus.
                On classe **_l’hyperkaliémie en 3 stades_**, du plus léger au plus sévère.\n
                    _Stade léger_ : entre 5,5 et 5,9 mmol/l\n
                    _Stade modéré_ : entre 6 et 6,5 mmol/l\n
                    _Stade sévère_ : supérieur à 6,5 mmol/l\n
                C’est lorsque le taux de potassium dans le sang atteint ou dépasse **_6,5 mmol/l que la présence de ce minéral dans le corps est vraiment dangereuse_**. Dans ce cas de figure, une hémodialyse est souvent réalisée (épuration du sang à l’aide d’un rein artificiel), afin d’éviter l’accident cardiaque.
                - **Haemoglobin (Taux d'hémoglobine) - hemo** : La concentration d'hémoglobine dans le sang.
                Le taux normal d’hémoglobine glyquée se situe entre 4,1 et 5,4 % chez les personnes non-diabétiques. Un taux supérieur à 5,4% peut être associé à un risque relatif de développer une maladie coronarienne ou un accident vasculaire cérébral.
                En revanche, un **_taux d’hémoglobine faible est synonyme d’anémie, et peut être la conséquence d’une insuffisance rénale_** (cf globules rouges et anémie).
                - **Packed Cell Volume (Volume de globules rouges tassés)** : La proportion du volume sanguin occupé par les globules rouges.
                - **White Blood Cell Count (Nombre de globules blancs) - wc** : Le nombre de globules blancs dans le sang, **_indiquant une possible réponse immunitaire ou une infection_**.
                Le nombre normal total se situe entre **_4 000 et 11 000 cellules par microlitre_**.
                - **Hypertension (Présence d'hypertension)** : La présence d'une pression artérielle élevée.
                - **Coronary Artery Disease (Présence de maladie coronarienne)** : La présence de maladie coronarienne, qui peut affecter la circulation sanguine vers les reins.
                - **Appetite (Niveaux d'appétit)** : Les niveaux d'appétit du patient.
                A mesure que la fonction rénale s’aggrave et que de plus en plus de déchets métaboliques s’accumulent dans le sang, la personne peut ressentir une asthénie, une faiblesse généralisée et des difficultés de concentration intellectuelle. Elle peut présenter une perte d’appétit et un essoufflement.
                - **Pedal Edema (Présence d'œdème des pieds)** : La présence d'œdème des pieds, pouvant être associée à une rétention d'eau liée à la maladie rénale.
                - **Anemia (Présence d'anémie)** : La présence d'anémie, souvent liée à des problèmes rénaux.
                Valeurs normales : 13 grammes par décilitre (g/dl) chez l'homme ; 12 g/dl chez la femme ; 10,5 g/dl chez la femme enceinte à partir du 2ème trimestre de grossesse."""
        )

    # Afficher une série de graphiques
        st.subheader("**Graphiques :**")

        selected_graph_type = st.selectbox("Sélectionnez le type de graphique", ["Boxplot", "Heatmap", "Analyse en Composantes Principales"])

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
        elif selected_graph_type == "Analyse en Composantes Principales":
            st.image("ACP rein.png", use_column_width=True)
        
    ## CANCER DU SEIN ##
    # Charger les données
    if selected_section == "Informations maladies" and selected_disease == "Cancer du sein":
        file_path = "dfcancersein.csv"
        sein = load_data(file_path)

    # Afficher la définition des variables
        st.subheader('Définition des variables :')
        st.write("""
                - **Radius Mean** : Moyenne des distances du centre aux points sur le périmètre : Il mesure la moyenne des distances du centre d'une tumeur aux points sur son périmètre. Cela peut donner des informations sur la taille de la tumeur.
                - **Area Mean** : Moyenne de la surface de la tumeur : Il représente la moyenne de la surface de la tumeur, ce qui peut également être indicatif de la taille de la tumeur.
                - **Compactness Mean** : Moyenne de (périmètre^2 / surface - 1,0) : C'est une mesure de la forme de la tumeur. Une valeur plus élevée peut indiquer une tumeur plus irrégulière.
                - **Concavity Mean** : Moyenne de la gravité des parties concaves du contour : Mesure la gravité des parties concaves de la tumeur, donnant une indication sur la forme des contours.
                - **Concave Points Mean** : Moyenne du nombre de parties concaves du contour : C'est la moyenne du nombre de parties concaves du contour. Cela pourrait aider à caractériser davantage la forme de la tumeur.
                - **Area Worst** : Aire la plus mauvaise (moyenne des trois plus grandes valeurs) de la tumeur : C'est la moyenne des trois plus grandes valeurs de la surface de la tumeur. Elle donne une indication de la taille maximale probable de la tumeur.
                - **Compactness Worst** : Compacité la plus mauvaise (moyenne des trois plus grandes valeurs) de la tumeur : Mesure la compacité maximale probable de la tumeur, basée sur la moyenne des trois plus grandes valeurs de la caractéristique de compacité.
                - **Concavity Worst** : Concavité la plus mauvaise (moyenne des trois plus grandes valeurs) de la tumeur : C'est la moyenne des trois plus grandes valeurs de la concavité, donnant une indication sur la gravité des parties concaves dans la tumeur.
                - **Area SE** : Erreur standard de la surface de la tumeur : Il représente l'erreur standard de la surface de la tumeur.
                - **Fractal Dimension SE** : Erreur standard de "l'approximation du littoral" - 1 : Mesure l'approximation fractale de la tumeur.
                - **Symmetry Worst** : Symétrie la pire (moyenne des trois plus grandes valeurs) de la tumeur :  C'est la moyenne des trois plus grandes valeurs de la symétrie de la tumeur.
                - **Fractal Dimension Worst (Dimension fractale la pire)** : C'est la moyenne des trois plus grandes valeurs de la dimension fractale de la tumeur."""
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

    # Définir la fonction pour créer le graphique
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

        elif selected_graph_type == "Analyse en Composantes Principales":
            st.image("ACP cancer sein.png", use_column_width=True)


    ## MALADIES DU FOIE ##
    # Charger les données
    if selected_section == "Informations maladies" and selected_disease == "Maladies du foie":
            file_path = "df_liver_M0F1 .csv"
            foie = load_data(file_path)

        # Afficher la définition des variables
            st.subheader('Définition des variables :')
            st.write("""
                    - **Age (age du patient)** :\n 
                    **_Cancer du foie_** : le cancer du foie est l’une des localisations de cancer dont l’incidence croissante contribue ces dernières décennies à l’augmentation des cancers en France.
                    On estime à 6 433 le nombre de nouveaux cas d’hépatocarcinomes survenus en 2005, dont **_79% chez l’homme_**. 95% des personnes atteintes avaient **_50 ans et plus_**, les âges moyens au diagnostic étant de **_71 ans chez la femme et 68 ans chez l’homme_**. L’évolution de l’incidence diffère chez l’homme et la femme. Chez celle-ci, la hausse est régulière sur l’ensemble de la période 1980-2005 : 4% de plus par an en moyenne. Chez l’homme, elle est moins marquée à la fin de la période qu’au début : 3,8% d’évolution annuelle entre 1980 et 2005 mais seulement 1,9% de plus par an entre 2000 et 2005 (INCa, 2009).\n
                    **_Cyrrhose du foie_** : La cirrhose du foie est une maladie du foie qui touche environ 200 000 personnes en France avec 15 000 décès chaque année. **_L'âge moyen de diagnostic est de 55 ans_**. 
                    - **Gender (Genre du patient)** : Depuis 20 ans, le nombre de **_cirrhose du foie augmente chez la femme_**. Elle aurait plus de risque de développer une maladie hépatique liée à l'alcool ou un "foie gras". 
                    - **Total Bilirubin (Bilirubine totale dans le sang)** : La bilirubine est produite par la dégradation des globules rouges, en particulier de l’hémoglobine. C’est une étape normale. Arrivés en fin de vie, les globules rouges sont détruits par la rate. La bilirubine est, quant à elle, captée dans la rate par une protéine et transportée vers le foie, puis excrétée dans le tube digestif où elle donne aux selles leur couleur. 
                    La mesure de la bilirubine dans le sang est prescrite en cas de jaunisse ou de suspicion d'une perturbation d'organes tels que le foie ou la vésicule biliaire
                    La quantité de bilirubine totale dans le sang est **_normalement comprise entre 0,3 et 1,9 mg / dl_** (milligrammes par décilitre).
                    La quantité de bilirubine conjuguée, appelée aussi **_bilirubine directe, est normalement comprise entre 0 et 0,3 mg / dl_**.
                    - **Alkaline Phosphotase (Taux de phosphatase alcaline dans le sang)** : 
                    Les phosphatases alcalines (PAL) sont des enzymes fabriquées par plusieurs tissus de l’organisme et plus particulièrement par le foie, les os, l’intestin et le placenta lors de la grossesse. Parfois bénin, un **_taux de phosphatases alcalines élevé sert également au diagnostic de maladies du foie et des os_**.
                    Un **_taux normal de phosphatases alcalines se situe entre : 
                    30 et 125 UI/L (unité internationale par litre) chez l’adulte,
                    70 et 450 UI/L chez l’enfant et l’adolescent_**.
                    Chez la femme enceinte, le taux de PAL tend à augmenter.
                    - **Alamine Aminotransferase (Taux d'alanine aminotransférase dans le sang)** : 
                    L'alanine aminotransférase est un enzyme nécessaire au bon fonctionnement de l'organisme. On le retrouve à divers endroits dans le corps, dont les muscles, le cœur, les reins et le foie. C'est au niveau du foie qu'on en retrouve les plus grandes quantités. **_L'augmentation de cette enzyme dans le sang est presque toujours associée à une atteinte hépatique_**. L'ALT est essentiellement un marqueur de dommage hépatocellulaire.
                    Valeurs cibles : 
                    chez l'homme : 10 à 32 U/L
                    chez la femme : 9 à 24 U/L
                    Les valeurs cibles varient en fonction de l'âge. Elles peuvent aussi varier d'un laboratoire à l'autre.
                    **_Son augmentation dans le plasma sanguin signe une cytolyse hépatique_**.
                    - **Albumin and Globulin Ratio (Rapport de l'albumine à la globuline dans le sang)** : 
                    L’albumine est présente en grande majorité dans le sang. 60 % de cette protéine sont dans les tissus, tandis que les 40 % restants sont dans le sang. Sa principale fonction est de transporter les vitamines, hormones, enzymes, médicaments, bilirubine non conjuguée, etc. dans les tissus, puisqu’elle circule facilement dans le sang. Elle joue aussi le rôle de "stabilisateur" du volume du sang, pour l’empêcher de perdre trop d’eau, car elle agit dans le maintien de la pression colloïdo-osmotique du sang.
                    Le test de protéines totales mesure l’albumine et les globulines. 
                    À titre indicatif, la **_valeur normale des protéines totales sériques est comprise entre 65 et 80 grammes/L_**. Le **_apport albumine/globuline se situe entre 1,2 et 1,8_**.
                    De faibles valeurs de protéines totales sont associées aux conditions suivantes : 
                    Leucémie,
                    Malabsorption intestinale (comme dans la maladie cœliaque ou la maladie intestinale inflammatoire),
                    Dénutrition,
                    Immunosuppression,
                    Troubles hématologiques,
                    Maladie rénale,
                    (Dans la glomérulonéphrite et le syndrome néphrotique, il y a perte de protéines dans les urines),
                    Maladie hépatique sévère,
                    (Il n’y a pas de production de protéines, comme dans le cas d’une cirrhose ou d’une insuffisance hépatique),
                    Insuffisance cardiaque congestive
                    (Il y a une augmentation du volume plasmatique, ce qui diminue la concentration des protéines par dilution).
                    Un test de protéines totales et de rapport albumine/globuline (A/G) mesure la quantité totale de protéines dans votre sang. Il existe deux grands types de protéines dans le sang :\n
                    **_Albumine_** , qui aide à empêcher le sang de s'échapper des vaisseaux sanguins. Il aide également à déplacer les hormones, les médicaments, les vitamines et d’autres substances importantes dans tout le corps. L'albumine est fabriquée dans le foie.\n
                    **_Globulines_** , qui aident à combattre les infections et à déplacer les nutriments dans tout le corps. Certaines globulines sont produites par le foie. D'autres sont fabriqués par le système immunitaire .
                    Le test compare également la quantité d’albumine dans votre sang à la quantité de globuline.
                    La comparaison s’appelle le rapport albumine/globuline (A/G).
                    Si vos niveaux de protéines totales ou vos résultats de rapport A/G ne sont pas normaux, cela peut être le signe d’un problème de santé grave."""
            )
    
    # Afficher une série de graphiques
            st.subheader("**Graphiques :**")

            selected_graph_type = st.selectbox("Sélectionnez le type de graphique", ["Boxplot", "Heatmap", "Analyse en Composantes Principales"])

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
                
            elif selected_graph_type == "Analyse en Composantes Principales":
                st.image("ACP liver.png", use_column_width=True)


    ## PREDICTION DIABETE ##
    if selected_section == "Prédiction" and selected_disease == "Diabète":
    
        # Charger les données
        file_path = "df_diabete_fin (1).csv"
        diabete = load_data(file_path)

        # Features and target variable
        feature_columns1 = ['Pregnancies', 'Glucose', 'BloodPressure', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        predicted_class1 = ['Outcome']

        # Split the data
        X = diabete[feature_columns1].values
        y = diabete[predicted_class1].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=10)

        # Train your model

        logistic_regression_model = LogisticRegression(random_state=10)
        logistic_regression_model.fit(X_train, y_train.ravel())

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
        Insulin = st.slider("Insulin", min_value=10.0, max_value=900.0, step=0.1, value=157.0, label_visibility="collapsed")
        st.markdown("<h6 style='font-size: 18px;'>BMI :</h6>", unsafe_allow_html=True)
        BMI = st.slider("BMI", min_value=10.0, max_value=70.0, step=0.1, value=32.0, label_visibility="collapsed")
        st.markdown("<h6 style='font-size: 18px;'>DiabetesPedigreeFonction :</h6>", unsafe_allow_html=True)
        DiabetesPedigreeFunction = st.slider("DiabetesPedigreeFonction", min_value=0.0, max_value=3.0, step=0.01, value=0.50, label_visibility="collapsed")
        st.markdown("<h6 style='font-size: 18px;'>Age :</h6>", unsafe_allow_html=True)
        Age = st.slider("Age", min_value=20, max_value=90, step=1, value=32, label_visibility="collapsed")

        # Make prediction
        input_data = [[Pregnancies, Glucose, BloodPressure, Insulin, BMI, DiabetesPedigreeFunction, Age]]
        prediction = logistic_regression_model.predict(input_data)

        # Display result
        st.subheader("_Prédiction_ :")
        if prediction[0] == 1:
            st.markdown("<h3 style='font-size: 25px;'>Avec les informations renseignées, le modèle prédit que le patient est susceptible d'être atteint d'un diabète.<br>Veuillez consulter un professionnel de santé.</h3>", unsafe_allow_html=True)
            st.image("https://www.arsenre.com/wp-content/uploads/2022/07/85477-580-doctor-6810751_640.png", use_column_width=True)
        else:
            st.markdown("<h3 style='font-size: 25px;'>Avec les informations renseignées, le modèle prédit que le patient n'est pas atteint d'un diabète.</h3>", unsafe_allow_html=True)
            st.image("diabète.png", use_column_width=True)
    
    ## PREDICTION MALADIES CARDIAQUES ##
    if selected_section == "Prédiction" and selected_disease == "Maladies cardiaques":
        file_path = "dfheartML.csv"
        coeur = load_data(file_path)

        # Features and target variable
        X = coeur[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca' , 'thal']]
        y = coeur.iloc[:, -1]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

        # Train your model
        RandomForest = RandomForestClassifier()
        RandomForest.fit(X_train, y_train)

        # Input fields
        st.subheader("_Entrez les informations du patient_ :")
        st.write("")
        st.markdown("<h6 style='font-size: 18px;'>Age :</h6>", unsafe_allow_html=True)
        age = st.slider("Age", min_value=0.0, max_value=125.0, step=1.0, value=40.0, label_visibility="collapsed")
        st.write("")
        st.markdown("<h6 style='font-size: 18px;'>Sex (0 : male, 1 : female):</h6>", unsafe_allow_html=True)
        sex = st.selectbox("Sex", [0.0, 1.0], label_visibility="collapsed")
        st.write("")
        st.markdown("<h6 style='font-size: 18px;'>cp :</h6>", unsafe_allow_html=True)
        cp = st.slider("cp", min_value=0.0, max_value=5.0, step=1.0, value=1.0, label_visibility="collapsed")
        st.write("")
        st.markdown("<h6 style='font-size: 18px;'>trestbps :</h6>", unsafe_allow_html=True)
        trestbps = st.slider("trestbps", min_value=50.0, max_value=250.0, step=5.0, value=125.0, label_visibility="collapsed")
        st.write("")
        st.markdown("<h6 style='font-size: 18px;'>chol :</h6>", unsafe_allow_html=True)
        chol = st.slider("chol", min_value=100.0, max_value=750.0, step=5.0, value=300.0, label_visibility="collapsed")
        st.write("")
        st.markdown("<h6 style='font-size: 18px;'>fbs (0 : No, 1 : Yes) :</h6>", unsafe_allow_html=True)
        fbs = st.selectbox("fbs", [0.0, 1.0], label_visibility="collapsed")
        st.write("")
        st.markdown("<h6 style='font-size: 18px;'>restecg :</h6>", unsafe_allow_html=True)
        restecg = st.slider("restecg", min_value=0.0, max_value=3.0, step=1.0, value=1.0, label_visibility="collapsed")
        st.write("")
        st.markdown("<h6 style='font-size: 18px;'>thalach :</h6>", unsafe_allow_html=True)
        thalach = st.slider("thalach", min_value=50.0, max_value=250.0, step=1.0, value=100.0, label_visibility="collapsed")
        st.write("")
        st.markdown("<h6 style='font-size: 18px;'>exang (0 : absence, 1 : presence) :</h6>", unsafe_allow_html=True)
        exang = st.selectbox("exang", [0.0, 1.0], label_visibility="collapsed")
        st.write("")
        st.markdown("<h6 style='font-size: 18px;'>oldpeak :</h6>", unsafe_allow_html=True)
        oldpeak = st.slider("oldpeak", min_value=0.0, max_value=10.0, step=0.1, value=1.0, label_visibility="collapsed")
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
        prediction = RandomForest.predict(input_data)

        # Display result
        st.subheader("_Prédiction_ :")
        if prediction[0] == 1:
            st.markdown("<h3 style='font-size: 25px;'>Avec les informations renseignées, le modèle prédit que le patient est susceptible d'être atteint d'une maladie cardiaque.<br>Veuillez consulter un professionnel de santé.</h3>", unsafe_allow_html=True)
            st.image("https://www.arsenre.com/wp-content/uploads/2022/07/85477-580-doctor-6810751_640.png", use_column_width=True)
        else:
            st.markdown("<h3 style='font-size: 25px;'>Avec les informations renseignées, le modèle prédit que le patient n'est pas atteint d'une maladie cardiaque.</h3>", unsafe_allow_html=True)
            st.image("https://haltemis.fr/wp-content/uploads/2022/01/healthy-lifestyle.png", use_column_width=True)

    ## PREDICTION CANCER DU SEIN ##
    if selected_section == "Prédiction" and selected_disease == "Cancer du sein":
        file_path = "dfcancersein.csv"
        sein = load_data(file_path)

        # Features and target variable
        feature_columns1 = ['radius_mean','texture_mean','smoothness_mean','compactness_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se','smoothness_se','compactness_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','smoothness_worst','compactness_worst','symmetry_worst','fractal_dimension_worst']

        # Split the data
        X = sein[feature_columns1].values
        y = sein.iloc[:, 0].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

        # Train your model
        random_forest_model = RandomForestClassifier()
        random_forest_model.fit(X_train, y_train.ravel())

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
        prediction = random_forest_model.predict(input_data)

        # Display result
        st.subheader("_Prédiction_ :")
        if prediction[0] == 0:
            st.markdown("<h3 style='font-size: 25px;'>Avec les informations renseignées, le modèle prédit que le patient est susceptible d'être atteint d'un cancer du sein.<br>Veuillez consulter un professionnel de santé.</h3>", unsafe_allow_html=True)
            st.image("https://www.arsenre.com/wp-content/uploads/2022/07/85477-580-doctor-6810751_640.png", use_column_width=True)
        else:
            st.markdown("<h3 style='font-size: 25px;'>Avec les informations renseignées, le modèle prédit que le patient n'est pas atteint d'un cancer du sein.</h3>", unsafe_allow_html=True)
            st.image("https://onconormandie.fr/wp-content/uploads/2023/09/Design-sans-titre-7.png", use_column_width=True)

    ## PREDICTION MALADIE REIN ##
    if selected_section == "Prédiction" and selected_disease == "Maladies rénales":
        file_path = "dfrein_ML.csv"
        rein = load_data(file_path)

    # Features and target variable
        X = rein[['sg', 'htn', 'hemo', 'dm', 'al', 'appet', 'rc', 'pc']]
        y = rein.iloc[:, -1]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.38, random_state=33)

        # Train your model
        GradientBoost = GradientBoostingClassifier()
        GradientBoost.fit(X_train, y_train)

        # Input fields
        st.subheader("_Entrez les informations du patient_ :")
        st.write("")
        st.markdown("<h6 style='font-size: 18px;'>Specific Gravity (sg) :</h6>", unsafe_allow_html=True)
        sg = st.slider("Specific Gravity", min_value=1.0, max_value=2.0, step=0.005, value=1.010, label_visibility="collapsed")
        st.write("")
        st.markdown("<h6 style='font-size: 18px;'>Hypertension (htn) - (0 : absence, 1 : presence) :</h6>", unsafe_allow_html=True)
        htn = st.selectbox("Hypertension", [0.0, 1.0], label_visibility="collapsed")
        st.write("")
        st.markdown("<h6 style='font-size: 18px;'>Hemoglobin (hemo) :</h6>", unsafe_allow_html=True)
        hemo = st.slider("Hemoglobin", min_value=0.0, max_value=20.0, step=0.1, value=12.0, label_visibility="collapsed")
        st.write("")
        st.markdown("<h6 style='font-size: 18px;'>Diabetes Mellitus (dm) - (0 : absence, 1 : presence) :</h6>", unsafe_allow_html=True)
        dm = st.selectbox("Diabetes Mellitus", [0.0, 1.0], label_visibility="collapsed")
        st.write("")
        st.markdown("<h6 style='font-size: 18px;'>Albumin (al) :</h6>", unsafe_allow_html=True)
        al = st.slider("Albumin", min_value=0.0, max_value=6.0, step=1.0, value=0.0, label_visibility="collapsed")
        st.write("")
        st.markdown("<h6 style='font-size: 18px;'>Appetite (appet) - (0 : poor, 1 : good) :</h6>", unsafe_allow_html=True)
        appet = st.selectbox("Appetite", [0.0, 1.0], label_visibility="collapsed")
        st.write("")
        st.markdown("<h6 style='font-size: 18px;'>Red Blood Cells (rc) :</h6>", unsafe_allow_html=True)
        rc = st.slider("Red Blood Cells", min_value=1.0, max_value=10.0, step=0.1, value=5.0, label_visibility="collapsed")
        st.write("")
        st.markdown("<h6 style='font-size: 18px;'>Pus Cell (pc) - (0 : normal, 1 : abnormal) :</h6>", unsafe_allow_html=True)
        pc = st.selectbox("Pus cell", [0.0, 1.0], label_visibility="collapsed")

        # Make prediction
        input_data = [[sg, htn, hemo, dm, al, appet, rc, pc]]
        prediction = GradientBoost.predict(input_data)

        # Display result
        st.subheader("_Prédiction_ :")
        if prediction[0] == 1:
            st.markdown("<h3 style='font-size: 25px;'>Avec les informations renseignées, le modèle prédit que le patient est susceptible d'être atteint d'une maladie rénale.<br>Veuillez consulter un professionnel de santé.</h3>", unsafe_allow_html=True)
            st.image("https://www.arsenre.com/wp-content/uploads/2022/07/85477-580-doctor-6810751_640.png", use_column_width=True)
        else:         
            st.markdown("<h3 style='font-size: 25px;'>Avec les informations renseignées, le modèle prédit que le patient n'est pas atteint d'une maladie rénale.</h3>", unsafe_allow_html=True)
            st.image("https://us.123rf.com/450wm/kahovsky/kahovsky1801/kahovsky180100062/92916263-%E3%81%8B%E3%82%8F%E3%81%84%E3%81%84%E9%9D%A2%E7%99%BD%E3%81%84%E3%80%81%E7%AC%91%E9%A1%94%E3%81%AE%E5%8C%BB%E8%80%85%E3%81%A8%E5%81%A5%E5%BA%B7%E5%B9%B8%E3%81%9B%E3%81%AA%E8%82%BA%E3%81%AE%E3%82%A4%E3%83%A9%E3%82%B9%E3%83%88%E3%80%82.jpg", use_column_width=True)
    

    ## PREDICTION MALADIES DU FOIE ##
    if selected_section == "Prédiction" and selected_disease == "Maladies du foie":
        file_path = "df_liver_M0F1 .csv"
        foie = load_data(file_path)

        # Features and target variable
        X = foie[['Age', 'Gender', 'Total_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 'Albumin_and_Globulin_Ratio']]
        y = foie.iloc[:, -1]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.45, random_state=42)

        # Train your model
        RandomForest = RandomForestClassifier()
        RandomForest.fit(X_train, y_train)

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
        Total_Bilirubin = st.number_input('Total bilirubin', min_value=0.0, max_value=35.0, value=2.52, step=0.01, label_visibility = 'collapsed')
        st.write("")
        st.markdown("<h6 style='font-size: 18px;'>Alkaline phosphotase :</h6>", unsafe_allow_html=True)
        Alkaline_Phosphotase = st.number_input("Alkaline phosphotase", min_value=0.0, max_value=2200.0, step=0.01, value=268.0, label_visibility = 'collapsed')
        st.write("")
        st.markdown("<h6 style='font-size: 18px;'>Alamine aminotransferase :</h6>", unsafe_allow_html=True)
        Alamine_Aminotransferase = st.number_input("Alamine aminotransferase", min_value=0.0, max_value=2000.0, step=0.01, value=62.52, label_visibility = 'collapsed')
        st.write("")
        st.markdown("<h6 style='font-size: 18px;'>Albumin and globulin ratio :</h6>", unsafe_allow_html=True)
        Albumin_and_Globulin_Ratio = st.number_input("Albumin and globulin ratio", min_value=0.0, max_value=10.0, step=0.01, value=0.97, label_visibility = 'collapsed')

        # Make prediction
        input_data = [[Age, Gender, Total_Bilirubin, Alkaline_Phosphotase, Alamine_Aminotransferase, Albumin_and_Globulin_Ratio]]
        prediction = RandomForest.predict(input_data)

        # Display result
        st.subheader("_Prédiction_ :")
        if prediction[0] == 1:
            st.markdown("<h3 style='font-size: 25px;'>Avec les informations renseignées, le modèle prédit que le patient est susceptible d'être atteint d'une maladie hépatique.<br>Veuillez consulter un professionnel de santé.</h3>", unsafe_allow_html=True)
            st.image("https://www.arsenre.com/wp-content/uploads/2022/07/85477-580-doctor-6810751_640.png", use_column_width=True)
        else:
            st.markdown("<h3 style='font-size: 25px;'>Avec les informations renseignées, le modèle prédit que le patient n'est pas atteint d'une maladie hépatique.</h3>", unsafe_allow_html=True)
            st.image("foie sain.png", use_column_width=True)

if __name__ == "__main__":
    main()






        







    