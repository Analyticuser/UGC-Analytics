import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.linear_model import LassoCV

DATA_PATH = Path(__file__).parent / 'data' / 'train.csv'


def chargement_verification(path):
    if not path.exists():
        sys.exit("Le fichier est introuvable, veuillez téléchargez le dataset Kaggle Toxic Comment Classification et le placer dans le dossier data ")
    df = pd.read_csv(path)
    # Y a-t-il des données manquantes ?
    if len(df[df.isnull().any(axis=1)]) == 0:
        print('Il ne manque aucune donnée !')
    else:
        print('Certaines données sont manquantes')

    # Y a-t-il des doublons ?
    if len(df[df.duplicated(keep=False)]) == 0:
        print('Il n\'y a aucun doublon !')
    else:
        print('Des doublons sont à supprimer...')

    return df

def ajout_col_toxique_ou_non(df):
    df = df.set_index('id')
    toxic_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    for col in toxic_cols:
        print(f"le nombre de commentaires '{col}' est de {len(df[df[col] == 1])}")

    df['no_toxic_comment'] = (df[toxic_cols] == 0).all(axis=1).astype(int)
    df['toxic_comment'] = (df[toxic_cols] == 1).any(axis=1).astype(int)

    print(f"\nIl y a {df['no_toxic_comment'].sum()} commentaires non toxiques")
    print(f"Il y a {df['toxic_comment'].sum()} commentaires toxiques")

    return df

def ajout_features_textuelles(df):
    df['length_comm'] = df['comment_text'].str.len()
    df['maj'] = df['comment_text'].str.count(r'[A-Z]')
    df['ponct'] = df['comment_text'].str.count(r'[.,?;:!\'\"()\-&~#^¨$]')

    return df

def schemas_ponctuation(df):
    plt.figure()
    sns.histplot(data=df, x='length_comm', bins=50, kde=True)
    plt.xlim(0, 1500)
    plt.title('Longueur des commentaires (caractères et espaces)')
    plt.savefig('Histogramme_lgr_comm.png',dpi=300,bbox_inches='tight')
    plt.show()

    plt.figure()
    sns.scatterplot(data=df, x='length_comm', y='ponct', hue='toxic_comment', alpha=0.5)
    plt.title('Longueur des commentaires en fonction de la ponctuation ')
    plt.savefig('Lgr_comm_en_fonction_ponctuation.png',dpi=300,bbox_inches='tight')
    plt.show()

def ajout_ratio(df):
    df['length_comm'] = df['length_comm'].replace(0, 1)  # pour éviter la division par zéro
    df['ratio_maj'] = df['maj'] / df['length_comm']
    df['ratio_ponct'] = df['ponct'] / df['length_comm']

    return df

def heatmaps(df):
    colonnes_corr = df[['toxic_comment', 'toxic', 'severe_toxic', 'obscene', 'threat',
                         'insult', 'identity_hate', 'length_comm', 'ratio_maj', 'ratio_ponct']]
    correlation_matrix = colonnes_corr.corr()

    labels = df[['toxic_comment', 'toxic', 'severe_toxic', 'obscene', 'threat',
                 'insult', 'identity_hate']]
    labels_corr = labels.corr()

    plt.figure()
    sns.heatmap(data=correlation_matrix, annot=True, linewidths=0.5, fmt='.2f')
    plt.title("Corrélation entre la toxicité et les commentaires")
    plt.savefig('Corr_toxic_comm.png',dpi=300,bbox_inches='tight')
    plt.show()

    plt.figure()
    sns.heatmap(data=labels_corr, linewidths=0.5, annot=True, fmt='.2f')
    plt.title('Corrélation entre les types de toxicité')
    plt.savefig('Corr_type_toxic.png',dpi=300,bbox_inches='tight')
    plt.show()

def nuage_de_mots(df):
    textes_toxiques = df[df['toxic_comment'] == 1]['comment_text']
    texte_entier = " ".join(textes_toxiques)
    nuage = WordCloud(width=800, height=400, background_color='black',
                       stopwords=STOPWORDS, max_words=100).generate(texte_entier)
    plt.figure()
    plt.imshow(nuage, interpolation='bilinear')
    plt.axis('off')
    plt.title("Mots les plus fréquents dans les commentaires toxiques")
    plt.savefig('nuage_toxic.png',dpi=300,bbox_inches='tight')
    plt.show()

    textes_sains = df[df['toxic_comment'] == 0]['comment_text']
    texte_clean = " ".join(textes_sains)
    nuage_sain = WordCloud(width=800, height=400, background_color='white',
                            stopwords=STOPWORDS, max_words=100).generate(texte_clean)
    plt.figure()
    plt.imshow(nuage_sain, interpolation='bilinear')
    plt.axis('off')
    plt.title("Mots les plus fréquents dans les commentaires NON toxiques")
    plt.savefig('nuage_sain.png',dpi=300,bbox_inches='tight')
    plt.show()

def entrainement_evaluation_model(df):
    df['severite_comm'] = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum(axis=1)
    X = df[['length_comm', 'ratio_maj', 'ratio_ponct']]
    y = df['severite_comm']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train, y_train)

    test_predictions = model.predict(X_test)

    MAE = mean_absolute_error(y_test, test_predictions)
    MSE = mean_squared_error(y_test, test_predictions)
    RMSE = np.sqrt(MSE)
    print(f"MAE = {MAE:.3f}, MSE = {MSE:.3f}, RMSE = {RMSE:.3f}")

    residus = y_test - test_predictions

    plt.figure()
    sns.scatterplot(x=y_test, y=residus)
    plt.axhline(y=0, color='red', ls='--')
    plt.xlabel('Vrai score de sévérité')
    plt.ylabel('Erreur de prédiction (Résidus)')
    plt.title('Diagramme résiduels entre les prédictions et les valeurs réelles')
    plt.savefig('Diag_résidus.png',dpi=300,bbox_inches='tight')
    plt.show()

    modele_lasso = LassoCV(cv=5, random_state=101)
    modele_lasso.fit(X_train, y_train)
    coefficients = pd.DataFrame({
        'Variable': X.columns,
        'Coefficient': modele_lasso.coef_
    })
    print(coefficients)

def main():
    df = chargement_verification(DATA_PATH)
    df = ajout_col_toxique_ou_non(df)
    df = ajout_features_textuelles(df)
    schemas_ponctuation(df)
    df = ajout_ratio(df)
    heatmaps(df)
    nuage_de_mots(df)
    entrainement_evaluation_model(df)


if __name__ == "__main__":
    main()
