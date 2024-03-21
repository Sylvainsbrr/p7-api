# Importation des bibliothèques
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

import mlflow
import mlflow.sklearn
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score

from gensim.models import Word2Vec
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt
import mlflow
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input
from tensorflow.keras.metrics import AUC, Accuracy, Recall
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Embedding
from tensorflow.keras.metrics import Recall, AUC

import numpy as np
import mlflow
import mlflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import mlflow
from mlflow.sklearn import log_model



# Charger les données
data = pd.read_csv('sampled_tweets.csv')

from gensim.models import KeyedVectors

# Chemin vers le fichier .bin téléchargé
model_path = 'GoogleNews-vectors-negative300.bin'

# Charger le modèle Word2Vec
word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)

# Initialisation de MLflow
mlflow.set_tracking_uri("mlruns")  # Spécifiez le chemin où vous souhaitez stocker les runs
mlflow.set_experiment("p7_simple_scripts")  # Remplacez par le nom de votre expérience

def clean_and_stem(text):
    # Suppression des URLs
    text = re.sub(r'http\S+', '', text)  # Supprime toutes les occurrences d'URLs
    
    # Suppression des mentions (@) et des hashtags (#)
    text = re.sub(r'@\w+|#\w+', '', text)  # Supprime les mots commençant par @ ou #

    # Suppression des caractères non alphabétiques (garder uniquement les lettres)
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Supprime tous les caractères qui ne sont pas des lettres

    # Conversion en minuscules pour standardiser le texte
    text = text.lower()

    # Tokenisation : division du texte en mots
    tokens = word_tokenize(text)

    # Stemmisation : réduction des mots à leur racine
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in tokens if word not in stopwords.words('english')]  # Supprime également les stopwords

    return ' '.join(stemmed_tokens)

# Appliquer la fonction de nettoyage
data['cleaned_text'] = data['text'].apply(clean_and_stem)

# Séparation initiale en ensemble d'entraînement et temporaire (validation + test)
train_data, temp_data, y_train, temp_labels = train_test_split(
    data['cleaned_text'],  # Les textes nettoyés
    data['target'],        # Les étiquettes associées, en utilisant 'target' comme nom de la colonne
    test_size=0.4,         # 40% du dataset pour validation + test
    random_state=42        # Pour la reproductibilité
)


# Séparation de l'ensemble temporaire en validation et test
validation_data, test_data, y_validation, y_test = train_test_split(
    temp_data,             # Ensemble temporaire
    temp_labels,           # Étiquettes temporaire
    test_size=0.5,         # Diviser en deux parties égales
    random_state=42        # Pour la reproductibilité
)


# Création d'un pipeline avec Bag of Words et régression logistique
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', LogisticRegression())
])

start_time = time.time()
# Entraînement du modèle sur l'ensemble d'entraînement
pipeline.fit(train_data, y_train)
end_time = time.time()
training_time = end_time - start_time

# Fonction pour calculer les métriques
def evaluate_model(model, X, y_true):
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred_proba)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)

    return accuracy, auc_roc, specificity

# Évaluation sur l'ensemble de validation
accuracy_val, auc_roc_val, specificity_val = evaluate_model(pipeline, validation_data, y_validation)

# Évaluation sur l'ensemble de test
accuracy_test, auc_roc_test, specificity_test = evaluate_model(pipeline, test_data, y_test)

print(f"Validation - Accuracy: {accuracy_val}, AUC ROC: {auc_roc_val}, Specificity: {specificity_val}")
print(f"Test - Accuracy: {accuracy_test}, AUC ROC: {auc_roc_test}, Specificity: {specificity_test}")

# Initialisation de MLflow et enregistrement des paramètres, métriques, et du modèle
mlflow.start_run(run_name="logistic_regression_model")

mlflow.log_param("vectorizer", "CountVectorizer")
mlflow.log_param("classifier", "LogisticRegression")
mlflow.log_metric("training_time_seconds", training_time)
mlflow.log_metric("validation_accuracy", accuracy_val)
mlflow.log_metric("validation_auc_roc", auc_roc_val)
mlflow.log_metric("validation_specificity", specificity_val)
mlflow.log_metric("test_accuracy", accuracy_test)
mlflow.log_metric("test_auc_roc", auc_roc_test)
mlflow.log_metric("test_specificity", specificity_test)

log_model(pipeline, "bag_of_words_model")

mlflow.end_run()


def train_evaluate_model(model, X_train, y_train, X_validation, y_validation, X_test, y_test, model_name):
    """
    Entraîne un modèle, l'évalue sur les ensembles de validation et de test, et enregistre les métriques avec MLflow.
    
    Parameters:
    - model: Le modèle à entraîner.
    - X_train, y_train: Données d'entraînement.
    - X_validation, y_validation: Données de validation.
    - X_test, y_test: Données de test.
    - model_name: Nom du modèle pour l'enregistrement MLflow.
    """
    with mlflow.start_run(run_name=model_name):

        # Enregistrer le temps de début
        start_time = time.time()
        
        # Entraînement du modèle
        model.fit(X_train, y_train)

        # Enregistrer le temps de fin
        end_time = time.time()

        # Calculer la durée d'entraînement
        training_duration = end_time - start_time
        
        # Évaluation sur l'ensemble de validation
        y_validation_pred = model.predict(X_validation)
        # Pour les modèles sklearn, assurez-vous que y_validation_pred est correctement formaté pour les métriques binaires
        y_validation_pred = np.round(y_validation_pred).astype(int)
        validation_proba = model.predict_proba(X_validation)[:, 1] if hasattr(model, "predict_proba") else y_validation_pred
        validation_metrics = calculate_metrics(y_validation, y_validation_pred, validation_proba)
        
        # Évaluation sur l'ensemble de test
        y_test_pred = model.predict(X_test)
        # Assurez-vous que y_test_pred est correctement formaté pour les métriques binaires
        y_test_pred = np.round(y_test_pred).astype(int)
        test_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_test_pred
        test_metrics = calculate_metrics(y_test, y_test_pred, test_proba)
        
        # Enregistrement des métriques de validation dans MLflow
        for key, value in validation_metrics.items():
            mlflow.log_metric("validation_" + key, value)
        
        # Enregistrement des métriques de test dans MLflow
        for key, value in test_metrics.items():
            mlflow.log_metric("test_" + key, value)

        # Enregistrement du temps d'entraînement dans MLflow
        mlflow.log_metric("training_time_seconds", training_duration)
        
        # Enregistrement des paramètres du modèle
        mlflow.log_params(model.get_params())
        mlflow.set_tag("model_name", model_name)
        # Adaptation pour les modèles non-Keras
        if hasattr(model, 'save'):
            mlflow.keras.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, model_name)

        print(f"Le temps d'entraînement du modèle est de {training_duration} secondes.")
        print("Validation Metrics:", validation_metrics)
        print("Test Metrics:", test_metrics)
        return y_test_pred

def calculate_metrics(y_true, y_pred, y_proba):
    """
    Calcule et retourne un dictionnaire des métriques d'évaluation, y compris la spécificité.
    """
    # Calcul des différentes métriques
    accuracy = accuracy_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_proba)
    
    # Calcul de la spécificité
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    
    # Retourne un dictionnaire avec toutes les métriques
    metrics = {
        "accuracy": accuracy,
        "auc_roc": auc_roc,
        "specificity": specificity  # Ajout de la spécificité
    }
    return metrics

def print_metrics(metrics, dataset_name):
    """
    Affiche les métriques d'évaluation.
    """
    print(f"{dataset_name} Metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print()



# Fonction pour vectoriser un tweet
def vectorize_tweet(tweet, model):
    words = tweet.split()
    word_vectors = [model[word] for word in words if word in model]
    if len(word_vectors) == 0:
        return np.zeros(300)  # 300 est la taille du vecteur dans le modèle Google News
    return np.mean(word_vectors, axis=0)

# Vectorisation avec Word2Vec
X_train_w2v = np.array([vectorize_tweet(tweet, word2vec_model) for tweet in train_data])
X_validation_w2v = np.array([vectorize_tweet(tweet, word2vec_model) for tweet in validation_data])
X_test_w2v = np.array([vectorize_tweet(tweet, word2vec_model) for tweet in test_data])

# Configuration des paramètres du modèle (exemple)
xgb_params = {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 4}

# Création du modèle XGBoost
xgb_model = XGBClassifier(**xgb_params)

# Entraînement, évaluation et récupération des prédictions de test
y_test_pred = train_evaluate_model(xgb_model, X_train_w2v, y_train, X_validation_w2v, y_validation, X_test_w2v, y_test, "XGBoost_Model")

# Configuration des paramètres du modèle RFC (exemple)
rfc_params = {
    'n_estimators': 100,  # Le nombre d'arbres dans la forêt
    'max_depth': None,    # La profondeur maximale des arbres
    'random_state': 42    # Pour la reproductibilité
}

# Création du modèle RFC avec les paramètres spécifiés
rfc_model = RandomForestClassifier(**rfc_params)

# Entraînement et évaluation du modèle RFC
# Remplacez X_train_vect, y_train, etc., par vos propres variables si elles ont des noms différents
y_test_pred_rfc = train_evaluate_model(rfc_model, X_train_w2v, y_train, X_validation_w2v, y_validation, X_test_w2v, y_test, "RFC_Model")

#%%

#%%
