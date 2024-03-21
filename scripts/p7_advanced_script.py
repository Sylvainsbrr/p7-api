# Importation des bibliothèques
import pandas as pd
import os
import nltk

import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import time
import numpy as np

import mlflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors

# Télécharger les stopwords de nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Charger les données
data = pd.read_csv('sampled_tweets.csv')

# Chemin vers le fichier .bin téléchargé
model_path = 'GoogleNews-vectors-negative300.bin'

# Charger le modèle Word2Vec
word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)

# Initialisation de MLflow
mlflow.set_tracking_uri("mlruns")  # Chemin où  stocker les runs
mlflow.set_experiment("p7_advanced_scripts")  # Le nom d' expérience


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
    stemmed_tokens = [stemmer.stem(word) for word in tokens if word not in stopwords.words('english')]

    return ' '.join(stemmed_tokens)


# Appliquer la fonction de nettoyage
data['cleaned_text'] = data['text'].apply(clean_and_stem)

# Séparation initiale en ensemble d'entraînement et temporaire (validation + test)
train_data, temp_data, y_train, temp_labels = train_test_split(
    data['cleaned_text'],  # Les textes nettoyés
    data['target'],  # Les étiquettes associées, en utilisant 'target' comme nom de la colonne
    test_size=0.4,  # 40% du dataset pour validation + test
    random_state=42  # Pour la reproductibilité
)

# Séparation de l'ensemble temporaire en validation et test
validation_data, test_data, y_validation, y_test = train_test_split(
    temp_data,  # Ensemble temporaire
    temp_labels,  # Étiquettes temporaire
    test_size=0.5,  # Diviser en deux parties égales
    random_state=42  # Pour la reproductibilité
)


# Fonction pour vectoriser un tweet
def vectorize_tweet_optimized(tweet, model, vector_size=300):
    # Tokenisation du tweet et filtration des mots non présents dans le modèle Word2Vec
    words = [word for word in tweet.split() if word in model]

    # Initialisation d'un tableau pour stocker les vecteurs des mots
    word_vectors = np.zeros((len(words), vector_size))

    for i, word in enumerate(words):
        word_vectors[i, :] = model[word]

    # Calcul du vecteur moyen si le tweet contient des mots connus, sinon vecteur zéro
    if len(words) >= 1:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(vector_size)


# Vectorisation avec Word2Vec
X_train_w2v = np.array([vectorize_tweet_optimized(tweet, word2vec_model) for tweet in train_data])
X_validation_w2v = np.array([vectorize_tweet_optimized(tweet, word2vec_model) for tweet in validation_data])
X_test_w2v = np.array([vectorize_tweet_optimized(tweet, word2vec_model) for tweet in test_data])


def calculate_metrics(y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    auc_roc = roc_auc_score(y_true, y_pred_proba)
    return specificity, auc_roc


def evaluate_model_performance(y_true, y_pred_proba, threshold=0.5):
    """
    Évalue la performance du modèle en calculant diverses métriques à partir des étiquettes réelles et des probabilités
    prédites.

    Args:
    - y_true (array-like): Vecteur des étiquettes réelles.
    - y_pred_proba (array-like): Vecteur des probabilités prédites pour la classe positive.
    - threshold (float): Seuil pour convertir les probabilités en prédictions de classe.

    Returns:
    - Un dictionnaire contenant les métriques calculées.
    """
    # Conversion des probabilités prédites en étiquettes de classe basées sur le seuil
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Calcul des métriques
    accuracy = accuracy_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred_proba)

    # Calcul de la spécificité
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)

    # Affichage des résultats
    metrics = {
        'Accuracy': accuracy,
        'AUC ROC': auc_roc,
        'Specificity': specificity
    }

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    return metrics


# Construction du modèle avec régularisation et dropout
model_dense_word2vec = Sequential([
    Dense(256, activation='relu', input_shape=(X_train_w2v.shape[1],)),
    Dropout(0.5),  # Ajout d'une couche de Dropout pour la régularisation
    Dense(128, activation='relu'),
    Dropout(0.5),  # Un autre Dropout
    Dense(1, activation='sigmoid')
])

model_dense_word2vec.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Callback pour l'arrêt prématuré
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Entraînement du modèle
start_time = time.time()
history = model_dense_word2vec.fit(X_train_w2v, y_train, validation_data=(X_validation_w2v, y_validation),
                                   epochs=20, batch_size=64, callbacks=[early_stopping])
end_time = time.time()
training_time = end_time - start_time

# Prédictions sur l'ensemble de test
y_pred_proba_test = model_dense_word2vec.predict(X_test_w2v).flatten()
y_pred_proba_validation = model_dense_word2vec.predict(X_validation_w2v).flatten()

# Utiliser la fonction evaluate_model_performance pour obtenir les métriques pour l'ensemble de test
metrics_test = evaluate_model_performance(y_test, y_pred_proba_test)

# Utiliser la fonction evaluate_model_performance pour obtenir les métriques pour l'ensemble de validation
metrics_validation = evaluate_model_performance(y_validation, y_pred_proba_validation)

# Logging avec MLflow
with mlflow.start_run(run_name="Dense_Model_Word2vec"):
    mlflow.log_params({"epochs": 50, "batch_size": 64, "layers": "3 (Dense)", "dropout_rate": 0.5, "optimizer": "Adam"})

    # Log des métriques pour l'ensemble de test
    mlflow.log_metrics({f"test_{key}": value for key, value in metrics_test.items()})

    # Log des métriques pour l'ensemble de validation
    mlflow.log_metrics({f"validation_{key}": value for key, value in metrics_validation.items()})

    mlflow.log_metric("training_time_seconds", training_time)

    mlflow.keras.log_model(model_dense_word2vec, "Dense_Model_with_Dropout_and_Regularization")

from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional

#### LSTM

# Redimensionnement de X_train_w2v pour ajouter une dimension temporelle (timesteps = 1)
X_train_w2v_reshaped = X_train_w2v.reshape((X_train_w2v.shape[0], 1, X_train_w2v.shape[1]))
X_validation_w2v_reshaped = X_validation_w2v.reshape((X_validation_w2v.shape[0], 1, X_validation_w2v.shape[1]))
X_test_w2v_reshaped = X_test_w2v.reshape((X_test_w2v.shape[0], 1, X_test_w2v.shape[1]))

model_lstm_word2vec = Sequential([
    # Utilisation d'une couche LSTM bidirectionnelle pour capturer le contexte dans les deux directions
    Bidirectional(LSTM(128, return_sequences=True, input_shape=(1, X_train_w2v.shape[1]))),
    Dropout(0.5),  # Ajout d'un dropout pour la régularisation
    Bidirectional(LSTM(64)),  # Une autre couche LSTM pour capturer des caractéristiques à un niveau plus élevé
    Dropout(0.5),
    Dense(64, activation='relu'),  # Une couche Dense pour la classification
    Dense(1, activation='sigmoid')  # Couche de sortie
])

model_lstm_word2vec.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# EarlyStopping pour optimiser le nombre d'epochs
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

start_time = time.time()
# Entraînement avec EarlyStopping
history_lstm = model_lstm_word2vec.fit(X_train_w2v_reshaped, y_train, epochs=20, batch_size=32,
                                       validation_data=(X_validation_w2v_reshaped, y_validation),
                                       callbacks=[early_stopping])
end_time = time.time()
training_time = end_time - start_time

# Assurez-vous d'utiliser les données redimensionnées pour faire des prédictions
y_pred_proba_test = model_lstm_word2vec.predict(X_test_w2v_reshaped).flatten()
y_pred_proba_validation = model_lstm_word2vec.predict(X_validation_w2v_reshaped).flatten()

# Utiliser la fonction evaluate_model_performance pour obtenir les métriques pour l'ensemble de test
metrics_test = evaluate_model_performance(y_test, y_pred_proba_test)

# Utiliser la fonction evaluate_model_performance pour obtenir les métriques pour l'ensemble de validation
metrics_validation = evaluate_model_performance(y_validation, y_pred_proba_validation)

epochs_effective = len(history_lstm.history['loss'])  # Nombre réel d'epochs effectués

with mlflow.start_run(run_name="LSTM_Model_Word2vec"):
    mlflow.log_params({"epochs": epochs_effective, "batch_size": 32, "model_type": "LSTM"})

    # Log des métriques pour l'ensemble de test
    mlflow.log_metrics({f"test_{key}": value for key, value in metrics_test.items()})

    # Log des métriques pour l'ensemble de validation
    mlflow.log_metrics({f"validation_{key}": value for key, value in metrics_validation.items()})

    mlflow.log_metric("training_time_seconds", training_time)

    mlflow.keras.log_model(model_lstm_word2vec, "LSTM_Model_with_Word2Vec")

# Dense avec GloVe

# Chemin vers le dossier contenant les fichiers GloVe
glove_dir = 'glove.6B'

# Choix du fichier d'embedding spécifique à charger
embedding_dim = 100  # ou 50, 200, 300 selon le fichier que vous souhaitez utiliser
glove_file = os.path.join(glove_dir, f'glove.6B.{embedding_dim}d.txt')

# Chargement des embeddings GloVe dans un dictionnaire
embeddings_index = {}
with open(glove_file, encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))


def text_to_avg_vector(text_list, embedding_index, dim=100):
    vectors = np.zeros((len(text_list), dim))
    for i, text in enumerate(text_list):
        embeddings = [embedding_index.get(word, np.zeros(dim)) for word in text.split() if word in embedding_index]
        if embeddings:
            vectors[i] = np.mean(embeddings, axis=0)
    return vectors


# Conversion des textes en vecteurs moyens d'embeddings
X_train_vec = text_to_avg_vector(train_data, embeddings_index, dim=embedding_dim)
X_validation_vec = text_to_avg_vector(validation_data, embeddings_index, dim=embedding_dim)
X_test_vec = text_to_avg_vector(test_data, embeddings_index, dim=embedding_dim)

model_dense_glove = Sequential([
    Dense(512, activation='relu', input_shape=(embedding_dim,)),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model_dense_glove.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

start_time = time.time()
history_dense = model_dense_glove.fit(X_train_vec, y_train, epochs=20, batch_size=64,
                                      validation_data=(X_validation_vec, y_validation),
                                      callbacks=[early_stopping])
end_time = time.time()
training_time = end_time - start_time

# Prédictions sur l'ensemble de test
y_pred_proba_test = model_dense_glove.predict(X_test_vec).flatten()
y_pred_proba_validation = model_dense_glove.predict(X_validation_vec).flatten()

# Utiliser la fonction evaluate_model_performance pour obtenir les métriques pour l'ensemble de test
metrics_test = evaluate_model_performance(y_test, y_pred_proba_test)

# Utiliser la fonction evaluate_model_performance pour obtenir les métriques pour l'ensemble de validation
metrics_validation = evaluate_model_performance(y_validation, y_pred_proba_validation)

# Logging avec MLflow
with mlflow.start_run(run_name="Dense_Model_Glove"):
    mlflow.log_params({"epochs": 20, "batch_size": 64, "layers": "4 (Dense)", "dropout_rate": 0.5, "optimizer": "Adam"})

    # Log des métriques pour l'ensemble de test
    mlflow.log_metrics({f"test_{key}": value for key, value in metrics_test.items()})

    # Log des métriques pour l'ensemble de validation
    mlflow.log_metrics({f"validation_{key}": value for key, value in metrics_validation.items()})

    mlflow.log_metric("training_time_seconds", training_time)

    mlflow.keras.log_model(model_dense_glove, "Dense_Model_Glove")

# LTSM with Glove

# Tokenisation des textes pour déterminer la taille du vocabulaire
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['cleaned_text'])
total_words = len(tokenizer.word_index) + 1

# Calcul de la longueur maximale des séquences
sequence_lengths = data['cleaned_text'].apply(lambda x: len(x.split()))
max_sequence_length = sequence_lengths.max()

print(f"Taille totale du vocabulaire: {total_words}")
print(f"Longueur maximale des séquences: {max_sequence_length}")

# Tokenisation et conversion des textes en séquences d'entiers pour chaque sous-ensemble
sequences_train = tokenizer.texts_to_sequences(train_data)
sequences_validation = tokenizer.texts_to_sequences(validation_data)
sequences_test = tokenizer.texts_to_sequences(test_data)

# Padding des séquences pour qu'elles aient toutes la même longueur
X_train_padded = pad_sequences(sequences_train, maxlen=max_sequence_length, padding='post')
X_validation_padded = pad_sequences(sequences_validation, maxlen=max_sequence_length, padding='post')
X_test_padded = pad_sequences(sequences_test, maxlen=max_sequence_length, padding='post')

vocab_size = total_words

embedding_matrix = np.zeros((vocab_size, 100))  # 100 pour embedding_dim
for word, i in tokenizer.word_index.items():
    if i < vocab_size:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Dropout
from tensorflow.keras.initializers import Constant

model_lstm_glove = Sequential([
    Embedding(input_dim=vocab_size, output_dim=100, embeddings_initializer=Constant(embedding_matrix),
              input_length=30, trainable=False),  # 30 pour max_sequence_length
    SpatialDropout1D(0.2),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model_lstm_glove.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Supposons que y_train, y_validation et y_test sont déjà définis
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

start_time = time.time()
history = model_lstm_glove.fit(
    X_train_padded, y_train,
    validation_data=(X_validation_padded, y_validation),
    epochs=50,
    batch_size=64,
    callbacks=[early_stopping]
)
end_time = time.time()
training_time = end_time - start_time

# Prédictions sur l'ensemble de test
y_pred_proba_test = model_lstm_glove.predict(X_test_padded).flatten()
y_pred_proba_validation = model_lstm_glove.predict(X_validation_padded).flatten()

# Utiliser la fonction evaluate_model_performance pour obtenir les métriques pour l'ensemble de test
metrics_test = evaluate_model_performance(y_test, y_pred_proba_test)

# Utiliser la fonction evaluate_model_performance pour obtenir les métriques pour l'ensemble de validation
metrics_validation = evaluate_model_performance(y_validation, y_pred_proba_validation)

# Logging avec MLflow
with mlflow.start_run(run_name="LTSM_Model_Glove"):
    mlflow.log_params({"epochs": 10, "batch_size": 64, "layers": "4", "dropout_rate": 0.5, "optimizer": "Adam"})

    # Log des métriques pour l'ensemble de test
    mlflow.log_metrics({f"test_{key}": value for key, value in metrics_test.items()})

    # Log des métriques pour l'ensemble de validation
    mlflow.log_metrics({f"validation_{key}": value for key, value in metrics_validation.items()})

    mlflow.log_metric("training_time_seconds", training_time)

    mlflow.keras.log_model(model_lstm_glove, "LTSM_Model_Glove")

# %%
