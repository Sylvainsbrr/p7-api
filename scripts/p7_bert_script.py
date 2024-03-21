import time

import mlflow
import numpy as np

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

import pandas as pd
from sklearn.model_selection import train_test_split


# Charger les données
data = pd.read_csv('sampled_tweets_bert.csv')

# Initialisation de MLflow
mlflow.set_tracking_uri("mlruns")  # Chemin où  stocker les runs
mlflow.set_experiment("BERT_classification")  # Le nom d' expérience

### Préparation des données

# Séparation des données en ensembles d'entraînement, de validation et de test
train_data, temp_data, y_train, temp_labels = train_test_split(
    data['text'], data['target'], test_size=0.2, random_state=42)

validation_data, test_data, y_validation, y_test = train_test_split(
    temp_data, temp_labels, test_size=0.5, random_state=42)

### Préparation des données poour BERT

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# Tokeniser les textes et calculer leurs longueurs
token_lengths = []
for text in data['text']:
    tokens = tokenizer.encode(text, add_special_tokens=True)
    token_lengths.append(len(tokens))

# Convertir la liste des longueurs en DataFrame pour analyse
token_lengths_df = pd.DataFrame(token_lengths, columns=['length'])

# Analyser la distribution des longueurs de tokens
description = token_lengths_df.describe(percentiles=[.75, .90, .95, .99])
print(description)


max_length = 38  # Ajustez selon vos besoins

def encode_data(tokenizer, data, max_length):
    return tokenizer(data.tolist(), padding=True, truncation=True, max_length=max_length, return_tensors="tf")

X_train = encode_data(tokenizer, train_data, max_length)
X_validation = encode_data(tokenizer, validation_data, max_length)
X_test = encode_data(tokenizer, test_data, max_length)


model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(y_train.unique()))

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

model.compile(optimizer=optimizer, loss=loss, metrics=[metric])



batch_size = 16
epochs = 3

start_time = time.time()
model.fit(
    {'input_ids': X_train['input_ids'], 'attention_mask': X_train['attention_mask']},
    y_train.values,
    validation_data=({'input_ids': X_validation['input_ids'], 'attention_mask': X_validation['attention_mask']}, y_validation.values),
    batch_size=batch_size,
    epochs=epochs
)
end_time = time.time()
training_time = end_time - start_time



# Début du suivi MLflow
mlflow.start_run(run_name='Bert_model')
# Évaluation du modèle
y_pred_probs = model.predict({'input_ids': X_test['input_ids'], 'attention_mask': X_test['attention_mask']})
y_pred = np.argmax(y_pred_probs.logits, axis=1)
accuracy = accuracy_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred_probs.logits[:,1])  # Pour classification binaire
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)

# Enregistrement des métriques
mlflow.log_metric("accuracy", accuracy)
mlflow.log_metric("auc_roc", auc_roc)
mlflow.log_metric("specificity", specificity)
mlflow.log_metric("training_time_seconds", training_time)

# Sauvegarde du modèle
model.save_pretrained("Bert_model")

# Fin du suivi MLflow
mlflow.end_run()






#%%
