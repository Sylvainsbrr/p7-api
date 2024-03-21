import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Chemin vers le modèle SavedModel
model_path = '../model/LTSM_Model_Glove'
model = load_model(model_path)

# Chemin vers le fichier tokenizer
tokenizer_path = '../model/tokenizer_glove.pickle'
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Texte de test
text_to_predict = "i love twitter and i love the world"

# Préparation des données de test
sequences = tokenizer.texts_to_sequences([text_to_predict])
padded_sequence = pad_sequences(sequences, maxlen=30)  # Assurez-vous que maxlen correspond à la configuration de votre modèle

# Pour la classification binaire
prediction = model.predict(padded_sequence)
pred_class = (prediction > 0.5).astype(int)  # Cela convertira la probabilité en 0 ou 1

print(f"Prédiction pour le texte '{text_to_predict}': {pred_class[0][0]}")
#%%
