from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import logging

app = Flask(__name__)

# Configurer le logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FlaskApp")

# Mettre à jour le chemin pour pointer vers le dossier SavedModel
model_path = '../model/LTSM_Model_Glove'
logger.info('Chargement du modèle à partir de : %s', model_path)
model = load_model(model_path)

# Charger le tokenizer (ajustez le chemin si nécessaire)
tokenizer_path = '../model/tokenizer_glove.pickle'
logger.info('Chargement du tokenizer à partir de : %s', tokenizer_path)
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

@app.route('/predict', methods=['POST'])
def predict():
    logger.info('Requête /predict reçue')
    data = request.json
    if not data or 'text' not in data:
        logger.warning('Aucun texte fourni dans la requête')
        return jsonify({"error": "No text provided"}), 400

    text = data['text']
    logger.info('Texte pour prédiction: %s', text[:50])  # Log les 50 premiers caractères pour éviter les logs trop longs

    # Convertir le texte en séquences et paddez-les pour correspondre à l'entrée du modèle
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequences, maxlen=30)  # Ajustez `maxlen` si nécessaire

    # Prédiction
    logger.info('Prédiction en cours...')
    prediction = model.predict(padded_sequence)
    pred_class = np.argmax(prediction, axis=1)  # Ajustez selon votre cas d'utilisation

    # Retourner la classe prédite
    logger.info('Prédiction terminée. Classe prédite: %d', pred_class[0])
    return jsonify({"prediction": int(pred_class[0])})

if __name__ == '__main__':
    app.run(debug=True)