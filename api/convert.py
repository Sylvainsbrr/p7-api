import tensorflow as tf

# Remplacer './mon_modele' par le chemin réel où votre modèle Keras est sauvegardé
model = tf.keras.models.load_model('./mon_modele')


# Convertir le modèle
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Sauvegarder le modèle converti en format TensorFlow Lite
with open('modele_optimise.tflite', 'wb') as f:
    f.write(tflite_model)