from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from fastapi.concurrency import run_in_threadpool
from google.cloud import storage
from google.oauth2 import service_account
import json
import os
import tempfile

app = FastAPI()

# Fonction pour créer un client GCS en utilisant les credentials JSON
def create_gcs_client():
    credentials_info = json.loads(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON'))
    credentials = service_account.Credentials.from_service_account_info(credentials_info)
    return storage.Client(credentials=credentials, project=credentials_info['project_id'])

# Fonction pour télécharger un fichier depuis GCS
def download_blob_to_tempfile(bucket_name, source_blob_name):
    client = create_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    # Créer un fichier temporaire et télécharger le blob dedans
    _, temp_local_path = tempfile.mkstemp()
    blob.download_to_filename(temp_local_path)
    return temp_local_path

# Charger le modèle et le tokenizer
def load_model_and_tokenizer():
    bucket_name = "p7-mlruns"
    model_blob_name = "LTSM_Model_Glove"
    tokenizer_blob_name = "tokenizer_glove.pickle"

    model_dir = download_blob_to_tempfile(bucket_name, model_blob_name + "saved_model.pb")  # Assurez-vous du chemin correct
    tokenizer_path = download_blob_to_tempfile(bucket_name, tokenizer_blob_name)

    # Charger le modèle
    model = load_model(model_dir)

    # Charger le tokenizer
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)

    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

class Item(BaseModel):
    text: str

@app.post("/predict/")
async def create_item(item: Item):
    text = item.text
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequences, maxlen=30)
    prediction = await run_in_threadpool(model.predict, padded_sequence)
    pred_class = (prediction > 0.5).astype(int)
    return {"prediction": int(pred_class[0][0])}

