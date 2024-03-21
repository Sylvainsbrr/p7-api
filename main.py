from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from fastapi.concurrency import run_in_threadpool

app = FastAPI()

# Load the model and tokenizer
model_path = '../model/LTSM_Model_Glove'
model = load_model(model_path)
tokenizer_path = '../model/tokenizer_glove.pickle'
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

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
#%%