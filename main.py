from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from fastapi.concurrency import run_in_threadpool

app = FastAPI()

# Load the tokenizer
tokenizer_path = '../model/tokenizer_glove.pickle'
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the TensorFlow SavedModel
model_path = '../model/LTSM_Model_Glove'
model = tf.saved_model.load(model_path)
infer = model.signatures["serving_default"]  # Adjust if your model has a different signature

class Item(BaseModel):
    text: str

@app.post("/predict/")
async def create_item(item: Item):
    text = item.text
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequences, maxlen=30)

    # Adapt the prediction call for the TensorFlow SavedModel
    # Convert input to expected format
    padded_sequence = tf.constant(padded_sequence, dtype=tf.float32)  # Ensure dtype matches model expectation
    prediction = await run_in_threadpool(infer, inputs=tf.constant(padded_sequence))  # Adjust input key if needed
    pred_class = (prediction['output_0'].numpy() > 0.5).astype(int)  # Adjust output key if needed

    return {"prediction": int(pred_class[0][0])}
