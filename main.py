from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
import numpy as np
import cv2

app = FastAPI()
model = load_model("model/brain_tumor_classifier_mobilenetv2.h5")

IMG_SIZE = 224

@app.get("/")
def root():
    return {"message": "Modelo activo"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 127.5 - 1.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0]
    label = "Tumor" if np.argmax(prediction) == 1 else "No Tumor"
    confidence = float(np.max(prediction))

    return {"prediction": label, "confidence": round(confidence, 4)}
