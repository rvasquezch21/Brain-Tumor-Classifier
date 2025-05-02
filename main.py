from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from io import BytesIO


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float


app = FastAPI()

# Cargar el modelo
model = load_model("model/brain_tumor_classifier_mobilenetv2.h5")

# Tamaño de imagen esperado para el modelo (MobileNetV2)
IMG_SIZE = 224


@app.get("/")
def root():
    return {"message": "Modelo activo"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # Validar el tipo MIME (solo imágenes JPEG y PNG)
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Solo se permiten imágenes JPEG o PNG")

    # Leer la imagen desde el archivo
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Error al leer la imagen")

    # Redimensionar y preprocesar la imagen
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 127.5 - 1.0  # Normalización entre -1 y 1
    img = np.expand_dims(img, axis=0)  # Asegurar la forma de entrada del modelo

    # Hacer la predicción
    prediction = model.predict(img)[0]
    label = "Tumor" if np.argmax(prediction) == 1 else "No Tumor"
    confidence = float(np.max(prediction))

    return {"prediction": label, "confidence": round(confidence, 4)}
