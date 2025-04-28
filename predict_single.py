import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Parámetros
IMG_SIZE = 224
MODEL_PATH = "model/brain_tumor_classifier_mobilenetv2.h5"
IMAGE_PATH = "test_images/test1.jpg"  # AQUI VA LA IMAGEN PARA PROBAR MAES PERO DESPUES

# Cargar modelo entrenado
model = load_model(MODEL_PATH)

# Cargar y preprocesar imagen
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"No se pudo cargar la imagen en: {IMAGE_PATH}")

img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = preprocess_input(img)
img = np.expand_dims(img, axis=0)

# Hacer predicción
prediction = model.predict(img)[0]
class_idx = np.argmax(prediction)
confidence = prediction[class_idx]

label_map = {0: "No Tumor", 1: "Tumor"}
print(f"\n🔍 Predicción: {label_map[class_idx]} (confianza: {confidence:.2%})")

