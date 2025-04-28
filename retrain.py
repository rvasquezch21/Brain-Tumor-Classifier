import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Parámetros
IMG_SIZE = 224
DATASET_PATH = "data/Brain Tumor Data Set"
CSV_PATH = "metadata/metadata_rgb_only.csv"
MODEL_OUTPUT_PATH = "model/brain_tumor_classifier_mobilenetv2.h5"

# Cargar metadata
df = pd.read_csv(CSV_PATH)

# Mapear clases a 0 y 1
df['label'] = df['class'].map({'tumor': 1, 'no_tumor': 0})

# Construir rutas completas a las imágenes
def get_image_path(row):
    folder = "Brain Tumor" if row['label'] == 1 else "Healthy"
    return os.path.join(DATASET_PATH, folder, row['image'])

df['path'] = df.apply(get_image_path, axis=1)

# Cargar imágenes y etiquetas
images = []
labels = []

for idx, row in df.iterrows():
    img_path = row['path']
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess_input(img)
        images.append(img)
        labels.append(row['label'])

X = np.array(images)
y = to_categorical(labels)

# Separar datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo base MobileNetV2
base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(2, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Congelar capas base
for layer in base_model.layers:
    layer.trainable = False

# Compilar modelo
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar modelo
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

# Guardar modelo entrenado
model.save(MODEL_OUTPUT_PATH)

print(f"\n✅ Modelo reentrenado y guardado en: {MODEL_OUTPUT_PATH}")
