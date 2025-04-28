import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Parámetros SE TIENEN QUE CAMBIAR
IMG_SIZE = 224
DATASET_PATH = "'/Users/pablogarro/BrainTumorClassifier/data /Brain Tumor Data Set'"
CSV_PATH = "/Users/pablogarro/BrainTumorClassifier/metadata/metadata_rgb_only.csv"

# Cargar metadata
df = pd.read_csv(CSV_PATH)
df['label'] = df['class'].map({'tumor': 1, 'no_tumor': 0})

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

# Dividir datos
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

# Aumentación de datos
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2,
                         height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

# Entrenamiento
model.fit(aug.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_test, y_test))

# Evaluación
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2%}")

# Guardar modelo
model.save("model/brain_tumor_classifier_mobilenetv2.h5")
