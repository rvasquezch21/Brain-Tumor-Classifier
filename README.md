🧠 Brain Tumor Classifier
Clasificador de tumores cerebrales basado en MobileNetV2 utilizando imágenes médicas y desplegado como API con FastAPI y Docker.

🗂 Estructura del Proyecto 

BrainTumorClassifier/ ├── data/ # Dataset original ├── metadata/ # CSV con información de imágenes ├── model/ # Modelo entrenado (.h5) ├── test_images/ # Imágenes para pruebas manuales ├── .github/workflows/ # Configuración de CI/CD con GitHub Actions │ └── deploy.yml ├── main.py # API principal (FastAPI) ├── train_model.py # Entrenamiento inicial del modelo ├── retrain.py # Reentrenamiento con nuevo dataset ├── predict_single.py # Predicción local desde consola ├── Dockerfile # Imagen Docker para despliegue ├── requirements.txt # Dependencias del proyecto └── README.md # Documentación general



🚀 Cómo ejecutar localmente

# 1. Clona el repositorio
git clone https://github.com/rvasquezch21/Brain-Tumor-Classifier.git
cd Brain-Tumor-Classifier

# 2. Instala dependencias (idealmente dentro de un entorno virtual)
pip install -r requirements.txt

# 3. Ejecuta la API
uvicorn main:app --reload

Accede a la documentación automática de la API en:
http://localhost:8000/docs

🧪 Probar predicciones localmente
Usa el script predict_single.py para probar una imagen individual:

python predict_single.py

Asegúrate de definir la ruta de la imagen en el archivo (IMAGE_PATH).

🐳 Despliegue con Docker

# Construir imagen
docker build -t brain-tumor-api .

# Correr contenedor
docker run -d -p 8000:8000 brain-tumor-api

🔁 CI/CD con GitHub Actions
Cada push a la rama develop:
Construye la imagen Docker.
La sube a DockerHub.
Se conecta por SSH a una instancia EC2.
Reemplaza el contenedor desplegado.

📁 Dataset
Este proyecto utiliza un subconjunto público del Brain Tumor Dataset.
Nota: el dataset completo no se encuentra en este repositorio por razones de tamaño.


