import os

print("""
=== Brain Tumor Classifier ===
1. Entrenar modelo
2. Predecir una imagen
3. Salir
""")

option = input("Selecciona una opción: ")

if option == "1":
    os.system("python train_model.py")
elif option == "2":
    os.system("python predict_single.py")
elif option == "3":
    print("Saliendo...")
else:
    print("Opción no válida")
