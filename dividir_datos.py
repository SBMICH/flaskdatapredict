import pandas as pd
from sklearn.model_selection import train_test_split

def split_arff(input_file, train_file, test_file, test_size=0.3):
    # Leer el archivo ARFF (solo datos)
    with open(input_file, "r") as file:
        lines = file.readlines()
    
    # Dividir encabezado y datos
    header = [line for line in lines if line.startswith("@")]
    data = [line for line in lines if not line.startswith("@")]
    
    # Crear un DataFrame desde los datos
    df = pd.DataFrame([line.strip().split(",") for line in data])
    
    # Dividir en conjunto de entrenamiento y prueba
    train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
    
    # Guardar el conjunto de entrenamiento
    with open(train_file, "w") as file:
        file.writelines(header)
        for row in train_data.values:
            file.write(",".join(row) + "\n")
    
    # Guardar el conjunto de prueba
    with open(test_file, "w") as file:
        file.writelines(header)
        for row in test_data.values:
            file.write(",".join(row) + "\n")
    
    print(f"División completada:\n- Entrenamiento: {train_file}\n- Prueba: {test_file}")

# Archivos de entrada y salida
input_arff = "FGR_dataset.arff"
train_arff = "FGR_training_set.arff"
test_arff = "FGR_testing_set.arff"

split_arff(input_arff, train_arff, test_arff)