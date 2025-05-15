import pandas as pd
from sklearn.model_selection import train_test_split
import arff  # Asegúrate de que liac-arff esté instalado con `pip install liac-arff`

def create_arff(df, output_file, relation_name):
    """
    Crea un archivo ARFF a partir de un DataFrame de pandas.
    """
    attributes = []
    for column in df.columns:
        if df[column].dtype == "object":
            # Atributos nominales
            attributes.append((column, df[column].unique().tolist()))
        else:
            # Atributos numéricos deben ser "NUMERIC"
            attributes.append((column, "NUMERIC"))
    
    # Crear el diccionario ARFF
    arff_data = {
        "relation": relation_name,
        "attributes": attributes,
        "data": df.values.tolist(),
    }
    
    # Guardar el archivo ARFF
    with open(output_file, "w") as f:
        arff.dump(arff_data, f)
    
    print(f"Archivo ARFF creado: {output_file}")

# Cargar el dataset completo (ajusta la ruta al archivo original)
file_path = "FGR_dataset.xlsx"  # Cambia a .xlsx si es necesario
df = pd.read_excel(file_path)  # Usa pd.read_excel si el archivo es Excel

# Dividir el dataset en entrenamiento (70%) y prueba (30%)
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

# Crear los archivos ARFF
create_arff(df, "FGR_dataset.arff", "FGR_dataset")           # Dataset completo
create_arff(train_df, "FGR_training_set.arff", "TrainingSet")  # Conjunto de entrenamiento
create_arff(test_df, "FGR_testing_set.arff", "TestingSet")    # Conjunto de prueba