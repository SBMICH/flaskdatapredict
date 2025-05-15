import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

def load_and_preprocess_data():
    try:
        # Cargar el Dataset
        data = pd.read_excel("FGR_dataset.xlsx")
    except FileNotFoundError:
        raise FileNotFoundError("Error: No se encontró el archivo 'FGR_dataset.xlsx'.")

    # Imputar valores faltantes con la media
    imputer = SimpleImputer(strategy='mean')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    # Verificar si hay valores faltantes después de la imputación
    if data_imputed.isnull().values.any():
        raise ValueError("Error: El dataset sigue teniendo valores faltantes.")

    # Preprocesar el Dataset
    X = data_imputed.drop(columns=["C31"])
    y = data_imputed["C31"]

    # Escalar las variables numéricas
    scaler = StandardScaler()  # o puedes probar MinMaxScaler() si es más adecuado
    X_scaled = scaler.fit_transform(X)

    # Dividir en conjunto de entrenamiento y prueba (estratificado)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, stratify=y)

    return X_train, X_test, y_train, y_test
