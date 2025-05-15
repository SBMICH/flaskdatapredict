import pandas as pd
import numpy as np
import pickle
import os

# Función sigmoide segura
def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

# Propagación FCM individual por instancia
def predict_fcm_instance(adj_matrix, input_vector, iterations=10):
    """
    Propaga una sola instancia de entrada por la red FCM.
    """
    if len(input_vector) != 30:
        raise ValueError("La entrada debe tener exactamente 30 características.")
    
    concepts = np.zeros(31)
    concepts[:30] = input_vector  # Establecer los primeros 30 nodos

    for _ in range(iterations):
        concepts = sigmoid(np.dot(adj_matrix, concepts))

    return concepts[30]  # Valor predicho del nodo 31

def main():
    # Cargar la matriz de adyacencia
    try:
        adj_matrix = pd.read_csv("random.csv", header=None).values
    except FileNotFoundError:
        print("No se encontró 'random.csv'")
        return

    if adj_matrix.shape != (31, 31):
        print("Error: la matriz debe ser 31x31.")
        return

    # Cargar el dataset
    try:
        df = pd.read_excel("FGR_dataset.xlsx")
    except FileNotFoundError:
        print("No se encontró 'FGR_dataset.xlsx'")
        return

    if df.shape[1] < 31:
        print("Error: se requieren al menos 31 columnas.")
        return

    # Separar entradas (30) y salida (nodo 31 real)
    X = df.iloc[:, :30].values
    y_real = df.iloc[:, 30].values

    # Normalizar entradas
    X_norm = (X - X.mean(axis=0)) / X.std(axis=0)

    # Predicciones
    y_pred = np.array([predict_fcm_instance(adj_matrix, x) for x in X_norm])

    # Guardar predicciones
    results_df = pd.DataFrame({
        "Real": y_real,
        "Predicho": y_pred
    })
    results_df.to_excel("predicciones_fcm.xlsx", index=False)

    # Guardar el modelo
    os.makedirs("models", exist_ok=True)
    with open("models/fcm_model_31.pkl", "wb") as f:
        pickle.dump(adj_matrix, f)

    print("Predicciones guardadas en 'predicciones_fcm.xlsx'")
    print("Modelo guardado en 'models/fcm_model_31.pkl'")

if __name__ == "__main__":
    main()
