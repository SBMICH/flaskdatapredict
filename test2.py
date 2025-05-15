import pickle
import numpy as np
import matplotlib.pyplot as plt

# Cargar la matriz de adyacencia desde el archivo
with open("models/fcm_model.pkl", "rb") as f:
    adj_matrix = pickle.load(f)

# Inspeccionar dimensiones
print(f"Dimensiones de la matriz de adyacencia: {adj_matrix.shape}")

# Mostrar algunos valores (opcional)
print("Matriz de adyacencia (primeros 5x5 elementos):")
print(adj_matrix[:5, :5])  # Muestra los primeros 5x5 elementos

# Verificar propiedades de la matriz
print("¿Es cuadrada?:", adj_matrix.shape[0] == adj_matrix.shape[1])
print("Valores máximos y mínimos:")
print(f"Máximo: {np.max(adj_matrix)}, Mínimo: {np.min(adj_matrix)}")

# Graficar la matriz de adyacencia como un mapa de calor
plt.figure(figsize=(8, 6))
plt.imshow(adj_matrix, cmap="viridis", interpolation="nearest")
plt.colorbar(label="Peso de la conexión")
plt.title("Mapa de calor de la matriz de adyacencia")
plt.xlabel("Nodo origen")
plt.ylabel("Nodo destino")
plt.show()