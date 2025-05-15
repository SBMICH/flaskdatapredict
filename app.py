import matplotlib
matplotlib.use("Agg")  # Usar un backend no interactivo para evitar problemas con la GUI

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns  # Requiere instalación: pip install seaborn
import io
import base64

# Crear la aplicación Flask
app = Flask(__name__)

# Cargar los modelos desde la carpeta 'models/'
with open("models/logistic_model.pkl", "rb") as f:
    logistic_model = pickle.load(f)

with open("models/svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)

ann_model = load_model("models/ann_model.keras")

with open("models/fcm_model.pkl", "rb") as f:
    fcm_adj_matrix = pickle.load(f)

# Inicializar el escalador estándar
scaler = StandardScaler()

# Función de activación sigmoide estable
def sigmoid(x):
    x = np.clip(x, -500, 500)  # Limitar valores extremos para evitar desbordamientos
    return 1 / (1 + np.exp(-x))

# Función para realizar predicciones con FCM (nodo 31)
def predict_fcm(adj_matrix, data):
    """
    Predice el valor del nodo 31 basándose en las características de entrada (primeros 30 nodos).
    Args:
        adj_matrix (np.ndarray): Matriz de adyacencia (31x31).
        data (np.ndarray): Características de entrada (30 nodos).
    Returns:
        float: Valor predicho para el nodo 31.
    """
    # Extraer los pesos hacia el nodo 31 (última fila, primeros 30 elementos)
    weights_to_node_31 = adj_matrix[30, :30]
    
    # Calcular el valor del nodo 31 usando los primeros 30 nodos
    node_31_value = sigmoid(np.dot(weights_to_node_31, data))
    return np.round(node_31_value)  # Retornar el valor redondeado (0 o 1)

# Ruta para el index.html
@app.route("/")
def home():
    return render_template("index.html")  # Página HTML para interactuar con la app

# Ruta para manejar las predicciones individuales
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Obtener los datos del formulario
        data = np.array([float(request.form[key]) for key in request.form if key != "model_type"])
        model_type = request.form["model_type"]
    except ValueError:
        return jsonify({"error": "Por favor, ingresa valores válidos en todos los campos."}), 400

    # Escalar los datos de entrada para modelos que no sean FCM
    if model_type != "fcm":
        scaled_data = scaler.fit_transform([data])

    # Realizar la predicción según el modelo seleccionado
    if model_type == "logistic":
        prediction = logistic_model.predict(scaled_data)[0]
    elif model_type == "svm":
        prediction = svm_model.predict(scaled_data)[0]
    elif model_type == "ann":
        prediction = ann_model.predict(scaled_data).flatten()[0]  # Usar flatten para obtener un escalar
        prediction = 1 if prediction >= 0.5 else 0  # Convertir probabilidad a clase (0 o 1)
    elif model_type == "fcm":
        # Para el modelo FCM, no se utiliza el escalador
        if fcm_adj_matrix.shape[0] - 1 != data.shape[0]:
            return jsonify({"error": "El número de características no coincide con los nodos del modelo FCM."}), 400
        prediction = predict_fcm(fcm_adj_matrix, data)  # Predice el nodo 31
    else:
        return jsonify({"error": "Modelo no reconocido."}), 400

    # Determinar el mensaje si es verdadero o falso
    message = "Verdadero" if int(prediction) == 1 else "Falso"

    # Retornar resultados sin probabilidad
    return jsonify({
        "prediction": int(prediction),
        "message": message  # Mensaje indicando si es Verdadero o Falso
    })

# Ruta para manejar predicciones por lote (batch)
@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    try:
        # Capturar el modelo seleccionado
        model_type = request.form.get("batch_model", "logistic")

        # Verificar si se subió un archivo
        if "file" not in request.files:
            return jsonify({"error": "No se subió ningún archivo."}), 400

        file = request.files["file"]

        # Verificar la extensión del archivo
        if not (file.filename.endswith(".csv") or file.filename.endswith(".xlsx")):
            return jsonify({"error": "Extensión de archivo no permitida. Solo se aceptan archivos .csv o .xlsx."}), 400

        # Leer el archivo como DataFrame
        if file.filename.endswith(".csv"):
            data = pd.read_csv(file)
        else:
            data = pd.read_excel(file)

        # Verificar valores nulos
        if data.isnull().values.any():
            return jsonify({"error": "El archivo contiene valores nulos o incompletos."}), 400

        # Verificar que el archivo tenga al menos 31 columnas
        if data.shape[1] < 31:
            return jsonify({"error": "El archivo debe contener al menos 31 columnas (30 características y 1 etiqueta)."}), 400

        # Separar características y etiquetas
        X = data.iloc[:, :30].values  # Primeras 30 columnas como características
        y_true = data.iloc[:, 30].values  # Última columna como etiqueta verdadera

        # Seleccionar el modelo
        if model_type == "logistic":
            model_to_use = logistic_model
        elif model_type == "svm":
            model_to_use = svm_model
        elif model_type == "ann":
            model_to_use = ann_model
        elif model_type == "fcm":
            model_to_use = fcm_adj_matrix
        else:
            return jsonify({"error": f"Modelo '{model_type}' no reconocido."}), 400

        # Procesar los datos de entrada
        if model_type != "fcm":
            # Escalar los datos para modelos que no sean FCM
            X_scaled = scaler.fit_transform(X)

            if model_type == "ann":
                # ANN: Convertir predicciones continuas a binarias
                y_pred = (model_to_use.predict(X_scaled).flatten() >= 0.5).astype(int)
            else:
                # Logistic Regression y SVM
                y_pred = model_to_use.predict(X_scaled)
        else:
            # Para FCM, no se utiliza el escalador
            y_pred = [predict_fcm(model_to_use, row) for row in X]

        # Calcular matriz de confusión y exactitud
        conf_matrix = confusion_matrix(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred) * 100

        # Crear el gráfico de la matriz de confusión
        plt.figure(figsize=(8, 6))
        try:
            labels = sorted(np.unique(y_true))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
            plt.xlabel("Predicción")
            plt.ylabel("Etiqueta Verdadera")
            plt.title(f"Matriz de Confusión ({model_type.upper()})")

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode("utf-8")
            buf.close()
        finally:
            plt.close()

        return jsonify({
            "accuracy": f"{accuracy:.2f}%",
            "graph": image_base64
        })
    except Exception as e:
        return jsonify({"error": f"Error procesando el archivo: {str(e)}"}), 500

# Ejecutar la aplicación
if __name__ == "__main__":
    app.run(debug=True)