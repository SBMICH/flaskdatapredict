import os

import matplotlib
matplotlib.use("Agg")  # Backend sin interfaz gráfica

import base64
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from flask import Flask, jsonify, render_template, request
from sklearn.metrics import accuracy_score, confusion_matrix

from modelos import FEATURES, LABEL, MODEL_NAMES, dataframe_from_values, get_predictor

MAX_FILAS_LOTE = 5000

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2 MB por archivo


def error(mensaje, codigo=400):
    return jsonify({"error": mensaje}), codigo


@app.errorhandler(413)
def archivo_muy_grande(_):
    return error("El archivo supera el límite de 2 MB.", 413)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/metrics")
def metrics():
    """Métricas de los modelos sobre el conjunto de prueba (generadas por trainers/train.py)."""
    return jsonify(get_predictor().metrics)


@app.route("/example")
def example():
    """Una fila al azar del dataset, para probar el formulario."""
    fila = pd.read_excel(os.path.join(os.path.dirname(__file__), "FGR_dataset.xlsx")).sample(1).iloc[0]
    return jsonify({"values": {c: round(float(fila[c]), 4) for c in FEATURES}, "label": int(fila[LABEL])})


@app.route("/predict", methods=["POST"])
def predict():
    model_type = request.form.get("model_type", "")
    if model_type not in MODEL_NAMES:
        return error("Modelo no reconocido.")

    faltantes = [c for c in FEATURES if not request.form.get(c, "").strip()]
    if faltantes:
        return error(f"Faltan campos: {', '.join(faltantes)}.")
    try:
        values = [float(request.form[c]) for c in FEATURES]
    except ValueError:
        return error("Por favor, ingresa valores numéricos válidos en todos los campos.")
    if not np.isfinite(values).all():
        return error("Los valores deben ser números finitos.")
    if max(abs(v) for v in values) > 100:
        return error("Hay valores fuera de rango: los modelos esperan datos estandarizados (aprox. entre -5 y +6).")

    prediction = int(get_predictor().predict(dataframe_from_values(values), model_type)[0])
    return jsonify({"prediction": prediction, "message": "Verdadero" if prediction == 1 else "Falso"})


def leer_archivo(file):
    nombre = (file.filename or "").lower()
    if nombre.endswith(".csv"):
        return pd.read_csv(file)
    if nombre.endswith(".xlsx"):
        return pd.read_excel(file)
    raise ValueError("Extensión no permitida. Solo se aceptan archivos .csv o .xlsx.")


@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    model_type = request.form.get("batch_model", "logistic")
    if model_type not in MODEL_NAMES:
        return error(f"Modelo '{model_type}' no reconocido.")
    if "file" not in request.files or not request.files["file"].filename:
        return error("No se subió ningún archivo.")

    try:
        data = leer_archivo(request.files["file"])
    except ValueError as e:
        return error(str(e))
    except Exception:
        return error("No se pudo leer el archivo. Revisa que sea un .csv o .xlsx válido.")

    faltantes = [c for c in FEATURES if c not in data.columns]
    if faltantes:
        return error(f"Faltan las columnas: {', '.join(faltantes)}. Se esperan C1..C30 y, opcionalmente, {LABEL}.")
    if len(data) == 0:
        return error("El archivo no tiene filas.")
    if len(data) > MAX_FILAS_LOTE:
        return error(f"El archivo tiene demasiadas filas (máximo {MAX_FILAS_LOTE}).")

    X = data[FEATURES].apply(pd.to_numeric, errors="coerce")
    if X.isnull().values.any():
        return error("Las columnas C1..C30 contienen valores vacíos o no numéricos.")

    y_pred = get_predictor().predict(X, model_type)
    respuesta = {"rows": int(len(data)), "positives": int(y_pred.sum())}

    if LABEL not in data.columns:
        respuesta["accuracy"] = None
        return jsonify(respuesta)

    y_true = pd.to_numeric(data[LABEL], errors="coerce")
    if y_true.isnull().any():
        return error(f"La columna {LABEL} tiene valores vacíos o no numéricos.")
    y_true = y_true.astype(int)

    labels = [0, 1]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    try:
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Etiqueta verdadera")
        ax.set_title(f"Matriz de confusión ({model_type.upper()})")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        grafico = base64.b64encode(buf.getvalue()).decode("utf-8")
    finally:
        plt.close(fig)

    respuesta.update({
        "accuracy": f"{accuracy_score(y_true, y_pred) * 100:.2f}%",
        "confusion_matrix": matrix.tolist(),
        "graph": grafico,
    })
    return jsonify(respuesta)


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
