import os

import matplotlib
matplotlib.use("Agg")  # Backend sin interfaz gráfica

import base64
import io
import json
import threading

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from flask import Flask, Response, jsonify, render_template, request
from sklearn.metrics import accuracy_score, confusion_matrix

from campos import COLUMNAS, fuera_de_rango
from modelos import FEATURES, LABEL, MODEL_NAMES, MODELS_DIR, dataframe_from_values, get_predictor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_FILAS_LOTE = 5000
DATASET = pd.read_excel(os.path.join(BASE_DIR, "FGR_dataset_real.xlsx"))
REALES = DATASET[DATASET["sintetica"] == 0]  # pacientes del estudio (sin las filas generadas con SMOTE)
N_PACIENTES = 72  # pacientes del estudio original (Yi et al., Ann Med 2022)
METRICS = json.loads((MODELS_DIR / "metrics.json").read_text(encoding="utf-8"))

# Información de cada modelo para la página (las métricas salen de models/metrics.json)
MODELOS = [
    {"key": "svm", "name": "SVM", "full": "Máquina de vectores de soporte", "icon": "fa-vector-square",
     "desc": "Busca la frontera que mejor separa las dos clases. Funciona bien con pocos datos."},
    {"key": "ann", "name": "Red neuronal", "full": "Red neuronal artificial (Keras)", "icon": "fa-network-wired",
     "desc": "Tres capas densas con regularización y dropout. Captura relaciones no lineales."},
    {"key": "logistic", "name": "Regresión logística", "full": "Regresión logística", "icon": "fa-chart-line",
     "desc": "Modelo lineal e interpretable. Sirve como punto de referencia."},
    {"key": "fcm", "name": "FCM", "full": "Mapa cognitivo difuso (experimental)", "icon": "fa-project-diagram",
     "desc": "Sus pesos no se aprendieron de los datos: rinde como el azar y se muestra solo como referencia.",
     "experimental": True},
]
for m in MODELOS:
    m["metrics"] = METRICS.get(m["key"], {})

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2 MB por archivo

# Carga los modelos en segundo plano al arrancar para que la primera predicción no tarde.
if os.environ.get("PRELOAD_MODELS", "1") == "1":
    threading.Thread(target=get_predictor, daemon=True).start()


def error(mensaje, codigo=400):
    return jsonify({"error": mensaje}), codigo


@app.errorhandler(413)
def archivo_muy_grande(_):
    return error("El archivo supera el límite de 2 MB.", 413)


@app.route("/")
def home():
    return render_template("index.html", columnas=COLUMNAS, modelos=MODELOS, metrics=METRICS,
                           n_pacientes=N_PACIENTES, n_filas=len(DATASET),
                           n_sinteticas=int(DATASET["sintetica"].sum()))


@app.route("/metrics")
def metrics():
    """Métricas de los modelos sobre el conjunto de prueba (generadas por trainers/train.py)."""
    return jsonify(METRICS)


@app.route("/example")
def example():
    """Una paciente real al azar del dataset, para probar el formulario."""
    fila = REALES.sample(1).iloc[0]
    valores = {c: (int(fila[c]) if float(fila[c]).is_integer() else float(fila[c])) for c in FEATURES}
    return jsonify({"values": valores, "label": int(fila[LABEL])})


@app.route("/plantilla.csv")
def plantilla():
    """Plantilla para la predicción por lote: columnas C1..C31 con dos filas reales de cada clase."""
    filas = pd.concat([REALES[REALES[LABEL] == 0].head(2), REALES[REALES[LABEL] == 1].head(2)])
    csv = filas[FEATURES + [LABEL]].to_csv(index=False)
    return Response(csv, mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=plantilla_lote.csv"})


@app.route("/predict", methods=["POST"])
def predict():
    model_type = request.form.get("model_type", "")
    if model_type not in MODEL_NAMES:
        return error("Elige un modelo válido.")

    faltantes = [c for c in FEATURES if not request.form.get(c, "").strip()]
    if faltantes:
        return error(f"Faltan campos: {', '.join(faltantes)}.")
    try:
        values = {c: float(request.form[c]) for c in FEATURES}
    except ValueError:
        return error("Por favor, ingresa valores numéricos válidos en todos los campos.")
    if not np.isfinite(list(values.values())).all():
        return error("Los valores deben ser números finitos.")
    fuera = fuera_de_rango(values)
    if fuera:
        return error(f"Valores fuera del rango aceptado en: {', '.join(fuera)}.")

    prediction = int(get_predictor().predict(dataframe_from_values(list(values.values())), model_type)[0])
    return jsonify({"prediction": prediction,
                    "message": "Con restricción del crecimiento" if prediction == 1 else "Peso fetal normal"})


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
