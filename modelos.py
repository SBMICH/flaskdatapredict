"""Carga de los modelos entrenados y funciones de predicción.

Todos los modelos reciben los datos ya preprocesados con el *mismo* preprocesador
que se ajustó durante el entrenamiento (models/preprocessor.pkl).

La red neuronal se entrena con TensorFlow (trainers/train.py), pero la app no lo necesita:
sus pesos se exportan a models/ann_weights.npz y la predicción se calcula con NumPy.
"""
import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")  # ver el comentario en app.py

import json
import pickle
import threading
from pathlib import Path

import numpy as np
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent / "models"
FEATURES = [f"C{i}" for i in range(1, 31)]  # 30 características de entrada
LABEL = "C31"                                # variable a predecir
MODEL_NAMES = ("logistic", "svm", "ann", "fcm")


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


class RedNeuronalNumpy:
    """Paso hacia adelante de la red densa entrenada con Keras, usando solo NumPy.

    Cada capa calcula  activación(x · W + b). El dropout solo actúa al entrenar, así que aquí no aparece.
    """

    def __init__(self, ruta):
        with np.load(ruta, allow_pickle=False) as datos:
            n = int(datos["n_capas"])
            self.capas = [(datos[f"W{i}"], datos[f"b{i}"], str(datos[f"act{i}"])) for i in range(n)]

    def probabilidad(self, X):
        x = np.asarray(X, dtype=np.float64)
        for W, b, activacion in self.capas:
            x = x @ W + b
            if activacion == "relu":
                x = np.maximum(x, 0)
            elif activacion == "sigmoid":
                x = sigmoid(x)
            elif activacion != "linear":
                raise ValueError(f"Activación no soportada: {activacion}")
        return x.flatten()

    def predict(self, X):
        return (self.probabilidad(X) >= 0.5).astype(int)


def exportar_pesos(modelo_keras, ruta):
    """Guarda los pesos de las capas densas de un modelo Keras en un .npz (lo usa trainers/train.py)."""
    datos, i = {}, 0
    for capa in modelo_keras.layers:
        pesos = capa.get_weights()
        if not pesos:  # Dropout, Input...: no tienen pesos
            continue
        datos[f"W{i}"], datos[f"b{i}"] = pesos
        datos[f"act{i}"] = np.array(capa.get_config()["activation"])
        i += 1
    datos["n_capas"] = np.array(i)
    np.savez(ruta, **datos)


def predict_fcm(adj_matrix, X, iterations=10):
    """Mapa cognitivo difuso: propaga las 30 entradas por la red y lee el nodo 31."""
    n = adj_matrix.shape[0]
    preds = []
    for row in np.atleast_2d(X):
        concepts = np.zeros(n)
        concepts[: len(row)] = row
        for _ in range(iterations):
            concepts = sigmoid(adj_matrix @ concepts)
        preds.append(concepts[n - 1])
    return (np.array(preds) >= 0.5).astype(int)


class Predictor:
    def __init__(self, models_dir=MODELS_DIR):
        models_dir = Path(models_dir)
        with open(models_dir / "preprocessor.pkl", "rb") as f:
            self.preprocessor = pickle.load(f)
        with open(models_dir / "logistic_model.pkl", "rb") as f:
            self.logistic = pickle.load(f)
        with open(models_dir / "svm_model.pkl", "rb") as f:
            self.svm = pickle.load(f)
        with open(models_dir / "fcm_model.pkl", "rb") as f:
            self.fcm_matrix = pickle.load(f)

        self.ann = RedNeuronalNumpy(models_dir / "ann_weights.npz")

        metrics_path = models_dir / "metrics.json"
        self.metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}

    def predict(self, df, model_name):
        """df: DataFrame con las columnas C1..C30. Devuelve un arreglo de 0/1."""
        if model_name not in MODEL_NAMES:
            raise ValueError(f"Modelo '{model_name}' no reconocido.")
        X = self.preprocessor.transform(df[FEATURES])
        if model_name == "logistic":
            return self.logistic.predict(X).astype(int)
        if model_name == "svm":
            return self.svm.predict(X).astype(int)
        if model_name == "ann":
            return self.ann.predict(X)
        return predict_fcm(self.fcm_matrix, X)


_predictor = None
_lock = threading.Lock()


def get_predictor():
    """Carga los modelos la primera vez que se necesitan (una sola vez, aunque lleguen varias peticiones)."""
    global _predictor
    with _lock:
        if _predictor is None:
            _predictor = Predictor()
    return _predictor


def dataframe_from_values(values):
    return pd.DataFrame([values], columns=FEATURES)
