"""Carga de los modelos entrenados y funciones de predicción.

Todos los modelos reciben los datos ya preprocesados con el *mismo* preprocesador
que se ajustó durante el entrenamiento (models/preprocessor.pkl).
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # Solo CPU
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # Menos mensajes de TensorFlow

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent / "models"
FEATURES = [f"C{i}" for i in range(1, 31)]  # 30 características de entrada
LABEL = "C31"                                # variable a predecir
MODEL_NAMES = ("logistic", "svm", "ann", "fcm")


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


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

        from tensorflow.keras.models import load_model  # import lento: solo una vez
        self.ann = load_model(models_dir / "ann_model.keras")

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
            return (self.ann.predict(X, verbose=0).flatten() >= 0.5).astype(int)
        return predict_fcm(self.fcm_matrix, X)


_predictor = None


def get_predictor():
    """Carga los modelos la primera vez que se necesitan."""
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor


def dataframe_from_values(values):
    return pd.DataFrame([values], columns=FEATURES)
