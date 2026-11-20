"""Entrena los cuatro modelos y guarda todo en models/.

Uso (desde la carpeta del proyecto):
    python trainers/train.py

Guarda: preprocessor.pkl, logistic_model.pkl, svm_model.pkl, ann_model.keras, ann_weights.npz
y metrics.json. Requiere TensorFlow: pip install -r requirements-train.txt
El FCM (fcm_model.pkl) no se entrena aquí: solo se evalúa con el mismo conjunto de prueba.
"""
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from modelos import FEATURES, LABEL, exportar_pesos, predict_fcm  # noqa: E402

SEED = 42
MODELS_DIR = ROOT / "models"


def cargar_datos():
    """Divide SOLO las pacientes reales en entrenamiento y prueba; las filas sintéticas (SMOTE) van
    únicamente al entrenamiento. Así las métricas se miden sobre pacientes reales."""
    data = pd.read_excel(ROOT / "FGR_dataset_real.xlsx")
    reales, sinteticas = data[data["sintetica"] == 0], data[data["sintetica"] == 1]
    train, test = train_test_split(reales, test_size=0.3, stratify=reales[LABEL], random_state=SEED)
    train = pd.concat([train, sinteticas])
    # El preprocesador se ajusta después, solo con el entrenamiento, para no filtrar información de la prueba.
    return train[FEATURES], test[FEATURES], train[LABEL].astype(int), test[LABEL].astype(int)


def entrenar_ann(X_train, y_train, X_test, y_test):
    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # solo CPU
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # menos mensajes de TensorFlow
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense, Dropout, Input
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.regularizers import l2

    tf.keras.utils.set_random_seed(SEED)
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(128, activation="relu", kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        Dense(64, activation="relu", kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        Dense(32, activation="relu", kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="binary_crossentropy", metrics=["accuracy"])
    # Se monitorea una parte del entrenamiento, no el conjunto de prueba.
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0,
              callbacks=[EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)])
    return model


def resumen(y_true, y_pred):
    return {"accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
            "f1": round(float(f1_score(y_true, y_pred)), 4)}


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    MODELS_DIR.mkdir(exist_ok=True)

    X_train, X_test, y_train, y_test = cargar_datos()
    print(f"Entrenamiento: {len(X_train)} filas | Prueba: {len(X_test)} filas")
    print(f"Proporción de la clase 1 en prueba: {y_test.mean():.2f}")

    preprocessor = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler())
    Xtr = preprocessor.fit_transform(X_train)
    Xte = preprocessor.transform(X_test)

    metrics = {"n_train": len(X_train), "n_test": len(X_test), "n_test_positive": int(y_test.sum()),
               "baseline_accuracy": round(float(max(y_test.mean(), 1 - y_test.mean())), 4)}

    logistic = LogisticRegression(solver="liblinear", C=1.0, max_iter=200).fit(Xtr, y_train)
    metrics["logistic"] = resumen(y_test, logistic.predict(Xte))
    metrics["logistic"]["cv_accuracy"] = round(float(cross_val_score(logistic, Xtr, y_train, cv=5).mean()), 4)

    svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=SEED).fit(Xtr, y_train)
    metrics["svm"] = resumen(y_test, svm.predict(Xte))
    metrics["svm"]["cv_accuracy"] = round(float(cross_val_score(svm, Xtr, y_train, cv=5).mean()), 4)

    ann = entrenar_ann(Xtr, y_train, Xte, y_test)
    metrics["ann"] = resumen(y_test, (ann.predict(Xte, verbose=0).flatten() >= 0.5).astype(int))

    with open(MODELS_DIR / "fcm_model.pkl", "rb") as f:
        fcm_matrix = pickle.load(f)
    metrics["fcm"] = resumen(y_test, predict_fcm(fcm_matrix, Xte))

    with open(MODELS_DIR / "preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)
    with open(MODELS_DIR / "logistic_model.pkl", "wb") as f:
        pickle.dump(logistic, f)
    with open(MODELS_DIR / "svm_model.pkl", "wb") as f:
        pickle.dump(svm, f)
    ann.save(MODELS_DIR / "ann_model.keras")
    exportar_pesos(ann, MODELS_DIR / "ann_weights.npz")  # lo que usa la app, sin TensorFlow
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
