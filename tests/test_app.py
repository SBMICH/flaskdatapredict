import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import app  # noqa: E402
from modelos import FEATURES, LABEL, MODEL_NAMES, predict_fcm  # noqa: E402


@pytest.fixture(scope="module")
def client():
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture(scope="module")
def dataset():
    return pd.read_excel(ROOT / "FGR_dataset.xlsx")


def formulario(fila, modelo):
    datos = {c: str(fila[c]) for c in FEATURES}
    datos["model_type"] = modelo
    return datos


def csv_bytes(df):
    return io.BytesIO(df.to_csv(index=False).encode())


@pytest.mark.parametrize("modelo", MODEL_NAMES)
def test_prediccion_individual(client, dataset, modelo):
    r = client.post("/predict", data=formulario(dataset.iloc[0], modelo))
    assert r.status_code == 200
    assert r.get_json()["prediction"] in (0, 1)


def test_prediccion_depende_de_la_entrada(client, dataset):
    """Regresión del bug del escalador: antes todas las entradas daban el mismo resultado."""
    resultados = set()
    for i in range(len(dataset)):
        r = client.post("/predict", data=formulario(dataset.iloc[i], "logistic"))
        resultados.add(r.get_json()["prediction"])
    assert resultados == {0, 1}


def test_modelo_desconocido(client, dataset):
    r = client.post("/predict", data=formulario(dataset.iloc[0], "inventado"))
    assert r.status_code == 400


def test_campo_faltante(client, dataset):
    datos = formulario(dataset.iloc[0], "svm")
    del datos["C7"]
    r = client.post("/predict", data=datos)
    assert r.status_code == 400
    assert "C7" in r.get_json()["error"]


@pytest.mark.parametrize("valor", ["abc", "1000", "nan", "inf"])
def test_valores_invalidos(client, dataset, valor):
    datos = formulario(dataset.iloc[0], "svm")
    datos["C2"] = valor
    assert client.post("/predict", data=datos).status_code == 400


def test_lote_con_etiquetas(client, dataset):
    r = client.post("/batch_predict", data={"batch_model": "logistic", "file": (csv_bytes(dataset), "datos.csv")},
                    content_type="multipart/form-data")
    assert r.status_code == 200
    cuerpo = r.get_json()
    assert cuerpo["rows"] == len(dataset)
    assert len(cuerpo["confusion_matrix"]) == 2
    assert cuerpo["graph"]


def test_lote_sin_etiquetas(client, dataset):
    r = client.post("/batch_predict",
                    data={"batch_model": "svm", "file": (csv_bytes(dataset.drop(columns=[LABEL])), "datos.csv")},
                    content_type="multipart/form-data")
    assert r.status_code == 200
    assert r.get_json()["accuracy"] is None


def test_lote_extension_no_permitida(client):
    r = client.post("/batch_predict", data={"batch_model": "svm", "file": (io.BytesIO(b"x"), "datos.txt")},
                    content_type="multipart/form-data")
    assert r.status_code == 400


def test_lote_columnas_faltantes(client, dataset):
    r = client.post("/batch_predict",
                    data={"batch_model": "svm", "file": (csv_bytes(dataset.drop(columns=["C5"])), "datos.csv")},
                    content_type="multipart/form-data")
    assert r.status_code == 400
    assert "C5" in r.get_json()["error"]


def test_lote_sin_archivo(client):
    assert client.post("/batch_predict", data={"batch_model": "svm"}).status_code == 400


def test_fcm_devuelve_binarios():
    matriz = np.random.default_rng(0).uniform(-1, 1, (31, 31))
    preds = predict_fcm(matriz, np.random.default_rng(1).normal(size=(5, 30)))
    assert set(preds) <= {0, 1}
    assert len(preds) == 5
