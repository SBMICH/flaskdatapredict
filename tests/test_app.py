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
    return pd.read_excel(ROOT / "FGR_dataset_real.xlsx")


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
    assert sum(map(sum, cuerpo["confusion_matrix"])) == len(dataset)


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


def test_valor_fuera_de_rango_indica_el_campo(client, dataset):
    datos = formulario(dataset.iloc[0], "svm")
    datos["C12"] = "-5"
    r = client.post("/predict", data=datos)
    assert r.status_code == 400
    assert "C12" in r.get_json()["error"]


def test_pagina_principal_tiene_los_30_campos(client):
    html = client.get("/").get_data(as_text=True)
    assert all(f'id="{c}"' in html for c in FEATURES)


def test_niveles_del_formulario_son_validos(client):
    """Cada opción de las listas debe ser aceptada por /predict."""
    from campos import GRUPOS
    datos = {f["col"]: str(f["options"][-1][1]) for g in GRUPOS for f in g["fields"]}
    datos["model_type"] = "logistic"
    assert client.post("/predict", data=datos).status_code == 200


def test_plantilla_csv(client):
    r = client.get("/plantilla.csv")
    assert r.status_code == 200
    columnas = r.get_data(as_text=True).splitlines()[0].split(",")
    assert columnas == FEATURES + [LABEL]


def test_ejemplo(client):
    cuerpo = client.get("/example").get_json()
    assert set(cuerpo["values"]) == set(FEATURES)
    assert cuerpo["label"] in (0, 1)


def test_fcm_devuelve_binarios():
    matriz = np.random.default_rng(0).uniform(-1, 1, (31, 31))
    preds = predict_fcm(matriz, np.random.default_rng(1).normal(size=(5, 30)))
    assert set(preds) <= {0, 1}
    assert len(preds) == 5


def test_conversion_coincide_con_el_articulo(dataset):
    """Las pacientes reales reconstruidas deben reproducir los rangos publicados por Yi et al. (2022)."""
    reales = dataset[dataset["sintetica"] == 0]
    publicados = {"C1": (18, 44), "C2": (18.6, 37.59), "C3": (26.2, 34), "C18": (112, 226), "C27": (18.7, 33.4)}
    for col, (minimo, maximo) in publicados.items():
        assert abs(reales[col].min() - minimo) <= 0.5, col
        assert abs(reales[col].max() - maximo) <= 0.5, col
    # Motivo del parto: 6 HELLP, 23 sufrimiento fetal, 17 hipertensión no controlada (Tabla 1 del artículo)
    conteo = reales["C20"].value_counts()
    assert (conteo[0], conteo[1], conteo[3]) == (6, 23, 17)


def test_red_numpy_igual_a_keras(dataset):
    """La app predice con NumPy; debe dar lo mismo que el modelo de TensorFlow del que salen los pesos.
    Se omite si TensorFlow no está instalado (por ejemplo, en el servidor de la demo)."""
    import pickle
    keras = pytest.importorskip("tensorflow.keras.models")
    from modelos import MODELS_DIR, RedNeuronalNumpy
    with open(MODELS_DIR / "preprocessor.pkl", "rb") as f:
        X = pickle.load(f).transform(dataset[FEATURES])
    esperado = keras.load_model(MODELS_DIR / "ann_model.keras").predict(X, verbose=0).flatten()
    obtenido = RedNeuronalNumpy(MODELS_DIR / "ann_weights.npz").probabilidad(X)
    assert np.allclose(esperado, obtenido, atol=1e-5)
