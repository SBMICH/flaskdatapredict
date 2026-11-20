"""Descripción de las 30 variables de entrada (en unidades clínicas reales) para el formulario.

Los valores vienen de FGR_dataset_real.xlsx, generado por herramientas/convertir_unidades.py a partir del
dataset estandarizado. Las variables continuas se escriben como número; las categóricas se eligen de una lista.
"""
from pathlib import Path

import pandas as pd

DATASET = Path(__file__).resolve().parent / "FGR_dataset_real.xlsx"

# Opciones de las variables categóricas: código -> texto
CATEGORIAS = {
    "C6": {0: "Edema", 1: "Restricción del crecimiento", 2: "Hipertensión"},
    "C16": {0: "No", 1: "Sí"},
    "C17": {0: "Ninguno", 1: "Hipertensión", 2: "Síndrome de ovario poliquístico"},
    "C20": {0: "Síndrome HELLP", 1: "Sufrimiento fetal", 2: "Disfunción orgánica",
            3: "Hipertensión no controlada", 4: "Edema grave", 5: "Restricción del crecimiento"},
    "C21": {0: "Cesárea", 1: "Parto vaginal"},
}

# (título, icono, [(columna, etiqueta, unidad, descripción)])
GRUPOS_BASE = [
    ("Datos de la paciente", "fa-user", [
        ("C1", "Edad", "años", ""),
        ("C2", "Índice de masa corporal", "kg/m²", "Peso (kg) / altura² (m²)."),
        ("C3", "Edad gestacional al parto", "semanas", ""),
        ("C4", "Gravidez", "embarazos", "Número total de embarazos, incluido el actual."),
        ("C5", "Paridad", "partos", "Número de partos previos."),
    ]),
    ("Síntoma inicial", "fa-stethoscope", [
        ("C6", "Síntoma inicial", "", "Primer síntoma observado antes del diagnóstico."),
        ("C7", "Semana de inicio del síntoma", "semanas", ""),
        ("C8", "Del síntoma al parto", "días", ""),
    ]),
    ("Hipertensión y edema", "fa-heart-pulse", [
        ("C9", "Semana de inicio de hipertensión", "semanas", ""),
        ("C10", "De la hipertensión al parto", "días", ""),
        ("C11", "Semana de inicio de edema", "semanas", ""),
        ("C12", "Del edema al parto", "días", ""),
    ]),
    ("Proteinuria", "fa-vial", [
        ("C13", "Semana de inicio de proteinuria", "semanas", ""),
        ("C14", "De la proteinuria al parto", "días", ""),
    ]),
    ("Tratamiento e historia", "fa-pills", [
        ("C15", "Tratamiento expectante", "días", "Días de vigilancia hospitalaria antes de terminar el embarazo."),
        ("C16", "Antihipertensivos antes del ingreso", "", ""),
        ("C17", "Antecedentes médicos", "", ""),
    ]),
    ("Presión arterial", "fa-gauge-high", [
        ("C18", "Presión sistólica máxima", "mmHg", ""),
        ("C19", "Presión diastólica máxima", "mmHg", ""),
    ]),
    ("Parto", "fa-baby", [
        ("C20", "Motivo del parto", "", "Indicación principal para terminar el embarazo."),
        ("C21", "Tipo de parto", "", ""),
    ]),
    ("Laboratorio", "fa-flask", [
        ("C22", "BNP máximo", "pg/mL", "Péptido natriurético tipo B."),
        ("C23", "Creatinina máxima", "µmol/L", ""),
        ("C24", "Ácido úrico máximo", "µmol/L", ""),
        ("C25", "Proteinuria máxima", "mg/24 h", ""),
        ("C26", "Proteína total mínima", "g/L", ""),
        ("C27", "Albúmina mínima", "g/L", ""),
        ("C28", "ALT máxima", "UI/L", "Alanina aminotransferasa."),
        ("C29", "AST máxima", "UI/L", "Aspartato aminotransferasa."),
        ("C30", "Plaquetas mínimas", "×10⁹/L", "Valor aproximado: la desviación publicada parece tener una errata."),
    ]),
]

PERCENTILES = [(0.10, "Muy bajo"), (0.25, "Bajo"), (0.50, "Típico"), (0.75, "Alto"), (0.90, "Muy alto")]
MARGEN = 0.25  # fracción del rango del dataset que se acepta por encima y por debajo


def _numero(v):
    v = float(v)
    return int(v) if v.is_integer() else round(v, 2)


def _niveles(serie):
    """Niveles (percentiles) para elegir sin escribir: [(etiqueta, valor)]."""
    niveles = {}
    for q, etiqueta in PERCENTILES:
        valor = _numero(serie.quantile(q, interpolation="nearest"))
        if valor not in niveles or abs(q - 0.5) < abs(niveles[valor][0] - 0.5):
            niveles[valor] = (q, etiqueta)
    return [(f"{etiqueta} ({valor})", valor) for valor, (q, etiqueta) in sorted(niveles.items())]


def _paso(serie):
    decimales = serie.map(lambda v: len(f"{v:g}".split(".")[1]) if "." in f"{v:g}" else 0).max()
    return 10 ** -min(int(decimales), 2) if decimales else 1


def _construir():
    data = pd.read_excel(DATASET)
    reales = data[data["sintetica"] == 0]
    grupos, rangos = [], {}
    for titulo, icono, campos in GRUPOS_BASE:
        lista = []
        for col, etiqueta, unidad, ayuda in campos:
            serie = reales[col]
            minimo, maximo = float(serie.min()), float(serie.max())
            if col in CATEGORIAS:
                opciones = [(texto, codigo) for codigo, texto in CATEGORIAS[col].items()]
                tipo, rango = "select", (minimo, maximo)
                defecto = int(serie.mode().iloc[0])
            else:
                opciones = _niveles(serie)
                tipo = "number"
                holgura = (maximo - minimo) * MARGEN
                rango = (max(0.0, minimo - holgura), maximo + holgura)
                defecto = _numero(serie.median())
            rangos[col] = rango
            lista.append({"col": col, "label": etiqueta, "unit": unidad, "help": ayuda, "kind": tipo,
                          "options": opciones, "default": defecto, "step": _paso(serie),
                          "min": _numero(rango[0]), "max": _numero(rango[1]),
                          "data_min": _numero(minimo), "data_max": _numero(maximo)})
        grupos.append({"title": titulo, "icon": icono, "fields": lista})
    return grupos, rangos


def _en_dos_columnas(grupos):
    """Reparte los grupos en dos columnas fijas, en orden, con una cantidad de campos parecida."""
    total = sum(len(g["fields"]) for g in grupos)
    izquierda, acumulado = [], 0
    for g in grupos:
        if acumulado + len(g["fields"]) / 2 > total / 2:
            break
        izquierda.append(g)
        acumulado += len(g["fields"])
    return [izquierda, grupos[len(izquierda):]]


GRUPOS, RANGOS = _construir()
COLUMNAS = _en_dos_columnas(GRUPOS)


def fuera_de_rango(valores):
    """valores: dict columna -> número. Devuelve las columnas fuera del rango aceptado."""
    return [c for c, v in valores.items() if not RANGOS[c][0] <= v <= RANGOS[c][1]]
