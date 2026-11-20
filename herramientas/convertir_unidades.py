"""Convierte FGR_dataset.xlsx (estandarizado) a unidades clínicas reales.

Origen de los datos: Yi J, Chen L, Meng X, Chen Y. "Risk factors and foetal growth restriction associated
with expectant treatment of early-onset preeclampsia". Ann Med. 2022;54(1):3250-3257.
https://pmc.ncbi.nlm.nih.gov/articles/PMC9673790/  (72 pacientes con preeclampsia de inicio temprano).

El archivo que tenemos es la versión usada en García et al., Informatics 2025 (dataset 2): cada variable
se estandarizó con la media y la desviación de las 72 pacientes, y después se añadieron filas sintéticas
con SMOTE para igualar las clases (56 + 56 = 112).

Para volver a las unidades reales se usa  valor = z * desviación + media  con las cifras publicadas en las
tablas 1 y 2 del artículo. Comprobación: los mínimos y máximos reconstruidos coinciden con los publicados.

Uso (desde la carpeta del proyecto):
    python herramientas/convertir_unidades.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ORIGEN = ROOT / "FGR_dataset.xlsx"
DESTINO = ROOT / "FGR_dataset_real.xlsx"

N = 72
POBLACIONAL = np.sqrt((N - 1) / N)  # el archivo se estandarizó con la desviación poblacional (n), no la muestral (n-1)

# Variables continuas: (media, desviación) publicadas, decimales para redondear
CONTINUAS = {
    "C1": (29.64, 5.23, 0),       # edad (años)
    "C2": (25.13, 3.93, 1),       # IMC (kg/m²)
    "C3": (31.58, 1.95, 1),       # edad gestacional al parto (semanas)
    "C7": (28.05, 2.65, 1),       # semana de inicio del síntoma inicial
    "C8": (25.64, 15.6, 0),       # días del síntoma inicial al parto
    "C9": (29.05, 2.86, 1),       # semana de inicio de hipertensión
    "C10": (18.58, 15.96, 0),     # días de hipertensión al parto
    "C13": (29.56, 2.74, 1),      # semana de inicio de proteinuria
    "C14": (14.93, 14.81, 0),     # días de proteinuria al parto
    "C18": (156.62, 18.34, 0),    # presión sistólica máxima (mmHg)
    "C19": (103.89, 11.38, 0),    # presión diastólica máxima (mmHg)
    "C22": (926.49, 1509.46, 0),  # BNP máximo (pg/mL)
    "C23": (61.63, 16.45, 1),     # creatinina máxima (µmol/L)
    "C24": (474.66, 94.98, 1),    # ácido úrico máximo (µmol/L)
    "C25": (4667.44, 2300.29, 0), # proteinuria máxima (mg/24 h)
    "C26": (50.14, 4.55, 1),      # proteína total mínima (g/L)
    "C27": (27.01, 3.39, 1),      # albúmina mínima (g/L)
    "C28": (31.31, 31.71, 1),     # ALT máxima (UI/L)
    "C29": (33.17, 21.7, 1),      # AST máxima (UI/L)
}

# Columnas cuya desviación publicada no reproduce el rango publicado. Se ajusta la escala para que
# el mínimo y el máximo de las pacientes reales coincidan con los publicados: (mínimo, máximo, decimales).
POR_RANGO = {
    "C11": (20, 33.1, 1),   # semana de inicio de edema
    "C12": (1, 83, 0),      # días de edema al parto
    "C30": (42, 356, 0),    # plaquetas mínimas (×10⁹/L); la desviación publicada (110,33) parece una errata
}

# Variables discretas: (valor z del código más bajo, distancia en z entre códigos, código más bajo, código más alto)
DISCRETAS = {
    "C4": (-1.0609, 0.8303, 1, 5),   # gravidez: 1, 2, 3...
    "C5": (-0.9036, 1.6682, 0, 2),   # paridad: 0, 1, 2
    "C6": (-1.0074, 1.0216, 0, 2),   # síntoma inicial: 0 edema, 1 FGR, 2 hipertensión
    "C15": (-0.9563, 0.13345, 1, 31),  # días de tratamiento expectante (1-31)
    "C17": (-0.4775, 1.8096, 0, 2),  # antecedentes: 0 ninguno, 1 hipertensión, 2 SOP
    "C20": (-1.5979, 0.7880, 0, 5),  # motivo del parto (ver campos.py)
    "C21": (-0.3015, 3.6181, 0, 1),  # tipo de parto: 0 cesárea, 1 vaginal
}

# C16 tiene 0 en 61-62 pacientes y valores distintos en 11: coincide con "antihipertensivos antes de la
# hospitalización" (11/72), pero el artículo no dice la unidad. Se guarda como sí (1) / no (0).
C16_CERO = -0.3585


def es_original(df):
    """Las filas reales tienen todas sus variables discretas exactamente en un código; las de SMOTE no."""
    ok = np.ones(len(df), dtype=bool)
    for col, (z0, paso, lo, hi) in DISCRETAS.items():
        if col == "C15":
            continue
        codigos = (df[col] - z0) / paso
        ok &= np.isclose(codigos, codigos.round(), atol=0.01)
    return ok


def convertir(df):
    z = df.round(4)
    original = es_original(z)
    real = pd.DataFrame(index=df.index)

    for col, (media, sd, dec) in CONTINUAS.items():
        real[col] = (z[col] * sd * POBLACIONAL + media).round(dec)

    for col, (minimo, maximo, dec) in POR_RANGO.items():
        zo = z.loc[original, col]
        escala = (maximo - minimo) / (zo.max() - zo.min())
        real[col] = (minimo + (z[col] - zo.min()) * escala).round(dec)

    for col, (z0, paso, lo, hi) in DISCRETAS.items():
        real[col] = ((z[col] - z0) / paso + lo).round().clip(lo, hi).astype(int)

    real["C16"] = (z["C16"] > C16_CERO + 0.01).astype(int)
    real["C31"] = df["C31"].astype(int)
    real["sintetica"] = (~original).astype(int)
    return real[[f"C{i}" for i in range(1, 32)] + ["sintetica"]]


def main():
    real = convertir(pd.read_excel(ORIGEN))
    real.to_excel(DESTINO, index=False)
    originales = real[real.sintetica == 0]
    print(f"Guardado {DESTINO.name}: {len(real)} filas ({len(originales)} reales, {real.sintetica.sum()} sintéticas)")
    print(originales.describe().T[["mean", "min", "max"]].round(2).to_string())


if __name__ == "__main__":
    main()
