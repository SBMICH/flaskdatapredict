# FGR Predict: restricción del crecimiento fetal con aprendizaje automático (Flask)

Aplicación web en Flask que estima si el bebé de una paciente con **preeclampsia de inicio temprano** tendrá
**restricción del crecimiento fetal (FGR)** o un peso normal, a partir de 30 variables clínicas. Compara cuatro
modelos y permite hacer predicciones individuales o subir un archivo `.csv`/`.xlsx` con un lote completo.

> **Proyecto educativo. No es una herramienta médica ni debe usarse para tomar decisiones clínicas.**

## Datos

Los datos provienen de un estudio retrospectivo con **72 pacientes** del Anhui Maternal and Child Health Care
Hospital: Yi J, Chen L, Meng X, Chen Y. *Risk factors and foetal growth restriction associated with expectant
treatment of early-onset preeclampsia*. Annals of Medicine 2022;54(1):3250-3257
([PMC9673790](https://pmc.ncbi.nlm.nih.gov/articles/PMC9673790/)).

El archivo con el que se construyó el proyecto (`FGR_dataset.xlsx`) es la versión usada en García et al.,
*Informatics* 2025 (dataset 2). Esa versión tiene dos particularidades:

1. **Venía estandarizada** (promedio 0, desviación 1), sin unidades.
2. **Tiene 112 filas, no 72:** se añadieron unas 40 filas sintéticas con SMOTE para igualar las clases
   (56 con FGR y 56 sin ella).

`herramientas/convertir_unidades.py` recupera las unidades reales usando los promedios y desviaciones publicados
en el artículo y genera `FGR_dataset_real.xlsx`, con una columna `sintetica` que marca las filas generadas.
Los mínimos, máximos y frecuencias reconstruidos coinciden con los publicados (hay un test que lo comprueba).
Excepciones: las **plaquetas** (la desviación publicada parece tener una errata) y el **edema** se reconstruyeron
ajustando el rango publicado, así que son aproximados. En el síntoma inicial, el orden entre «edema» e
«hipertensión» se dedujo de las frecuencias (35 y 34 pacientes).

La etiqueta `C31` vale 1 si hubo restricción del crecimiento fetal (peso estimado por debajo del percentil 10)
y 0 si el peso fue normal.

## Resultados

Para no inflar las métricas, **las filas sintéticas solo se usan para entrenar**. La evaluación se hace con el
30 % de las pacientes reales (23 pacientes: 17 con FGR y 6 sin ella). Como las clases están desbalanceadas, la
métrica principal es la **exactitud balanceada** (el promedio de aciertos en cada clase; adivinar da 50 %).

| Modelo | Exactitud balanceada | Exactitud | F1 |
|---|---|---|---|
| SVM (kernel RBF) | 85,8 % | 87,0 % | 0,91 |
| Red neuronal (Keras) | 85,8 % | 87,0 % | 0,91 |
| Regresión logística | 79,9 % | 78,3 % | 0,84 |
| FCM (experimental) | 50,0 % | 73,9 % | 0,85 |

Responder siempre «FGR» daría 73,9 % de exactitud, por eso la exactitud simple engaña aquí. Con tan pocos datos
las cifras cambian bastante según la división; tómalas como orientativas.

**Sobre el FCM:** la matriz de pesos de `models/fcm_model.pkl` no se aprendió de los datos y su resultado es el
del azar (clasifica todo como FGR). Se deja en la app solo como referencia y está marcado como experimental.

Las métricas se regeneran con `python trainers/train.py` y se guardan en `models/metrics.json`.

## Cómo ejecutarla

```bash
python -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Abre http://127.0.0.1:5000. Para depurar, define `FLASK_DEBUG=1` (por defecto está desactivado).

## Cómo usarla

- **Predicción individual:** cada campo se escribe en su unidad clínica (años, mmHg, µmol/L...) y empieza en el
  valor típico de las pacientes del estudio. Debajo se muestra el rango observado. Las variables categóricas
  (síntoma inicial, antecedentes, motivo y tipo de parto...) se eligen de una lista. El interruptor
  **Usar niveles en todo** convierte todos los campos en listas de niveles (*muy bajo*, *bajo*, *típico*, *alto*,
  *muy alto*: percentiles 10, 25, 50, 75 y 90). También se puede:
  - cargar una **paciente de ejemplo** real y ver si el modelo acierta su resultado;
  - **comparar los 4 modelos** con los mismos datos.
- **Predicción por lote:** el archivo debe tener las columnas `C1..C30` en las mismas unidades y, opcionalmente,
  `C31`. Si incluye `C31` se muestran la exactitud y la matriz de confusión. Máximo 5.000 filas y 2 MB.
  La app ofrece una plantilla descargable en `/plantilla.csv`.

## Estructura

```
app.py                Rutas Flask y validación de entradas
campos.py             Descripción, unidades y rangos de las 30 variables
modelos.py            Carga de modelos y funciones de predicción (incluye el FCM)
trainers/train.py     Entrena los modelos, guarda el preprocesador y las métricas
models/               Modelos entrenados, preprocessor.pkl y metrics.json
templates/, static/   Interfaz (HTML, CSS y JavaScript sin frameworks)
herramientas/         Conversión a unidades reales y exportación a ARFF
FGR_dataset.xlsx      Dataset original (estandarizado)
FGR_dataset_real.xlsx Dataset en unidades reales (generado)
tests/                Pruebas con pytest
```

## Decisiones y correcciones

- El preprocesador (imputación + estandarización) se ajusta **solo con los datos de entrenamiento**, se guarda en
  `models/preprocessor.pkl` y es el mismo que usa la app. La primera versión lo reajustaba en cada petición con una
  sola fila, lo que dejaba todas las entradas en cero y hacía que las predicciones no significaran nada.
- Las filas sintéticas no se usan para evaluar.
- Semilla fija (`42`) para que el entrenamiento sea reproducible.
- Validación de campos, tipos, rango de cada variable (el del estudio con un margen), columnas y tamaño de archivo.
- Los modelos se cargan en segundo plano al arrancar, para que la primera predicción no tarde.

## Pruebas

```bash
pip install -r requirements-dev.txt
python -m pytest
```

## Limitaciones

- Solo 72 pacientes reales: hay riesgo de sobreajuste y las métricas son inestables.
- Todas las pacientes tenían preeclampsia precoz: los modelos no aplican a otros embarazos.
- Algunas variables se reconstruyeron de forma aproximada (ver la sección Datos).
- Los modelos se cargan con `pickle`: solo carga archivos `.pkl` de fuentes que controles.
