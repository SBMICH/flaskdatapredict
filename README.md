# Predicción con modelos de aprendizaje automático (Flask)

Aplicación web en Flask que clasifica casos del conjunto de datos **FGR** (112 pacientes, 30 variables de
entrada `C1..C30` y una etiqueta binaria `C31`) con cuatro modelos y permite hacer predicciones
individuales o subir un archivo `.csv`/`.xlsx` para evaluar un lote completo.

> **Proyecto educativo. No es una herramienta médica ni debe usarse para tomar decisiones clínicas.**

## Resultados

Evaluados sobre el 30 % de los datos que no se usó para entrenar (34 filas, clases balanceadas 50/50).
Con tan pocos datos las cifras varían mucho según la división; tómalas como orientativas.

| Modelo | Exactitud (prueba) | F1 |
|---|---|---|
| Regresión logística | 76,5 % | 0,75 |
| SVM (kernel RBF) | 85,3 % | 0,86 |
| Red neuronal (Keras) | 85,3 % | 0,83 |
| FCM (experimental) | 50,0 % | 0,67 |

**Sobre el FCM:** la matriz de pesos de `models/fcm_model.pkl` no se aprendió de los datos, y su resultado es el
del azar (clasifica todo como clase 1). Se deja en la app solo como referencia y está marcado como experimental.
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

- **Predicción individual:** el dataset original ya venía *estandarizado* (promedio 0, desviación 1), así que los
  campos esperan valores en esa escala y no en las unidades que aparecen en los textos de ayuda. El botón
  "Cargar una fila de ejemplo" rellena el formulario con una fila real del dataset.
- **Predicción por lote:** el archivo debe tener las columnas `C1..C30` y, opcionalmente, `C31`. Si incluye `C31`
  se muestran la exactitud y la matriz de confusión. Máximo 5.000 filas y 2 MB.

## Estructura

```
app.py              Rutas Flask y validación de entradas
modelos.py          Carga de modelos y funciones de predicción (incluye el FCM)
trainers/train.py   Entrena los modelos, guarda el preprocesador y las métricas
models/             Modelos entrenados, preprocessor.pkl y metrics.json
herramientas/       Script para exportar el dataset a formato ARFF
tests/              Pruebas con pytest
```

## Decisiones y correcciones de esta versión

- El preprocesador (imputación + estandarización) se ajusta **solo con los datos de entrenamiento**, se guarda en
  `models/preprocessor.pkl` y es el mismo que usa la app. Antes se reajustaba en cada petición con una sola fila,
  lo que dejaba todas las entradas en cero y hacía que las predicciones no significaran nada.
- Semilla fija (`42`) para que el entrenamiento sea reproducible.
- Validación de campos, tipos, rangos, columnas y tamaño de archivo.

## Pruebas

```bash
pip install -r requirements-dev.txt
python -m pytest
```

## Limitaciones

- El dataset es muy pequeño (112 filas): hay riesgo de sobreajuste y las métricas son inestables.
- Los modelos se cargan con `pickle`: solo carga archivos `.pkl` de fuentes que controles.
- El origen y la descripción de cada variable del dataset deben documentarse antes de cualquier uso más allá
  del aprendizaje.
