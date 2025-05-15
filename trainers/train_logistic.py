import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from preprocessing import load_and_preprocess_data
from sklearn.preprocessing import StandardScaler

# Cargar y Preprocesar Datos
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Escalar los datos (si no lo hiciste antes)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar Regresión Logística con ajustes de parámetros
try:
    logistic_model = LogisticRegression(solver='liblinear', C=1.0, max_iter=200)
    logistic_model.fit(X_train_scaled, y_train)

    # Predicción y exactitud
    logistic_y_pred = logistic_model.predict(X_test_scaled)
    logistic_acc = accuracy_score(y_test, logistic_y_pred)

    print(f"Exactitud (Regresión Logística): {logistic_acc}")

    # Mostrar un reporte de clasificación
    print("\nReporte de Clasificación:\n", classification_report(y_test, logistic_y_pred))

    # Validación cruzada para evaluar el rendimiento generalizado
    cross_val_scores = cross_val_score(logistic_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"Precisión promedio con validación cruzada: {cross_val_scores.mean()}")

    # Guardar el modelo
    with open("logistic_model.pkl", "wb") as f:
        pickle.dump(logistic_model, f)

except Exception as e:
    print(f"Error entrenando Regresión Logística: {e}")
