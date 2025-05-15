import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from preprocessing import load_and_preprocess_data

# Cargar y Preprocesar Datos
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar Máquina de Soporte Vectorial con ajustes de parámetros
try:
    svm_model = SVC(kernel="rbf", C=1.0, gamma='scale', probability=True)
    svm_model.fit(X_train_scaled, y_train)

    # Predicción y exactitud
    svm_y_pred = svm_model.predict(X_test_scaled)
    svm_acc = accuracy_score(y_test, svm_y_pred)

    print(f"Exactitud (SVM): {svm_acc}")

    # Mostrar un reporte de clasificación
    print("\nReporte de Clasificación:\n", classification_report(y_test, svm_y_pred))

    # Validación cruzada para evaluar el rendimiento generalizado
    cross_val_scores = cross_val_score(svm_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"Precisión promedio con validación cruzada: {cross_val_scores.mean()}")

    # Guardar el modelo
    with open("svm_model.pkl", "wb") as f:
        pickle.dump(svm_model, f)

except Exception as e:
    print(f"Error entrenando SVM: {e}")
