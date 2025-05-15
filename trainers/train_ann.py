import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from preprocessing import load_and_preprocess_data

# Cargar y Preprocesar Datos
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Entrenar Red Neuronal Artificial
try:
    ann_model = Sequential([
        Dense(128, activation="relu", input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        Dense(64, activation="relu", kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        Dense(32, activation="relu", kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])
    ann_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                      loss="binary_crossentropy", metrics=["accuracy"])

    # Early Stopping para detener el entrenamiento si no mejora
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    # Entrenamiento
    ann_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

    ann_acc = ann_model.evaluate(X_test, y_test, verbose=0)[1]
    print(f"Exactitud (Red Neuronal): {ann_acc}")

    # Guardar el modelo
    ann_model.save("ann_model.keras")
except Exception as e:
    print(f"Error entrenando Red Neuronal: {e}")
