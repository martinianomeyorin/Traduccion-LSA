import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle

def entrenar_modelo():
    print("Cargando dataset...")
    df = pd.read_csv("dataset_manos_lsa_con_palabras.csv")
    print("Dataset cargado correctamente.")
    X = df.drop("label", axis=1).values
    y = df["label"].values

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(len(set(y_encoded)), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y_encoded, epochs=20, batch_size=32)

    model.save("modelo_gestos.h5")
    with open("labels.pkl", "wb") as f:
        pickle.dump(encoder.classes_, f)

if __name__ == "__main__":
    entrenar_modelo()

