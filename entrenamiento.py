import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np

def entrenar_modelo():
    print("🚀 ENTRENANDO MODELO CON 2 MANOS")
    print("=" * 60)
    
    # Cargar dataset
    print("\n📂 Cargando dataset...")
    df = pd.read_csv("dataset_manos_lsa_v2_con_palabras.csv")
    print(f"✓ Dataset cargado: {len(df)} muestras")
    print(f"✓ Columnas: {df.shape[1]}")
    print(f"✓ Clases únicas: {df['label'].nunique()}")
    
    # Mostrar distribución de uso de manos
    solo_izq = df[(df['has_left'] == 1) & (df['has_right'] == 0)].shape[0]
    solo_der = df[(df['has_left'] == 0) & (df['has_right'] == 1)].shape[0]
    ambas = df[(df['has_left'] == 1) & (df['has_right'] == 1)].shape[0]
    
    print(f"\n👐 Distribución de manos:")
    print(f"   Solo izquierda: {solo_izq} ({solo_izq/len(df)*100:.1f}%)")
    print(f"   Solo derecha: {solo_der} ({solo_der/len(df)*100:.1f}%)")
    print(f"   Ambas manos: {ambas} ({ambas/len(df)*100:.1f}%)")
    
    # Separar features y labels
    X = df.drop("label", axis=1).values
    y = df["label"].values
    
    print(f"\n📊 Shape de X: {X.shape}")
    print(f"   (debe ser: muestras × 128)")
    print(f"   - 63 features mano izquierda")
    print(f"   - 63 features mano derecha")
    print(f"   - 2 flags (has_left, has_right)")
    
    # Verificar que tenemos 128 features
    if X.shape[1] != 128:
        print(f"\n⚠️ ADVERTENCIA: Se esperaban 128 features pero hay {X.shape[1]}")
        print("   Verifica que el dataset fue generado correctamente")
    
    # Codificar labels
    print("\n🔤 Codificando etiquetas...")
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    num_clases = len(encoder.classes_)
    print(f"✓ {num_clases} clases codificadas")
    
    # Dividir en train/test
    print("\n✂️ Dividiendo dataset (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"✓ Train: {len(X_train)} muestras")
    print(f"✓ Test: {len(X_test)} muestras")
    
    # Crear modelo
    print("\n🧠 Construyendo modelo...")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128,)),  # 128 features de entrada
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_clases, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✓ Modelo creado")
    model.summary()
    
    # Entrenar modelo
    print("\n🎯 Entrenando modelo...")
    print("=" * 60)
    
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluar modelo
    print("\n📈 Evaluando modelo...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"✓ Precisión en test: {test_accuracy*100:.2f}%")
    print(f"✓ Loss en test: {test_loss:.4f}")
    
    # Guardar modelo y etiquetas
    print("\n💾 Guardando modelo y etiquetas...")
    model.save("modelo_gestos_v2.h5")
    print("✓ Modelo guardado: modelo_gestos_v2.h5")
    
    with open("labels_v2.pkl", "wb") as f:
        pickle.dump(encoder.classes_, f)
    print("✓ Etiquetas guardadas: labels_v2.pkl")
    
    # Resumen final
    print("\n" + "=" * 60)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"📊 Total muestras: {len(df)}")
    print(f"📊 Clases: {num_clases}")
    print(f"📊 Precisión final: {test_accuracy*100:.2f}%")
    print(f"📊 Features: 128 (2 manos + flags)")
    print("=" * 60)
    
    return model, history

if __name__ == "__main__":
    entrenar_modelo()
    print("\n🎉 ¡Proceso completado!")