"""Prepara y entrena un reconocedor temporal con los videos LSA64.

Uso:
    python entrenar_lsa64_temporal.py --archive C:/ruta/lsa64_raw.zip

Cada muestra conserva 32 fotogramas de landmarks por video. Los participantes
1-8 se usan para entrenamiento, el 9 para validación y el 10 para prueba.
"""
import argparse
import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf


FRAME_COUNT = 32
FEATURE_COUNT = 128
VIDEO_NAME = re.compile(r"^all/(\d{3})_(\d{3})_(\d{3})\.mp4$", re.IGNORECASE)

# Orden oficial de las 64 clases del conjunto, traducidas al español.
LABELS = [
    "Opaco", "Rojo", "Verde", "Amarillo", "Brillante", "Azul claro", "Colores", "Rosa",
    "Mujer", "Enemigo", "Hijo", "Hombre", "Lejos", "Cajón", "Nacido", "Aprender",
    "Llamar", "Espumadera", "Amargo", "Dulce de leche", "Leche", "Agua", "Comida", "Argentina",
    "Uruguay", "País", "Apellido", "Dónde", "Imitar", "Cumpleaños", "Desayuno", "Foto",
    "Hambriento", "Mapa", "Moneda", "Música", "Barco", "Ninguno", "Nombre", "Paciencia",
    "Perfume", "Sordo", "Trampa", "Arroz", "Parrilla", "Caramelo", "Chicle", "Fideos",
    "Yogur", "Aceptar", "Gracias", "Cerrar", "Aparecer", "Aterrizar", "Atrapar", "Ayuda",
    "Bailar", "Bañarse", "Comprar", "Copiar", "Correr", "Darse cuenta", "Dar", "Encontrar",
]


def landmarks_mano(hand_landmarks):
    return [value for point in hand_landmarks.landmark for value in (point.x, point.y, point.z)]


def extraer_secuencia(video_path, detector):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_total < 1:
            return None
        targets = set(np.linspace(0, frame_total - 1, FRAME_COUNT).round().astype(int).tolist())
        frames = []
        for index in range(frame_total):
            ok, frame = cap.read()
            if not ok:
                break
            if index not in targets:
                continue

            # Mantener la misma convención de espejo que usa la cámara en la app.
            frame = cv2.flip(frame, 1)
            result = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            left, right, has_left, has_right = [0.0] * 63, [0.0] * 63, 0, 0
            for hand, handedness in zip(result.multi_hand_landmarks or [], result.multi_handedness or []):
                side = handedness.classification[0].label
                if side == "Right":
                    right, has_right = landmarks_mano(hand), 1
                else:
                    left, has_left = landmarks_mano(hand), 1
            frames.append(left + right + [has_left, has_right])

        if not frames:
            return None
        # Videos breves o con recuentos de fotogramas inexactos se completan
        # repitiendo muestras uniformes de la secuencia extraída.
        if len(frames) != FRAME_COUNT:
            indices = np.linspace(0, len(frames) - 1, FRAME_COUNT).round().astype(int)
            frames = [frames[index] for index in indices]
        sequence = np.asarray(frames, dtype=np.float32)
        if not np.any(sequence[:, 126:128]):
            return None
        return sequence
    finally:
        cap.release()


def preparar_dataset(archive_path, cache_path):
    sequences, labels, signers, processed = [], [], [], set()
    if cache_path.exists():
        with np.load(cache_path) as data:
            sequences = list(data["sequences"])
            labels = list(data["labels"])
            signers = list(data["signers"])
            processed = set(data["processed"].astype(str).tolist())
        print(f"Reanudando desde caché: {len(processed)} videos ya procesados")

    def guardar_checkpoint():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            sequences=np.asarray(sequences, dtype=np.float32).reshape((len(sequences), FRAME_COUNT, FEATURE_COUNT)),
            labels=np.asarray(labels, dtype=np.int64),
            signers=np.asarray(signers, dtype=np.int64),
            processed=np.asarray(sorted(processed), dtype=np.str_),
        )

    with zipfile.ZipFile(archive_path) as archive:
        members = sorted(
            (info for info in archive.infolist() if VIDEO_NAME.fullmatch(info.filename)),
            key=lambda info: info.filename,
        )
        if len(members) != 3200:
            raise ValueError(f"Se esperaban 3200 videos LSA64; se encontraron {len(members)}")

        failures = []
        hands_api = mp.solutions.hands
        pending = [member for member in members if member.filename not in processed]
        with hands_api.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.5,
        ) as detector, tempfile.TemporaryDirectory(prefix="lsa64-") as temp_dir:
            video_path = Path(temp_dir) / "sample.mp4"
            for number, member in enumerate(pending, start=1):
                match = VIDEO_NAME.fullmatch(member.filename)
                sign_id, signer_id, _repetition = map(int, match.groups())
                with archive.open(member) as source, video_path.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
                sequence = extraer_secuencia(video_path, detector)
                video_path.unlink(missing_ok=True)
                if sequence is None:
                    failures.append(member.filename)
                else:
                    sequences.append(sequence)
                    labels.append(sign_id - 1)
                    signers.append(signer_id)
                processed.add(member.filename)
                if number % 100 == 0 or number == len(pending):
                    guardar_checkpoint()
                    print(f"Procesados {len(processed)}/{len(members)}; secuencias válidas: {len(sequences)}")

        if failures:
            print(f"Aviso: {len(failures)} videos no produjeron landmarks utilizables.")
            print("Primeros casos:", ", ".join(failures[:8]))
        if not sequences:
            raise ValueError("No se pudo extraer ninguna secuencia de landmarks")

    sequences = np.asarray(sequences, dtype=np.float32).reshape((len(sequences), FRAME_COUNT, FEATURE_COUNT))
    labels = np.asarray(labels, dtype=np.int64)
    signers = np.asarray(signers, dtype=np.int64)
    guardar_checkpoint()
    print(f"Caché guardada en {cache_path}")
    return sequences, labels, signers


def crear_modelo():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(FRAME_COUNT, FEATURE_COUNT)),
        tf.keras.layers.Masking(mask_value=0.0),
        tf.keras.layers.GRU(96, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.GRU(64),
        tf.keras.layers.Dense(96, activation="relu"),
        tf.keras.layers.Dropout(0.35),
        tf.keras.layers.Dense(len(LABELS), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def entrenar(sequences, labels, signers, output_dir):
    train = np.isin(signers, np.arange(1, 9))
    validation = signers == 9
    test = signers == 10
    if not train.any() or not validation.any() or not test.any():
        raise ValueError("El dataset debe incluir participantes del 1 al 10 para esta separación")

    model = crear_modelo()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ]
    model.fit(
        sequences[train], labels[train],
        validation_data=(sequences[validation], labels[validation]),
        epochs=60,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )
    loss, accuracy = model.evaluate(sequences[test], labels[test], verbose=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "modelo_lsa64_temporal.keras")
    (output_dir / "labels_lsa64_temporal.json").write_text(
        json.dumps(LABELS, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metrics = {
        "test_signers": [10],
        "train_videos": int(train.sum()),
        "validation_videos": int(validation.sum()),
        "test_videos": int(test.sum()),
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
        "class_count": len(LABELS),
        "frames_per_video": FRAME_COUNT,
    }
    (output_dir / "metricas_lsa64_temporal.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Entrena un clasificador temporal con LSA64")
    parser.add_argument("--archive", required=True, type=Path, help="Ruta al ZIP oficial lsa64_raw.zip")
    parser.add_argument("--cache", type=Path, default=Path("dataset_lsa64_temporal.npz"))
    parser.add_argument("--output", type=Path, default=Path("modelos/lsa64"))
    args = parser.parse_args()
    if not args.archive.is_file():
        parser.error(f"No se encontró el ZIP: {args.archive}")
    sequences, labels, signers = preparar_dataset(args.archive, args.cache)
    entrenar(sequences, labels, signers, args.output)


if __name__ == "__main__":
    main()
