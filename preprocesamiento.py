import os
import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm

# Configuración de MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

dataset = []

def procesar_video(ruta_video, etiqueta):
    """
    Procesa un video y extrae landmarks de las manos
    """
    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        print(f"Error al abrir: {ruta_video}")
        return 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    manos_detectadas = 0
    
    # Barra de progreso para cada video
    pbar = tqdm(total=total_frames, desc=f"{etiqueta} - {os.path.basename(ruta_video)}", leave=False)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        pbar.update(1)
        
        # Convertir a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        # Si hay manos detectadas
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extraer coordenadas (x, y, z) de los 21 puntos
                puntos = []
                for landmark in hand_landmarks.landmark:
                    puntos.extend([landmark.x, landmark.y, landmark.z])
                
                # Agregar al dataset
                dataset.append(puntos + [etiqueta])
                manos_detectadas += 1
    
    pbar.close()
    cap.release()
    return manos_detectadas

def main():
    carpeta_dataset = "LSA"
    
    print("🚀 GENERANDO DATASET DE LENGUAJE DE SEÑAS")
    print("=" * 50)
    
    # Verificar estructura de carpetas y contar videos
    print(f"🔍 Explorando carpeta: {carpeta_dataset}")
    
    if not os.path.exists(carpeta_dataset):
        print(f"❌ Carpeta no encontrada: {carpeta_dataset}")
        return
    
    contenido = os.listdir(carpeta_dataset)
    print(f"📁 Contenido encontrado: {len(contenido)} elementos")
    
    total_videos = 0
    clases = []
    todos_los_videos = []
    
    # Verificar si los videos están directamente en LSA o en subcarpetas
    videos_directos = [f for f in contenido if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if videos_directos:
        print(f"📹 Videos encontrados directamente en LSA: {len(videos_directos)}")
        # Si los videos están directos, extraer clase del nombre del archivo
        for video in videos_directos:
            # Asumiendo formato como "001_001_001.mp4" donde "001" es la clase
            clase = video.split('_')[0]  # Extrae los primeros 3 dígitos
            ruta_video = os.path.join(carpeta_dataset, video)
            todos_los_videos.append((ruta_video, clase))
            if clase not in clases:
                clases.append(clase)
        total_videos = len(videos_directos)
    else:
        # Videos en subcarpetas (estructura original)
        for clase in contenido:
            ruta_clase = os.path.join(carpeta_dataset, clase)
            if os.path.isdir(ruta_clase):
                videos_clase = [f for f in os.listdir(ruta_clase) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
                if videos_clase:
                    for video in videos_clase:
                        ruta_video = os.path.join(ruta_clase, video)
                        todos_los_videos.append((ruta_video, clase))
                    total_videos += len(videos_clase)
                    clases.append(clase)
    
    print(f"📊 Total de videos: {total_videos}")
    print(f"📊 Clases encontradas: {len(clases)}")
    print(f"📊 Clases: {', '.join(clases[:10])}{'...' if len(clases) > 10 else ''}")
    print()
    
    # Procesar videos
    videos_procesados = 0
    total_detecciones = 0
    
    if not todos_los_videos:
        print("❌ No se encontraron videos para procesar")
        return
    
    # Barra de progreso general
    pbar_general = tqdm(total=total_videos, desc="Progreso general")
    
    for ruta_video, clase in todos_los_videos:
        detecciones = procesar_video(ruta_video, clase)
        total_detecciones += detecciones
        videos_procesados += 1
        pbar_general.update(1)
    
    pbar_general.close()
    
    # Guardar dataset
    print(f"\n💾 GUARDANDO DATASET...")
    
    if len(dataset) > 0:
        # Crear columnas
        columnas = [f"{eje}{i}" for i in range(21) for eje in ['x', 'y', 'z']] + ["label"]
        df = pd.DataFrame(dataset, columns=columnas)
        
        # Guardar CSV
        archivo_salida = "dataset_manos_lsa.csv"
        df.to_csv(archivo_salida, index=False)
        
        # Estadísticas finales
        print(f"✅ ¡Dataset generado exitosamente!")
        print(f"   📄 Archivo: {archivo_salida}")
        print(f"   📊 Total muestras: {len(df):,}")
        print(f"   📊 Videos procesados: {videos_procesados}")
        print(f"   📊 Detecciones totales: {total_detecciones:,}")
        print(f"   📊 Clases únicas: {df['label'].nunique()}")
        
        print(f"\n📈 Distribución por clase:")
        distribucion = df['label'].value_counts().sort_index()
        for clase, count in distribucion.head(10).items():
            print(f"   {clase}: {count:,} muestras")
        
        if len(distribucion) > 10:
            print(f"   ... y {len(distribucion) - 10} clases más")
        
        print(f"\n🎯 Promedio de muestras por video: {len(df) / videos_procesados:.1f}")
        
    else:
        print("❌ Dataset vacío. Verifica que los videos contengan manos visibles.")

if __name__ == "__main__":
    main()
    print("\n🎉 ¡Proceso completado!")

    