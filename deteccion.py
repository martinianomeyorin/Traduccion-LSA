import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import time
import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()

# ============= CONFIGURACIÓN DE LA API DE OPENAI =============
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

if not OPENAI_API_KEY:
    print("❌ Error: No se encontró la variable de entorno OPENAI_API_KEY.")
    print("Por favor, configúrala antes de ejecutar el script.")
    exit()

def mejorar_oracion_con_chatgpt(oracion_primitiva):
    """Envía la oración a ChatGPT para mejorarla con coherencia gramatical"""
    if not oracion_primitiva.strip():
        return oracion_primitiva
    
    prompt = f"""Se te pasarán oraciones primitivas, carentes de conjugaciones, conectores y artículos. 
Debes transformar esa oración y darle un sentido más natural, conservando la idea principal que representan las palabras originales, 
pero si es necesario añade también sustantivos y verbos para conservar naturalidad. 
Solo me tenés que pasar una oración resultante y además tiene que estar en la lengua argentina (argentinismos, voseo, etc).

Oración primitiva: {oracion_primitiva}

Oración mejorada:"""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Sos un asistente que mejora oraciones en español argentino."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 150
    }
    
    try:
        print(f"\n📤 Enviando a ChatGPT: '{oracion_primitiva}'")
        response = requests.post(OPENAI_API_URL, headers=headers, json=data, timeout=10)
        
        if response.status_code == 200:
            resultado = response.json()
            oracion_mejorada = resultado['choices'][0]['message']['content'].strip()
            oracion_mejorada = oracion_mejorada.replace("Oración mejorada:", "").strip()
            print(f"✅ ChatGPT respondió: '{oracion_mejorada}'")
            return oracion_mejorada
        else:
            print(f"❌ Error API: {response.status_code} - {response.text}")
            return oracion_primitiva
            
    except requests.exceptions.Timeout:
        print("⏱️ Timeout: ChatGPT tardó demasiado en responder")
        return oracion_primitiva
    except Exception as e:
        print(f"❌ Error al conectar con ChatGPT: {e}")
        return oracion_primitiva

def extraer_landmarks_mano(hand_landmarks):
    """Extrae los 63 valores (21 puntos × 3 coordenadas) de una mano"""
    puntos = []
    for landmark in hand_landmarks.landmark:
        puntos.extend([landmark.x, landmark.y, landmark.z])
    return puntos

# ============= CÓDIGO PRINCIPAL =============
def detectar_en_tiempo_real():
    print("Cargando modelo...")
    try:
        model = tf.keras.models.load_model("modelo_gestos_v2.h5")
        print("✓ Modelo cargado correctamente")
        
        input_shape = model.input_shape[1]
        print(f"✓ El modelo espera {input_shape} características")
        
        if input_shape != 128:
            print(f"⚠️ ADVERTENCIA: Este código espera un modelo de 128 features (2 manos)")
            print(f"   Pero el modelo cargado tiene {input_shape} features")
            return
            
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        return

    print("Cargando etiquetas...")
    try:
        with open("labels_v2.pkl", "rb") as f:
            etiquetas = pickle.load(f)
        print(f"✓ Etiquetas cargadas: {len(etiquetas)} clases")
    except Exception as e:
        print(f"❌ Error al cargar etiquetas: {e}")
        return

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara")
        return

    print("\n" + "="*60)
    print("CÁMARA INICIADA - TRADUCTOR LSA CON IA (2 MANOS)")
    print("="*60)
    print("Controles:")
    print("  - Q o ESC: Salir")
    print("  - F: Pantalla completa")
    print("  - C: Limpiar oración")
    print("  - ESPACIO: Agregar espacio")
    print("  - M: Mejorar oración con ChatGPT")
    print("="*60 + "\n")
    
    window_name = "Traductor LSA con IA (2 Manos)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    pantalla_completa = False
    UMBRAL_CONFIANZA = 0.90
    
    oracion = []
    oracion_mejorada = ""
    ultima_palabra = None
    contador_misma_palabra = 0
    tiempo_ultima_deteccion = 0
    TIEMPO_ESPERA = 1.5
    mostrar_mejorada = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer el frame")
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        mejor_prediccion = None
        mejor_probabilidad = 0
        mensaje_estado = ""

        if results.multi_hand_landmarks and results.multi_handedness:
            num_manos = len(results.multi_hand_landmarks)
            
            # Dibujar landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
            
            # Preparar datos para predicción
            mano_izquierda = [0.0] * 63
            mano_derecha = [0.0] * 63
            tiene_izquierda = 0
            tiene_derecha = 0
            
            # Clasificar cada mano detectada
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                landmarks = extraer_landmarks_mano(hand_landmarks)
                
                if label == "Right":
                    mano_derecha = landmarks
                    tiene_derecha = 1
                else:  # "Left"
                    mano_izquierda = landmarks
                    tiene_izquierda = 1
            
            # Crear vector de entrada: [63 izq] + [63 der] + [has_left, has_right]
            fila_combinada = mano_izquierda + mano_derecha + [tiene_izquierda, tiene_derecha]
            
            if len(fila_combinada) == 128:
                try:
                    prediccion = model.predict(np.array([fila_combinada]), verbose=0)[0]
                    prob = np.max(prediccion)
                    
                    if prob >= UMBRAL_CONFIANZA:
                        mejor_prediccion = etiquetas[np.argmax(prediccion)]
                        mejor_probabilidad = prob
                        
                        if num_manos == 2:
                            mensaje_estado = "Seña bimanual - 2 manos detectadas"
                        else:
                            mensaje_estado = "1 mano detectada"
                except Exception as e:
                    print(f"Error al predecir: {e}")
            
            # Mostrar estado
            if mensaje_estado:
                cv2.putText(frame, mensaje_estado, 
                          (10, frame.shape[0] - 230), 
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.6, 
                          (255, 200, 0), 
                          2)
            
            # Mostrar predicción
            if mejor_prediccion is not None:
                porcentaje = mejor_probabilidad * 100
                
                cv2.putText(frame, f"Sena: {mejor_prediccion}", 
                          (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          1.0, 
                          (0, 255, 0), 
                          3)
                
                cv2.putText(frame, f"Confianza: {porcentaje:.1f}%", 
                          (10, 70), 
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.7, 
                          (0, 255, 0), 
                          2)
                
                # Barra de confianza
                barra_ancho = 300
                barra_lleno = int((porcentaje / 100) * barra_ancho)
                cv2.rectangle(frame, (10, 85), (10 + barra_ancho, 105), (100, 100, 100), 2)
                cv2.rectangle(frame, (10, 85), (10 + barra_lleno, 105), (0, 255, 0), -1)
                
                tiempo_actual = time.time()
                
                # Contar estabilidad
                if ultima_palabra != mejor_prediccion:
                    contador_misma_palabra = 1
                    ultima_palabra = mejor_prediccion
                else:
                    contador_misma_palabra += 1
                
                progreso = min(contador_misma_palabra, 10)
                cv2.putText(frame, f"Estabilidad: {progreso}/10", 
                          (10, 130), 
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.6, 
                          (255, 255, 0), 
                          2)
                
                # Agregar palabra si es estable
                if (contador_misma_palabra >= 10 and 
                    (tiempo_actual - tiempo_ultima_deteccion) > TIEMPO_ESPERA):
                    if oracion:
                        oracion.append(" ")
                    oracion.append(mejor_prediccion)
                    print(f"✓ Palabra agregada: {mejor_prediccion}")
                    print(f"   Oración primitiva: {''.join(oracion)}")
                    tiempo_ultima_deteccion = tiempo_actual
                    contador_misma_palabra = 0
                    mostrar_mejorada = False
        else:
            contador_misma_palabra = 0
            ultima_palabra = None
            cv2.putText(frame, "No se detectan manos", 
                      (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 
                      0.8, 
                      (0, 0, 255), 
                      2)

        # Mostrar oración
        if mostrar_mejorada and oracion_mejorada:
            texto_a_mostrar = oracion_mejorada
            color_texto = (100, 255, 100)
            etiqueta = "Oracion mejorada con IA:"
        else:
            texto_a_mostrar = ''.join(oracion) if oracion else "[Esperando senas...]"
            color_texto = (255, 255, 255)
            etiqueta = "Oracion primitiva:"
        
        ancho_frame = frame.shape[1]
        caracteres_por_linea = int(ancho_frame / 12)
        
        palabras_mostrar = [etiqueta] + texto_a_mostrar.split()
        lineas = []
        linea_actual = ""
        
        for i, palabra in enumerate(palabras_mostrar):
            if i == 0:
                linea_actual = palabra
            elif len(linea_actual + " " + palabra) <= caracteres_por_linea:
                linea_actual += " " + palabra
            else:
                if linea_actual:
                    lineas.append(linea_actual)
                linea_actual = palabra
        
        if linea_actual:
            lineas.append(linea_actual)
        
        # Fondo semitransparente
        y_inicial = frame.shape[0] - 30 - (len(lineas) * 40)
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, y_inicial - 15), (ancho_frame - 5, frame.shape[0] - 5), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        for i, linea in enumerate(lineas):
            y_pos = y_inicial + (i * 40)
            cv2.putText(frame, linea, 
                       (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.9, 
                       color_texto, 
                       2)
        
        # Instrucciones
        cv2.putText(frame, "M: Mejorar con IA | F: Pantalla | C: Limpiar | ESPACIO: Espacio | Q: Salir", 
                   (10, frame.shape[0] - 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, 
                   (200, 200, 200), 
                   1)

        cv2.imshow(window_name, frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:
            break
        elif key == ord('m'):
            if oracion:
                oracion_primitiva = ''.join(oracion).strip()
                print("\n" + "="*60)
                print("🤖 MEJORANDO ORACIÓN CON CHATGPT...")
                print("="*60)
                oracion_mejorada = mejorar_oracion_con_chatgpt(oracion_primitiva)
                mostrar_mejorada = True
            else:
                print("⚠ No hay oración para mejorar")
        elif key == ord('f'):
            if pantalla_completa:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                pantalla_completa = False
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                pantalla_completa = True
        elif key == ord('c'):
            oracion = []
            oracion_mejorada = ""
            ultima_palabra = None
            contador_misma_palabra = 0
            mostrar_mejorada = False
            print("→ Oración limpiada")
        elif key == 32:
            if oracion:
                oracion.append(" ")
                mostrar_mejorada = False
        
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except:
            break

    # Resumen final
    if oracion:
        oracion_primitiva_final = ''.join(oracion).strip()
        print("\n" + "="*60)
        print("📝 RESUMEN FINAL:")
        print("="*60)
        print(f"Oración primitiva: {oracion_primitiva_final}")
        if oracion_mejorada:
            print(f"Oración mejorada:  {oracion_mejorada}")
        print("="*60)
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Aplicación cerrada")

if __name__ == "__main__":
    detectar_en_tiempo_real()