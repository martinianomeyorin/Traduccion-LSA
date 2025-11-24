import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import time
import requests
import os
import threading
from flask import Flask, render_template, Response, jsonify, request
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ============= CONFIGURACIÓN GLOBAL =============
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print("🚀 Cargando modelo...")
model = tf.keras.models.load_model("modelo_gestos_v2.h5")
with open("labels_v2.pkl", "rb") as f:
    etiquetas = pickle.load(f)
print(f"✅ Modelo cargado: {len(etiquetas)} clases")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# ============= ESTADO GLOBAL =============
estado_app = {
    "palabras": [],
    "oracion_mejorada": "",
    "ultima_deteccion": "-",
    "estado_ia": "idle",
    "confianza_actual": 0,
    "manos_detectadas": 0
}

# Variables de control
variables_control = {
    "ultima_palabra": None,
    "contador_misma_palabra": 0,
    "tiempo_ultima_deteccion": 0,
    "UMBRAL_CONFIANZA": 0.75,  # BAJADO de 0.90 a 0.75
    "TIEMPO_ESPERA": 1.5
}

def extraer_landmarks_mano(hand_landmarks):
    """Extrae los 63 valores (21 puntos × 3 coordenadas) de una mano"""
    puntos = []
    for landmark in hand_landmarks.landmark:
        puntos.extend([landmark.x, landmark.y, landmark.z])
    return puntos

def consultar_chatgpt_async():
    """Lógica de ChatGPT en hilo separado"""
    if not estado_app["palabras"]:
        estado_app["estado_ia"] = "error"
        return

    oracion_primitiva = " ".join(estado_app["palabras"]).strip()
    estado_app["estado_ia"] = "cargando"
    
    print(f"📤 Consultando ChatGPT: '{oracion_primitiva}'")
    
    headers = {
        "Content-Type": "application/json", 
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    prompt = f"""Se te pasarán oraciones primitivas, carentes de conjugaciones, conectores y artículos. 
Debes transformar esa oración y darle un sentido más natural, conservando la idea principal que representan las palabras originales, 
pero si es necesario añade también sustantivos y verbos para conservar naturalidad. 
Solo me tenés que pasar una oración resultante y además tiene que estar en la lengua argentina (argentinismos, voseo, etc).

Oración primitiva: {oracion_primitiva}

Oración mejorada:"""
    
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
        response = requests.post(
            "https://api.openai.com/v1/chat/completions", 
            headers=headers, 
            json=data, 
            timeout=10
        )
        
        if response.status_code == 200:
            res = response.json()
            mejorada = res['choices'][0]['message']['content'].strip()
            mejorada = mejorada.replace("Oración mejorada:", "").strip()
            estado_app["oracion_mejorada"] = mejorada
            estado_app["estado_ia"] = "listo"
            print(f"✅ ChatGPT: '{mejorada}'")
        else:
            print(f"❌ Error ChatGPT: {response.status_code}")
            estado_app["estado_ia"] = "error"
    except Exception as e:
        print(f"❌ Error ChatGPT: {e}")
        estado_app["estado_ia"] = "error"

# ============= GENERADOR DE VIDEO =============
def generar_frames():
    cap = cv2.VideoCapture(2)
    
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara")
        return
    
    print("✅ Cámara iniciada")
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        estado_app["manos_detectadas"] = 0
        mensaje_debug = "Sin manos detectadas"

        if results.multi_hand_landmarks and results.multi_handedness:
            estado_app["manos_detectadas"] = len(results.multi_hand_landmarks)
            num_manos = len(results.multi_hand_landmarks)

            # Dibujar landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )

            # Extracción de datos (2 manos)
            mano_izquierda = [0.0] * 63
            mano_derecha = [0.0] * 63
            tiene_izquierda = 0
            tiene_derecha = 0

            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                landmarks = extraer_landmarks_mano(hand_landmarks)
                
                if label == "Right":
                    mano_derecha = landmarks
                    tiene_derecha = 1
                else:
                    mano_izquierda = landmarks
                    tiene_izquierda = 1
            
            fila_combinada = mano_izquierda + mano_derecha + [tiene_izquierda, tiene_derecha]
            
            if len(fila_combinada) == 128:
                try:
                    prediccion = model.predict(np.array([fila_combinada]), verbose=0)[0]
                    prob = np.max(prediccion)
                    
                    # Actualizar confianza
                    estado_app["confianza_actual"] = int(prob * 100)
                    
                    if prob >= variables_control["UMBRAL_CONFIANZA"]:
                        idx = np.argmax(prediccion)
                        mejor_prediccion = etiquetas[idx]
                        mensaje_debug = f"{mejor_prediccion} ({int(prob*100)}%)"

                        tiempo_actual = time.time()

                        # Lógica de estabilidad
                        if variables_control["ultima_palabra"] != mejor_prediccion:
                            variables_control["contador_misma_palabra"] = 1
                            variables_control["ultima_palabra"] = mejor_prediccion
                        else:
                            variables_control["contador_misma_palabra"] += 1
                        
                        # Mostrar progreso de estabilidad
                        progreso = min(variables_control["contador_misma_palabra"], 10)
                        mensaje_debug += f" [{progreso}/10]"
                        
                        # Agregar palabra cuando sea estable
                        if (variables_control["contador_misma_palabra"] >= 10 and 
                           (tiempo_actual - variables_control["tiempo_ultima_deteccion"]) > variables_control["TIEMPO_ESPERA"]):
                            
                            estado_app["palabras"].append(mejor_prediccion)
                            estado_app["ultima_deteccion"] = mejor_prediccion
                            
                            variables_control["tiempo_ultima_deteccion"] = tiempo_actual
                            variables_control["contador_misma_palabra"] = 0
                            
                            print(f"✅ Palabra agregada: {mejor_prediccion}")

                except Exception as e:
                    print(f"❌ Error predicción: {e}")
                    mensaje_debug = f"Error: {str(e)}"
        else:
            variables_control["contador_misma_palabra"] = 0
            variables_control["ultima_palabra"] = None

        # Dibujar info en video
        cv2.putText(frame, mensaje_debug, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Mostrar número de manos
        cv2.putText(frame, f"Manos: {estado_app['manos_detectadas']}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Codificar
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ============= RUTAS FLASK =============
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generar_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_data')
def get_data():
    # IMPORTANTE: Join con espacio para formar oración legible
    oracion_raw = " ".join(estado_app["palabras"]) if estado_app["palabras"] else "..."
    
    return jsonify({
        "palabras": estado_app["palabras"],
        "oracion_raw": oracion_raw,
        "oracion_mejorada": estado_app["oracion_mejorada"],
        "estado_ia": estado_app["estado_ia"],
        "manos": estado_app["manos_detectadas"],
        "confianza": estado_app["confianza_actual"],
        "ultima_signo": estado_app["ultima_deteccion"]
    })

@app.route('/accion', methods=['POST'])
def accion():
    req = request.json
    tipo = req.get('tipo')
    
    print(f"🔧 Acción recibida: {tipo}")
    
    if tipo == 'limpiar':
        estado_app["palabras"] = []
        estado_app["oracion_mejorada"] = ""
        estado_app["estado_ia"] = "idle"
        estado_app["ultima_deteccion"] = "-"
        variables_control["ultima_palabra"] = None
        variables_control["contador_misma_palabra"] = 0
        print("🗑️ Datos limpiados")
        
    elif tipo == 'espacio':
        estado_app["palabras"].append(" ")
        print("⎵ Espacio agregado")
        
    elif tipo == 'mejorar':
        if estado_app["palabras"]:
            threading.Thread(target=consultar_chatgpt_async).start()
            print("🤖 Mejorando con IA...")
        else:
            print("⚠️ No hay palabras para mejorar")
        
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌐 SERVIDOR WEB LSA INICIADO")
    print("="*60)
    print(f"📊 Clases: {len(etiquetas)}")
    print(f"🎯 Umbral confianza: {variables_control['UMBRAL_CONFIANZA']*100}%")
    print(f"⏱️ Tiempo espera: {variables_control['TIEMPO_ESPERA']}s")
    print(f"🤖 ChatGPT: {'✅' if OPENAI_API_KEY else '❌'}")
    print("\n🌐 Abrir en navegador: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)