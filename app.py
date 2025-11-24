import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import time
import requests
import os
import threading
from flask import Flask, render_template, Response, jsonify, request, session, redirect, url_for
from dotenv import load_dotenv


from Models.UserModels import db, crear_usuario, autenticar_usuario
from Models.HistorialModels import guardar_historial, borrar_mensaje, HistorialMensaje
load_dotenv()

app = Flask(__name__)

# ============= CONFIGURACIÓN DE BASE DE DATOS =============
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:marti123@127.0.0.1:5432/LSA'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'marti123' 

db.init_app(app)

with app.app_context():
    db.create_all()
    print("💾 Conectado a PostgreSQL (Base: LSA). Tablas verificadas.")

# ============= CONFIGURACIÓN GLOBAL IA =============
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print("🚀 Cargando modelo de gestos...")
try:
    model = tf.keras.models.load_model("modelo_gestos_v2.h5")
    with open("labels_v2.pkl", "rb") as f:
        etiquetas = pickle.load(f)
    print(f"✅ Modelo cargado: {len(etiquetas)} clases")
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    model = None
    etiquetas = []

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

# Variables de control para la lógica de detección
variables_control = {
    "ultima_palabra": None,
    "contador_misma_palabra": 0,
    "tiempo_ultima_deteccion": 0,
    "UMBRAL_CONFIANZA": 0.75,
    "TIEMPO_ESPERA": 1.5
}

def extraer_landmarks_mano(hand_landmarks):
    puntos = []
    for landmark in hand_landmarks.landmark:
        puntos.extend([landmark.x, landmark.y, landmark.z])
    return puntos

# === FUNCIÓN ASÍNCRONA PARA CONSULTAR CHATGPT ===
def consultar_chatgpt_async(app_instance, user_email):
    # 1. Validación inicial
    if not estado_app["palabras"]:
        estado_app["estado_ia"] = "error"
        return

    oracion_primitiva = " ".join(estado_app["palabras"]).strip()
    estado_app["estado_ia"] = "cargando"
    
    # 2. Configuración OpenAI
    headers = {
        "Content-Type": "application/json", 
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    prompt = f"""Se te pasarán oraciones primitivas, carentes de conjugaciones, conectores y artículos. 
Debes transformar esa oración y darle un sentido más natural en español argentino.
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
        # 3. Petición a OpenAI
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data, timeout=10)
        
        if response.status_code == 200:
            res = response.json()
            mejorada = res['choices'][0]['message']['content'].strip().replace("Oración mejorada:", "").strip()
            
            # Actualizamos estado global
            estado_app["oracion_mejorada"] = mejorada
            estado_app["estado_ia"] = "listo"

            # === SECCIÓN DE GUARDADO CON DEBUG ===
            print(f"\n--- INTENTO DE GUARDAR HISTORIAL ---")
            print(f"📧 Email recibido: '{user_email}'")
            print(f"📝 Mensaje: '{mejorada}'")

            if user_email:
                try:
                    # Usamos el contexto de la aplicación para conectar a la BD
                    with app_instance.app_context():
                        # Llamamos a la función y capturamos el resultado
                        exito = guardar_historial(user_email, mejorada)
                        
                        if exito:
                            print("✅ ÉXITO: Mensaje guardado en la base de datos.")
                        else:
                            print("❌ ERROR: guardar_historial devolvió False. (Revisa si el usuario existe en la tabla usuarios)")
                except Exception as db_error:
                    print(f"❌ CRASH BD: Falló la conexión dentro del hilo: {db_error}")
            else:
                print("⚠️ ALERTA: No se guardó porque el user_email es None o vacío.")
            
            print("------------------------------------\n")
            # ======================================

        else:
            print(f"Error OpenAI status: {response.status_code}")
            estado_app["estado_ia"] = "error"
            
    except Exception as e:
        print(f"Error General OpenAI: {e}")
        estado_app["estado_ia"] = "error"

def generar_frames():
    cap = cv2.VideoCapture(2) 
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened(): return
    
    while True:
        success, frame = cap.read()
        if not success: break
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        estado_app["manos_detectadas"] = 0
        mensaje_debug = "Sin manos detectadas"

        if results.multi_hand_landmarks and results.multi_handedness:
            estado_app["manos_detectadas"] = len(results.multi_hand_landmarks)
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if model is not None:
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
                        estado_app["confianza_actual"] = int(prob * 100)
                        
                        if prob >= variables_control["UMBRAL_CONFIANZA"]:
                            idx = np.argmax(prediccion)
                            mejor_prediccion = etiquetas[idx]
                            mensaje_debug = f"{mejor_prediccion} ({int(prob*100)}%)"
                            tiempo_actual = time.time()

                            if variables_control["ultima_palabra"] != mejor_prediccion:
                                variables_control["contador_misma_palabra"] = 1
                                variables_control["ultima_palabra"] = mejor_prediccion
                            else:
                                variables_control["contador_misma_palabra"] += 1
                            
                            if (variables_control["contador_misma_palabra"] >= 10 and 
                               (tiempo_actual - variables_control["tiempo_ultima_deteccion"]) > variables_control["TIEMPO_ESPERA"]):
                                estado_app["palabras"].append(mejor_prediccion)
                                estado_app["ultima_deteccion"] = mejor_prediccion
                                variables_control["tiempo_ultima_deteccion"] = tiempo_actual
                                variables_control["contador_misma_palabra"] = 0
                    except Exception: pass
            else:
                 mensaje_debug = "Modelo no cargado"
        else:
            variables_control["contador_misma_palabra"] = 0
            variables_control["ultima_palabra"] = None

        # Dibujar info en pantalla (opcional para debug)
        cv2.putText(frame, mensaje_debug, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ============= RUTAS =============

@app.route('/')
def login_page():
    if 'user_email' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/register')
def register_page():
    return render_template('register.html')

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.json
    identificador = data.get('email')
    password = data.get('password')
    exito, usuario = autenticar_usuario(identificador, password)
    if exito:
        session['user_email'] = usuario.email
        session['username'] = usuario.nombre
        return jsonify({"success": True, "message": "Login exitoso"})
    else:
        return jsonify({"success": False, "message": "Credenciales inválidas"}), 401

@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.json
    nombre = data.get('username') 
    email = data.get('email')
    password = data.get('password')
    
    if not nombre or not email or not password:
        return jsonify({"success": False, "message": "Faltan datos"}), 400
    
    # Intenta crear usuario
    exito, mensaje = crear_usuario(nombre, email, password)
    
    if exito:
        return jsonify({"success": True, "message": mensaje})
    else:
        return jsonify({"success": False, "message": mensaje}), 400

@app.route('/logout')
def logout():
    global estado_app, variables_control
    estado_app["palabras"] = []
    estado_app["oracion_mejorada"] = ""
    estado_app["ultima_deteccion"] = "-"
    estado_app["estado_ia"] = "idle"
    estado_app["confianza_actual"] = 0
    estado_app["manos_detectadas"] = 0
    variables_control["ultima_palabra"] = None
    variables_control["contador_misma_palabra"] = 0
    session.clear()
    return redirect(url_for('login_page'))

@app.route('/dashboard')
def dashboard():
    if 'user_email' not in session:
        return redirect(url_for('login_page'))
    return render_template('dashboard.html', username=session.get('username'))

@app.route('/video_feed')
def video_feed():
    return Response(generar_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_data')
def get_data():
    oracion_raw = " ".join(estado_app["palabras"]) if estado_app["palabras"] else "..."
    return jsonify({
        "palabras": estado_app["palabras"],
        "oracion_mejorada": estado_app["oracion_mejorada"],
        "estado_ia": estado_app["estado_ia"],
        "manos": estado_app["manos_detectadas"],
        "confianza": estado_app["confianza_actual"],
        "ultima_signo": estado_app["ultima_deteccion"],
        "oracion_raw": oracion_raw
    })

@app.route('/accion', methods=['POST'])
def accion():
    try:
        # Validación básica
        if not request.is_json:
            return jsonify({"status": "error", "message": "JSON requerido"}), 400

        req = request.json
        tipo = req.get('tipo')
        
        if tipo == 'limpiar':
            estado_app["palabras"] = []
            estado_app["oracion_mejorada"] = ""
            estado_app["estado_ia"] = "idle"
            estado_app["ultima_deteccion"] = "-"
            variables_control["contador_misma_palabra"] = 0
            
        elif tipo == 'espacio':
            estado_app["palabras"].append(" ")
            
        elif tipo == 'mejorar':
            if estado_app["palabras"]:
                user_email = session.get('user_email')
                
                if user_email:
                    threading.Thread(
                        target=consultar_chatgpt_async, 
                        args=(app, user_email) 
                    ).start()
                else:
                    print("⚠️ Intento de mejorar oración sin usuario en sesión")
                    return jsonify({"status": "error", "message": "Sesión expirada"}), 401
                    
        return jsonify({"status": "ok"})

    except Exception as e:
        print(f"❌ ERROR CRÍTICO EN /accion: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    
# === RUTAS DE HISTORIAL (API) ===

@app.route('/api/historial', methods=['GET'])
def get_historial_api():
    if 'user_email' not in session:
        return jsonify([])
    
    mensajes = HistorialMensaje.query.filter_by(email=session['user_email'])\
        .order_by(HistorialMensaje.fecha_hora.desc()).limit(20).all()
        
    return jsonify([m.to_json() for m in mensajes])

@app.route('/api/historial/<int:id>', methods=['DELETE'])
def delete_historial_api(id):
    if 'user_email' not in session:
        return jsonify({"success": False}), 401
    
    exito = borrar_mensaje(id, session['user_email'])
    if exito:
        return jsonify({"success": True})
    else:
        return jsonify({"success": False, "message": "Error al borrar"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)