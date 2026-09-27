"""Aplicación web para traducir señas de LSA.

Configuración por variables de entorno: SECRET_KEY, DATABASE_URL y OLLAMA_BASE_URL.
"""
import os
import pickle
import secrets
import threading
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import requests
import tensorflow as tf
from dotenv import load_dotenv
from authlib.integrations.base_client.errors import OAuthError
from authlib.integrations.flask_client import OAuth
from flask import Flask, Response, flash, jsonify, redirect, render_template, request, session, url_for
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from Models.HistorialModels import HistorialMensaje, borrar_mensaje, guardar_historial
from Models.UserModels import GoogleAccount, User, autenticar_usuario, crear_usuario, db

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.getenv("SECRET_KEY") or os.urandom(32),
    SQLALCHEMY_DATABASE_URI=os.getenv("DATABASE_URL", f"sqlite:///{(BASE_DIR / 'lsa.db').as_posix()}"),
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.getenv("COOKIE_SECURE", "false").lower() == "true",
    MAX_CONTENT_LENGTH=16 * 1024,
)
db.init_app(app)
oauth = OAuth(app)
google_login_enabled = bool(os.getenv("GOOGLE_CLIENT_ID") and os.getenv("GOOGLE_CLIENT_SECRET"))
if google_login_enabled:
    oauth.register(
        name="google",
        client_id=os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )
with app.app_context():
    db.create_all()

# State belongs to the signed-in user; only one local camera/model instance is used.
_states = {}
_state_lock = threading.RLock()
_camera_lock = threading.Lock()
_ia_jobs = set()
_ia_lock = threading.Lock()

def _state(email):
    with _state_lock:
        return _states.setdefault(email, {
            "palabras": [], "oracion_mejorada": "", "ultima_deteccion": "-",
            "estado_ia": "idle", "confianza_actual": 0, "manos_detectadas": 0,
            "error_ia": "", "camera_error": "", "ultima_palabra": None, "contador": 0, "ultimo_tiempo": 0,
        })

try:
    model = tf.keras.models.load_model(BASE_DIR / "modelo_gestos_v2.h5")
    with (BASE_DIR / "labels_v2.pkl").open("rb") as labels_file:
        etiquetas = pickle.load(labels_file)
    if model.output_shape[-1] != len(etiquetas):
        raise ValueError("La cantidad de etiquetas no coincide con las salidas del modelo")
    app.logger.info("Modelo de gestos cargado (%d clases)", len(etiquetas))
except Exception:
    app.logger.exception("No se pudo cargar el modelo de gestos")
    model, etiquetas = None, []

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

def extraer_landmarks_mano(hand_landmarks):
    return [coordenada for landmark in hand_landmarks.landmark for coordenada in (landmark.x, landmark.y, landmark.z)]

def _mejorar_oracion(email, oracion):
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    ollama_model = os.getenv("OLLAMA_MODEL", "gemma3:4b")
    try:
        response = requests.post(
            f"{ollama_url}/api/chat",
            json={"model": ollama_model, "stream": False, "messages": [
                {"role": "system", "content": "Reescribí frases primitivas de Lengua de Señas Argentina como español argentino natural. Conservá el sentido y no agregues información."},
                {"role": "user", "content": oracion}], "options": {"temperature": 0.4, "num_predict": 150}},
            timeout=float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180")),
        )
        if not response.ok:
            try:
                provider_message = response.json().get("error", "")
            except (ValueError, AttributeError):
                provider_message = ""
            app.logger.error("Ollama respondió HTTP %s: %s", response.status_code, str(provider_message)[:500])
            if response.status_code == 404:
                message = f"Ollama no encuentra el modelo {ollama_model}. Descargalo con: ollama pull {ollama_model}"
            elif response.status_code >= 500:
                message = "Ollama tuvo un problema al generar la respuesta. Revisá que el modelo esté instalado y volvé a intentar."
            else:
                message = f"Ollama rechazó la solicitud (HTTP {response.status_code}). Revisá la salida del servidor."
            with _state_lock:
                _state(email).update(estado_ia="error", error_ia=message)
            return
        mejorada = response.json()["message"]["content"].strip()
        if not mejorada:
            raise ValueError("La IA devolvió una respuesta vacía")
        with _state_lock:
            state = _state(email)
            state["oracion_mejorada"], state["estado_ia"], state["error_ia"] = mejorada, "listo", ""
        with app.app_context():
            if not guardar_historial(email, mejorada):
                app.logger.error("No se pudo guardar el historial para %s", email)
    except requests.RequestException as error:
        app.logger.warning("No se pudo conectar con Ollama: %s", error)
        with _state_lock:
            _state(email).update(estado_ia="error", error_ia="No se pudo conectar con Ollama. Confirmá que la aplicación de Ollama esté abierta y volvé a intentar.")
    except Exception:
        app.logger.exception("Error al procesar la respuesta de Ollama")
        with _state_lock:
            _state(email).update(estado_ia="error", error_ia="No se pudo procesar la respuesta de Ollama. Revisá la salida del servidor.")
    finally:
        with _ia_lock:
            _ia_jobs.discard(email)

def generar_frames(email):
    # Serializes camera access so multiple tabs cannot open competing devices.
    with _camera_lock:
        requested_index = int(os.getenv("CAMERA_INDEX", "2"))
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        cap = None
        for camera_index in dict.fromkeys((requested_index, 2, 0, 1)):
            candidate = cv2.VideoCapture(camera_index, backend)
            if candidate.isOpened():
                cap = candidate
                app.logger.info("Cámara abierta en el índice %s", camera_index)
                break
            candidate.release()
        if cap is None:
            message = "No se pudo abrir ninguna cámara. Cerrá otras apps que la estén usando o configurá CAMERA_INDEX en .env."
            app.logger.error(message)
            with _state_lock:
                _state(email)["camera_error"] = message
            yield b""
            return
        with _state_lock:
            _state(email)["camera_error"] = ""
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    with _state_lock:
                        _state(email)["camera_error"] = "La cámara se abrió, pero dejó de enviar imagen. Revisá la conexión y los permisos de cámara."
                    break
                frame = cv2.flip(frame, 1)
                results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                state = _state(email)
                state["manos_detectadas"] = len(results.multi_hand_landmarks or [])
                debug = "Sin manos detectadas"
                if results.multi_hand_landmarks:
                    for landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
                if results.multi_hand_landmarks and results.multi_handedness and model is not None:
                    left, right, has_left, has_right = [0.0] * 63, [0.0] * 63, 0, 0
                    for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        label = handedness.classification[0].label
                        if label == "Right":
                            right, has_right = extraer_landmarks_mano(landmarks), 1
                        else:
                            left, has_left = extraer_landmarks_mano(landmarks), 1
                    try:
                        prediction = model(np.asarray([left + right + [has_left, has_right]], dtype=np.float32), training=False).numpy()[0]
                        probability, index = float(np.max(prediction)), int(np.argmax(prediction))
                        state["confianza_actual"] = round(probability * 100)
                        if probability >= 0.75:
                            word, now = str(etiquetas[index]), time.monotonic()
                            debug = f"{word} ({round(probability * 100)}%)"
                            if state["ultima_palabra"] == word:
                                state["contador"] += 1
                            else:
                                state["ultima_palabra"], state["contador"] = word, 1
                            if state["contador"] >= 10 and now - state["ultimo_tiempo"] > 1.5:
                                if state["palabras"] and state["palabras"][-1] != " ":
                                    state["palabras"].append(" ")
                                state["palabras"].append(word)
                                state["ultima_deteccion"], state["ultimo_tiempo"] = word, now
                                state["contador"] = 0
                        else:
                            state["contador"], state["ultima_palabra"] = 0, None
                    except Exception:
                        app.logger.exception("Error durante la inferencia")
                elif model is None:
                    debug = "Modelo no cargado"
                cv2.putText(frame, debug, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                encoded, buffer = cv2.imencode(".jpg", frame)
                if encoded:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        finally:
            cap.release()

def _usuario_google(google_sub, email, display_name):
    linked_account = GoogleAccount.query.filter_by(google_sub=google_sub).first()
    if linked_account:
        user = User.query.filter_by(email=linked_account.email).first()
        if user:
            return user
        raise ValueError("La cuenta de Google está vinculada a un usuario que ya no existe")

    user = User.query.filter_by(email=email).first()
    if user is None:
        base_name = (display_name or email.partition("@")[0]).strip()[:100] or "Usuario"
        username, suffix = base_name, 2
        while User.query.filter_by(nombre=username).first():
            suffix_text = f" ({suffix})"
            username = f"{base_name[:100 - len(suffix_text)]}{suffix_text}"
            suffix += 1
        user = User(email=email, nombre=username)
        user.set_password(secrets.token_urlsafe(48))
        db.session.add(user)
        db.session.flush()

    db.session.add(GoogleAccount(google_sub=google_sub, email=user.email))
    try:
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        linked_account = GoogleAccount.query.filter_by(google_sub=google_sub).first()
        user = User.query.filter_by(email=linked_account.email).first() if linked_account else None
        if user is None:
            raise
    return user

@app.get("/")
def login_page():
    return redirect(url_for("dashboard")) if session.get("user_email") else render_template("login.html", google_login_enabled=google_login_enabled)

@app.get("/register")
def register_page():
    return render_template("register.html")

@app.post("/api/login")
def api_login():
    data = request.get_json(silent=True) or {}
    email, password = data.get("email"), data.get("password")
    if not isinstance(email, str) or not isinstance(password, str) or not email.strip() or not password:
        return jsonify(success=False, message="Ingresá tu usuario y contraseña"), 400
    success, user = autenticar_usuario(email.strip(), password)
    if not success:
        return jsonify(success=False, message="Credenciales inválidas"), 401
    session.clear()
    session.update(user_email=user.email, username=user.nombre)
    return jsonify(success=True, message="Login exitoso")

@app.post("/api/register")
def api_register():
    data = request.get_json(silent=True) or {}
    name, email, password = data.get("username"), data.get("email"), data.get("password")
    if not all(isinstance(value, str) for value in (name, email, password)):
        return jsonify(success=False, message="Completá todos los campos"), 400
    if not name.strip() or not email.strip() or len(name.strip()) > 100 or len(email.strip()) > 255:
        return jsonify(success=False, message="Revisá el nombre y el correo ingresados"), 400
    if len(password) < 8:
        return jsonify(success=False, message="La contraseña debe tener al menos 8 caracteres"), 400
    success, message = crear_usuario(name.strip(), email.strip(), password)
    return jsonify(success=success, message=message), (201 if success else 400)

@app.get("/auth/google")
def google_login():
    if not google_login_enabled:
        flash("El inicio con Google todavía no está configurado. Agregá GOOGLE_CLIENT_ID y GOOGLE_CLIENT_SECRET al archivo .env.")
        return redirect(url_for("login_page"))
    redirect_uri = os.getenv("GOOGLE_REDIRECT_URI") or url_for("google_callback", _external=True)
    return oauth.google.authorize_redirect(redirect_uri)

@app.get("/auth/google/callback")
def google_callback():
    if not google_login_enabled:
        return redirect(url_for("login_page"))
    try:
        token = oauth.google.authorize_access_token()
        claims = token.get("userinfo") or oauth.google.userinfo()
        email = claims.get("email", "").strip().lower()
        google_sub = claims.get("sub")
        if not email or not google_sub or claims.get("email_verified") is not True:
            flash("Google no devolvió un correo verificado. No se inició sesión.")
            return redirect(url_for("login_page"))
        user = _usuario_google(google_sub, email, claims.get("name", ""))
        session.clear()
        session.update(user_email=user.email, username=user.nombre)
        return redirect(url_for("dashboard"))
    except OAuthError as error:
        db.session.rollback()
        app.logger.warning("Falló el flujo de inicio de sesión de Google: %s", error.error)
        flash("No se pudo completar el inicio de sesión con Google. Intentá nuevamente.")
    except Exception:
        db.session.rollback()
        app.logger.exception("Error procesando el inicio de sesión de Google")
        flash("Ocurrió un error al iniciar sesión con Google.")
    return redirect(url_for("login_page"))

@app.get("/logout")
def logout():
    email = session.get("user_email")
    if email:
        with _state_lock:
            _states.pop(email, None)
    session.clear()
    return redirect(url_for("login_page"))

@app.get("/dashboard")
def dashboard():
    if not session.get("user_email"):
        return redirect(url_for("login_page"))
    return render_template("dashboard.html", username=session.get("username"))

@app.get("/video_feed")
def video_feed():
    if not session.get("user_email"):
        return "", 401
    return Response(generar_frames(session["user_email"]), mimetype="multipart/x-mixed-replace; boundary=frame", headers={"Cache-Control": "no-store"})

@app.get("/get_data")
def get_data():
    email = session.get("user_email")
    if not email:
        return jsonify(success=False), 401
    with _state_lock:
        state = _state(email).copy()
    return jsonify(palabras=state["palabras"], oracion_mejorada=state["oracion_mejorada"], estado_ia=state["estado_ia"], error_ia=state["error_ia"], camera_error=state["camera_error"], manos=state["manos_detectadas"], confianza=state["confianza_actual"], ultima_signo=state["ultima_deteccion"], oracion_raw="".join(state["palabras"]) or "...")

@app.post("/accion")
def accion():
    email = session.get("user_email")
    if not email:
        return jsonify(status="error", message="Sesión expirada"), 401
    data = request.get_json(silent=True)
    if not data or data.get("tipo") not in {"limpiar", "espacio", "mejorar"}:
        return jsonify(status="error", message="Acción inválida"), 400
    state = _state(email)
    with _state_lock:
        action = data["tipo"]
        if action == "limpiar":
            state.update(palabras=[], oracion_mejorada="", estado_ia="idle", error_ia="", confianza_actual=0, ultima_deteccion="-", contador=0, ultima_palabra=None)
        elif action == "espacio":
            if state["palabras"] and state["palabras"][-1] != " ":
                state["palabras"].append(" ")
        else:
            source = "".join(state["palabras"]).strip()
            if not source:
                return jsonify(status="error", message="Primero agregá una seña"), 400
            with _ia_lock:
                if email in _ia_jobs:
                    return jsonify(status="error", message="Ya estamos mejorando una oración"), 409
                _ia_jobs.add(email)
            state["estado_ia"] = "cargando"
            state["error_ia"] = ""
            threading.Thread(target=_mejorar_oracion, args=(email, source), daemon=True).start()
    return jsonify(status="ok")

@app.get("/api/historial")
def get_historial_api():
    email = session.get("user_email")
    if not email:
        return jsonify(success=False, message="Sesión expirada"), 401
    messages = HistorialMensaje.query.filter_by(email=email).order_by(HistorialMensaje.fecha_hora.desc()).limit(20).all()
    return jsonify([message.to_json() for message in messages])

@app.delete("/api/historial/<int:message_id>")
def delete_historial_api(message_id):
    email = session.get("user_email")
    if not email:
        return jsonify(success=False, message="Sesión expirada"), 401
    if not borrar_mensaje(message_id, email):
        return jsonify(success=False, message="Mensaje inexistente"), 404
    return jsonify(success=True)

@app.get("/health")
def health():
    try:
        db.session.execute(text("SELECT 1"))
        return jsonify(status="ok", database="ok", model="ready" if model is not None else "unavailable")
    except SQLAlchemyError:
        app.logger.exception("Health check de base de datos falló")
        return jsonify(status="error", database="unavailable"), 503

if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "5000")), debug=False)
