"""Aplicación web para traducir señas de LSA.

Configuración por variables de entorno: SECRET_KEY, DATABASE_URL y OLLAMA_BASE_URL.
"""
import os
import pickle
import re
import secrets
import hashlib
import hmac
import smtplib
import threading
import time
import unicodedata
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from zoneinfo import ZoneInfo

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
from Models.UserModels import GoogleAccount, PasswordResetCode, PracticeSign, User, autenticar_usuario, crear_usuario, db, validar_password

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Una grabación de referencia (participante 1, repetición 1) por cada clase LSA64.
LSA64_VIDEO_IDS = {
    "opaco": 1, "rojo": 2, "verde": 3, "amarillo": 4, "brillante": 5, "azul claro": 6,
    "bandera": 7, "rosa": 8, "mujer": 9, "enemigo": 10, "hijo": 11, "hombre": 12,
    "lejos": 13, "cajon": 14, "nacido": 15, "aprender": 16, "llamar": 17,
    "desnatadora": 18, "amargo": 19, "dulce de leche": 20, "leche": 21, "agua": 22,
    "comida": 23, "argentina": 24, "uruguay": 25, "pais": 26, "apellido": 27, "donde": 28,
    "imitar": 29, "cumpleanos": 30, "desayuno": 31, "foto": 32, "hambriento": 33,
    "mapa": 34, "acunar": 35, "musica": 36, "barco": 37, "ninguno": 38, "nombre": 39,
    "paciencia": 40, "perfume": 41, "sordo": 42, "trampa": 43, "arroz": 44,
    "parrilla": 45, "dulce": 46, "chicle": 47, "fideos": 48, "yogur": 49, "aceptar": 50,
    "gracias": 51, "cerrar": 52, "aparecer": 53, "aterrizar": 54, "atrapar": 55,
    "ayuda": 56, "bailar": 57, "banarse": 58, "comprar": 59, "copiar": 60, "correr": 61,
    "darse cuenta": 62, "dar": 63, "encontrar": 64,
}

PRACTICE_DAY_MILESTONES = (10, 20, 30, 40, 50, 64)
PRACTICE_STREAK_MILESTONES = (2, 3, 5, 7, 14, 30)
ARGENTINA_TZ = ZoneInfo("America/Argentina/Buenos_Aires")


def _normalizar_nombre_sena(label):
    decomposed = unicodedata.normalize("NFD", str(label).casefold())
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def _racha_maxima(dias):
    mejor = actual = 0
    anterior = None
    for dia in sorted(dias):
        actual = actual + 1 if anterior and (dia - anterior).days == 1 else 1
        mejor = max(mejor, actual)
        anterior = dia
    return mejor


def _progreso_practica(email):
    registros = PracticeSign.query.filter_by(email=email).all()
    conteos = {}
    for registro in registros:
        conteos.setdefault(registro.practice_date, set()).add(_normalizar_nombre_sena(registro.sign_label))

    hoy = datetime.now(ARGENTINA_TZ).date()
    cantidad_hoy = len(conteos.get(hoy, set()))
    dias = set(conteos)
    racha_maxima = _racha_maxima(dias)
    inicio_racha = hoy if hoy in dias else hoy - timedelta(days=1)
    racha_actual = 0
    while inicio_racha in dias:
        racha_actual += 1
        inicio_racha -= timedelta(days=1)

    mejor_dia = max((len(senas) for senas in conteos.values()), default=0)
    logros = []
    for umbral in PRACTICE_DAY_MILESTONES:
        logros.append({
            "id": f"day-{umbral}", "category": "daily", "title": f"Coleccionista de señas {umbral}",
            "description": f"Aprendé {umbral} señas distintas en un día.", "threshold": umbral,
            "progress": min(cantidad_hoy, umbral), "unlocked": mejor_dia >= umbral,
        })
    for umbral in PRACTICE_STREAK_MILESTONES:
        logros.append({
            "id": f"streak-{umbral}", "category": "streak", "title": f"Racha de {umbral} días",
            "description": f"Practicá señas {umbral} días consecutivos.", "threshold": umbral,
            "progress": min(racha_actual, umbral), "unlocked": racha_maxima >= umbral,
        })
    return {
        "today_count": cantidad_hoy,
        "best_day": mejor_dia,
        "current_streak": racha_actual,
        "unlocked_count": sum(1 for logro in logros if logro["unlocked"]),
        "achievements": logros,
    }

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

def _now_utc_naive():
    return datetime.now(timezone.utc).replace(tzinfo=None)

def _hash_reset_code(email, code):
    secret = app.config["SECRET_KEY"]
    secret = secret.encode("utf-8") if isinstance(secret, str) else secret
    payload = f"password-reset:{email}:{code}".encode("utf-8")
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()

def _send_reset_email(email, code):
    server = os.getenv("MAIL_SERVER", "").strip()
    username = os.getenv("MAIL_USERNAME", "").strip()
    password = os.getenv("MAIL_PASSWORD", "")
    sender = os.getenv("MAIL_FROM", "").strip() or username
    if not all((server, username, password, sender)):
        raise RuntimeError("Falta configurar MAIL_SERVER, MAIL_USERNAME, MAIL_PASSWORD o MAIL_FROM")

    message = EmailMessage()
    message["Subject"] = "Código para restablecer tu contraseña"
    message["From"] = sender
    message["To"] = email
    message.set_content(
        "Recibimos una solicitud para restablecer la contraseña de tu cuenta.\n\n"
        f"Tu código de verificación es: {code}\n\n"
        "El código vence en 10 minutos y solo se puede usar una vez. "
        "Si no solicitaste este cambio, podés ignorar este correo."
    )

    port = int(os.getenv("MAIL_PORT", "587"))
    use_ssl = os.getenv("MAIL_USE_SSL", "false").lower() == "true"
    use_tls = os.getenv("MAIL_USE_TLS", "true").lower() == "true"
    timeout = float(os.getenv("MAIL_TIMEOUT_SECONDS", "20"))
    if use_ssl:
        with smtplib.SMTP_SSL(server, port, timeout=timeout) as smtp:
            smtp.login(username, password)
            smtp.send_message(message)
    else:
        with smtplib.SMTP(server, port, timeout=timeout) as smtp:
            smtp.ehlo()
            if use_tls:
                smtp.starttls()
                smtp.ehlo()
            smtp.login(username, password)
            smtp.send_message(message)

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
                {"role": "system", "content": (
                    "Sos un traductor de palabras reconocidas de Lengua de Señas Argentina a español argentino. "
                    "La entrada es una secuencia de palabras señadas, no una pregunta sobre el significado de una palabra. "
                    "Convertí únicamente esa secuencia en una sola oración breve y natural, conservando su significado "
                    "y agregando solo la gramática indispensable. No expliques la seña ni definas palabras; no agregues "
                    "contexto, interpretaciones, alternativas, introducciones, listas ni comentarios. Si la secuencia ya "
                    "es comprensible, repetila con puntuación. Ejemplo: `opaco gracias` → `Opaco, gracias.` "
                    "Respondé exclusivamente con la oración traducida."
                )},
                {"role": "user", "content": oracion}], "options": {"temperature": 0.1, "num_predict": 48}},
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
            oracion_pendiente = oracion.strip()
            oracion_actual = "".join(state["palabras"]).strip()
            if oracion_actual == oracion_pendiente:
                state["palabras"] = []
            elif oracion_actual.startswith(oracion_pendiente):
                palabras_nuevas = oracion_actual[len(oracion_pendiente):].strip()
                state["palabras"] = [palabras_nuevas] if palabras_nuevas else []
            if not state["palabras"]:
                state["contador"], state["ultima_palabra"] = 0, None
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
                        vector = np.asarray([left + right + [has_left, has_right]], dtype=np.float32)
                        prediction = model(vector, training=False).numpy()[0]
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
    is_json_request = request.is_json
    data = request.get_json(silent=True) or {} if is_json_request else request.form
    email, password = data.get("email"), data.get("password")
    if not isinstance(email, str) or not isinstance(password, str) or not email.strip() or not password:
        if not is_json_request:
            flash("Ingresá tu usuario y contraseña")
            return redirect(url_for("login_page"))
        return jsonify(success=False, message="Ingresá tu usuario y contraseña"), 400
    success, user = autenticar_usuario(email.strip(), password)
    if not success:
        if not is_json_request:
            flash("Credenciales inválidas")
            return redirect(url_for("login_page"))
        return jsonify(success=False, message="Credenciales inválidas"), 401
    session.clear()
    session.update(user_email=user.email, username=user.nombre)
    if not is_json_request:
        return redirect(url_for("dashboard"))
    return jsonify(success=True, message="Login exitoso")

@app.post("/api/register")
def api_register():
    data = request.get_json(silent=True) or {}
    name, email, password = data.get("username"), data.get("email"), data.get("password")
    if not all(isinstance(value, str) for value in (name, email, password)):
        return jsonify(success=False, message="Completá todos los campos"), 400
    if not name.strip() or not email.strip() or len(name.strip()) > 100 or len(email.strip()) > 255:
        return jsonify(success=False, message="Revisá el nombre y el correo ingresados"), 400
    password_error = validar_password(password)
    if password_error:
        return jsonify(success=False, message=password_error), 400
    success, message = crear_usuario(name.strip(), email.strip(), password)
    return jsonify(success=success, message=message), (201 if success else 400)

@app.post("/api/password-reset/request")
def request_password_reset():
    data = request.get_json(silent=True) if request.is_json else request.form
    data = data or {}
    email = data.get("email")
    if not isinstance(email, str) or len(email.strip()) > 255 or not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email.strip()):
        return jsonify(success=False, message="Ingresá un correo válido."), 400

    mail_configured = all(os.getenv(key) for key in ("MAIL_SERVER", "MAIL_USERNAME", "MAIL_PASSWORD"))
    if not mail_configured:
        return jsonify(success=False, message="El envío de correo todavía no está configurado en el servidor."), 503

    email = email.strip().lower()
    generic_message = "Si existe una cuenta con ese correo, enviaremos un código para restablecer la contraseña."
    user = User.query.filter_by(email=email).first()
    if user is None:
        return jsonify(success=True, message=generic_message)

    now = _now_utc_naive()
    reset = PasswordResetCode.query.filter_by(email=email).first()
    if reset and now - reset.last_sent_at < timedelta(seconds=60):
        return jsonify(success=True, message=generic_message)

    code = f"{secrets.randbelow(1_000_000):06d}"
    if reset is None:
        reset = PasswordResetCode(email=email, code_hash=_hash_reset_code(email, code), expires_at=now + timedelta(minutes=10), last_sent_at=now, attempts=0)
        db.session.add(reset)
    else:
        reset.code_hash = _hash_reset_code(email, code)
        reset.expires_at = now + timedelta(minutes=10)
        reset.last_sent_at = now
        reset.attempts = 0
    db.session.commit()

    try:
        _send_reset_email(email, code)
    except Exception:
        app.logger.exception("No se pudo enviar un código de restablecimiento")
        # In local development, give the user actionable feedback. On a public
        # deployment keep the generic response to avoid account enumeration.
        db.session.delete(reset)
        db.session.commit()
        if request.remote_addr in {"127.0.0.1", "::1"}:
            return jsonify(
                success=False,
                message="No se pudo enviar el correo. Revisá la configuración SMTP y la consola donde ejecutaste app.py.",
            ), 503
    return jsonify(success=True, message=generic_message)

@app.post("/api/password-reset/confirm")
def confirm_password_reset():
    data = request.get_json(silent=True) if request.is_json else request.form
    data = data or {}
    email, code = data.get("email"), data.get("code")
    password, confirmation = data.get("password"), data.get("password_confirmation")
    if not all(isinstance(value, str) for value in (email, code, password, confirmation)):
        return jsonify(success=False, message="Completá todos los campos."), 400
    email, code = email.strip().lower(), code.strip()
    password_error = validar_password(password)
    if password_error:
        return jsonify(success=False, message=password_error + "."), 400
    if password != confirmation:
        return jsonify(success=False, message="Las contraseñas no coinciden."), 400
    if len(email) > 255 or len(code) != 6 or not code.isdigit():
        return jsonify(success=False, message="El código es inválido o venció. Solicitá uno nuevo."), 400

    reset = PasswordResetCode.query.filter_by(email=email).first()
    now = _now_utc_naive()
    if reset is None or now >= reset.expires_at or reset.attempts >= 5:
        return jsonify(success=False, message="El código es inválido o venció. Solicitá uno nuevo."), 400
    if not hmac.compare_digest(reset.code_hash, _hash_reset_code(email, code)):
        reset.attempts += 1
        db.session.commit()
        return jsonify(success=False, message="El código es inválido o venció. Solicitá uno nuevo."), 400

    user = User.query.filter_by(email=email).first()
    if user is None:
        db.session.delete(reset)
        db.session.commit()
        return jsonify(success=False, message="El código es inválido o venció. Solicitá uno nuevo."), 400
    user.set_password(password)
    db.session.delete(reset)
    db.session.commit()
    return jsonify(success=True, message="Contraseña actualizada. Ya podés iniciar sesión.")

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
    practice_videos = {}
    for label in etiquetas:
        sign_id = LSA64_VIDEO_IDS.get(_normalizar_nombre_sena(label))
        if sign_id is not None:
            practice_videos[str(label)] = url_for("static", filename=f"videos/lsa64/{sign_id:03}.mp4")
    return render_template(
        "dashboard.html",
        username=session.get("username"),
        labels=[str(label) for label in etiquetas],
        practice_videos=practice_videos,
    )

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
    return jsonify(palabras=state["palabras"], oracion_mejorada=state["oracion_mejorada"], estado_ia=state["estado_ia"], error_ia=state["error_ia"], camera_error=state["camera_error"], manos=state["manos_detectadas"], confianza=state["confianza_actual"], ultima_signo=state["ultima_deteccion"], oracion_raw="".join(state["palabras"]) or "...", modelo=model is not None)

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


@app.get("/api/practica/logros")
def get_practice_achievements():
    email = session.get("user_email")
    if not email:
        return jsonify(success=False, message="Sesión expirada"), 401
    return jsonify(_progreso_practica(email))


@app.post("/api/practica/acierto")
def record_practice_success():
    email = session.get("user_email")
    if not email:
        return jsonify(success=False, message="Sesión expirada"), 401

    data = request.get_json(silent=True) or {}
    raw_label = data.get("sena")
    if not isinstance(raw_label, str):
        return jsonify(success=False, message="Seña inválida"), 400
    label = next((str(item) for item in etiquetas if _normalizar_nombre_sena(item) == _normalizar_nombre_sena(raw_label)), None)
    if not label:
        return jsonify(success=False, message="Seña inválida"), 400

    hoy = datetime.now(ARGENTINA_TZ).date()
    registro = PracticeSign.query.filter_by(email=email, practice_date=hoy, sign_label=label).first()
    progreso_antes = _progreso_practica(email)
    if not registro:
        try:
            db.session.add(PracticeSign(email=email, practice_date=hoy, sign_label=label))
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
        except SQLAlchemyError:
            db.session.rollback()
            app.logger.exception("No se pudo guardar el progreso de práctica para %s", email)
            return jsonify(success=False, message="No se pudo guardar el progreso"), 500

    progreso = _progreso_practica(email)
    ya_desbloqueados = {item["id"] for item in progreso_antes["achievements"] if item["unlocked"]}
    progreso["newly_unlocked"] = [
        item for item in progreso["achievements"]
        if item["unlocked"] and item["id"] not in ya_desbloqueados
    ]
    return jsonify(progreso)

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
