from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from sqlalchemy import UniqueConstraint

# Inicializamos la extensión de base de datos
db = SQLAlchemy()

# Función para obtener la hora local de Argentina
def arg_now():
    # Guardamos UTC para evitar fechas ambiguas al cambiar de zona horaria.
    return datetime.now(timezone.utc).replace(tzinfo=None)

class User(db.Model):
    __tablename__ = 'usuarios'

    email = db.Column(db.String(255), primary_key=True)  
    nombre = db.Column(db.String(100), nullable=False)
    contraseña = db.Column(db.String(255), nullable=False)
    fecha_alta = db.Column(db.DateTime, default=arg_now, nullable=False)

    def set_password(self, password):
        """Encripta la contraseña y la guarda en la columna 'contraseña'"""
        self.contraseña = generate_password_hash(password)

    def check_password(self, password):
        """Verifica si la contraseña es correcta comparando con 'contraseña'"""
        return check_password_hash(self.contraseña, password)

    def to_json(self):
        fecha_local = self.fecha_alta.replace(tzinfo=timezone.utc).astimezone(ZoneInfo("America/Argentina/Buenos_Aires"))
        return {
            "email": self.email,
            "nombre": self.nombre,
            "fecha_alta": fecha_local.strftime('%Y-%m-%d %H:%M:%S')
        }


class GoogleAccount(db.Model):
    """Vincula un usuario local con el identificador estable de Google (sub)."""
    __tablename__ = "cuentas_google"

    google_sub = db.Column(db.String(255), primary_key=True)
    email = db.Column(db.String(255), db.ForeignKey("usuarios.email", ondelete="CASCADE"), nullable=False, index=True)


class PasswordResetCode(db.Model):
    __tablename__ = "codigos_restablecimiento"

    email = db.Column(db.String(255), db.ForeignKey("usuarios.email", ondelete="CASCADE"), primary_key=True)
    code_hash = db.Column(db.String(64), nullable=False)
    expires_at = db.Column(db.DateTime, nullable=False)
    last_sent_at = db.Column(db.DateTime, nullable=False)
    attempts = db.Column(db.Integer, nullable=False, default=0)


class PracticeSign(db.Model):
    """Una seña distinta practicada por usuario en una fecha local."""
    __tablename__ = "senas_practicadas"
    __table_args__ = (
        UniqueConstraint("email", "practice_date", "sign_label", name="uq_practice_sign_per_day"),
    )

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), db.ForeignKey("usuarios.email", ondelete="CASCADE"), nullable=False, index=True)
    practice_date = db.Column(db.Date, nullable=False, index=True)
    sign_label = db.Column(db.String(100), nullable=False)

# --- MÉTODOS AUXILIARES ---

def validar_password(password):
    if not isinstance(password, str):
        return "La contraseña no es válida"
    if len(password) < 8:
        return "La contraseña debe tener al menos 8 caracteres"
    if len(password) > 128:
        return "La contraseña no puede superar los 128 caracteres"
    if not any(caracter.isupper() for caracter in password):
        return "La contraseña debe incluir al menos una mayúscula"
    if not any(not caracter.isalnum() and not caracter.isspace() for caracter in password):
        return "La contraseña debe incluir al menos un carácter especial (por ejemplo: ! o #)"
    return None

def crear_usuario(nombre, email, password):
    nombre, email = nombre.strip(), email.strip().lower()
    error_password = validar_password(password)
    if not nombre or not email or len(nombre) > 100 or len(email) > 255 or error_password:
        return False, error_password or "Revisá los datos ingresados"
    # La restricción única de la base de datos sigue siendo la garantía final
    # ante registros concurrentes.
    if User.query.filter_by(email=email).first():
        return False, "El email ya está registrado"
    if User.query.filter_by(nombre=nombre).first():
        return False, "El nombre de usuario ya está en uso"

    try:
        nuevo_usuario = User(nombre=nombre, email=email)
        nuevo_usuario.set_password(password)
        
        db.session.add(nuevo_usuario)
        db.session.commit()
        return True, "Usuario creado exitosamente"
    except Exception:
        db.session.rollback()
        return False, "No se pudo crear la cuenta. Revisá si el correo o el nombre ya están registrados"

def autenticar_usuario(identificador, password):
    """
    Autenticación híbrida: Busca por Email O por Nombre.
    'identificador' puede ser cualquiera de los dos.
    """
    identificador = identificador.strip()
    usuario = User.query.filter((User.email == identificador.lower()) | (User.nombre == identificador)).first()
    
    if usuario and usuario.check_password(password):
        return True, usuario
    else:
        return False, None
