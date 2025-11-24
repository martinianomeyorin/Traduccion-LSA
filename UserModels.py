from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

# Inicializamos la extensión de base de datos
db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'usuarios'

    email = db.Column(db.String(255), primary_key=True)  
    nombre = db.Column(db.String(100), nullable=False)
    contraseña = db.Column(db.String(255), nullable=False)
    fecha_alta = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        """Encripta la contraseña y la guarda en la columna 'contraseña'"""
        self.contraseña = generate_password_hash(password)

    def check_password(self, password):
        """Verifica si la contraseña es correcta comparando con 'contraseña'"""
        return check_password_hash(self.contraseña, password)

    def to_json(self):
        return {
            "email": self.email,
            "nombre": self.nombre,
            "fecha_alta": self.fecha_alta.strftime('%Y-%m-%d %H:%M:%S')
        }

# --- MÉTODOS AUXILIARES ---

def crear_usuario(nombre, email, password):
    # 1. Verificar si el email ya existe (PK)
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
    except Exception as e:
        db.session.rollback()
        return False, f"Error de base de datos: {str(e)}"

def autenticar_usuario(identificador, password):
    """
    Autenticación híbrida: Busca por Email O por Nombre.
    'identificador' puede ser cualquiera de los dos.
    """
    usuario = User.query.filter((User.email == identificador) | (User.nombre == identificador)).first()
    
    if usuario and usuario.check_password(password):
        return True, usuario
    else:
        return False, None