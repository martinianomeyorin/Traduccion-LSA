from datetime import datetime
import pytz  # <-- IMPORTANTE: agregado
# --- CAMBIO IMPORTANTE: ---
# No hacemos "db = SQLAlchemy()". 
# Importamos la "db" que ya creaste en UserModels para usar la misma conexión.
from Models.UserModels import db 

# Función para obtener hora de Argentina
def arg_now():
    return datetime.now(pytz.timezone("America/Argentina/Buenos_Aires"))

class HistorialMensaje(db.Model):
    __tablename__ = 'historial_mensajes'

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), db.ForeignKey('usuarios.email'), nullable=False)
    mensaje = db.Column(db.Text, nullable=False)
    fecha_hora = db.Column(db.DateTime, default=arg_now)  # <-- CAMBIADO

    def to_json(self):
        return {
            "id": self.id,
            "mensaje": self.mensaje,
            "fecha": self.fecha_hora.strftime('%d/%m %H:%M') 
        }

# --- FUNCIONES AUXILIARES ---

def guardar_historial(email, texto):
    try:
        nuevo_msg = HistorialMensaje(email=email, mensaje=texto)
        db.session.add(nuevo_msg)
        db.session.commit()
        return True
    except Exception as e:
        print(f"❌ Error guardando historial: {e}")
        db.session.rollback()
        return False

def borrar_mensaje(id, mail):
    try:
        msg = HistorialMensaje.query.filter_by(id=id, email=mail).first()
        if msg:
            db.session.delete(msg)
            db.session.commit()
            return True
        else:
            return False
    except Exception as e:
        print(f"❌ Error borrando mensaje: {e}")
        db.session.rollback()
        return False
