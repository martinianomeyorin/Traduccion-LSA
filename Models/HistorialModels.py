from datetime import datetime, timezone
from zoneinfo import ZoneInfo
# --- CAMBIO IMPORTANTE: ---
# No hacemos "db = SQLAlchemy()". 
# Importamos la "db" que ya creaste en UserModels para usar la misma conexión.
from Models.UserModels import db 

# Función para obtener hora de Argentina
def arg_now():
    return datetime.now(timezone.utc).replace(tzinfo=None)

class HistorialMensaje(db.Model):
    __tablename__ = 'historial_mensajes'

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), db.ForeignKey('usuarios.email', ondelete='CASCADE'), nullable=False, index=True)
    mensaje = db.Column(db.Text, nullable=False)
    fecha_hora = db.Column(db.DateTime, default=arg_now, nullable=False, index=True)

    def to_json(self):
        fecha_local = self.fecha_hora.replace(tzinfo=timezone.utc).astimezone(ZoneInfo("America/Argentina/Buenos_Aires"))
        return {
            "id": self.id,
            "mensaje": self.mensaje,
            "fecha": fecha_local.strftime('%d/%m %H:%M')
        }

# --- FUNCIONES AUXILIARES ---

def guardar_historial(email, texto):
    try:
        nuevo_msg = HistorialMensaje(email=email, mensaje=texto)
        db.session.add(nuevo_msg)
        db.session.commit()
        return True
    except Exception:
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
    except Exception:
        db.session.rollback()
        return False
