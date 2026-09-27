# Traductor de Lengua de Señas Argentina

Aplicación web que reconoce un conjunto de señas estáticas de LSA mediante los landmarks de ambas manos y un modelo TensorFlow. Permite armar una oración, solicitar una reformulación en español argentino y consultar o borrar el historial asociado a la cuenta.

> El modelo incluido reconoce las clases con las que fue entrenado; no es un traductor general ni interpreta por sí solo gramática, movimiento continuo o expresiones no representadas en el dataset.

## Requisitos

- Python 3.10 u 3.11 (TensorFlow y MediaPipe deben ser compatibles con la plataforma).
- Cámara web.
- Ollama instalado y ejecutándose para habilitar la reformulación local.
- Para producción, PostgreSQL y una clave de sesión secreta.

## Instalación

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Editá `.env` y configurá `SECRET_KEY` y, si corresponde, `DATABASE_URL`. En desarrollo, la aplicación usa SQLite en `lsa.db` si no se especifica `DATABASE_URL`; así puede iniciarse sin instalar PostgreSQL. La reformulación usa Ollama local: instalalo, iniciá el servicio y descargá el modelo con `ollama pull gemma3:4b`. El modelo predeterminado ocupa aproximadamente 3,3 GB; podés cambiarlo con `OLLAMA_MODEL`.

## Ejecución

```powershell
python app.py
```

Abrí <http://127.0.0.1:5000>. La tabla se crea al iniciar si no existe. Para seleccionar otra cámara, definí `CAMERA_INDEX` (por defecto `2`, luego prueba los índices `0` y `1`). `OLLAMA_BASE_URL`, `OLLAMA_MODEL` y `OLLAMA_TIMEOUT_SECONDS` configuran el servicio local de IA. `COOKIE_SECURE=true` debe usarse cuando la aplicación está detrás de HTTPS. `HOST` y `PORT` permiten ajustar la interfaz de escucha.

## Inicio de sesión con Google

Para habilitarlo, creá un cliente OAuth de tipo **Aplicación web** en Google Cloud Console. Configurá como URI de redireccionamiento autorizado `http://127.0.0.1:5000/auth/google/callback`, copiá el ID y el secreto del cliente en `GOOGLE_CLIENT_ID` y `GOOGLE_CLIENT_SECRET` dentro de `.env`, y reiniciá Flask. Entrá a la aplicación usando `http://127.0.0.1:5000` para que el dominio coincida con el callback. Si la pantalla de consentimiento está en modo de prueba, agregá tu cuenta Google como usuario de prueba. Para publicar la app, usá un callback HTTPS con el dominio público exacto.

El ingreso usa OpenID Connect y requiere que Google confirme que el correo está verificado. Las cuentas locales existentes con el mismo correo verificado se vinculan automáticamente; los nuevos usuarios se crean al primer ingreso. Se guarda el identificador estable de Google en la tabla `cuentas_google`.

## Restablecimiento de contraseña

La opción «Olvidé mi contraseña» envía un código de 6 dígitos por SMTP. Configurá `MAIL_SERVER`, `MAIL_PORT`, `MAIL_USERNAME`, `MAIL_PASSWORD` y `MAIL_FROM` en `.env`. Con Gmail se puede usar `smtp.gmail.com` en el puerto 587 con TLS; la cuenta que envía debe usar una contraseña de aplicación, no la contraseña habitual. [Ayuda oficial de Gmail para SMTP](https://support.google.com/a/answer/176600).

Los códigos vencen a los 10 minutos, se guardan hasheados, tienen hasta 5 intentos y las solicitudes se limitan a una por minuto para cada dirección. La respuesta de solicitud no revela si el correo está registrado. La tabla de códigos se crea automáticamente al iniciar la aplicación.

Ejemplo de configuración para PostgreSQL:

```dotenv
SECRET_KEY=una-clave-aleatoria-larga
DATABASE_URL=postgresql://usuario:clave@localhost:5432/lsa
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=gemma3:4b
OLLAMA_TIMEOUT_SECONDS=180
CAMERA_INDEX=0
COOKIE_SECURE=false
```

## Estructura

- `app.py`: servidor Flask, cámara, inferencia y API.
- `entrenar_lsa64_temporal.py`: extracción de secuencias y entrenamiento temporal opcional.
- `static/videos/lsa64/`: ejemplos locales para el modo práctica (64 videos, aproximadamente 43 MB). Ver [ATRIBUCION_LSA64.md](ATRIBUCION_LSA64.md) para la fuente y licencia.
- `Models/`: usuarios e historial SQLAlchemy.
- `templates/` y `static/`: interfaz.
- `modelo_gestos_v2.h5`, `labels_v2.pkl`: modelo y etiquetas usados en ejecución.
- `preprocesamiento.py`, `entrenamiento.py`, `deteccion.py`: herramientas del flujo de datos y entrenamiento.

Los scripts de entrenamiento existentes trabajan con señas estáticas. Para entrenar con movimiento, se puede usar el conjunto LSA64: descarga los videos desde [la página del dataset](https://facundoq.github.io/datasets/lsa64/) y conserva el ZIP original. El conjunto contiene videos de 64 señas realizadas por 10 participantes y está publicado bajo licencia CC BY-NC-SA 4.0: úsalo solo de forma no comercial, atribuye a sus autores y comparte derivados bajo la misma licencia. Para uso comercial hace falta autorización de los titulares.

Con un entorno virtual que tenga instaladas las dependencias de `requirements.txt`, ejecutá desde PowerShell:

```powershell
python entrenar_lsa64_temporal.py --archive "$HOME\Downloads\lsa64_raw.zip"
```

El preprocesamiento conserva 32 fotogramas por video y extrae los landmarks de ambas manos. Guarda un checkpoint en `dataset_lsa64_temporal.npz` cada 100 videos, por lo que se puede interrumpir y reanudar ejecutando el mismo comando. Se entrena con los participantes 1–8, se valida con el 9 y se reserva el 10 para la prueba. El modelo experimental y las métricas quedan en `modelos/lsa64/`. La aplicación en vivo sigue usando `modelo_gestos_v2.h5` y `labels_v2.pkl`; el modelo temporal no está conectado a la inferencia de la app.

La evaluación con una persona reservada mide generalización a ese participante, pero una única partición no alcanza para afirmar que el modelo funcione igual con cualquier persona, iluminación o cámara. LSA64 contiene señas aisladas, así que este modelo tampoco resuelve todavía traducción continua ni gramática completa de LSA.

## Privacidad y operación

La cámara y la reformulación se procesan localmente en el servidor. Las oraciones mejoradas se guardan en el historial de la cuenta. Cada sesión conserva su propio borrador; la cámara es un recurso local compartido y atiende una conexión a la vez. El endpoint `/health` informa el estado de la base y del modelo.

En despliegues públicos, usá HTTPS, `SECRET_KEY` aleatoria y una base PostgreSQL; ejecutá detrás de un servidor WSGI y no uses el servidor de desarrollo de Flask. Los cambios de esquema existentes requieren una migración de base de datos: `create_all` solo crea tablas que todavía no existen.
