"""
DPS Voice Manager - Configuración
===================================
Servicio de gestión de voces (Voice ID) con diarización e identificación de speakers.
Puerto: 3010

Funcionalidades:
- Banco de voces: registro de locutores conocidos con embeddings vectoriales
- Diarización: segmentación de audio por speaker (Pyannote 3.1)
- Identificación: asociar segmentos a locutores conocidos (ECAPA-TDNN)
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# CONFIGURACIÓN DEL SERVICIO
# ============================================================================

SERVICE_NAME = "DPS Voice Manager"
SERVICE_PORT = int(os.getenv("VOICE_MANAGER_PORT", "3010"))
HOST = os.getenv("HOST", "0.0.0.0")

# ============================================================================
# DIRECTORIOS
# ============================================================================

BASE_DIR = Path(__file__).parent
TEMP_DIR = BASE_DIR / "temp_uploads"
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"
LOGS_DIR = BASE_DIR / "logs"

# Crear directorios necesarios
for dir_path in [TEMP_DIR, MODELS_DIR, STATIC_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# HUGGING FACE (requerido para Pyannote)
# ============================================================================

HF_TOKEN = os.getenv("HF_TOKEN", "")

# ============================================================================
# ELASTICSEARCH LOCAL (para Voice IDs)
# ============================================================================

ES_LOCAL_HOST = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
ES_VOICE_INDEX = "voice_persons"

# ============================================================================
# CONFIGURACIÓN DE VOICE EMBEDDING (SpeechBrain ECAPA-TDNN)
# ============================================================================

# Modelo: speechbrain/spkrec-ecapa-voxceleb
# Produce embeddings de 192 dimensiones, estado del arte en speaker verification
EMBEDDING_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
EMBEDDING_DIMS = 192

# Dispositivo
DEVICE = os.getenv("VOICE_DEVICE", "cuda")  # "cuda" o "cpu"

# ============================================================================
# CONFIGURACIÓN DE DIARIZACIÓN (Pyannote 3.1)
# ============================================================================

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

# Rango de speakers esperados (None = auto-detectar)
MIN_SPEAKERS = None
MAX_SPEAKERS = None

# ============================================================================
# CONFIGURACIÓN DE REGISTRO DE VOCES
# ============================================================================

# Clips de audio para registro
MIN_AUDIO_CLIPS = 1
RECOMMENDED_AUDIO_CLIPS = 3
MAX_AUDIO_CLIPS = 10

# Duración de clips de referencia (segundos)
MIN_CLIP_DURATION = 3.0      # Al menos 3 segundos de habla
MAX_CLIP_DURATION = 120.0    # Máximo 2 minutos por clip
RECOMMENDED_CLIP_DURATION = 15.0  # Ideal: 10-30 segundos

# Porcentaje mínimo de habla en el clip (VAD)
MIN_SPEECH_RATIO = 0.3  # Al menos 30% del clip debe ser habla

# Tamaño máximo de archivo (50MB)
MAX_FILE_SIZE = 50 * 1024 * 1024

# Formatos permitidos
ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac', '.wma'}

# ============================================================================
# CONFIGURACIÓN DE IDENTIFICACIÓN
# ============================================================================

# Umbral de similitud coseno para match positivo (0.0 - 1.0)
# NOTA: Segmentos cortos de diarización producen embeddings con más varianza
# que clips largos de registro. Un threshold de 0.45 es razonable para
# identificación robusta con ECAPA-TDNN.
SIMILARITY_THRESHOLD = float(os.getenv("VOICE_SIMILARITY_THRESHOLD", "0.45"))

# Umbral mínimo para considerar un match como "probable"
PROBABLE_THRESHOLD = 0.35

# Duración mínima de segmento para intentar identificación (segundos)
MIN_SEGMENT_DURATION_FOR_ID = 2.0

# ============================================================================
# ROLES DE LOCUTORES
# ============================================================================

SPEAKER_ROLES = [
    "locutor",
    "periodista",
    "comentarista",
    "conductor",
    "corresponsal",
    "invitado",
    "analista",
    "reportero",
    "otro"
]

# ============================================================================
# LOGGING
# ============================================================================

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "json": {
            "format": '{"time": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "json",
            "filename": str(LOGS_DIR / "voice_manager.jsonl"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"]
    }
}
