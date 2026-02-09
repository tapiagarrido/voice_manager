# DPS Voice Manager# DPS Voice Manager



**Puerto:** 3010 · **Python:** 3.9+ · **GPU:** CUDA 12.1 recomendado**Puerto:** 3010



Servicio de gestión de voces para diarización e identificación de speakers. Registra locutores conocidos en un banco de voces (embeddings vectoriales 192d) y los identifica automáticamente cuando aparecen en nuevos audios.Servicio de gestión de voces para diarización e identificación de speakers. Permite registrar locutores conocidos en un banco de voces (embeddings vectoriales) y luego identificarlos automáticamente cuando aparecen en nuevos audios.



---## Arquitectura



## Arquitectura```

Audio → Pyannote 3.1 (Diarización) → Segmentos por speaker (SPEAKER_00, SPEAKER_01...)

```                                            ↓

┌──────────────────────────────────────────────────────────────────────┐                                    ECAPA-TDNN (Embedding 192d)

│                        DPS Voice Manager (:3010)                     │                                            ↓

│                                                                      │                                    Elasticsearch (cosine similarity)

│  ┌────────────┐    ┌──────────────────┐    ┌─────────────────────┐  │                                            ↓

│  │  FastAPI    │───▶│  Diarization Svc  │───▶│  Voice Embedding    │  │                                    "Juan Pérez" (sim=0.87) ✅

│  │  + WebUI    │    │  (Pyannote 3.1)   │    │  Service            │  │```

│  └─────┬──────┘    └────────┬─────────┘    │  (ECAPA-TDNN 192d)  │  │

│        │                    │               └──────────┬──────────┘  │### Modelos

│        │                    ▼                          │             │

│        │           Segmentos por speaker               │             │| Modelo | Uso | Dims | VRAM |

│        │           (SPEAKER_00, 01...)                 ▼             │|--------|-----|------|------|

│        │                    │               Embedding por speaker    │| `pyannote/speaker-diarization-3.1` | Segmentación por speaker | - | ~1GB |

│        │                    ▼                          │             │| `speechbrain/spkrec-ecapa-voxceleb` | Embeddings de voz | 192 | ~0.5GB |

│        │           ┌──────────────────┐               │             │

│        └──────────▶│  Elasticsearch    │◀──────────────┘             │### Flujo

│                    │  Service          │                             │

│                    │  (cosine search)  │                             │1. **Registro**: Subir clips de audio de un locutor conocido → se extrae embedding promedio → se guarda en Elasticsearch con nombre, rol, aliases

│                    └──────────────────┘                              │2. **Diarización**: Enviar audio largo → Pyannote segmenta por speakers → para cada speaker se extrae embedding → se compara contra banco → se etiqueta con nombre real si match > threshold

└──────────────────────────────────────────────────────────────────────┘3. **Integración**: `dps_audio_processor` envía audio vía HTTP → recibe segmentos con speakers identificados → los alinea con los segmentos de Whisper → cada línea de transcripción tiene `speaker: "Nombre"`

                              │

                              ▼## Setup

                     Elasticsearch 8.x

                     Índice: voice_persons### Requisitos

                     dense_vector (192d, cosine)

```- Python 3.10+

- CUDA (recomendado, funciona en CPU pero lento)

### Flujo de Registro- `ffmpeg` instalado

- Token de HuggingFace (`HF_TOKEN`) con acceso a:

```  - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

Clips de audio ──▶ Pyannote (diarización) ──▶ Aislar speaker dominante  - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

                                                        │

                                    Concatenar waveforms del speaker ◀──┘### Instalación

                                                        │

                                              ECAPA-TDNN (embedding 192d)```bash

                                                        │# Configurar HF_TOKEN

                                              Elasticsearch (guardar)echo "HF_TOKEN=hf_xxxxxxxxxxxxx" > dps_voice_manager/.env

```

# Ejecutar setup

1. Se reciben 1-10 clips de audio de un locutorcd dps_voice_manager

2. Cada clip se diariza para aislar al **speaker dominante** (evita contaminación por otros hablantes)chmod +x setup.sh

3. Se concatenan los waveforms del speaker dominante de todos los clips./setup.sh

4. Se extrae **un solo embedding** del audio concatenado → alta calidad```

5. Se guarda en Elasticsearch con nombre, rol y aliases

### Iniciar

### Flujo de Identificación

```bash

```source ../venvs/voice_manager/bin/activate

Audio largo ──▶ Pyannote (diarización) ──▶ Segmentos por speakercd dps_voice_manager

                                                    │python main.py

                              Para cada speaker: ◀──┘```

                              Concatenar waveforms (≥0.5s c/u)

                                                    │El servicio estará en `http://localhost:3010`

                                          ECAPA-TDNN (embedding)

                                                    │## API Endpoints

                                          Elasticsearch (cosine search)

                                                    │### Voice Bank (CRUD)

                                    sim ≥ 0.45 → "Tomás Mosciatti" ✅

                                    sim < 0.35 → "SPEAKER_02" (desconocido)| Método | Ruta | Descripción |

```|--------|------|-------------|

| `GET` | `/api/health` | Health check con estado de subsistemas |

1. Pyannote segmenta el audio por speaker (labels anónimos)| `GET` | `/api/persons` | Listar voces registradas |

2. Para cada speaker se **concatenan los waveforms** de todos sus segmentos (≥0.5s cada uno)| `GET` | `/api/persons/{id}` | Obtener detalle de persona |

3. Se extrae un embedding del audio concatenado| `PUT` | `/api/persons/{id}` | Actualizar nombre/aliases/rol |

4. Se busca en Elasticsearch por similitud coseno| `DELETE` | `/api/persons/{id}` | Eliminar persona |

5. Si supera el umbral → se reemplaza el label anónimo por el nombre real| `POST` | `/api/search` | Buscar por nombre/alias (fuzzy) |

| `POST` | `/api/register` | Registrar voz con clips de audio |

---| `GET` | `/api/roles` | Roles disponibles |



## Estructura del Proyecto### Diarización



```| Método | Ruta | Descripción |

dps_voice_manager/|--------|------|-------------|

├── config.py                      # Configuración centralizada| `POST` | `/api/diarize` | Diarizar audio + identificar speakers |

├── main.py                        # FastAPI app + endpoints| `POST` | `/api/identify-speaker` | Identificar speaker en clip corto |

├── requirements.txt               # Dependencias con versiones fijas

├── setup.sh                       # Script de instalación automática### UI Web

├── .env                           # Variables de entorno (HF_TOKEN, etc.)

├── services/| Ruta | Descripción |

│   ├── __init__.py|------|-------------|

│   ├── diarization_service.py     # Diarización + identificación (Pyannote)| `GET /` | Interfaz web completa (registro, lista, test) |

│   ├── voice_embedding_service.py # Embeddings de voz (ECAPA-TDNN)| `WS /ws/progress/{id}` | WebSocket para progreso de registro |

│   └── elasticsearch_service.py   # CRUD banco de voces (ES 8.x)

├── models/## Registro de voz

│   └── ecapa_tdnn/                # Modelo descargado (auto en primer uso)

│       ├── embedding_model.ckpt```bash

│       ├── classifier.ckpt# Registrar con 3 clips de audio

│       ├── hyperparams.yamlcurl -X POST http://localhost:3010/api/register \

│       ├── label_encoder.ckpt  -F "name=Juan Pérez" \

│       └── mean_var_norm_emb.ckpt  -F "role=periodista" \

├── static/  -F "aliases=Juanito,JP" \

│   ├── index.html                 # UI web para gestión  -F "audio_files=@clip1.wav" \

│   └── functions.js               # Lógica frontend  -F "audio_files=@clip2.wav" \

├── temp_uploads/                  # Archivos temporales (auto-limpieza)  -F "audio_files=@clip3.wav"

├── logs/                          # Logs en formato JSONL (rotación 10MB)```

└── registro/                      # Clips de audio para registro manual

```**Recomendaciones para clips:**

- Duración: 10-30 segundos cada uno

---- Solo la persona hablando (sin otros speakers)

- Audio limpio, sin mucha música de fondo

## Librerías y Dependencias- 3-5 clips de diferentes contextos dan mejor resultado



### Core — Deep Learning## Diarización



| Librería | Versión | Descripción |```bash

|----------|---------|-------------|# Diarizar y identificar

| **torch** | 2.5.1+cu121 | Framework de deep learning (PyTorch). Requerido por Pyannote y SpeechBrain. Usar build CUDA 12.1. |curl -X POST http://localhost:3010/api/diarize \

| **torchaudio** | 2.5.1+cu121 | Procesamiento de audio con PyTorch. Carga, resampleo y manipulación de waveforms. |  -F "audio=@noticiario.wav" \

  -F "identify=true" \

> ⚠️ **CUDA:** Instalar PyTorch con soporte CUDA desde `https://download.pytorch.org/whl/cu121`. Sin GPU el servicio funciona pero es significativamente más lento.  -F "threshold=0.70"

```

### Diarización — Pyannote

Respuesta:

| Librería | Versión | Descripción |```json

|----------|---------|-------------|{

| **pyannote.audio** | 3.4.0 | Pipeline de speaker diarization. Segmenta audio por hablante usando modelos neuronales. Requiere token de HuggingFace. |  "segments": [

    {"start": 0.5, "end": 12.3, "speaker": "Juan Pérez", "confidence": 0.87, "role": "periodista"},

> ⚠️ **HuggingFace Token:** Se necesita `HF_TOKEN` con acceso aceptado en:    {"start": 12.5, "end": 25.1, "speaker": "SPEAKER_01", "confidence": 0},

> - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)    {"start": 25.3, "end": 45.0, "speaker": "Juan Pérez", "confidence": 0.87, "role": "periodista"}

> - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)  ],

>  "speakers": {

> Aceptar las condiciones de uso en cada página antes de usar el servicio.    "Juan Pérez": {"person_id": "abc12345", "similarity": 0.87, "total_time": 56.8, "identified": true},

    "SPEAKER_01": {"total_time": 12.6, "identified": false}

### Embeddings de Voz — SpeechBrain  },

  "total_speakers": 2,

| Librería | Versión | Descripción |  "identified_speakers": 1,

|----------|---------|-------------|  "unidentified_speakers": 1

| **speechbrain** | 1.0.3 | Toolkit de speech processing. Se usa el modelo ECAPA-TDNN para extraer embeddings de 192 dimensiones que representan la identidad vocal del hablante. |}

```

> El modelo `speechbrain/spkrec-ecapa-voxceleb` se descarga automáticamente en `models/ecapa_tdnn/` en el primer uso.

## Integración con Audio Processor

### Audio

El `dps_audio_processor` (puerto 3009) se comunica automáticamente con este servicio. Cuando el voice manager está corriendo:

| Librería | Versión | Descripción |

|----------|---------|-------------|1. Audio Processor recibe un audio para analizar

| **soundfile** | 0.13.1 | Lectura/escritura de archivos de audio (WAV, FLAC, OGG). Backend de torchaudio. |2. Whisper transcribe → segmentos con texto y timestamps

| **librosa** | 0.11.0 | Análisis de audio: duración, resampleo, detección de actividad vocal (VAD). |3. **Nuevo:** El audio se envía a Voice Manager vía HTTP

4. Voice Manager diariza e identifica speakers

### Web Framework5. Audio Processor alinea temporalmente los speakers con la transcripción

6. Cada segmento de transcripción tiene campo `speaker` con el nombre

| Librería | Versión | Descripción |

|----------|---------|-------------|Si Voice Manager no está corriendo, el pipeline sigue funcionando sin diarización (graceful fallback).

| **fastapi** | 0.128.5 | Framework web async para la API REST y WebSocket. |

| **uvicorn** | 0.39.0 | Servidor ASGI para FastAPI. |## Elasticsearch

| **python-multipart** | 0.0.20 | Parsing de formularios multipart (upload de archivos de audio). |

| **websockets** | 15.0.1 | Soporte WebSocket para notificaciones de progreso en tiempo real. |**Índice:** `voice_persons`



### HTTP / IO| Campo | Tipo | Descripción |

|-------|------|-------------|

| Librería | Versión | Descripción || `person_id` | keyword | UUID corto |

|----------|---------|-------------|| `name` | text + keyword | Nombre completo |

| **httpx** | 0.28.1 | Cliente HTTP sync/async. Usado para comunicación directa con Elasticsearch. || `aliases` | text + keyword | Nombres alternativos |

| **aiofiles** | 25.1.0 | IO asíncrono de archivos para escritura de uploads sin bloquear el event loop. || `role` | keyword | Rol (locutor, periodista, etc.) |

| `embedding` | dense_vector(192) | Vector de voz ECAPA-TDNN |

### Utilidades| `sample_count` | integer | Clips usados |

| `total_speech_duration` | float | Segundos totales de habla |

| Librería | Versión | Descripción |

|----------|---------|-------------|## Configuración

| **numpy** | 2.0.2 | Operaciones numéricas sobre embeddings (normalización, promedios). |

| **pydantic** | 2.12.5 | Validación de datos en modelos de request/response de FastAPI. |Variables de entorno (`.env`):

| **python-dotenv** | 1.2.1 | Carga variables de entorno desde archivo `.env`. |

```env

### Dependencias del SistemaHF_TOKEN=hf_xxxxxxxxxxxxx          # Requerido para Pyannote

VOICE_MANAGER_PORT=3010             # Puerto del servicio

| Herramienta | Descripción |ELASTICSEARCH_HOST=http://localhost:9200

|-------------|-------------|VOICE_DEVICE=cuda                   # cuda o cpu

| **ffmpeg** | Decodificación de audio (MP3, AAC, etc.). Requerido por torchaudio/librosa. |VOICE_SIMILARITY_THRESHOLD=0.70     # Umbral de match (0.0-1.0)

| **Elasticsearch 8.x** | Motor de búsqueda vectorial para el banco de voces. |```

| **CUDA Toolkit 12.1** | Aceleración GPU (opcional pero muy recomendado). |

---

## Setup

### Requisitos Previos

- Python 3.9+
- NVIDIA GPU con CUDA 12.1 (recomendado)
- `ffmpeg` instalado (`sudo apt install ffmpeg` / `sudo dnf install ffmpeg`)
- Elasticsearch 8.x corriendo en `localhost:9200`
- Token de HuggingFace con acceso a Pyannote

### Instalación Rápida

```bash
# 1. Configurar variables de entorno
cat > .env << EOF
HF_TOKEN=hf_xxxxxxxxxxxxx
ELASTICSEARCH_HOST=http://localhost:9200
VOICE_DEVICE=cuda
EOF

# 2. Ejecutar setup automático
chmod +x setup.sh
./setup.sh

# 3. Iniciar el servicio
source venv/bin/activate
python main.py
```

### Instalación Manual

```bash
# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar PyTorch con CUDA 12.1
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Instalar el resto de dependencias
pip install -r requirements.txt
```

El servicio estará disponible en `http://localhost:3010`

---

## API Endpoints

### Voice Bank (CRUD)

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/api/health` | Health check con estado de subsistemas |
| `GET` | `/api/persons` | Listar voces registradas |
| `GET` | `/api/persons/{id}` | Detalle de persona |
| `PUT` | `/api/persons/{id}` | Actualizar nombre/aliases/rol |
| `DELETE` | `/api/persons/{id}` | Eliminar persona del banco |
| `POST` | `/api/search` | Buscar por nombre/alias (fuzzy) |
| `POST` | `/api/register` | Registrar voz con clips de audio |
| `GET` | `/api/roles` | Roles disponibles |

### Diarización

| Método | Ruta | Descripción |
|--------|------|-------------|
| `POST` | `/api/diarize` | Diarizar audio + identificar speakers |
| `POST` | `/api/identify-speaker` | Identificar speaker en clip corto |

### UI Web

| Ruta | Descripción |
|------|-------------|
| `GET /` | Interfaz web (registro, listado, test) |
| `WS /ws/progress/{id}` | WebSocket para progreso de registro |

---

## Registro de Voz

```bash
curl -X POST http://localhost:3010/api/register \
  -F "name=Juan Pérez" \
  -F "role=periodista" \
  -F "aliases=Juanito,JP" \
  -F "audio_files=@clip1.wav" \
  -F "audio_files=@clip2.wav" \
  -F "audio_files=@clip3.wav"
```

**Recomendaciones para clips:**
- Duración: 10-60 segundos cada uno
- Idealmente solo la persona hablando (el sistema aísla al speaker dominante automáticamente)
- Audio limpio, sin mucha música de fondo
- 3-5 clips de diferentes contextos dan mejor resultado

## Diarización e Identificación

```bash
curl -X POST http://localhost:3010/api/diarize \
  -F "audio=@noticiario.wav" \
  -F "identify=true"
```

Respuesta:
```json
{
  "segments": [
    {"start": 0.5, "end": 12.3, "speaker": "Juan Pérez", "confidence": 0.87, "role": "periodista"},
    {"start": 12.5, "end": 25.1, "speaker": "SPEAKER_01", "confidence": 0},
    {"start": 25.3, "end": 45.0, "speaker": "Juan Pérez", "confidence": 0.87, "role": "periodista"}
  ],
  "speakers": {
    "Juan Pérez": {"person_id": "abc123", "similarity": 0.87, "total_time": 56.8, "identified": true},
    "SPEAKER_01": {"total_time": 12.6, "identified": false}
  },
  "total_speakers": 2,
  "identified_speakers": 1,
  "unidentified_speakers": 1
}
```

---

## Configuración

Variables de entorno en `.env`:

| Variable | Default | Descripción |
|----------|---------|-------------|
| `HF_TOKEN` | *(requerido)* | Token de HuggingFace para Pyannote |
| `VOICE_MANAGER_PORT` | `3010` | Puerto del servicio |
| `ELASTICSEARCH_HOST` | `http://localhost:9200` | URL de Elasticsearch |
| `VOICE_DEVICE` | `cuda` | Dispositivo: `cuda` o `cpu` |
| `VOICE_SIMILARITY_THRESHOLD` | `0.45` | Umbral de match (cosine similarity) |

### Umbrales de Identificación

| Umbral | Valor | Significado |
|--------|-------|-------------|
| `SIMILARITY_THRESHOLD` | 0.45 | Match positivo → se asigna nombre |
| `PROBABLE_THRESHOLD` | 0.35 | Match probable → se registra pero no se asigna |
| `MIN_SEGMENT_DURATION_FOR_ID` | 2.0s | Duración mínima total para intentar identificar un speaker |

> Los umbrales son más bajos que lo típico (0.70) porque los segmentos de diarización son cortos y producen embeddings con más varianza que clips largos de registro. La estrategia de concatenación de waveforms compensa parcialmente esto.

---

## Elasticsearch

**Índice:** `voice_persons` (se crea automáticamente)

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `person_id` | keyword | UUID corto (8 chars) |
| `name` | text + keyword | Nombre completo |
| `aliases` | text + keyword | Nombres alternativos |
| `role` | keyword | Rol (locutor, periodista, etc.) |
| `embedding` | dense_vector(192) | Vector de voz normalizado |
| `sample_count` | integer | Clips usados en registro |
| `total_speech_duration` | float | Segundos totales de habla procesados |
| `created_at` | date | Fecha de registro |
| `updated_at` | date | Última actualización |

---

## Integración con Audio Processor

El `dps_audio_processor` (puerto 3009) se comunica automáticamente:

1. Audio Processor recibe audio para analizar
2. Whisper transcribe → segmentos con texto y timestamps
3. Audio se envía a Voice Manager vía HTTP
4. Voice Manager diariza e identifica speakers
5. Audio Processor alinea speakers con la transcripción
6. Cada línea queda con `speaker: "Nombre Real"`

Si Voice Manager no está corriendo, el pipeline funciona sin diarización (graceful fallback).

---

## VRAM Estimada

| Componente | VRAM |
|------------|------|
| Pyannote speaker-diarization-3.1 | ~1.0 GB |
| ECAPA-TDNN (SpeechBrain) | ~0.5 GB |
| **Total** | **~1.5 GB** |
