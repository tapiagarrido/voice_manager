# DPS Voice Manager - Arquitectura Refactorizada

## 📚 Índice
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Arquitectura](#arquitectura)
- [Módulos Principales](#módulos-principales)
- [API Endpoints](#api-endpoints)
- [Ejecución](#ejecución)

## 🏗️ Estructura del Proyecto

```
dps_voice_manager/
├── main.py                          # Entrada principal (178 líneas)
├── config.py                        # Configuración centralizada
├── requirements.txt
├── README.md
├── REFACTORING.md                   # Detalles de refactorización
├── REFACTORING_SUMMARY.md           # Resumen ejecutivo
│
├── api/                             # API REST
│   └── routers/                     # Routers organizados por dominio
│       ├── health.py                # Health check + roles
│       ├── persons.py               # CRUD de personas
│       ├── search.py                # Búsqueda
│       ├── voice_registration.py   # Registro de voces
│       └── diarization.py           # Diarización e identificación
│
├── models/                          # Modelos de datos
│   ├── schemas.py                   # Pydantic models
│   └── ecapa_tdnn/                  # Modelos ML preentrenados
│
├── services/                        # Capa de servicios
│   ├── elasticsearch_service.py    # Interacción con ES
│   ├── voice_embedding_service.py  # Extracción de embeddings
│   ├── diarization_service.py      # Diarización con Pyannote
│   └── websocket_service.py        # WebSocket en tiempo real
│
├── utils/                           # Utilidades
│   └── validators.py                # Validaciones de audio
│
├── static/                          # Frontend
│   ├── index.html                   # UI web
│   └── functions.js
│
├── logs/                            # Logs rotativos
├── temp_uploads/                    # Archivos temporales
└── registro/                        # Embeddings persistidos
```

## 🎯 Arquitectura

### Capas

1. **Capa de Presentación** (`api/routers/`)
   - Manejo de requests/responses HTTP
   - Validación de entrada
   - Documentación automática con FastAPI

2. **Capa de Lógica de Negocio** (`services/`)
   - Elasticsearch: Almacenamiento de embeddings
   - Voice Embedding: Extracción con ECAPA-TDNN
   - Diarization: Segmentación con Pyannote
   - WebSocket: Notificaciones en tiempo real

3. **Capa de Datos** (`models/`)
   - Schemas Pydantic para validación
   - Modelos ML preentrenados

4. **Capa de Utilidades** (`utils/`)
   - Validadores reutilizables
   - Helpers comunes

### Flujo de Datos

```
Cliente (UI/API) 
    ↓
FastAPI Routers (validación entrada)
    ↓
Servicios (lógica de negocio)
    ↓
Elasticsearch / ML Models
    ↓
Respuesta al cliente
```

## 📦 Módulos Principales

### `api/routers/health.py`
- **GET** `/api/health` - Estado del sistema
- **GET** `/api/roles` - Roles disponibles

### `api/routers/persons.py`
- **GET** `/api/persons` - Listar personas
- **GET** `/api/persons/{id}` - Obtener persona
- **PUT** `/api/persons/{id}` - Actualizar persona
- **DELETE** `/api/persons/{id}` - Eliminar persona

### `api/routers/search.py`
- **POST** `/api/search` - Buscar por nombre/alias

### `api/routers/voice_registration.py`
- **POST** `/api/register` - Registrar voz con clips

### `api/routers/diarization.py`
- **POST** `/api/diarize` - Diarizar + identificar
- **POST** `/api/identify-speaker` - Identificar speaker

### `services/websocket_service.py`
- **WS** `/ws/progress/{id}` - Progreso de registro

### `models/schemas.py`
```python
class PersonResponse(BaseModel):
    person_id: str
    name: str
    aliases: List[str]
    role: str
    sample_count: int
    avg_confidence: float
    total_speech_duration: float
    created_at: str

class SearchRequest(BaseModel):
    query: str
    limit: int = 10

class UpdatePersonRequest(BaseModel):
    name: Optional[str]
    aliases: Optional[List[str]]
    role: Optional[str]

@dataclass
class ProcessingProgress:
    current: int
    total: int
    stage: str
    message: str
    percentage: float
```

### `utils/validators.py`
```python
def validate_audio_clips_count(audio_files)
def validate_speaker_role(role) -> str
def validate_audio_extension(filename) -> str
def validate_file_size(content, filename)
def validate_audio_duration(duration) -> tuple[bool, str]
```

## 🚀 Ejecución

### Desarrollo
```bash
python main.py
```

### Producción
```bash
uvicorn main:app --host 0.0.0.0 --port 3010 --workers 4
```

### Docker
```bash
docker build -t dps-voice-manager .
docker run -p 3010:3010 dps-voice-manager
```

## 🔧 Configuración

Variables de entorno en `config.py`:

- `SERVICE_PORT` - Puerto del servicio (default: 3010)
- `ES_HOST` - Host de Elasticsearch
- `HF_TOKEN` - Token de Hugging Face (para Pyannote)
- `MIN_AUDIO_CLIPS` - Mínimo clips para registro
- `SIMILARITY_THRESHOLD` - Umbral de similitud

## 📖 Documentación API

Accede a la documentación interactiva:
- Swagger UI: `http://localhost:3010/docs`
- ReDoc: `http://localhost:3010/redoc`

## 🧪 Testing

```bash
pytest tests/
```

## 📊 Métricas

El servicio expone métricas en el health check:
```json
{
  "status": "healthy",
  "service": "DPS Voice Manager",
  "port": 3010,
  "subsystems": {
    "elasticsearch": true,
    "voice_embedding": true,
    "diarization": true
  },
  "voice_bank": {
    "total_persons": 42
  }
}
```

## 🤝 Contribuir

1. Cada router debe tener su dominio claramente definido
2. Los servicios no deben importar routers
3. Validaciones en `utils/validators.py`
4. Type hints obligatorios
5. Docstrings en funciones públicas

## 📝 Changelog

Ver `REFACTORING.md` para detalles completos de cambios.

## 📄 Licencia

[Especificar licencia]
