# Refactorización de DPS Voice Manager

## 📋 Resumen de Cambios

El archivo `main.py` original tenía **1306 líneas** con múltiples responsabilidades mezcladas. Se ha refactorizado en una arquitectura modular y mantenible.

## 🏗️ Nueva Estructura

```
dps_voice_manager/
├── main.py                          # 170 líneas (antes: 1306)
├── config.py
├── models/
│   ├── __init__.py
│   └── schemas.py                   # Modelos Pydantic
├── api/
│   ├── __init__.py
│   └── routers/
│       ├── __init__.py
│       ├── health.py                # Health check + roles
│       ├── persons.py               # CRUD de personas
│       ├── search.py                # Búsqueda
│       ├── voice_registration.py   # Registro de voces
│       └── diarization.py           # Diarización e identificación
├── services/
│   ├── __init__.py
│   ├── elasticsearch_service.py
│   ├── voice_embedding_service.py
│   ├── diarization_service.py
│   └── websocket_service.py         # Gestión de WebSocket
├── utils/
│   ├── __init__.py
│   └── validators.py                # Validaciones de audio
└── static/
    ├── index.html                   # UI (ya existía)
    └── functions.js
```

## 📦 Módulos Creados

### 1. **models/schemas.py**
- `PersonResponse`: Respuesta con info de persona
- `SearchRequest`: Request de búsqueda
- `UpdatePersonRequest`: Request de actualización
- `ProcessingProgress`: Dataclass para progreso

### 2. **services/websocket_service.py**
- Gestión de conexiones WebSocket activas
- Función `send_progress()` para notificaciones en tiempo real

### 3. **utils/validators.py**
- `validate_audio_clips_count()`: Validar número de clips
- `validate_speaker_role()`: Validar y normalizar roles
- `validate_audio_extension()`: Validar formato de archivo
- `validate_file_size()`: Validar tamaño
- `validate_audio_duration()`: Validar duración de clips

### 4. **api/routers/**

#### health.py
- `GET /api/health`: Health check del sistema
- `GET /api/roles`: Obtener roles disponibles

#### persons.py
- `GET /api/persons`: Listar personas
- `GET /api/persons/{id}`: Obtener persona por ID
- `PUT /api/persons/{id}`: Actualizar persona
- `DELETE /api/persons/{id}`: Eliminar persona

#### search.py
- `POST /api/search`: Buscar por nombre/alias

#### voice_registration.py
- `POST /api/register`: Registrar nueva voz con clips

#### diarization.py
- `POST /api/diarize`: Diarizar audio + identificar
- `POST /api/identify-speaker`: Identificar speaker en clip corto

## ✅ Beneficios

1. **Mantenibilidad**: Código organizado por responsabilidad
2. **Testabilidad**: Módulos independientes fáciles de probar
3. **Escalabilidad**: Fácil agregar nuevos endpoints o funcionalidades
4. **Legibilidad**: main.py pasó de 1306 a 170 líneas
5. **Separación de Concerns**: Lógica de negocio separada de endpoints
6. **Reutilización**: Utilidades y validadores centralizados

## 🔄 Migración

El archivo original se respaldó como `main_old.py.bak`. La nueva versión es compatible:
- Todos los endpoints mantienen las mismas rutas
- La API es idéntica desde el punto de vista del cliente
- No se requieren cambios en el frontend

## 🚀 Ejecución

```bash
# Igual que antes
python main.py

# O con uvicorn
uvicorn main:app --host 0.0.0.0 --port 3010 --reload
```

## 📝 Notas

- **HTML embebido eliminado**: Ya existe `static/index.html` con la UI completa
- **Inyección de dependencias**: Los routers reciben referencias a servicios mediante funciones `set_services()`
- **Logging consistente**: Cada módulo usa su propio logger
- **Type hints**: Mantenidos en todos los módulos
