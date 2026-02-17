"""
DPS Voice Manager - Main Application
======================================
Servicio de gestión de voces (Voice ID) con diarización e identificación.

Puerto: 3010

Endpoints principales documentados en cada router:
- /api/health              → Health check
- /api/register            → Registrar voz
- /api/persons/*           → CRUD de personas
- /api/search              → Búsqueda
- /api/diarize             → Diarización + identificación
- /api/identify-speaker    → Identificar speaker
- /ws/progress/{id}        → WebSocket para progreso
"""
import logging
import logging.config
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from config import (
    SERVICE_NAME,
    SERVICE_PORT,
    HOST,
    LOGGING_CONFIG,
    STATIC_DIR
)
from services.voice_embedding_service import get_voice_embedding_service
from services.elasticsearch_service import get_elasticsearch_voice_service
from services.diarization_service import get_diarization_service
from services.websocket_service import websocket_progress_endpoint

# Importar routers
from api.routers import health, persons, search, voice_registration, diarization

# Configurar logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


# ============================================================================
# SERVICIOS GLOBALES
# ============================================================================

es_service = None
embedding_service = None
diarization_service = None


# ============================================================================
# LIFECYCLE
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestión del ciclo de vida"""
    global es_service, embedding_service, diarization_service
    
    logger.info(f"🚀 Iniciando {SERVICE_NAME} en puerto {SERVICE_PORT}")
    
    try:
        # 1. Elasticsearch
        es_service = get_elasticsearch_voice_service()
        es_ok = await es_service.initialize()
        if not es_ok:
            logger.error("❌ No se pudo conectar a Elasticsearch")
        
        # 2. Voice Embedding (ECAPA-TDNN)
        embedding_service = get_voice_embedding_service()
        emb_ok = await embedding_service.initialize()
        if not emb_ok:
            logger.warning("⚠️ Voice Embedding en modo degradado")
        
        # 3. Diarization (Pyannote)
        diarization_service = get_diarization_service()
        diar_ok = await diarization_service.initialize()
        if not diar_ok:
            logger.warning("⚠️ Diarización no disponible (HF_TOKEN requerido)")
        
        # Configurar servicios en routers
        health.set_services(es_service, embedding_service, diarization_service)
        persons.set_es_service(es_service)
        search.set_es_service(es_service)
        voice_registration.set_services(es_service, embedding_service, diarization_service)
        diarization.set_services(diarization_service, es_service, embedding_service)
        
        logger.info("✅ Voice Manager listo")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Error en startup: {e}")
        raise
    finally:
        logger.info("🛑 Cerrando servicio...")
        if es_service:
            await es_service.close()
        logger.info("✅ Servicio cerrado")


# ============================================================================
# APLICACIÓN FASTAPI
# ============================================================================

app = FastAPI(
    title="DPS Voice Manager",
    description="Gestión de voces para diarización e identificación de speakers",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar archivos estáticos
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ============================================================================
# REGISTRAR ROUTERS
# ============================================================================

app.include_router(health.router)
app.include_router(persons.router)
app.include_router(search.router)
app.include_router(voice_registration.router)
app.include_router(diarization.router)


# ============================================================================
# WEBSOCKET
# ============================================================================

@app.websocket("/ws/progress/{registration_id}")
async def websocket_progress(websocket, registration_id: str):
    """WebSocket para progreso de registro en tiempo real"""
    await websocket_progress_endpoint(websocket, registration_id)


# ============================================================================
# UI WEB
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Servir la interfaz web"""
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return html_path.read_text()
    return "<h1>DPS Voice Manager</h1><p>Interfaz no disponible. Coloca index.html en /static</p>"


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=HOST,
        port=SERVICE_PORT,
        reload=True,
        reload_excludes=["logs", "temp_uploads", "__pycache__", ".git"],
        log_level="info"
    )
