"""
DPS Voice Manager - Main Application
======================================
Servicio de gestión de voces (Voice ID) con diarización e identificación.

Puerto: 3010

Endpoints:
- GET  /                         → UI web
- GET  /api/health               → Health check
- POST /api/register             → Registrar voz con clips de audio
- GET  /api/persons              → Listar voces registradas
- GET  /api/persons/{id}         → Obtener persona
- PUT  /api/persons/{id}         → Actualizar nombre/aliases/role
- DELETE /api/persons/{id}       → Eliminar persona
- POST /api/search               → Buscar por nombre
- POST /api/diarize              → Diarizar audio + identificar speakers
- POST /api/identify-speaker     → Identificar speaker en clip corto
- WS   /ws/progress/{id}         → WebSocket para progreso de registro
"""
import os
import asyncio
import logging
import logging.config
import uuid
import json
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import (
    SERVICE_NAME,
    SERVICE_PORT,
    HOST,
    LOGGING_CONFIG,
    STATIC_DIR,
    TEMP_DIR,
    MIN_AUDIO_CLIPS,
    RECOMMENDED_AUDIO_CLIPS,
    MAX_AUDIO_CLIPS,
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE,
    MIN_CLIP_DURATION,
    MAX_CLIP_DURATION,
    SIMILARITY_THRESHOLD,
    SPEAKER_ROLES
)
from services.voice_embedding_service import VoiceEmbeddingService, get_voice_embedding_service
from services.elasticsearch_service import ElasticsearchVoiceService, get_elasticsearch_voice_service
from services.diarization_service import DiarizationService, get_diarization_service

# Configurar logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# Almacén de conexiones WebSocket
active_connections: Dict[str, WebSocket] = {}


# ============================================================================
# MODELOS PYDANTIC
# ============================================================================

class PersonResponse(BaseModel):
    person_id: str
    name: str
    aliases: List[str] = []
    role: str = "locutor"
    sample_count: int = 0
    avg_confidence: float = 0.0
    total_speech_duration: float = 0.0
    created_at: str = ""


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int = Field(default=10, ge=1, le=50)


class UpdatePersonRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=2, max_length=100)
    aliases: Optional[List[str]] = None
    role: Optional[str] = None


@dataclass
class ProcessingProgress:
    current: int
    total: int
    stage: str
    message: str
    percentage: float


# ============================================================================
# LIFECYCLE
# ============================================================================

es_service: Optional[ElasticsearchVoiceService] = None
embedding_service: Optional[VoiceEmbeddingService] = None
diarization_service: Optional[DiarizationService] = None


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
# WEBSOCKET PARA PROGRESO
# ============================================================================

@app.websocket("/ws/progress/{registration_id}")
async def websocket_progress(websocket: WebSocket, registration_id: str):
    """WebSocket para progreso de registro en tiempo real"""
    await websocket.accept()
    active_connections[registration_id] = websocket
    
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if registration_id in active_connections:
            del active_connections[registration_id]


async def send_progress(registration_id: str, progress: ProcessingProgress):
    """Enviar progreso a cliente WebSocket"""
    if registration_id in active_connections:
        try:
            await active_connections[registration_id].send_json({
                "current": progress.current,
                "total": progress.total,
                "stage": progress.stage,
                "message": progress.message,
                "percentage": progress.percentage
            })
        except Exception as e:
            logger.warning(f"Error enviando progreso: {e}")


# ============================================================================
# ENDPOINTS - HEALTH
# ============================================================================

@app.get("/api/health")
async def health_check():
    """Health check del servicio"""
    return {
        "status": "healthy",
        "service": SERVICE_NAME,
        "port": SERVICE_PORT,
        "subsystems": {
            "elasticsearch": es_service.is_available if es_service else False,
            "voice_embedding": embedding_service._initialized if embedding_service else False,
            "diarization": diarization_service._initialized if diarization_service else False
        },
        "voice_bank": {
            "total_persons": await es_service.get_persons_count() if es_service and es_service.is_available else 0
        }
    }


# ============================================================================
# ENDPOINTS - VOICE BANK CRUD
# ============================================================================

@app.get("/api/persons")
async def list_persons(limit: int = Query(100, ge=1, le=500)):
    """Listar todas las voces registradas"""
    if not es_service or not es_service.is_available:
        raise HTTPException(status_code=503, detail="Elasticsearch no disponible")
    
    persons = await es_service.get_all_persons(limit=limit)
    return persons


@app.get("/api/persons/{person_id}")
async def get_person(person_id: str):
    """Obtener una persona por ID"""
    if not es_service or not es_service.is_available:
        raise HTTPException(status_code=503, detail="Elasticsearch no disponible")
    
    person = await es_service.get_person(person_id)
    if not person:
        raise HTTPException(status_code=404, detail="Persona no encontrada")
    
    return person


@app.put("/api/persons/{person_id}")
async def update_person(person_id: str, request: UpdatePersonRequest):
    """Actualizar nombre/aliases/role de una persona"""
    if not es_service or not es_service.is_available:
        raise HTTPException(status_code=503, detail="Elasticsearch no disponible")
    
    # Verificar que existe
    existing = await es_service.get_person(person_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Persona no encontrada")
    
    success = await es_service.update_person(
        person_id=person_id,
        name=request.name,
        aliases=request.aliases,
        role=request.role
    )
    
    if not success:
        raise HTTPException(status_code=500, detail="Error actualizando persona")
    
    updated = await es_service.get_person(person_id)
    return {"success": True, "message": "Persona actualizada", "person": updated}


@app.delete("/api/persons/{person_id}")
async def delete_person(person_id: str):
    """Eliminar persona del banco de voces"""
    if not es_service or not es_service.is_available:
        raise HTTPException(status_code=503, detail="Elasticsearch no disponible")
    
    success = await es_service.delete_person(person_id)
    if not success:
        raise HTTPException(status_code=404, detail="Persona no encontrada")
    
    return {"success": True, "message": "Persona eliminada"}


@app.post("/api/search")
async def search_persons(request: SearchRequest):
    """Buscar por nombre/alias"""
    if not es_service or not es_service.is_available:
        raise HTTPException(status_code=503, detail="Elasticsearch no disponible")
    
    results = await es_service.search_by_name(request.query, request.limit)
    return {"results": results, "count": len(results)}


# ============================================================================
# ENDPOINTS - REGISTRO DE VOZ
# ============================================================================

@app.post("/api/register")
async def register_voice(
    name: str = Form(...),
    aliases: str = Form(""),
    role: str = Form("locutor"),
    audio_files: List[UploadFile] = File(...)
):
    """
    Registrar una nueva voz con clips de audio.
    
    - name: Nombre del locutor
    - aliases: Aliases separados por coma
    - role: Rol (locutor, periodista, comentarista, etc.)
    - audio_files: 1-10 clips de audio (WAV, MP3, OGG, etc.)
    """
    if not es_service or not es_service.is_available:
        raise HTTPException(status_code=503, detail="Elasticsearch no disponible")
    
    if not embedding_service or not embedding_service._initialized:
        raise HTTPException(status_code=503, detail="Voice Embedding no disponible")
    
    # Validar número de clips
    if len(audio_files) < MIN_AUDIO_CLIPS:
        raise HTTPException(
            status_code=400,
            detail=f"Se requiere al menos {MIN_AUDIO_CLIPS} clip de audio. Recibidos: {len(audio_files)}"
        )
    
    if len(audio_files) > MAX_AUDIO_CLIPS:
        raise HTTPException(
            status_code=400,
            detail=f"Máximo {MAX_AUDIO_CLIPS} clips. Recibidos: {len(audio_files)}"
        )
    
    # Validar role
    if role not in SPEAKER_ROLES:
        role = "otro"
    
    # Generar ID de registro para WebSocket
    registration_id = str(uuid.uuid4())[:8]
    
    # Parsear aliases
    alias_list = [a.strip() for a in aliases.split(",") if a.strip()] if aliases else []
    
    # Procesar clips
    embeddings = []
    sample_durations = []
    errors = []
    total = len(audio_files)
    
    for i, audio_file in enumerate(audio_files):
        # Validar extensión
        ext = Path(audio_file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            errors.append(f"Clip {i+1}: Formato no permitido ({ext})")
            continue
        
        # Leer y guardar temporalmente
        content = await audio_file.read()
        
        if len(content) > MAX_FILE_SIZE:
            errors.append(f"Clip {i+1}: Archivo muy grande ({len(content) // 1024 // 1024}MB)")
            continue
        
        # Guardar archivo temporal
        temp_path = TEMP_DIR / f"register_{registration_id}_{i}{ext}"
        try:
            temp_path.write_bytes(content)
            
            # Progreso
            await send_progress(registration_id, ProcessingProgress(
                current=i + 1,
                total=total,
                stage="extracting",
                message=f"Procesando clip {i+1}/{total}: {audio_file.filename}",
                percentage=((i + 1) / total) * 100
            ))
            
            # Verificar duración
            audio_info = await embedding_service.get_audio_info(str(temp_path))
            duration = audio_info.get("duration", 0)
            
            if duration < MIN_CLIP_DURATION:
                errors.append(
                    f"Clip {i+1}: Demasiado corto ({duration:.1f}s, mínimo {MIN_CLIP_DURATION}s)"
                )
                continue
            
            if duration > MAX_CLIP_DURATION:
                errors.append(
                    f"Clip {i+1}: Demasiado largo ({duration:.1f}s, máximo {MAX_CLIP_DURATION}s)"
                )
                continue
            
            # Extraer embedding — usa diarización para aislar hablante dominante
            # Esto evita contaminar el embedding si el clip tiene múltiples hablantes
            diar_pipeline = None
            if diarization_service and diarization_service._initialized:
                diar_pipeline = diarization_service._pipeline
            
            embedding = await embedding_service.extract_embedding_for_registration(
                str(temp_path), diarization_pipeline=diar_pipeline
            )
            
            if embedding is not None:
                embeddings.append(embedding)
                sample_durations.append(round(duration, 2))
                logger.info(f"  ✅ Clip {i+1}: embedding extraído ({duration:.1f}s)")
            else:
                errors.append(f"Clip {i+1}: No se pudo extraer embedding (¿hay habla en el audio?)")
                
        except Exception as e:
            errors.append(f"Clip {i+1}: Error procesando ({str(e)})")
            logger.error(f"Error procesando clip {i+1}: {e}")
            
        finally:
            # Limpiar temporal
            if temp_path.exists():
                temp_path.unlink()
    
    # Verificar que tenemos al menos 1 embedding válido
    if not embeddings:
        raise HTTPException(
            status_code=400,
            detail=f"No se pudo extraer ningún embedding válido. Errores: {errors}"
        )
    
    # Promediar embeddings (mantener float32 para consistencia con ES y búsquedas)
    avg_embedding = np.mean(embeddings, axis=0).astype(np.float32)
    norm = np.linalg.norm(avg_embedding)
    if norm > 0:
        avg_embedding = avg_embedding / norm
    
    # Calcular confianza (la consistencia entre embeddings)
    if len(embeddings) > 1:
        similarities = []
        for emb in embeddings:
            sim = float(np.dot(avg_embedding, emb))
            similarities.append(sim)
        avg_confidence = float(np.mean(similarities))
    else:
        avg_confidence = 1.0
    
    total_speech = sum(sample_durations)
    
    # Registrar en Elasticsearch
    register_result = await es_service.register_person(
        name=name,
        embedding=avg_embedding,
        role=role,
        aliases=alias_list,
        sample_count=len(embeddings),
        avg_confidence=avg_confidence,
        sample_durations=sample_durations,
        total_speech_duration=total_speech
    )
    
    if not register_result["success"]:
        raise HTTPException(
            status_code=500,
            detail=register_result.get("error", "Error guardando en Elasticsearch")
        )
    
    logger.info(
        f"✅ Voz registrada: {name} (ID: {register_result['person_id']}, "
        f"{len(embeddings)} clips, {total_speech:.1f}s habla)"
    )
    
    return {
        "success": True,
        "person_id": register_result["person_id"],
        "name": name,
        "role": role,
        "aliases": alias_list,
        "sample_count": len(embeddings),
        "avg_confidence": round(avg_confidence, 3),
        "total_speech_duration": round(total_speech, 2),
        "errors": errors,
        "message": f"Voz de '{name}' registrada con {len(embeddings)} clips ({total_speech:.1f}s de habla)"
    }


# ============================================================================
# ENDPOINTS - DIARIZACIÓN
# ============================================================================

@app.post("/api/diarize")
async def diarize_audio(
    audio: UploadFile = File(...),
    num_speakers: Optional[int] = Form(None),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
    identify: bool = Form(True),
    threshold: Optional[float] = Form(None)
):
    """
    Diarizar audio y opcionalmente identificar speakers.
    
    - audio: Archivo de audio
    - num_speakers: Número exacto de speakers (None = auto)
    - min_speakers/max_speakers: Rango de speakers
    - identify: Si True, buscar speakers en banco de voces
    - threshold: Umbral de similitud (default: 0.70)
    """
    if not diarization_service or not diarization_service._initialized:
        raise HTTPException(status_code=503, detail="Diarización no disponible")
    
    # Validar archivo
    ext = Path(audio.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Formato no soportado: {ext}")
    
    # Guardar temporalmente
    temp_path = TEMP_DIR / f"diarize_{uuid.uuid4().hex[:8]}{ext}"
    
    try:
        content = await audio.read()
        temp_path.write_bytes(content)
        
        logger.info(f"🎙️ Diarizando: {audio.filename} ({len(content) // 1024}KB)")
        
        # Ejecutar diarización + identificación
        result = await diarization_service.diarize_and_identify(
            audio_path=str(temp_path),
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            threshold=threshold,
            identify=identify
        )
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error en diarización: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.post("/api/identify-speaker")
async def identify_speaker(
    audio: UploadFile = File(...),
    threshold: Optional[float] = Query(None)
):
    """
    Identificar un speaker en un clip de audio corto.
    Compara contra el banco de voces.
    """
    if not embedding_service or not embedding_service._initialized:
        raise HTTPException(status_code=503, detail="Voice Embedding no disponible")
    
    if not es_service or not es_service.is_available:
        raise HTTPException(status_code=503, detail="Elasticsearch no disponible")
    
    ext = Path(audio.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Formato no soportado: {ext}")
    
    temp_path = TEMP_DIR / f"identify_{uuid.uuid4().hex[:8]}{ext}"
    
    try:
        content = await audio.read()
        temp_path.write_bytes(content)
        
        matches = await diarization_service.identify_speaker(
            audio_path=str(temp_path),
            threshold=threshold
        )
        
        return {
            "matches": matches,
            "speakers_checked": await es_service.get_persons_count()
        }
        
    except Exception as e:
        logger.error(f"❌ Error identificando speaker: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.get("/api/roles")
async def get_available_roles():
    """Obtener roles disponibles para locutores"""
    return {"roles": SPEAKER_ROLES}


# ============================================================================
# UI WEB
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Servir la interfaz web"""
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return html_path.read_text()
    return get_embedded_html()


def get_embedded_html() -> str:
    """HTML embebido como fallback"""
    return '''<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DPS Voice Manager</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .dropzone { transition: all 0.3s ease; }
        .dropzone.dragover { border-color: #8b5cf6; background-color: #f5f3ff; }
        .audio-preview { max-width: 100%; }
        .speaker-tag { display: inline-block; padding: 2px 8px; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; }
        .timeline-segment { height: 30px; position: absolute; border-radius: 4px; opacity: 0.8; cursor: pointer; }
        .timeline-segment:hover { opacity: 1; }
    </style>
</head>
<body class="bg-gray-100 min-h-screen">
    <div class="container mx-auto px-4 py-8 max-w-5xl">
        <!-- Header -->
        <div class="text-center mb-8">
            <h1 class="text-3xl font-bold text-gray-800">🎙️ DPS Voice Manager</h1>
            <p class="text-gray-600 mt-2">Banco de voces, diarización e identificación de speakers</p>
            <div id="healthStatus" class="mt-2 text-sm text-gray-500"></div>
        </div>

        <!-- Tabs -->
        <div class="flex mb-6 border-b overflow-x-auto">
            <button onclick="showTab('register')" id="tab-register" 
                    class="px-6 py-3 font-medium text-purple-600 border-b-2 border-purple-600 whitespace-nowrap">
                🎤 Registrar Voz
            </button>
            <button onclick="showTab('persons')" id="tab-persons"
                    class="px-6 py-3 font-medium text-gray-500 hover:text-gray-700 whitespace-nowrap">
                👥 Voces Registradas
            </button>
            <button onclick="showTab('test')" id="tab-test"
                    class="px-6 py-3 font-medium text-gray-500 hover:text-gray-700 whitespace-nowrap">
                🧪 Probar Identificación
            </button>
        </div>

        <!-- ==================== TAB: REGISTRAR VOZ ==================== -->
        <div id="panel-register" class="bg-white rounded-lg shadow-md p-6">
            <form id="registerForm" onsubmit="submitRegistration(event)">
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                    <div>
                        <label class="block text-gray-700 font-medium mb-2">Nombre *</label>
                        <input type="text" id="personName" required minlength="2"
                               class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                               placeholder="Ej: Mónica Rincón">
                    </div>
                    <div>
                        <label class="block text-gray-700 font-medium mb-2">Rol</label>
                        <select id="personRole" class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-purple-500">
                            <option value="locutor">Locutor</option>
                            <option value="periodista">Periodista</option>
                            <option value="comentarista">Comentarista</option>
                            <option value="conductor">Conductor</option>
                            <option value="corresponsal">Corresponsal</option>
                            <option value="invitado">Invitado</option>
                            <option value="analista">Analista</option>
                            <option value="reportero">Reportero</option>
                            <option value="otro">Otro</option>
                        </select>
                    </div>
                </div>

                <div class="mb-4">
                    <label class="block text-gray-700 font-medium mb-2">Aliases (opcional)</label>
                    <input type="text" id="personAliases"
                           class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-purple-500"
                           placeholder="Ej: La Rincón, Moni (separados por coma)">
                </div>

                <!-- Dropzone Audio -->
                <div class="mb-4">
                    <label class="block text-gray-700 font-medium mb-2">
                        Clips de audio * <span class="text-sm text-gray-500">(mínimo 1, recomendado 3-5 clips de 10-30s)</span>
                    </label>
                    <div id="dropzone" 
                         class="dropzone border-2 border-dashed border-gray-300 rounded-lg p-8 text-center cursor-pointer hover:border-purple-400">
                        <input type="file" id="fileInput" multiple accept="audio/*" class="hidden"
                               onchange="handleFiles(this.files)">
                        <div class="text-gray-500">
                            <svg class="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 48 48">
                                <path d="M24 4v28m0 0l-8-8m8 8l8-8M8 36v4a4 4 0 004 4h24a4 4 0 004-4v-4" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                            <p class="mt-2">Arrastra clips de audio aquí o <span class="text-purple-500 font-medium">haz clic para seleccionar</span></p>
                            <p class="text-sm text-gray-400 mt-1">WAV, MP3, OGG, FLAC, M4A • 3s-120s cada clip • Máx 50MB</p>
                        </div>
                    </div>
                </div>

                <!-- Preview de audios -->
                <div id="audioPreview" class="space-y-2 mb-4"></div>
                <div id="audioCount" class="text-sm text-gray-600 mb-4"></div>

                <!-- Progreso -->
                <div id="progressContainer" class="hidden mb-4">
                    <div class="flex justify-between text-sm text-gray-600 mb-1">
                        <span id="progressMessage">Procesando...</span>
                        <span id="progressPercent">0%</span>
                    </div>
                    <div class="w-full bg-gray-200 rounded-full h-3">
                        <div id="progressBar" class="bg-purple-600 h-3 rounded-full transition-all duration-300" style="width: 0%"></div>
                    </div>
                </div>

                <button type="submit" id="submitBtn"
                        class="w-full bg-purple-600 text-white py-3 rounded-lg font-medium hover:bg-purple-700 disabled:bg-gray-400 disabled:cursor-not-allowed">
                    🎤 Registrar Voz
                </button>
            </form>

            <div id="resultContainer" class="hidden mt-4 p-4 rounded-lg"></div>
        </div>

        <!-- ==================== TAB: VOCES REGISTRADAS ==================== -->
        <div id="panel-persons" class="hidden bg-white rounded-lg shadow-md p-6">
            <div class="flex justify-between items-center mb-4">
                <h2 class="text-xl font-semibold">Voces Registradas</h2>
                <button onclick="loadPersons()" class="text-purple-600 hover:text-purple-800">🔄 Actualizar</button>
            </div>

            <div class="mb-4">
                <input type="text" id="searchInput" placeholder="Buscar por nombre..."
                       class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-purple-500"
                       onkeyup="filterPersons(this.value)">
            </div>

            <div id="personsList" class="space-y-3">
                <p class="text-gray-500 text-center py-8">Cargando...</p>
            </div>
        </div>

        <!-- ==================== TAB: PROBAR IDENTIFICACIÓN ==================== -->
        <div id="panel-test" class="hidden bg-white rounded-lg shadow-md p-6">
            <h2 class="text-xl font-semibold mb-4">🧪 Probar Diarización e Identificación</h2>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div>
                    <label class="block text-gray-700 font-medium mb-2">Audio para analizar</label>
                    <input type="file" id="testAudioInput" accept="audio/*"
                           class="w-full px-3 py-2 border rounded-lg text-sm">
                </div>
                <div class="flex items-end gap-2">
                    <div class="flex-1">
                        <label class="block text-gray-700 font-medium mb-2">Modo</label>
                        <select id="testMode" class="w-full px-4 py-2 border rounded-lg">
                            <option value="diarize">Diarizar + Identificar</option>
                            <option value="identify">Solo Identificar Speaker</option>
                        </select>
                    </div>
                    <button onclick="runTest()" id="testBtn"
                            class="px-6 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:bg-gray-400">
                        Analizar
                    </button>
                </div>
            </div>

            <!-- Opciones avanzadas -->
            <details class="mb-4">
                <summary class="text-sm text-gray-500 cursor-pointer hover:text-gray-700">Opciones avanzadas</summary>
                <div class="grid grid-cols-3 gap-4 mt-2 p-3 bg-gray-50 rounded-lg">
                    <div>
                        <label class="text-sm text-gray-600">Nº Speakers</label>
                        <input type="number" id="testNumSpeakers" min="1" max="20" placeholder="Auto"
                               class="w-full px-3 py-1 border rounded text-sm">
                    </div>
                    <div>
                        <label class="text-sm text-gray-600">Threshold</label>
                        <input type="number" id="testThreshold" min="0.1" max="1.0" step="0.05" placeholder="0.70"
                               class="w-full px-3 py-1 border rounded text-sm">
                    </div>
                    <div class="flex items-end">
                        <label class="flex items-center gap-2 text-sm text-gray-600">
                            <input type="checkbox" id="testIdentify" checked class="rounded">
                            Identificar
                        </label>
                    </div>
                </div>
            </details>

            <!-- Resultados -->
            <div id="testLoading" class="hidden text-center py-8">
                <div class="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-purple-600"></div>
                <p class="text-gray-500 mt-2">Procesando audio... esto puede tomar unos segundos</p>
            </div>

            <div id="testResults" class="hidden">
                <!-- Resumen -->
                <div id="testSummary" class="mb-4 p-4 bg-purple-50 rounded-lg"></div>

                <!-- Timeline visual -->
                <div id="testTimeline" class="mb-4"></div>

                <!-- Segmentos detallados -->
                <div id="testSegments" class="space-y-1"></div>
            </div>
        </div>

        <!-- ==================== MODAL EDITAR ==================== -->
        <div id="editModal" class="hidden fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div class="bg-white rounded-lg shadow-xl p-6 w-full max-w-md mx-4">
                <div class="flex justify-between items-center mb-4">
                    <h3 class="text-lg font-semibold">✏️ Editar Persona</h3>
                    <button onclick="closeEditModal()" class="text-gray-500 hover:text-gray-700 text-2xl">&times;</button>
                </div>
                <form id="editForm" onsubmit="submitEdit(event)">
                    <input type="hidden" id="editPersonId">
                    
                    <div class="mb-4">
                        <label class="block text-gray-700 font-medium mb-2">Nombre</label>
                        <input type="text" id="editName" required minlength="2"
                               class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-purple-500">
                    </div>
                    
                    <div class="mb-4">
                        <label class="block text-gray-700 font-medium mb-2">Aliases <span class="text-sm text-gray-500">(separados por coma)</span></label>
                        <input type="text" id="editAliases"
                               class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-purple-500">
                    </div>

                    <div class="mb-4">
                        <label class="block text-gray-700 font-medium mb-2">Rol</label>
                        <select id="editRole" class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-purple-500">
                            <option value="locutor">Locutor</option>
                            <option value="periodista">Periodista</option>
                            <option value="comentarista">Comentarista</option>
                            <option value="conductor">Conductor</option>
                            <option value="corresponsal">Corresponsal</option>
                            <option value="invitado">Invitado</option>
                            <option value="analista">Analista</option>
                            <option value="reportero">Reportero</option>
                            <option value="otro">Otro</option>
                        </select>
                    </div>
                    
                    <p class="text-sm text-gray-500 mb-4">💡 El embedding de voz no se modifica. Para actualizar la voz, elimine y re-registre.</p>
                    
                    <div class="flex gap-3">
                        <button type="button" onclick="closeEditModal()" 
                                class="flex-1 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50">Cancelar</button>
                        <button type="submit" id="editSubmitBtn"
                                class="flex-1 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700">Guardar</button>
                    </div>
                </form>
            </div>
        </div>
    </div>

    <script>
        let selectedFiles = [];
        const SPEAKER_COLORS = [
            '#8b5cf6', '#ef4444', '#3b82f6', '#10b981', '#f59e0b',
            '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1'
        ];

        // ===== TABS =====
        function showTab(tab) {
            ['register', 'persons', 'test'].forEach(t => {
                document.getElementById('panel-' + t).classList.toggle('hidden', t !== tab);
                const tabBtn = document.getElementById('tab-' + t);
                tabBtn.classList.toggle('text-purple-600', t === tab);
                tabBtn.classList.toggle('border-purple-600', t === tab);
                tabBtn.classList.toggle('border-b-2', t === tab);
                tabBtn.classList.toggle('text-gray-500', t !== tab);
            });
            if (tab === 'persons') loadPersons();
        }

        // ===== HEALTH CHECK =====
        async function checkHealth() {
            try {
                const r = await fetch('/api/health');
                const data = await r.json();
                const s = data.subsystems;
                const parts = [];
                parts.push(s.elasticsearch ? '✅ ES' : '❌ ES');
                parts.push(s.voice_embedding ? '✅ Embedding' : '❌ Embedding');
                parts.push(s.diarization ? '✅ Diarización' : '⚠️ Diarización');
                parts.push(`📊 ${data.voice_bank.total_persons} voces registradas`);
                document.getElementById('healthStatus').innerHTML = parts.join(' • ');
            } catch (e) {
                document.getElementById('healthStatus').innerHTML = '❌ Servicio no disponible';
            }
        }

        // ===== DROPZONE =====
        const dropzone = document.getElementById('dropzone');
        const fileInput = document.getElementById('fileInput');
        dropzone.addEventListener('click', () => fileInput.click());
        dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
        dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
        dropzone.addEventListener('drop', (e) => { e.preventDefault(); dropzone.classList.remove('dragover'); handleFiles(e.dataTransfer.files); });

        function handleFiles(files) {
            for (const file of files) {
                if (file.type.startsWith('audio/') && selectedFiles.length < 10) {
                    selectedFiles.push(file);
                }
            }
            updateAudioPreview();
        }

        function updateAudioPreview() {
            const preview = document.getElementById('audioPreview');
            const count = document.getElementById('audioCount');
            
            preview.innerHTML = selectedFiles.map((file, i) => `
                <div class="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <div class="flex items-center gap-3">
                        <span class="text-purple-500">🎵</span>
                        <div>
                            <p class="text-sm font-medium">${file.name}</p>
                            <p class="text-xs text-gray-400">${(file.size / 1024 / 1024).toFixed(1)} MB</p>
                        </div>
                    </div>
                    <div class="flex items-center gap-2">
                        <audio controls class="h-8" style="max-width: 200px">
                            <source src="${URL.createObjectURL(file)}">
                        </audio>
                        <button type="button" onclick="removeFile(${i})" class="text-red-500 hover:text-red-700 text-lg">×</button>
                    </div>
                </div>
            `).join('');
            
            const n = selectedFiles.length;
            let color = n < 1 ? 'text-red-500' : n < 3 ? 'text-yellow-600' : 'text-green-600';
            count.innerHTML = `<span class="${color}">${n} clip${n !== 1 ? 's' : ''} seleccionado${n !== 1 ? 's' : ''}</span>`;
            document.getElementById('submitBtn').disabled = n < 1;
        }

        function removeFile(index) {
            selectedFiles.splice(index, 1);
            updateAudioPreview();
        }

        // ===== REGISTRO =====
        async function submitRegistration(e) {
            e.preventDefault();
            if (selectedFiles.length < 1) { alert('Se requiere al menos 1 clip'); return; }
            
            const formData = new FormData();
            formData.append('name', document.getElementById('personName').value);
            formData.append('aliases', document.getElementById('personAliases').value);
            formData.append('role', document.getElementById('personRole').value);
            selectedFiles.forEach(file => formData.append('audio_files', file));
            
            document.getElementById('progressContainer').classList.remove('hidden');
            document.getElementById('submitBtn').disabled = true;
            
            try {
                const response = await fetch('/api/register', { method: 'POST', body: formData });
                const result = await response.json();
                
                const container = document.getElementById('resultContainer');
                container.classList.remove('hidden');
                
                if (response.ok && result.success) {
                    container.className = 'mt-4 p-4 rounded-lg bg-green-100 text-green-800';
                    container.innerHTML = `
                        <p class="font-medium">✅ ${result.message}</p>
                        <p class="text-sm mt-1">ID: ${result.person_id} • Confianza: ${(result.avg_confidence * 100).toFixed(0)}%</p>
                        ${result.errors.length ? `<p class="text-sm mt-1 text-yellow-700">⚠️ Warnings: ${result.errors.join(', ')}</p>` : ''}
                    `;
                    document.getElementById('registerForm').reset();
                    selectedFiles = [];
                    updateAudioPreview();
                    checkHealth();
                } else {
                    container.className = 'mt-4 p-4 rounded-lg bg-red-100 text-red-800';
                    container.innerHTML = `<p class="font-medium">❌ ${result.detail || result.error || 'Error desconocido'}</p>`;
                }
            } catch (error) {
                alert('Error de conexión: ' + error.message);
            } finally {
                document.getElementById('progressContainer').classList.add('hidden');
                document.getElementById('progressBar').style.width = '0%';
                document.getElementById('submitBtn').disabled = false;
            }
        }

        // ===== VOCES REGISTRADAS =====
        let allPersons = [];

        async function loadPersons() {
            const list = document.getElementById('personsList');
            list.innerHTML = '<p class="text-gray-500 text-center py-8">Cargando...</p>';
            try {
                const r = await fetch('/api/persons');
                allPersons = await r.json();
                renderPersons(allPersons);
            } catch (e) {
                list.innerHTML = '<p class="text-red-500 text-center py-8">Error cargando voces</p>';
            }
        }

        const ROLE_EMOJIS = {
            'locutor': '🎙️', 'periodista': '📰', 'comentarista': '💬',
            'conductor': '📺', 'corresponsal': '🌍', 'invitado': '🎤',
            'analista': '🔍', 'reportero': '📡', 'otro': '👤'
        };

        function renderPersons(persons) {
            const list = document.getElementById('personsList');
            if (!persons.length) {
                list.innerHTML = '<p class="text-gray-500 text-center py-8">No hay voces registradas</p>';
                return;
            }
            
            list.innerHTML = persons.map(p => `
                <div class="flex items-center justify-between p-4 bg-gray-50 rounded-lg hover:bg-gray-100">
                    <div class="flex items-center gap-4">
                        <div class="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center text-xl">
                            ${ROLE_EMOJIS[p.role] || '🎙️'}
                        </div>
                        <div>
                            <p class="font-medium text-gray-800">${p.name}</p>
                            <div class="flex items-center gap-2 text-sm text-gray-500">
                                <span class="speaker-tag bg-purple-100 text-purple-700">${p.role || 'locutor'}</span>
                                ${p.aliases && p.aliases.length ? ` <span>• ${p.aliases.join(', ')}</span>` : ''}
                            </div>
                            <p class="text-xs text-gray-400">${p.sample_count || 0} clips • ${(p.total_speech_duration || 0).toFixed(0)}s habla • Consistencia: ${((p.avg_confidence || 0) * 100).toFixed(0)}%</p>
                        </div>
                    </div>
                    <div class="flex gap-2">
                        <button onclick="openEditModal('${p.person_id}', '${p.name.replace(/'/g, "\\'")}', '${(p.aliases || []).join(', ').replace(/'/g, "\\'")}', '${p.role || 'locutor'}')" 
                                class="text-blue-500 hover:text-blue-700 p-2" title="Editar">✏️</button>
                        <button onclick="deletePerson('${p.person_id}', '${p.name.replace(/'/g, "\\'")}')" 
                                class="text-red-500 hover:text-red-700 p-2" title="Eliminar">🗑️</button>
                    </div>
                </div>
            `).join('');
        }

        function filterPersons(query) {
            const q = query.toLowerCase();
            const filtered = allPersons.filter(p => 
                p.name.toLowerCase().includes(q) ||
                (p.role || '').toLowerCase().includes(q) ||
                (p.aliases && p.aliases.some(a => a.toLowerCase().includes(q)))
            );
            renderPersons(filtered);
        }

        async function deletePerson(id, name) {
            if (!confirm(`¿Eliminar la voz de "${name}"?`)) return;
            try {
                const r = await fetch(`/api/persons/${id}`, { method: 'DELETE' });
                if (r.ok) { loadPersons(); checkHealth(); }
                else alert('Error eliminando voz');
            } catch (e) { alert('Error de conexión'); }
        }

        // ===== EDITAR =====
        function openEditModal(personId, name, aliases, role) {
            document.getElementById('editPersonId').value = personId;
            document.getElementById('editName').value = name;
            document.getElementById('editAliases').value = aliases;
            document.getElementById('editRole').value = role;
            document.getElementById('editModal').classList.remove('hidden');
        }

        function closeEditModal() {
            document.getElementById('editModal').classList.add('hidden');
        }

        async function submitEdit(e) {
            e.preventDefault();
            const personId = document.getElementById('editPersonId').value;
            const name = document.getElementById('editName').value.trim();
            const aliasesStr = document.getElementById('editAliases').value.trim();
            const aliases = aliasesStr ? aliasesStr.split(',').map(a => a.trim()).filter(a => a) : [];
            const role = document.getElementById('editRole').value;
            
            document.getElementById('editSubmitBtn').disabled = true;
            try {
                const r = await fetch(`/api/persons/${personId}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name, aliases, role })
                });
                if (r.ok) { closeEditModal(); loadPersons(); }
                else { const res = await r.json(); alert('Error: ' + (res.detail || 'Error actualizando')); }
            } catch (e) { alert('Error de conexión'); }
            finally { document.getElementById('editSubmitBtn').disabled = false; }
        }

        // ===== TEST DIARIZACIÓN =====
        async function runTest() {
            const fileInput = document.getElementById('testAudioInput');
            if (!fileInput.files.length) { alert('Selecciona un archivo de audio'); return; }
            
            const mode = document.getElementById('testMode').value;
            const file = fileInput.files[0];
            
            document.getElementById('testBtn').disabled = true;
            document.getElementById('testLoading').classList.remove('hidden');
            document.getElementById('testResults').classList.add('hidden');
            
            try {
                const formData = new FormData();
                formData.append('audio', file);
                
                let url, response;
                
                if (mode === 'diarize') {
                    const numSpeakers = document.getElementById('testNumSpeakers').value;
                    const threshold = document.getElementById('testThreshold').value;
                    const identify = document.getElementById('testIdentify').checked;
                    
                    if (numSpeakers) formData.append('num_speakers', numSpeakers);
                    if (threshold) formData.append('threshold', threshold);
                    formData.append('identify', identify);
                    
                    response = await fetch('/api/diarize', { method: 'POST', body: formData });
                } else {
                    response = await fetch('/api/identify-speaker', { method: 'POST', body: formData });
                }
                
                const result = await response.json();
                
                if (!response.ok) {
                    alert('Error: ' + (result.detail || 'Error procesando'));
                    return;
                }
                
                document.getElementById('testResults').classList.remove('hidden');
                
                if (mode === 'diarize') {
                    renderDiarizationResults(result);
                } else {
                    renderIdentifyResults(result);
                }
                
            } catch (e) {
                alert('Error: ' + e.message);
            } finally {
                document.getElementById('testBtn').disabled = false;
                document.getElementById('testLoading').classList.add('hidden');
            }
        }

        function renderDiarizationResults(result) {
            // Summary
            const summary = document.getElementById('testSummary');
            summary.innerHTML = `
                <div class="grid grid-cols-3 gap-4 text-center">
                    <div>
                        <p class="text-2xl font-bold text-purple-600">${result.total_speakers}</p>
                        <p class="text-sm text-gray-600">Speakers</p>
                    </div>
                    <div>
                        <p class="text-2xl font-bold text-green-600">${result.identified_speakers}</p>
                        <p class="text-sm text-gray-600">Identificados</p>
                    </div>
                    <div>
                        <p class="text-2xl font-bold text-orange-500">${result.unidentified_speakers}</p>
                        <p class="text-sm text-gray-600">Desconocidos</p>
                    </div>
                </div>
                <div class="mt-3 flex flex-wrap gap-2">
                    ${Object.entries(result.speakers).map(([name, info], i) => `
                        <span class="speaker-tag text-white" style="background-color: ${SPEAKER_COLORS[i % SPEAKER_COLORS.length]}">
                            ${info.identified ? '✅' : '❓'} ${name} (${info.total_time}s)
                            ${info.similarity ? ' • ' + (info.similarity * 100).toFixed(0) + '%' : ''}
                        </span>
                    `).join('')}
                </div>
            `;

            // Timeline
            if (result.segments.length > 0) {
                const maxTime = Math.max(...result.segments.map(s => s.end));
                const speakers = [...new Set(result.segments.map(s => s.speaker))];
                const speakerColors = {};
                speakers.forEach((s, i) => speakerColors[s] = SPEAKER_COLORS[i % SPEAKER_COLORS.length]);
                
                const timeline = document.getElementById('testTimeline');
                timeline.innerHTML = `
                    <p class="text-sm font-medium text-gray-700 mb-2">Timeline (${maxTime.toFixed(1)}s)</p>
                    <div class="relative bg-gray-200 rounded h-10 overflow-hidden">
                        ${result.segments.map(s => {
                            const left = (s.start / maxTime * 100).toFixed(2);
                            const width = ((s.end - s.start) / maxTime * 100).toFixed(2);
                            return `<div class="timeline-segment" title="${s.speaker}: ${s.start.toFixed(1)}s - ${s.end.toFixed(1)}s"
                                         style="left: ${left}%; width: ${width}%; background-color: ${speakerColors[s.speaker]}; top: 5px; height: 20px;"></div>`;
                        }).join('')}
                    </div>
                    <div class="flex justify-between text-xs text-gray-400 mt-1">
                        <span>0:00</span>
                        <span>${Math.floor(maxTime / 60)}:${String(Math.floor(maxTime % 60)).padStart(2, '0')}</span>
                    </div>
                `;
            }

            // Segments table
            const segsDiv = document.getElementById('testSegments');
            segsDiv.innerHTML = `
                <p class="text-sm font-medium text-gray-700 mb-2">Segmentos (${result.segments.length})</p>
                <div class="max-h-96 overflow-y-auto">
                    <table class="w-full text-sm">
                        <thead class="bg-gray-100 sticky top-0">
                            <tr>
                                <th class="px-3 py-2 text-left">Inicio</th>
                                <th class="px-3 py-2 text-left">Fin</th>
                                <th class="px-3 py-2 text-left">Duración</th>
                                <th class="px-3 py-2 text-left">Speaker</th>
                                <th class="px-3 py-2 text-left">Confianza</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${result.segments.map((s, i) => {
                                const speakers = [...new Set(result.segments.map(x => x.speaker))];
                                const colorIdx = speakers.indexOf(s.speaker);
                                return `<tr class="border-b hover:bg-gray-50">
                                    <td class="px-3 py-1">${formatTime(s.start)}</td>
                                    <td class="px-3 py-1">${formatTime(s.end)}</td>
                                    <td class="px-3 py-1">${s.duration.toFixed(1)}s</td>
                                    <td class="px-3 py-1">
                                        <span class="speaker-tag text-white" style="background-color: ${SPEAKER_COLORS[colorIdx % SPEAKER_COLORS.length]}">${s.speaker}</span>
                                        ${s.role ? `<span class="text-xs text-gray-400">${s.role}</span>` : ''}
                                    </td>
                                    <td class="px-3 py-1">${s.confidence ? (s.confidence * 100).toFixed(0) + '%' : '-'}</td>
                                </tr>`;
                            }).join('')}
                        </tbody>
                    </table>
                </div>
            `;
        }

        function renderIdentifyResults(result) {
            const summary = document.getElementById('testSummary');
            if (result.matches && result.matches.length > 0) {
                summary.innerHTML = `
                    <p class="font-medium text-green-700 mb-3">✅ ${result.matches.length} coincidencia(s) encontrada(s)</p>
                    ${result.matches.map((m, i) => `
                        <div class="flex items-center justify-between p-3 ${i === 0 ? 'bg-green-50' : 'bg-gray-50'} rounded-lg mb-2">
                            <div>
                                <p class="font-medium">${m.name} <span class="text-sm text-gray-500">${m.role || ''}</span></p>
                                <p class="text-xs text-gray-400">ID: ${m.person_id}</p>
                            </div>
                            <div class="text-right">
                                <p class="text-lg font-bold ${m.similarity >= 0.7 ? 'text-green-600' : 'text-yellow-600'}">
                                    ${(m.similarity * 100).toFixed(1)}%
                                </p>
                                <p class="text-xs text-gray-400">similitud</p>
                            </div>
                        </div>
                    `).join('')}
                `;
            } else {
                summary.innerHTML = '<p class="text-yellow-700">❓ No se encontraron coincidencias en el banco de voces</p>';
            }
            document.getElementById('testTimeline').innerHTML = '';
            document.getElementById('testSegments').innerHTML = '';
        }

        function formatTime(seconds) {
            const m = Math.floor(seconds / 60);
            const s = (seconds % 60).toFixed(1);
            return `${m}:${s.padStart(4, '0')}`;
        }

        // ===== INIT =====
        document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeEditModal(); });
        updateAudioPreview();
        checkHealth();
    </script>
</body>
</html>'''


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
