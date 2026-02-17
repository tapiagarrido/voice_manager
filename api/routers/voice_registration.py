"""
Router para registro de voces
"""
import logging
import uuid
from pathlib import Path
from typing import List, Optional
import numpy as np

from fastapi import APIRouter, HTTPException, File, UploadFile, Form

from config import TEMP_DIR
from models.schemas import ProcessingProgress
from services.voice_embedding_service import VoiceEmbeddingService
from services.elasticsearch_service import ElasticsearchVoiceService
from services.diarization_service import DiarizationService
from services.websocket_service import send_progress
from utils.validators import (
    validate_audio_clips_count,
    validate_speaker_role,
    validate_audio_extension,
    validate_audio_duration
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["registration"])

# Referencias a servicios (se configuran en main.py)
es_service: Optional[ElasticsearchVoiceService] = None
embedding_service: Optional[VoiceEmbeddingService] = None
diarization_service: Optional[DiarizationService] = None


def set_services(es, emb, diar):
    """Configurar referencias a servicios"""
    global es_service, embedding_service, diarization_service
    es_service = es
    embedding_service = emb
    diarization_service = diar


@router.post("/register")
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
    
    # Validaciones
    validate_audio_clips_count(audio_files)
    role = validate_speaker_role(role)
    
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
        try:
            ext = validate_audio_extension(audio_file.filename)
        except HTTPException as e:
            errors.append(f"Clip {i+1}: {e.detail}")
            continue
        
        # Leer contenido
        content = await audio_file.read()
        
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
            
            is_valid, error_msg = validate_audio_duration(duration)
            if not is_valid:
                errors.append(f"Clip {i+1}: {error_msg}")
                continue
            
            # Extraer embedding — usa diarización para aislar hablante dominante
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
    
    # Promediar embeddings (mantener float32 para consistencia)
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
