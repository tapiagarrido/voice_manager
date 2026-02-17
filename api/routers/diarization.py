"""
Router para diarización e identificación de speakers
"""
import logging
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Query

from config import TEMP_DIR
from services.diarization_service import DiarizationService
from services.elasticsearch_service import ElasticsearchVoiceService
from services.voice_embedding_service import VoiceEmbeddingService
from utils.validators import validate_audio_extension

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["diarization"])

# Referencias a servicios (se configuran en main.py)
diarization_service: Optional[DiarizationService] = None
es_service: Optional[ElasticsearchVoiceService] = None
embedding_service: Optional[VoiceEmbeddingService] = None


def set_services(diar, es, emb):
    """Configurar referencias a servicios"""
    global diarization_service, es_service, embedding_service
    diarization_service = diar
    es_service = es
    embedding_service = emb


@router.post("/diarize")
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
    ext = validate_audio_extension(audio.filename)
    
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


@router.post("/identify-speaker")
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
    
    ext = validate_audio_extension(audio.filename)
    
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
