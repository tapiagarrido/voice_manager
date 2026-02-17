"""
Router para health check y estado del sistema
"""
from fastapi import APIRouter
from typing import Optional

from config import SERVICE_NAME, SERVICE_PORT
from services.elasticsearch_service import ElasticsearchVoiceService
from services.voice_embedding_service import VoiceEmbeddingService
from services.diarization_service import DiarizationService

router = APIRouter(prefix="/api", tags=["health"])

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


@router.get("/health")
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


@router.get("/roles")
async def get_available_roles():
    """Obtener roles disponibles para locutores"""
    from config import SPEAKER_ROLES
    return {"roles": SPEAKER_ROLES}
