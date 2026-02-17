"""
Router para búsqueda de personas
"""
from fastapi import APIRouter, HTTPException
from typing import Optional

from models.schemas import SearchRequest
from services.elasticsearch_service import ElasticsearchVoiceService

router = APIRouter(prefix="/api", tags=["search"])

# Referencia a servicio (se configura en main.py)
es_service: Optional[ElasticsearchVoiceService] = None


def set_es_service(es):
    """Configurar referencia al servicio de Elasticsearch"""
    global es_service
    es_service = es


@router.post("/search")
async def search_persons(request: SearchRequest):
    """Buscar por nombre/alias"""
    if not es_service or not es_service.is_available:
        raise HTTPException(status_code=503, detail="Elasticsearch no disponible")
    
    results = await es_service.search_by_name(request.query, request.limit)
    return {"results": results, "count": len(results)}
