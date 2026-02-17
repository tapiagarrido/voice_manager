"""
Router para CRUD de personas (banco de voces)
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from models.schemas import SearchRequest, UpdatePersonRequest
from services.elasticsearch_service import ElasticsearchVoiceService

router = APIRouter(prefix="/api/persons", tags=["persons"])

# Referencia a servicio (se configura en main.py)
es_service: Optional[ElasticsearchVoiceService] = None


def set_es_service(es):
    """Configurar referencia al servicio de Elasticsearch"""
    global es_service
    es_service = es


@router.get("")
async def list_persons(limit: int = Query(100, ge=1, le=500)):
    """Listar todas las voces registradas"""
    if not es_service or not es_service.is_available:
        raise HTTPException(status_code=503, detail="Elasticsearch no disponible")
    
    persons = await es_service.get_all_persons(limit=limit)
    return persons


@router.get("/{person_id}")
async def get_person(person_id: str):
    """Obtener una persona por ID"""
    if not es_service or not es_service.is_available:
        raise HTTPException(status_code=503, detail="Elasticsearch no disponible")
    
    person = await es_service.get_person(person_id)
    if not person:
        raise HTTPException(status_code=404, detail="Persona no encontrada")
    
    return person


@router.put("/{person_id}")
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


@router.delete("/{person_id}")
async def delete_person(person_id: str):
    """Eliminar persona del banco de voces"""
    if not es_service or not es_service.is_available:
        raise HTTPException(status_code=503, detail="Elasticsearch no disponible")
    
    success = await es_service.delete_person(person_id)
    if not success:
        raise HTTPException(status_code=404, detail="Persona no encontrada")
    
    return {"success": True, "message": "Persona eliminada"}
