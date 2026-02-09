"""
Elasticsearch Voice Service
=============================
Gestiona el índice de voces registradas en Elasticsearch LOCAL.
Usa httpx directamente para compatibilidad con ES 8.x.

Índice: voice_persons
- Almacena embeddings de voz como dense_vector para búsqueda KNN
- Permite búsqueda por nombre, alias, rol, similitud de embedding
"""
import logging
import uuid
import unicodedata
from typing import Dict, Any, List, Optional
from datetime import datetime

import numpy as np
import httpx

from config import ES_LOCAL_HOST, ES_VOICE_INDEX, EMBEDDING_DIMS

logger = logging.getLogger(__name__)


class ElasticsearchVoiceService:
    """
    Servicio para gestionar Voice IDs en Elasticsearch local.
    Réplica del patrón de ElasticsearchFaceService usando httpx síncrono.
    """
    
    def __init__(self, host: str = None):
        self.host = (host or ES_LOCAL_HOST).rstrip('/')
        self.is_available = False
        self._client: Optional[httpx.Client] = None
        
    def _get_client(self) -> httpx.Client:
        """Obtener cliente HTTP síncrono"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(timeout=30.0)
        return self._client
        
    async def initialize(self) -> bool:
        """Inicializar conexión y crear índices si no existen"""
        try:
            client = self._get_client()
            
            # Verificar conexión
            response = client.get(f"{self.host}/")
            if response.status_code != 200:
                raise Exception(f"ES no disponible: {response.status_code}")
            
            info = response.json()
            logger.info(f"✅ Conectado a Elasticsearch {info['version']['number']} en {self.host}")
            
            # Crear índice de voces
            await self._ensure_voice_index()
            
            self.is_available = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Error conectando a Elasticsearch: {e}")
            self.is_available = False
            return False
    
    async def _ensure_voice_index(self):
        """Crear índice voice_persons si no existe"""
        client = self._get_client()
        
        try:
            response = client.head(f"{self.host}/{ES_VOICE_INDEX}")
            
            if response.status_code == 404:
                mapping = {
                    "settings": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0
                    },
                    "mappings": {
                        "properties": {
                            "person_id": {"type": "keyword"},
                            "name": {
                                "type": "text",
                                "fields": {"keyword": {"type": "keyword"}}
                            },
                            "normalized_name": {"type": "keyword"},
                            "aliases": {
                                "type": "text",
                                "fields": {"keyword": {"type": "keyword"}}
                            },
                            "role": {"type": "keyword"},
                            "embedding": {
                                "type": "dense_vector",
                                "dims": EMBEDDING_DIMS,
                                "index": True,
                                "similarity": "cosine"
                            },
                            "sample_count": {"type": "integer"},
                            "avg_confidence": {"type": "float"},
                            "sample_durations": {"type": "float"},
                            "total_speech_duration": {"type": "float"},
                            "created_at": {"type": "date"},
                            "updated_at": {"type": "date"},
                            "metadata": {"type": "object", "enabled": False}
                        }
                    }
                }
                
                response = client.put(
                    f"{self.host}/{ES_VOICE_INDEX}",
                    json=mapping
                )
                
                if response.status_code in (200, 201):
                    logger.info(f"✅ Índice {ES_VOICE_INDEX} creado ({EMBEDDING_DIMS} dims)")
                else:
                    logger.warning(f"⚠️ Error creando índice: {response.text}")
            else:
                logger.info(f"ℹ️ Índice {ES_VOICE_INDEX} ya existe")
                
        except Exception as e:
            logger.warning(f"⚠️ Error verificando índice: {e}")
    
    def _normalize_name(self, name: str) -> str:
        """Normaliza nombre (sin tildes, lowercase, underscores)"""
        normalized = unicodedata.normalize('NFD', name)
        normalized = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
        return normalized.lower().strip().replace(' ', '_')
    
    def _embedding_to_list(self, embedding: np.ndarray) -> List[float]:
        """Convierte numpy array a lista de floats (float32 para consistencia)"""
        return embedding.astype(np.float32).tolist()
    
    def _list_to_embedding(self, embedding_list: List[float]) -> np.ndarray:
        """Convierte lista a numpy array"""
        return np.array(embedding_list, dtype=np.float32)
    
    # ========================================================================
    # CRUD
    # ========================================================================
    
    async def register_person(
        self,
        name: str,
        embedding: np.ndarray,
        role: str = "locutor",
        aliases: List[str] = None,
        sample_count: int = 1,
        avg_confidence: float = 0.0,
        sample_durations: List[float] = None,
        total_speech_duration: float = 0.0,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Registrar una nueva voz con su embedding."""
        person_id = str(uuid.uuid4())[:8]
        now = datetime.utcnow().isoformat()
        
        doc = {
            "person_id": person_id,
            "name": name,
            "normalized_name": self._normalize_name(name),
            "aliases": aliases or [],
            "role": role,
            "embedding": self._embedding_to_list(embedding),
            "sample_count": sample_count,
            "avg_confidence": avg_confidence,
            "sample_durations": sample_durations or [],
            "total_speech_duration": total_speech_duration,
            "created_at": now,
            "updated_at": now,
            "metadata": metadata or {}
        }
        
        try:
            client = self._get_client()
            response = client.put(
                f"{self.host}/{ES_VOICE_INDEX}/_doc/{person_id}?refresh=true",
                json=doc
            )
            
            if response.status_code in (200, 201):
                logger.info(f"✅ Voz registrada: {name} (ID: {person_id}, rol: {role})")
                return {
                    "success": True,
                    "person_id": person_id,
                    "name": name,
                    "message": f"Voz de '{name}' registrada exitosamente"
                }
            else:
                logger.error(f"Error registrando: {response.text}")
                return {"success": False, "error": response.text}
            
        except Exception as e:
            logger.error(f"❌ Error registrando voz: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_person(self, person_id: str) -> Optional[Dict[str, Any]]:
        """Obtener una persona por ID (sin embedding)"""
        try:
            client = self._get_client()
            response = client.post(
                f"{self.host}/{ES_VOICE_INDEX}/_search",
                json={
                    "size": 1,
                    "query": {"term": {"person_id": person_id}},
                    "_source": {"excludes": ["embedding"]}
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                hits = data.get("hits", {}).get("hits", [])
                if hits:
                    return hits[0]["_source"]
            return None
            
        except Exception as e:
            logger.error(f"Error obteniendo persona {person_id}: {e}")
            return None
    
    async def get_person_with_embedding(self, person_id: str) -> Optional[Dict[str, Any]]:
        """Obtener persona incluyendo su embedding (para comparación)"""
        try:
            client = self._get_client()
            response = client.get(f"{self.host}/{ES_VOICE_INDEX}/_doc/{person_id}")
            
            if response.status_code == 200:
                data = response.json()
                person = data.get("_source", {})
                if "embedding" in person:
                    person["embedding"] = self._list_to_embedding(person["embedding"])
                return person
            return None
            
        except Exception as e:
            logger.error(f"Error obteniendo persona con embedding: {e}")
            return None
    
    async def get_all_persons(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtener todas las voces registradas (sin embeddings)"""
        try:
            client = self._get_client()
            response = client.post(
                f"{self.host}/{ES_VOICE_INDEX}/_search",
                json={
                    "size": limit,
                    "sort": [{"created_at": "desc"}],
                    "_source": {
                        "excludes": ["embedding"]
                    }
                }
            )
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            persons = []
            
            for hit in data.get("hits", {}).get("hits", []):
                person = hit["_source"]
                person["_id"] = hit["_id"]
                persons.append(person)
            
            return persons
            
        except Exception as e:
            logger.error(f"Error obteniendo voces: {e}")
            return []
    
    async def search_by_name(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Buscar personas por nombre o alias (fuzzy)"""
        try:
            client = self._get_client()
            response = client.post(
                f"{self.host}/{ES_VOICE_INDEX}/_search",
                json={
                    "size": limit,
                    "query": {
                        "bool": {
                            "should": [
                                {
                                    "match": {
                                        "name": {
                                            "query": query,
                                            "fuzziness": "AUTO"
                                        }
                                    }
                                },
                                {
                                    "match": {
                                        "aliases": {
                                            "query": query,
                                            "fuzziness": "AUTO"
                                        }
                                    }
                                },
                                {
                                    "term": {
                                        "normalized_name": self._normalize_name(query)
                                    }
                                }
                            ],
                            "minimum_should_match": 1
                        }
                    },
                    "_source": {
                        "excludes": ["embedding"]
                    }
                }
            )
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            persons = []
            
            for hit in data.get("hits", {}).get("hits", []):
                person = hit["_source"]
                person["_id"] = hit["_id"]
                person["_score"] = hit["_score"]
                persons.append(person)
            
            return persons
            
        except Exception as e:
            logger.error(f"Error buscando por nombre: {e}")
            return []
    
    async def search_by_embedding(
        self,
        embedding: np.ndarray,
        k: int = 5,
        min_score: float = 0.55
    ) -> List[Dict[str, Any]]:
        """
        Buscar voces similares por embedding usando script_score (cosine).
        
        Args:
            embedding: Vector de 192 dims normalizado
            k: Número máximo de resultados
            min_score: Umbral mínimo de similitud (0-1)
            
        Returns:
            Lista de personas con campo 'similarity' añadido
        """
        try:
            client = self._get_client()
            response = client.post(
                f"{self.host}/{ES_VOICE_INDEX}/_search",
                json={
                    "size": k,
                    "query": {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                "params": {
                                    "query_vector": self._embedding_to_list(embedding)
                                }
                            }
                        }
                    },
                    "_source": {
                        "excludes": ["embedding"]
                    }
                }
            )
            
            if response.status_code != 200:
                logger.warning(f"Error en búsqueda por embedding: {response.text}")
                return []
            
            data = response.json()
            persons = []
            
            for hit in data.get("hits", {}).get("hits", []):
                # Convertir score (0-2) a similitud (0-1)
                similarity = (hit["_score"] - 1.0)
                
                person = hit["_source"]
                name = person.get("name", "?")
                logger.debug(
                    f"  🔍 Candidato: {name} → sim={similarity:.4f} "
                    f"(min_score={min_score})"
                )
                
                if similarity >= min_score:
                    person["_id"] = hit["_id"]
                    person["similarity"] = round(similarity, 4)
                    persons.append(person)
            
            if not persons:
                # Log de todos los candidatos aunque no pasen el filtro
                all_hits = data.get("hits", {}).get("hits", [])
                if all_hits:
                    best = all_hits[0]
                    best_sim = best["_score"] - 1.0
                    best_name = best["_source"].get("name", "?")
                    logger.info(
                        f"  ⚠️ Sin matches sobre min_score={min_score}. "
                        f"Mejor candidato: {best_name} (sim={best_sim:.4f})"
                    )
            
            return persons
            
        except Exception as e:
            logger.error(f"Error buscando por embedding: {e}")
            return []
    
    async def update_person(
        self,
        person_id: str,
        name: str = None,
        aliases: List[str] = None,
        role: str = None
    ) -> bool:
        """Actualizar datos de una persona (sin cambiar embedding)"""
        try:
            update_doc = {"updated_at": datetime.utcnow().isoformat()}
            
            if name:
                update_doc["name"] = name
                update_doc["normalized_name"] = self._normalize_name(name)
            if aliases is not None:
                update_doc["aliases"] = aliases
            if role:
                update_doc["role"] = role
            
            client = self._get_client()
            response = client.post(
                f"{self.host}/{ES_VOICE_INDEX}/_update/{person_id}?refresh=true",
                json={"doc": update_doc}
            )
            
            if response.status_code == 200:
                logger.info(f"✏️ Voz actualizada: {person_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error actualizando voz: {e}")
            return False
    
    async def update_embedding(
        self,
        person_id: str,
        embedding: np.ndarray,
        sample_count: int,
        avg_confidence: float,
        sample_durations: List[float],
        total_speech_duration: float
    ) -> bool:
        """Actualizar embedding de una persona (re-registro de muestras)"""
        try:
            update_doc = {
                "embedding": self._embedding_to_list(embedding),
                "sample_count": sample_count,
                "avg_confidence": avg_confidence,
                "sample_durations": sample_durations,
                "total_speech_duration": total_speech_duration,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            client = self._get_client()
            response = client.post(
                f"{self.host}/{ES_VOICE_INDEX}/_update/{person_id}?refresh=true",
                json={"doc": update_doc}
            )
            
            if response.status_code == 200:
                logger.info(f"🔄 Embedding actualizado: {person_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error actualizando embedding: {e}")
            return False
    
    async def delete_person(self, person_id: str) -> bool:
        """Eliminar una persona del banco de voces"""
        try:
            client = self._get_client()
            response = client.delete(
                f"{self.host}/{ES_VOICE_INDEX}/_doc/{person_id}?refresh=true"
            )
            
            if response.status_code in (200, 204):
                logger.info(f"🗑️ Voz eliminada: {person_id}")
                return True
            
            logger.warning(f"Voz no encontrada o error: {response.status_code}")
            return False
            
        except Exception as e:
            logger.error(f"Error eliminando voz: {e}")
            return False
    
    async def get_persons_count(self) -> int:
        """Obtener el número total de voces registradas"""
        try:
            client = self._get_client()
            response = client.get(f"{self.host}/{ES_VOICE_INDEX}/_count")
            
            if response.status_code == 200:
                return response.json().get("count", 0)
            return 0
        except:
            return 0
    
    async def close(self):
        """Cerrar conexión"""
        if self._client and not self._client.is_closed:
            self._client.close()
            logger.info("🔌 Conexión Elasticsearch cerrada")


# ============================================================================
# SINGLETON
# ============================================================================

_instance: Optional[ElasticsearchVoiceService] = None

def get_elasticsearch_voice_service() -> ElasticsearchVoiceService:
    global _instance
    if _instance is None:
        _instance = ElasticsearchVoiceService()
    return _instance
