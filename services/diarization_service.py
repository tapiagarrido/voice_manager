"""
Diarization Service
====================
Segmentación de audio por speaker usando Pyannote 3.1.
Opcionalmente identifica speakers contra el banco de voces.

Pipeline:
1. Pyannote speaker-diarization-3.1 → segmentos con labels anónimos (SPEAKER_00, etc.)
2. Para cada speaker único, extraer embedding con ECAPA-TDNN
3. Comparar cada embedding contra el banco de voces (Elasticsearch cosine)
4. Si similitud > threshold, reemplazar label anónimo con nombre real

Salida:
{
    "segments": [{"start": 0.5, "end": 3.2, "speaker": "Juan Pérez", "confidence": 0.85}],
    "speakers": {
        "Juan Pérez": {"person_id": "abc12345", "similarity": 0.85, "total_time": 45.3},
        "SPEAKER_02": {"total_time": 12.1, "identified": false}
    },
    "total_speakers": 3,
    "identified_speakers": 2,
    "unidentified_speakers": 1
}
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from config import (
    DIARIZATION_MODEL,
    HF_TOKEN,
    DEVICE,
    SIMILARITY_THRESHOLD,
    PROBABLE_THRESHOLD,
    MIN_SEGMENT_DURATION_FOR_ID,
    MIN_SPEAKERS,
    MAX_SPEAKERS
)
from services.voice_embedding_service import VoiceEmbeddingService, get_voice_embedding_service
from services.elasticsearch_service import ElasticsearchVoiceService, get_elasticsearch_voice_service

logger = logging.getLogger(__name__)


class DiarizationService:
    """
    Servicio de diarización e identificación de speakers.
    
    Usa Pyannote 3.1 para segmentar audio por hablante y
    ECAPA-TDNN + banco de voces para identificar quién es cada uno.
    """
    
    def __init__(self):
        self._initialized = False
        self._pipeline = None
        self._device = DEVICE
        
    async def initialize(self) -> bool:
        """Inicializar pipeline de Pyannote"""
        try:
            if not HF_TOKEN:
                logger.warning("⚠️ HF_TOKEN no configurado. Diarización no disponible.")
                logger.warning("   Configura HF_TOKEN en .env y acepta las condiciones en HuggingFace")
                return False
            
            logger.info(f"🎙️ Cargando pipeline de diarización: {DIARIZATION_MODEL}")
            
            self._pipeline = await asyncio.to_thread(self._load_pipeline)
            
            if self._pipeline is not None:
                self._initialized = True
                logger.info(f"✅ Diarization Service inicializado (device={self._device})")
                return True
            else:
                logger.error("❌ No se pudo cargar el pipeline de diarización")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error inicializando Diarization Service: {e}")
            return False
    
    def _load_pipeline(self):
        """Cargar pipeline de Pyannote (bloqueante)"""
        try:
            from pyannote.audio import Pipeline
            import torch
            
            pipeline = Pipeline.from_pretrained(
                DIARIZATION_MODEL,
                use_auth_token=HF_TOKEN
            )
            
            # Mover a GPU si disponible
            if self._device == "cuda" and torch.cuda.is_available():
                import torch
                pipeline.to(torch.device("cuda"))
                logger.info("  → Pipeline en GPU (CUDA)")
            else:
                self._device = "cpu"
                logger.info("  → Pipeline en CPU")
            
            return pipeline
            
        except Exception as e:
            logger.error(f"Error cargando pipeline de diarización: {e}")
            
            # Información de ayuda
            if "401" in str(e) or "403" in str(e) or "token" in str(e).lower():
                logger.error("   → Verifica tu HF_TOKEN y que hayas aceptado las condiciones:")
                logger.error("     https://huggingface.co/pyannote/speaker-diarization-3.1")
                logger.error("     https://huggingface.co/pyannote/segmentation-3.0")
            
            return None
    
    async def diarize(
        self,
        audio_path: str,
        num_speakers: int = None,
        min_speakers: int = None,
        max_speakers: int = None
    ) -> List[Dict[str, Any]]:
        """
        Ejecutar diarización pura (sin identificación).
        
        Args:
            audio_path: Ruta al archivo de audio
            num_speakers: Número exacto de speakers (None = auto)
            min_speakers: Mínimo de speakers esperados
            max_speakers: Máximo de speakers esperados
            
        Returns:
            Lista de segmentos: [{"start", "end", "speaker": "SPEAKER_00"}]
        """
        if not self._initialized:
            logger.error("Servicio de diarización no inicializado")
            return []
        
        try:
            logger.info(f"🎙️ Diarizando audio: {audio_path}")
            
            segments = await asyncio.to_thread(
                self._diarize_sync, audio_path, num_speakers, min_speakers, max_speakers
            )
            
            logger.info(f"  → {len(segments)} segmentos encontrados")
            return segments
            
        except Exception as e:
            logger.error(f"Error en diarización: {e}")
            return []
    
    def _diarize_sync(
        self,
        audio_path: str,
        num_speakers: int = None,
        min_speakers: int = None,
        max_speakers: int = None
    ) -> List[Dict[str, Any]]:
        """Ejecutar diarización (bloqueante)"""
        try:
            # Configurar parámetros
            params = {}
            if num_speakers is not None:
                params["num_speakers"] = num_speakers
            else:
                if min_speakers is not None or MIN_SPEAKERS is not None:
                    params["min_speakers"] = min_speakers or MIN_SPEAKERS
                if max_speakers is not None or MAX_SPEAKERS is not None:
                    params["max_speakers"] = max_speakers or MAX_SPEAKERS
            
            diarization = self._pipeline(audio_path, **params)
            
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    "start": round(turn.start, 3),
                    "end": round(turn.end, 3),
                    "speaker": speaker,
                    "duration": round(turn.end - turn.start, 3)
                })
            
            return segments
            
        except Exception as e:
            logger.error(f"Error en diarización sync: {e}")
            return []
    
    async def diarize_and_identify(
        self,
        audio_path: str,
        num_speakers: int = None,
        min_speakers: int = None,
        max_speakers: int = None,
        threshold: float = None,
        identify: bool = True
    ) -> Dict[str, Any]:
        """
        Diarización + identificación de speakers contra el banco de voces.
        
        Pipeline:
        1. Diarizar → segmentos con labels anónimos
        2. Para cada speaker único: extraer embedding del segmento más largo
        3. Buscar en banco de voces (cosine similarity)
        4. Si match > threshold: etiquetar con nombre real
        
        Args:
            audio_path: Ruta al audio
            num_speakers: Número exacto de speakers
            min_speakers: Mínimo de speakers
            max_speakers: Máximo de speakers
            threshold: Umbral de similitud (default: SIMILARITY_THRESHOLD)
            identify: Si True, buscar en banco. Si False, solo diarizar.
            
        Returns:
            Dict con segments, speakers, conteos
        """
        if threshold is None:
            threshold = SIMILARITY_THRESHOLD
        
        # Paso 1: Diarizar
        raw_segments = await self.diarize(
            audio_path, num_speakers, min_speakers, max_speakers
        )
        
        if not raw_segments:
            return {
                "segments": [],
                "speakers": {},
                "total_speakers": 0,
                "identified_speakers": 0,
                "unidentified_speakers": 0
            }
        
        # Obtener speakers únicos y sus segmentos
        speaker_segments: Dict[str, List[Dict]] = {}
        for seg in raw_segments:
            speaker = seg["speaker"]
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append(seg)
        
        total_speakers = len(speaker_segments)
        logger.info(f"  → {total_speakers} speakers detectados")
        
        # Si no queremos identificar, devolver con labels anónimos
        if not identify:
            speakers_info = {}
            for speaker, segs in speaker_segments.items():
                total_time = sum(s["duration"] for s in segs)
                speakers_info[speaker] = {
                    "total_time": round(total_time, 2),
                    "segment_count": len(segs),
                    "identified": False
                }
            
            return {
                "segments": raw_segments,
                "speakers": speakers_info,
                "total_speakers": total_speakers,
                "identified_speakers": 0,
                "unidentified_speakers": total_speakers
            }
        
        # Paso 2: Identificar cada speaker
        embedding_service = get_voice_embedding_service()
        es_service = get_elasticsearch_voice_service()
        
        if not embedding_service._initialized or not es_service.is_available:
            logger.warning("⚠️ Servicios de identificación no disponibles, devolviendo sin identificar")
            speakers_info = {}
            for speaker, segs in speaker_segments.items():
                total_time = sum(s["duration"] for s in segs)
                speakers_info[speaker] = {
                    "total_time": round(total_time, 2),
                    "segment_count": len(segs),
                    "identified": False
                }
            return {
                "segments": raw_segments,
                "speakers": speakers_info,
                "total_speakers": total_speakers,
                "identified_speakers": 0,
                "unidentified_speakers": total_speakers
            }
        
        # Para cada speaker, extraer embedding y buscar en banco
        speaker_mapping: Dict[str, Dict[str, Any]] = {}  # speaker_label → info
        identified_count = 0
        
        for speaker_label, segs in speaker_segments.items():
            total_time = sum(s["duration"] for s in segs)
            
            # Usar TODOS los segmentos del speaker para concatenación de waveforms.
            # Incluir segmentos >= 0.5s ya que serán concatenados para formar un audio
            # más largo. El filtro de duración mínima total se aplica después en el
            # método de extracción de embedding.
            sorted_segs = sorted(segs, key=lambda s: s["duration"], reverse=True)
            
            # Mínimo 0.5s por segmento para concatenación (no necesitan ser largos)
            MIN_SEG_FOR_CONCAT = 0.5
            best_segments = [
                (s["start"], s["end"]) 
                for s in sorted_segs
                if s["duration"] >= MIN_SEG_FOR_CONCAT
            ]
            
            if not best_segments or total_time < MIN_SEGMENT_DURATION_FOR_ID:
                # Segmentos demasiado cortos para identificar
                speaker_mapping[speaker_label] = {
                    "total_time": round(total_time, 2),
                    "segment_count": len(segs),
                    "identified": False,
                    "reason": "segments_too_short"
                }
                continue
            
            # Extraer embedding concatenando waveforms de los segmentos
            speaker_embedding = await embedding_service.extract_embeddings_from_segments(
                audio_path, best_segments
            )
            
            if speaker_embedding is None:
                speaker_mapping[speaker_label] = {
                    "total_time": round(total_time, 2),
                    "segment_count": len(segs),
                    "identified": False,
                    "reason": "embedding_extraction_failed"
                }
                continue
            
            logger.info(
                f"  🔊 {speaker_label}: embedding extraído de {len(best_segments)} segmentos "
                f"(norma={np.linalg.norm(speaker_embedding):.4f})"
            )
            
            # Buscar en banco de voces
            matches = await es_service.search_by_embedding(
                embedding=speaker_embedding,
                k=3,
                min_score=PROBABLE_THRESHOLD
            )
            
            if matches and matches[0]["similarity"] >= threshold:
                best_match = matches[0]
                speaker_mapping[speaker_label] = {
                    "person_id": best_match["person_id"],
                    "name": best_match["name"],
                    "role": best_match.get("role", ""),
                    "similarity": best_match["similarity"],
                    "total_time": round(total_time, 2),
                    "segment_count": len(segs),
                    "identified": True,
                    "alternatives": [
                        {"name": m["name"], "similarity": m["similarity"]}
                        for m in matches[1:] if m["similarity"] >= PROBABLE_THRESHOLD
                    ]
                }
                identified_count += 1
                logger.info(
                    f"  ✅ {speaker_label} → {best_match['name']} "
                    f"(sim={best_match['similarity']:.3f})"
                )
            else:
                # No identificado
                info = {
                    "total_time": round(total_time, 2),
                    "segment_count": len(segs),
                    "identified": False,
                    "reason": "below_threshold"
                }
                # Añadir posibles candidatos
                if matches:
                    info["possible_matches"] = [
                        {"name": m["name"], "similarity": m["similarity"]}
                        for m in matches
                    ]
                speaker_mapping[speaker_label] = info
                logger.info(f"  ❓ {speaker_label} → no identificado")
        
        # Paso 3: Reemplazar labels en segmentos
        final_segments = []
        for seg in raw_segments:
            new_seg = dict(seg)
            info = speaker_mapping.get(seg["speaker"], {})
            
            if info.get("identified"):
                new_seg["speaker"] = info["name"]
                new_seg["person_id"] = info.get("person_id")
                new_seg["confidence"] = info.get("similarity", 0)
                new_seg["role"] = info.get("role", "")
            else:
                new_seg["confidence"] = 0
            
            final_segments.append(new_seg)
        
        # Construir speakers_info con nombres reales como keys
        speakers_info = {}
        for label, info in speaker_mapping.items():
            key = info.get("name", label)
            speakers_info[key] = info
        
        result = {
            "segments": final_segments,
            "speakers": speakers_info,
            "total_speakers": total_speakers,
            "identified_speakers": identified_count,
            "unidentified_speakers": total_speakers - identified_count
        }
        
        logger.info(
            f"🎙️ Diarización completada: "
            f"{total_speakers} speakers, {identified_count} identificados"
        )
        
        return result
    
    async def identify_speaker(
        self,
        audio_path: str,
        threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        Identificar un speaker en un clip de audio corto.
        
        Útil para test rápido: subir clip → quién es.
        
        Args:
            audio_path: Ruta al clip de audio
            threshold: Umbral de similitud
            
        Returns:
            Lista de matches: [{"person_id", "name", "similarity", "role"}]
        """
        if threshold is None:
            threshold = PROBABLE_THRESHOLD
        
        embedding_service = get_voice_embedding_service()
        es_service = get_elasticsearch_voice_service()
        
        if not embedding_service._initialized:
            return []
        
        # Extraer embedding del clip
        embedding = await embedding_service.extract_embedding(audio_path)
        if embedding is None:
            return []
        
        # Buscar en banco
        if es_service.is_available:
            matches = await es_service.search_by_embedding(
                embedding=embedding,
                k=5,
                min_score=threshold
            )
            return matches
        
        return []


# ============================================================================
# SINGLETON
# ============================================================================

_instance: Optional[DiarizationService] = None

def get_diarization_service() -> DiarizationService:
    global _instance
    if _instance is None:
        _instance = DiarizationService()
    return _instance
