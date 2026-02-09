"""
Voice Embedding Service
========================
Genera embeddings vectoriales de voz usando SpeechBrain ECAPA-TDNN.

- Modelo: speechbrain/spkrec-ecapa-voxceleb
- Embeddings: 192 dimensiones
- Normalizado a norma unitaria (para similitud coseno directa)

Los embeddings capturan las características biométricas de la voz
independientemente del contenido hablado (text-independent speaker verification).
"""
import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
import torchaudio

from config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIMS,
    DEVICE,
    MODELS_DIR,
    MIN_CLIP_DURATION,
    MIN_SPEECH_RATIO,
    MIN_SEGMENT_DURATION_FOR_ID
)

logger = logging.getLogger(__name__)


class VoiceEmbeddingService:
    """
    Servicio de embeddings de voz usando SpeechBrain ECAPA-TDNN.
    
    ECAPA-TDNN produce vectores de 192 dimensiones que representan
    la identidad vocal de un hablante. Estos vectores se comparan
    con similitud coseno para verificación/identificación.
    """
    
    def __init__(self):
        self._initialized = False
        self._model = None
        self._device = DEVICE
        
    async def initialize(self) -> bool:
        """Inicializar el modelo de embeddings"""
        try:
            logger.info(f"🧠 Cargando modelo de embeddings de voz: {EMBEDDING_MODEL}")
            
            # Cargar modelo en thread separado (es bloqueante)
            self._model = await asyncio.to_thread(self._load_model)
            
            if self._model is not None:
                self._initialized = True
                logger.info(f"✅ Voice Embedding Service inicializado ({EMBEDDING_DIMS} dims, device={self._device})")
                return True
            else:
                logger.error("❌ No se pudo cargar el modelo de embeddings")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error inicializando Voice Embedding Service: {e}")
            return False
    
    def _load_model(self):
        """Cargar modelo SpeechBrain (bloqueante)"""
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            
            model = EncoderClassifier.from_hparams(
                source=EMBEDDING_MODEL,
                savedir=str(MODELS_DIR / "ecapa_tdnn"),
                run_opts={"device": self._device}
            )
            
            logger.info(f"  → Modelo cargado en {self._device}")
            return model
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            
            # Fallback a CPU si CUDA falla
            if self._device == "cuda":
                logger.warning("⚠️ Intentando fallback a CPU...")
                try:
                    from speechbrain.inference.speaker import EncoderClassifier
                    
                    self._device = "cpu"
                    model = EncoderClassifier.from_hparams(
                        source=EMBEDDING_MODEL,
                        savedir=str(MODELS_DIR / "ecapa_tdnn"),
                        run_opts={"device": "cpu"}
                    )
                    logger.info("  → Modelo cargado en CPU (fallback)")
                    return model
                except Exception as e2:
                    logger.error(f"Error en fallback CPU: {e2}")
            
            return None
    
    async def extract_embedding(self, audio_path: str) -> Optional[np.ndarray]:
        """
        Extraer embedding de un archivo de audio completo.
        
        Args:
            audio_path: Ruta al archivo de audio
            
        Returns:
            np.ndarray de shape (192,) normalizado, o None si falla
        """
        if not self._initialized:
            logger.error("Servicio no inicializado")
            return None
            
        try:
            embedding = await asyncio.to_thread(
                self._extract_embedding_sync, audio_path
            )
            return embedding
        except Exception as e:
            logger.error(f"Error extrayendo embedding de {audio_path}: {e}")
            return None
    
    def _extract_embedding_sync(self, audio_path: str) -> Optional[np.ndarray]:
        """Extraer embedding (bloqueante)"""
        try:
            # Cargar audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convertir a mono si es estéreo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resamplear a 16kHz si es necesario (requerido por ECAPA-TDNN)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Verificar duración mínima
            duration = waveform.shape[1] / 16000
            if duration < MIN_CLIP_DURATION:
                logger.warning(f"Audio demasiado corto: {duration:.1f}s (mínimo {MIN_CLIP_DURATION}s)")
                return None
            
            # Extraer embedding con SpeechBrain
            # El modelo espera waveform de shape (batch, time)
            embedding = self._model.encode_batch(waveform)
            
            # Convertir a numpy y aplanar: (1, 1, 192) → (192,)
            emb_np = embedding.squeeze().cpu().numpy()
            
            # Normalizar a norma unitaria
            norm = np.linalg.norm(emb_np)
            if norm > 0:
                emb_np = emb_np / norm
            
            return emb_np
            
        except Exception as e:
            logger.error(f"Error en extracción de embedding: {e}")
            return None
    
    async def extract_embedding_from_segment(
        self,
        audio_path: str,
        start: float,
        end: float
    ) -> Optional[np.ndarray]:
        """
        Extraer embedding de un segmento temporal específico.
        
        Args:
            audio_path: Ruta al archivo de audio
            start: Tiempo de inicio en segundos
            end: Tiempo de fin en segundos
            
        Returns:
            np.ndarray de shape (192,) normalizado, o None
        """
        if not self._initialized:
            return None
            
        try:
            embedding = await asyncio.to_thread(
                self._extract_segment_embedding_sync, audio_path, start, end
            )
            return embedding
        except Exception as e:
            logger.error(f"Error extrayendo embedding de segmento [{start:.1f}-{end:.1f}]: {e}")
            return None
    
    def _extract_segment_embedding_sync(
        self, audio_path: str, start: float, end: float
    ) -> Optional[np.ndarray]:
        """Extraer embedding de segmento (bloqueante)"""
        try:
            # Verificar duración mínima
            duration = end - start
            if duration < MIN_SEGMENT_DURATION_FOR_ID:
                return None
            
            # Cargar audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convertir a mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resamplear a 16kHz si es necesario
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                sample_rate = 16000
            
            # Recortar al segmento
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            
            # Validar rango
            if start_sample >= waveform.shape[1]:
                return None
            end_sample = min(end_sample, waveform.shape[1])
            
            segment = waveform[:, start_sample:end_sample]
            
            # Verificar que el segmento tiene contenido
            if segment.shape[1] < int(MIN_SEGMENT_DURATION_FOR_ID * sample_rate):
                return None
            
            # Extraer embedding
            embedding = self._model.encode_batch(segment)
            emb_np = embedding.squeeze().cpu().numpy()
            
            # Normalizar
            norm = np.linalg.norm(emb_np)
            if norm > 0:
                emb_np = emb_np / norm
            
            return emb_np
            
        except Exception as e:
            logger.error(f"Error extrayendo embedding de segmento: {e}")
            return None
    
    async def extract_embeddings_from_segments(
        self,
        audio_path: str,
        segments: List[Tuple[float, float]]
    ) -> Optional[np.ndarray]:
        """
        Extraer embedding robusto concatenando waveforms de múltiples segmentos.
        
        En lugar de extraer embeddings individuales de segmentos cortos (ruidosos)
        y promediarlos, este método CONCATENA las waveforms de todos los segmentos
        del mismo speaker y extrae UN SOLO embedding del audio combinado.
        Esto produce un embedding mucho más estable y representativo.
        
        Args:
            audio_path: Ruta al archivo de audio
            segments: Lista de (start, end) en segundos
            
        Returns:
            np.ndarray normalizado de shape (192,), o None
        """
        if not self._initialized:
            return None

        try:
            embedding = await asyncio.to_thread(
                self._extract_concatenated_embedding_sync, audio_path, segments
            )
            return embedding
        except Exception as e:
            logger.error(f"Error extrayendo embedding concatenado: {e}")
            return None

    def _extract_concatenated_embedding_sync(
        self,
        audio_path: str,
        segments: List[Tuple[float, float]]
    ) -> Optional[np.ndarray]:
        """Concatenar waveforms de segmentos y extraer un solo embedding (bloqueante)"""
        try:
            # Cargar audio una sola vez
            waveform, sample_rate = torchaudio.load(audio_path)

            # Convertir a mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Resamplear a 16kHz si es necesario
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                sample_rate = 16000

            # Concatenar todos los segmentos válidos
            # Para concatenación, aceptamos segmentos desde 0.5s ya que
            # lo que importa es la duración TOTAL concatenada, no la individual
            MIN_SEG_FOR_CONCAT = 0.5
            concatenated_parts = []
            total_duration = 0.0

            for start, end in segments:
                seg_duration = end - start
                if seg_duration < MIN_SEG_FOR_CONCAT:
                    continue

                start_sample = int(start * sample_rate)
                end_sample = min(int(end * sample_rate), waveform.shape[1])

                if start_sample >= waveform.shape[1]:
                    continue

                segment = waveform[:, start_sample:end_sample]
                if segment.shape[1] > 0:
                    concatenated_parts.append(segment)
                    total_duration += seg_duration

            if not concatenated_parts:
                logger.warning("No se encontraron segmentos válidos para concatenar")
                return None

            # Concatenar todas las partes en un solo tensor
            combined_waveform = torch.cat(concatenated_parts, dim=1)

            # Verificar duración mínima total
            combined_duration = combined_waveform.shape[1] / sample_rate
            if combined_duration < MIN_SEGMENT_DURATION_FOR_ID:
                logger.warning(
                    f"Audio concatenado demasiado corto: {combined_duration:.1f}s"
                )
                return None

            # Extraer UN SOLO embedding del audio concatenado
            embedding = self._model.encode_batch(combined_waveform)
            emb_np = embedding.squeeze().cpu().numpy().astype(np.float32)

            # Normalizar
            norm = np.linalg.norm(emb_np)
            if norm > 0:
                emb_np = emb_np / norm

            logger.info(
                f"  📐 Embedding concatenado: {len(concatenated_parts)} segmentos, "
                f"{combined_duration:.1f}s total, norma={np.linalg.norm(emb_np):.6f}"
            )

            return emb_np

        except Exception as e:
            logger.error(f"Error en extracción concatenada: {e}")
            return None
    
    @staticmethod
    def compare_embeddings(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calcular similitud coseno entre dos embeddings.
        
        Como ambos están normalizados, el dot product = coseno.
        
        Args:
            emb1: Embedding 1 (192,)
            emb2: Embedding 2 (192,)
            
        Returns:
            Similitud coseno entre 0.0 y 1.0
        """
        similarity = float(np.dot(emb1, emb2))
        # Clampear a [0, 1] por posibles errores de precisión
        return max(0.0, min(1.0, similarity))
    
    async def get_audio_info(self, audio_path: str) -> dict:
        """
        Obtener información del audio (duración, sample_rate, etc.)
        
        Returns:
            Dict con duration, sample_rate, channels
        """
        try:
            info = await asyncio.to_thread(torchaudio.info, audio_path)
            return {
                "duration": info.num_frames / info.sample_rate,
                "sample_rate": info.sample_rate,
                "channels": info.num_channels,
                "num_frames": info.num_frames
            }
        except Exception as e:
            logger.error(f"Error obteniendo info de audio: {e}")
            return {"duration": 0, "sample_rate": 0, "channels": 0, "num_frames": 0}

    async def extract_embedding_for_registration(
        self,
        audio_path: str,
        diarization_pipeline=None
    ) -> Optional[np.ndarray]:
        """
        Extraer embedding robusto para REGISTRO de una persona.
        
        A diferencia de extract_embedding (que usa todo el audio), este método:
        1. Si hay pipeline de diarización disponible, diariza el clip
        2. Identifica al hablante dominante (mayor tiempo de habla)
        3. Extrae embeddings solo de los segmentos de ese hablante
        4. Promedia para un embedding limpio de una sola persona
        
        Esto evita que clips con múltiples hablantes contaminen el embedding.
        Si no hay diarización disponible, usa el audio completo (fallback).
        
        Args:
            audio_path: Ruta al archivo de audio
            diarization_pipeline: Pipeline de Pyannote (opcional)
            
        Returns:
            np.ndarray de shape (192,) normalizado, o None
        """
        if not self._initialized:
            logger.error("Servicio no inicializado")
            return None

        try:
            # Si tenemos pipeline de diarización, intentar aislar hablante dominante
            if diarization_pipeline is not None:
                return await asyncio.to_thread(
                    self._extract_dominant_speaker_embedding_sync,
                    audio_path, diarization_pipeline
                )
            
            # Fallback: usar audio completo
            logger.info("  ℹ️ Sin diarización, usando audio completo para registro")
            return await self.extract_embedding(audio_path)
            
        except Exception as e:
            logger.error(f"Error en extracción para registro: {e}")
            # Fallback al método original
            return await self.extract_embedding(audio_path)

    def _extract_dominant_speaker_embedding_sync(
        self,
        audio_path: str,
        diarization_pipeline
    ) -> Optional[np.ndarray]:
        """
        Diarizar clip, encontrar hablante dominante y extraer su embedding.
        (Bloqueante)
        """
        try:
            # Diarizar el clip
            diarization = diarization_pipeline(audio_path)
            
            # Agrupar segmentos por speaker y calcular tiempo total
            speaker_times = {}
            speaker_segments = {}
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                duration = turn.end - turn.start
                if speaker not in speaker_times:
                    speaker_times[speaker] = 0.0
                    speaker_segments[speaker] = []
                speaker_times[speaker] += duration
                speaker_segments[speaker].append((turn.start, turn.end))
            
            if not speaker_times:
                logger.warning("  ⚠️ Diarización no encontró hablantes en el clip")
                return self._extract_embedding_sync(audio_path)
            
            # Encontrar hablante dominante
            dominant_speaker = max(speaker_times, key=speaker_times.get)
            dominant_time = speaker_times[dominant_speaker]
            total_speakers = len(speaker_times)
            
            logger.info(
                f"  🎯 Clip diarizado: {total_speakers} hablantes detectados, "
                f"dominante={dominant_speaker} ({dominant_time:.1f}s)"
            )
            
            if total_speakers > 1:
                other_time = sum(t for s, t in speaker_times.items() if s != dominant_speaker)
                logger.info(f"  ⚠️ Filtrando {other_time:.1f}s de otros hablantes")
            
            # Extraer embeddings de los segmentos del hablante dominante
            segments = speaker_segments[dominant_speaker]
            
            # Cargar audio una sola vez
            waveform, sample_rate = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                sample_rate = 16000
            
            embeddings = []
            for start, end in segments:
                seg_duration = end - start
                if seg_duration < MIN_SEGMENT_DURATION_FOR_ID:
                    continue
                
                start_sample = int(start * sample_rate)
                end_sample = min(int(end * sample_rate), waveform.shape[1])
                segment = waveform[:, start_sample:end_sample]
                
                if segment.shape[1] < int(MIN_SEGMENT_DURATION_FOR_ID * sample_rate):
                    continue
                
                embedding = self._model.encode_batch(segment)
                emb_np = embedding.squeeze().cpu().numpy().astype(np.float32)
                
                norm = np.linalg.norm(emb_np)
                if norm > 0:
                    emb_np = emb_np / norm
                    embeddings.append(emb_np)
            
            if not embeddings:
                logger.warning("  ⚠️ No se obtuvieron embeddings del hablante dominante, usando audio completo")
                return self._extract_embedding_sync(audio_path)
            
            # Promediar y normalizar
            avg_embedding = np.mean(embeddings, axis=0).astype(np.float32)
            norm = np.linalg.norm(avg_embedding)
            if norm > 0:
                avg_embedding = avg_embedding / norm
            
            logger.info(
                f"  ✅ Embedding de registro: {len(embeddings)} segmentos del hablante dominante "
                f"({dominant_time:.1f}s de habla limpia)"
            )
            
            return avg_embedding
            
        except Exception as e:
            logger.error(f"Error en extracción de hablante dominante: {e}")
            return self._extract_embedding_sync(audio_path)


# ============================================================================
# SINGLETON
# ============================================================================

_instance: Optional[VoiceEmbeddingService] = None

def get_voice_embedding_service() -> VoiceEmbeddingService:
    global _instance
    if _instance is None:
        _instance = VoiceEmbeddingService()
    return _instance
