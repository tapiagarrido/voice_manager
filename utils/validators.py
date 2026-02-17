"""
Utilidades para validación de archivos de audio y otros
"""
from pathlib import Path
from typing import List
from fastapi import HTTPException, UploadFile

from config import (
    MIN_AUDIO_CLIPS,
    MAX_AUDIO_CLIPS,
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE,
    MIN_CLIP_DURATION,
    MAX_CLIP_DURATION,
    SPEAKER_ROLES
)


def validate_audio_clips_count(audio_files: List[UploadFile]):
    """Validar número de clips de audio"""
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


def validate_speaker_role(role: str) -> str:
    """Validar y normalizar rol de speaker"""
    if role not in SPEAKER_ROLES:
        return "otro"
    return role


def validate_audio_extension(filename: str) -> str:
    """Validar extensión de archivo de audio"""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Formato no soportado: {ext}"
        )
    return ext


def validate_file_size(content: bytes, filename: str):
    """Validar tamaño de archivo"""
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Archivo {filename} muy grande ({len(content) // 1024 // 1024}MB). Máximo: {MAX_FILE_SIZE // 1024 // 1024}MB"
        )


def validate_audio_duration(duration: float) -> tuple[bool, str]:
    """
    Validar duración de clip de audio.
    Retorna (es_valido, mensaje_error)
    """
    if duration < MIN_CLIP_DURATION:
        return False, f"Demasiado corto ({duration:.1f}s, mínimo {MIN_CLIP_DURATION}s)"
    
    if duration > MAX_CLIP_DURATION:
        return False, f"Demasiado largo ({duration:.1f}s, máximo {MAX_CLIP_DURATION}s)"
    
    return True, ""
