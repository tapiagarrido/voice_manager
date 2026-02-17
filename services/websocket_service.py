"""
Servicio para gestión de WebSocket y progreso en tiempo real
"""
import logging
from typing import Dict
from fastapi import WebSocket, WebSocketDisconnect

from models.schemas import ProcessingProgress

logger = logging.getLogger(__name__)

# Almacén de conexiones WebSocket activas
active_connections: Dict[str, WebSocket] = {}


async def websocket_progress_endpoint(websocket: WebSocket, registration_id: str):
    """Handler para WebSocket de progreso"""
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
