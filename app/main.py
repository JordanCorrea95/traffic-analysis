"""
API principal de FastAPI para análisis de tráfico vehicular.
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import shutil
import uuid
import logging
import asyncio
from datetime import datetime

from app.config import settings
from app.services import VideoProcessor

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Crear aplicación FastAPI
app = FastAPI(
    title="Traffic Analysis API",
    description="API para análisis de tráfico vehicular usando YOLOv26s",
    version="1.0.0"
)

# Asegurar que existen los directorios necesarios
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Diccionario para almacenar estado de procesamiento
processing_status = {}


@app.get("/")
async def root():
    """Endpoint raíz con información de la API."""
    return {
        "message": "Traffic Analysis API",
        "version": "1.0.0",
        "model": "YOLOv26s-seg",
        "endpoints": {
            "upload": "/api/upload",
            "download": "/api/download/{video_id}",
            "status": "/api/status/{video_id}",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Verifica el estado de salud de la API."""
    model_exists = settings.YOLO_MODEL_PATH.exists()

    return {
        "status": "healthy" if model_exists else "unhealthy",
        "model_loaded": model_exists,
        "model_path": str(settings.YOLO_MODEL_PATH),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Video file to process")
):
    """
    Endpoint para subir un video y procesarlo.

    Args:
        file: Archivo de video a procesar

    Returns:
        JSON con ID del video y estado inicial
    """
    logger.info(f"Recibiendo video: {file.filename}")

    # Validar tipo de archivo
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.SUPPORTED_VIDEO_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Formato no soportado. Use: {settings.SUPPORTED_VIDEO_FORMATS}"
        )

    # Validar tamaño (si es posible)
    file.file.seek(0, 2)  # Ir al final del archivo
    file_size = file.file.tell()
    file.file.seek(0)  # Volver al inicio

    if file_size > settings.MAX_VIDEO_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"Video muy grande. Máximo: {settings.MAX_VIDEO_SIZE_MB}MB"
        )

    # Generar ID único para el video
    video_id = str(uuid.uuid4())

    # Guardar archivo de entrada
    input_path = settings.UPLOAD_DIR / f"{video_id}{file_ext}"
    output_path = settings.PROCESSED_DIR / f"{video_id}_processed.mp4"

    try:
        # Guardar archivo subido
        with input_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Video guardado: {input_path}")

        # Inicializar estado de procesamiento
        processing_status[video_id] = {
            "status": "queued",
            "progress": 0,
            "total_frames": 0,
            "current_frame": 0,
            "statistics": None,
            "error": None,
            "created_at": datetime.now().isoformat(),
            "input_filename": file.filename
        }

        # Agregar tarea de procesamiento en background
        background_tasks.add_task(
            process_video_task,
            video_id,
            input_path,
            output_path
        )

        return {
            "video_id": video_id,
            "status": "queued",
            "message": "Video en cola para procesamiento",
            "filename": file.filename
        }

    except Exception as e:
        logger.error(f"Error al subir video: {e}")
        # Limpiar archivo si hubo error
        if input_path.exists():
            input_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status/{video_id}")
async def get_status(video_id: str):
    """
    Obtiene el estado de procesamiento de un video.

    Args:
        video_id: ID del video

    Returns:
        JSON con estado actual del procesamiento
    """
    if video_id not in processing_status:
        raise HTTPException(status_code=404, detail="Video no encontrado")

    status = processing_status[video_id]

    response = {
        "video_id": video_id,
        "status": status["status"],
        "progress": status["progress"],
        "input_filename": status.get("input_filename")
    }

    if status["status"] == "processing":
        response["current_frame"] = status["current_frame"]
        response["total_frames"] = status["total_frames"]

    if status["status"] == "completed":
        response["statistics"] = status["statistics"]

    if status["error"]:
        response["error"] = status["error"]

    return response


@app.get("/api/download/{video_id}")
async def download_video(video_id: str):
    """
    Descarga el video procesado.

    Args:
        video_id: ID del video

    Returns:
        Archivo de video procesado
    """
    if video_id not in processing_status:
        raise HTTPException(status_code=404, detail="Video no encontrado")

    status = processing_status[video_id]

    if status["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Video no está listo. Estado actual: {status['status']}"
        )

    output_path = settings.PROCESSED_DIR / f"{video_id}_processed.mp4"

    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Archivo procesado no encontrado")

    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        filename=f"processed_{status.get('input_filename', 'video.mp4')}"
    )


@app.delete("/api/delete/{video_id}")
async def delete_video(video_id: str):
    """
    Elimina un video y sus archivos asociados.

    Args:
        video_id: ID del video

    Returns:
        JSON con confirmación de eliminación
    """
    if video_id not in processing_status:
        raise HTTPException(status_code=404, detail="Video no encontrado")

    # Eliminar archivos
    for ext in settings.SUPPORTED_VIDEO_FORMATS + [".mp4"]:
        input_path = settings.UPLOAD_DIR / f"{video_id}{ext}"
        if input_path.exists():
            input_path.unlink()

    output_path = settings.PROCESSED_DIR / f"{video_id}_processed.mp4"
    if output_path.exists():
        output_path.unlink()

    # Eliminar del estado
    del processing_status[video_id]

    return {"message": "Video eliminado correctamente", "video_id": video_id}


async def process_video_task(
    video_id: str,
    input_path: Path,
    output_path: Path
):
    """
    Tarea de background para procesar el video.

    Args:
        video_id: ID del video
        input_path: Ruta del video de entrada
        output_path: Ruta del video de salida
    """
    try:
        logger.info(f"Iniciando procesamiento de video: {video_id}")

        # Actualizar estado
        processing_status[video_id]["status"] = "processing"

        # Callback para actualizar progreso
        def progress_callback(current_frame: int, total_frames: int):
            processing_status[video_id]["current_frame"] = current_frame
            processing_status[video_id]["total_frames"] = total_frames
            processing_status[video_id]["progress"] = int((current_frame / total_frames) * 100)

        # Crear procesador y ejecutar
        processor = VideoProcessor()
        result = await asyncio.to_thread(
            processor.process_video,
            input_path,
            output_path,
            progress_callback=progress_callback
        )

        # Actualizar estado a completado
        processing_status[video_id]["status"] = "completed"
        processing_status[video_id]["progress"] = 100
        processing_status[video_id]["statistics"] = result["statistics"]
        processing_status[video_id]["completed_at"] = datetime.now().isoformat()

        logger.info(f"Procesamiento completado: {video_id}")

    except Exception as e:
        logger.error(f"Error al procesar video {video_id}: {e}")
        processing_status[video_id]["status"] = "failed"
        processing_status[video_id]["error"] = str(e)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
