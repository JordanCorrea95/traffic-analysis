"""
Configuración de la aplicación de análisis de tráfico.
"""
from pathlib import Path

# Directorios base
BASE_DIR = Path(__file__).resolve().parent.parent.parent
STORAGE_DIR = BASE_DIR / "storage"
MODELS_DIR = BASE_DIR / "models"

# Directorios de almacenamiento
UPLOAD_DIR = STORAGE_DIR / "uploads"
PROCESSED_DIR = STORAGE_DIR / "processed"

# Modelo YOLO
YOLO_MODEL_PATH = MODELS_DIR / "yolo26m-seg.pt"
YOLO_CONFIDENCE_THRESHOLD = 0.5
YOLO_IOU_THRESHOLD = 0.5

# Clases de vehículos (COCO dataset)
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# Configuración de video
MAX_VIDEO_SIZE_MB = 500
SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv"]

# Configuración de procesamiento
FPS_PROCESS = 30  # FPS para procesar
TRACKING_MAX_AGE = 90  # Frames máximos sin detección antes de eliminar track (3 segundos a 30fps)

# Colores para visualización (BGR)
COLORS = {
    "car": (255, 0, 0),         # Verde
    "motorcycle": (0, 255, 0),  # Azul
    "bus": (0, 165, 255),       # Naranja
    "truck": (0, 0, 255)        # Rojo
}

# Configuración de carriles
# Estas líneas deben ajustarse según el video específico
LANE_LINES = {
    "horizontal": [
        {"y": 450, "name": "REFERENCIA"}  # Línea de referencia para conteo
    ],
    "vertical": []
}
