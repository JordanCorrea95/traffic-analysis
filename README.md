# Traffic Analysis API

Aplicación para conteo, identificación y segmentación de vehículos por dirección y carril usando FastAPI, OpenCV y YOLOv26m-seg.

## Características

- ✅ Detección y segmentación de vehículos usando YOLOv26m-seg
- ✅ Seguimiento multi-objeto (tracking) de vehículos con persistencia temporal
- ✅ **Estabilización de Clase (Class Smoothing)**: Algoritmo de ponderación lineal para evitar fluctuaciones de identificación (ej. auto ↔ camión)
- ✅ Conteo por carril y dirección (norte, sur, este, oeste) mediante cruce de línea de referencia
- ✅ Procesamiento asíncrono de videos mediante cola de tareas
- ✅ API REST con FastAPI y documentación automática
- ✅ Visualización rica: bounding boxes, máscaras de segmentación, IDs de tracking y estadísticas en pantalla
- ✅ Estadísticas detalladas exportables por clase, dirección y combinadas

## Requisitos

- Python 3.10+
- CUDA (recomendado para procesamiento fluido en GPU)

## Instalación

### 1. Clonar repositorio y crear entorno virtual

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Descargar modelo YOLOv26m-seg

Asegúrate de colocar el modelo en la carpeta `models/`:

```bash
mkdir -p models
# Descarga el modelo oficial de Ultralytics (o usa el configurado en settings.py)
curl -L -o models/yolo26m-seg.pt "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-seg.pt" 
```
*Nota: El nombre del archivo debe coincidir con `YOLO_MODEL_PATH` en `app/config/settings.py`.*

## Estructura del Proyecto

```
traffic-analysis/
├── app/
│   ├── config/          # Configuración (settings.py)
│   ├── models/          # Modelos de datos (Pydantic, dataclasses)
│   ├── services/        # Lógica de negocio
│   │   ├── vehicle_detector.py    # Detector YOLO y Lógica de Conteo
│   │   └── video_processor.py     # Procesador y Anotador de Video
│   ├── utils/           # Utilidades de dibujo y video
│   └── main.py          # Puntos de entrada de la API FastAPI
├── models/              # Modelos YOLO (.pt)
├── storage/
│   ├── uploads/         # Videos originales
│   └── processed/       # Videos resultantes con anotaciones
└── requirements.txt
```

## Configuración

Edita `app/config/settings.py` para personalizar el comportamiento:

- **Detección**: `YOLO_CONFIDENCE_THRESHOLD` (confianza mínima), `YOLO_IOU_THRESHOLD`.
- **Estabilización**: `TRACKING_MAX_AGE` (frames para mantener un track perdido).
- **Líneas de Conteo**: `LANE_LINES` define las coordenadas Y (horizontal) o X (vertical) de las líneas de referencia.
- **Clases**: `VEHICLE_CLASSES` mapea IDs de COCO a nombres legibles.

### Configurar Línea de Referencia

Por defecto, se usa una línea horizontal para el conteo:

```python
LANE_LINES = {
    "horizontal": [
        {"y": 450, "name": "REFERENCIA"}
    ],
    "vertical": []
}
```

## Ejecutar la API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

La API estará disponible en: `http://localhost:8000`
Documentación interactiva (Swagger): `http://localhost:8000/docs`

## Endpoints Principales

### 1. Health Check
`GET /health` - Verifica el estado del sistema.

### 2. Subir Video
`POST /api/upload` - Sube un archivo `.mp4`, `.avi` o `.mov`.
```bash
curl -X POST "http://localhost:8000/api/upload" -F "file=@mi_trafico.mp4"
```

### 3. Consultar Estado y Estadísticas
`GET /api/status/{video_id}` - Devuelve el progreso (0-100%) y, al finalizar, el resumen estadístico.

### 4. Descargar Resultado
`GET /api/download/{video_id}` - Descarga el video anotado.

## Visualización y Colores

El video procesado incluye:
- **Máscaras**: Overlay al 30% de opacidad.
- **Centroides**: 
    - ⚪ con borde 🟢: Vehículo ya contabilizado.
    - Color de clase sólido: Vehículo aún no contabilizado.
- **Estabilización**: La etiqueta muestra la "Clase Estable" calculada por el historial.

### Código de Colores (BGR)
- 🔵 **Car**: Azul `(255, 0, 0)`
- 🟢 **Motorcycle**: Verde `(0, 255, 0)`
- 🟠 **Bus**: Naranja `(0, 165, 255)`
- 🔴 **Truck**: Rojo `(0, 0, 255)`

## Lógica de Estabilización de Clase (Technical Detail)

Para evitar el "parpadeo" de etiquetas (ej. que un auto sea detectado como camión en un solo frame ruidoso), el sistema implementa **Linear Weighted Smoothing**:
1. Se mantiene un historial de las últimas 30 clasificaciones para cada `track_id`.
2. Al momento del cruce de línea, se calcula la clase predominante asignando pesos incrementales: la detección más reciente tiene 30 veces más peso que la primera.
3. Esto garantiza que el conteo estadístico sea extremadamente robusto y refleje la identidad real del vehículo.

## Tecnologías

- **FastAPI** & **Uvicorn**
- **YOLOv26m-seg** (vía Ultralytics)
- **OpenCV** & **NumPy**

## Licencia
MIT
