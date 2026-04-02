# Traffic Analysis API

Aplicación para conteo, identificación y segmentación de vehículos por dirección y carril usando FastAPI, OpenCV y YOLOv26s.

## Características

- ✅ Detección y segmentación de vehículos usando YOLOv26s
- ✅ Seguimiento multi-objeto (tracking) de vehículos
- ✅ Conteo por carril y dirección (norte, sur, este, oeste)
- ✅ Procesamiento asíncrono de videos
- ✅ API REST con FastAPI
- ✅ Visualización con bounding boxes y máscaras de segmentación
- ✅ Estadísticas detalladas por clase de vehículo

## Requisitos

- Python 3.10+
- CUDA (opcional, para aceleración GPU)

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

### 3. Descargar modelo YOLOv26s-seg

El modelo ya ha sido descargado en `models/yolo26s-seg.pt`. Si necesitas descargarlo nuevamente:

```bash
curl -L -o models/yolo26s-seg.pt "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-seg.pt"
```

## Estructura del Proyecto

```
traffic-analysis/
├── app/
│   ├── config/          # Configuración de la aplicación
│   ├── models/          # Modelos de datos
│   ├── services/        # Lógica de negocio
│   │   ├── vehicle_detector.py    # Detector YOLO
│   │   └── video_processor.py     # Procesador de video
│   ├── utils/           # Utilidades
│   └── main.py          # API FastAPI
├── models/              # Modelos YOLO
├── storage/
│   ├── uploads/         # Videos subidos
│   └── processed/       # Videos procesados
└── requirements.txt
```

## Configuración

Edita `app/config/settings.py` para ajustar:

- **Umbrales de detección**: `YOLO_CONFIDENCE_THRESHOLD`, `YOLO_IOU_THRESHOLD`
- **Configuración de carriles**: `LANE_LINES` (ajusta las coordenadas según tu video)
- **Clases de vehículos**: `VEHICLE_CLASSES`
- **Tamaño máximo de video**: `MAX_VIDEO_SIZE_MB`

### Configurar Carriles

Por defecto, la configuración incluye líneas de ejemplo. Ajústalas según tu video:

```python
LANE_LINES = {
    "horizontal": [
        {"y": 300, "direction": "north"},
        {"y": 500, "direction": "south"}
    ],
    "vertical": [
        {"x": 400, "direction": "west"},
        {"x": 600, "direction": "east"}
    ]
}
```

## Ejecutar la API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

La API estará disponible en: `http://localhost:8000`

Documentación interactiva: `http://localhost:8000/docs`

## Endpoints

### 1. Health Check

```bash
GET /health
```

Verifica el estado de la API y si el modelo está cargado.

### 2. Subir Video

```bash
POST /api/upload
```

Sube un video para procesamiento.

**Ejemplo con curl:**

```bash
curl -X POST "http://localhost:8000/api/upload" \
  -F "file=@/ruta/a/tu/video.mp4"
```

**Respuesta:**

```json
{
  "video_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "message": "Video en cola para procesamiento",
  "filename": "video.mp4"
}
```

### 3. Consultar Estado

```bash
GET /api/status/{video_id}
```

Consulta el progreso del procesamiento.

**Ejemplo:**

```bash
curl "http://localhost:8000/api/status/550e8400-e29b-41d4-a716-446655440000"
```

**Respuesta (en progreso):**

```json
{
  "video_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "progress": 45,
  "current_frame": 450,
  "total_frames": 1000,
  "input_filename": "video.mp4"
}
```

**Respuesta (completado):**

```json
{
  "video_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "progress": 100,
  "input_filename": "video.mp4",
  "statistics": {
    "total_vehicles": 57,
    "by_class": {
      "car": 45,
      "truck": 8,
      "bus": 4
    },
    "by_direction": {
      "north": 15,
      "south": 18,
      "east": 12,
      "west": 12
    },
    "by_lane": {
      "lane_0": 20,
      "lane_1": 37
    }
  }
}
```

### 4. Descargar Video Procesado

```bash
GET /api/download/{video_id}
```

Descarga el video procesado.

**Ejemplo:**

```bash
curl -L "http://localhost:8000/api/download/550e8400-e29b-41d4-a716-446655440000" \
  -o video_procesado.mp4
```

### 5. Eliminar Video

```bash
DELETE /api/delete/{video_id}
```

Elimina un video y sus archivos asociados.

## Visualización del Video Procesado

Cada vehículo detectado se muestra con:

- **Máscara de segmentación**: Overlay semitransparente de color
- **Bounding box**: Rectángulo alrededor del vehículo
- **Etiqueta**: Clase del vehículo (CAR, TRUCK, BUS, MOTORCYCLE) con probabilidad
- **ID de tracking**: Identificador único del vehículo
- **Centroide**: Punto central del vehículo
- **Líneas de carril**: Líneas de referencia para conteo
- **Estadísticas en tiempo real**: Contador en pantalla

### Colores por Clase

- 🟢 **Car**: Verde
- 🔴 **Truck**: Rojo
- 🟠 **Bus**: Naranja
- 🔵 **Motorcycle**: Azul

## Ejemplo de Uso Completo

```bash
# 1. Subir video
response=$(curl -s -X POST "http://localhost:8000/api/upload" \
  -F "file=@traffic_video.mp4")

video_id=$(echo $response | jq -r '.video_id')

# 2. Monitorear progreso
while true; do
  status=$(curl -s "http://localhost:8000/api/status/$video_id" | jq -r '.status')
  progress=$(curl -s "http://localhost:8000/api/status/$video_id" | jq -r '.progress')
  echo "Status: $status - Progress: $progress%"

  if [ "$status" = "completed" ]; then
    break
  fi

  sleep 5
done

# 3. Descargar video procesado
curl -L "http://localhost:8000/api/download/$video_id" -o processed_video.mp4

# 4. Ver estadísticas
curl -s "http://localhost:8000/api/status/$video_id" | jq '.statistics'
```

## Arquitectura

### Componentes Principales

1. **VehicleDetector** (`app/services/vehicle_detector.py`)
   - Ejecuta YOLOv26s para detección y segmentación
   - Gestiona el tracking de vehículos
   - Detecta cruces de línea y actualiza contadores

2. **VideoProcessor** (`app/services/video_processor.py`)
   - Procesa videos frame por frame
   - Genera visualizaciones
   - Coordina detección y tracking

3. **FastAPI App** (`app/main.py`)
   - Endpoints REST
   - Procesamiento asíncrono en background
   - Gestión de estado de trabajos

### Flujo de Procesamiento

```
Upload Video → Save to Storage → Queue Processing →
  ↓
Process Frames (YOLO Detection → Tracking → Count Crossing) →
  ↓
Annotate Frames → Write Output Video → Update Stats →
  ↓
Complete → Ready for Download
```

## Personalización

### Añadir Nuevas Clases de Vehículos

Edita `app/config/settings.py`:

```python
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    # Añade más clases del dataset COCO
}
```

### Ajustar Parámetros de Detección

```python
YOLO_CONFIDENCE_THRESHOLD = 0.5  # Aumentar para menos falsos positivos
YOLO_IOU_THRESHOLD = 0.45
TRACKING_MAX_AGE = 30  # Frames sin detección antes de eliminar
```

## Troubleshooting

### El modelo no se encuentra

Asegúrate de haber descargado el modelo:
```bash
ls -lh models/yolo26s-seg.pt
```

### Video muy grande

Ajusta `MAX_VIDEO_SIZE_MB` en `settings.py` o comprime el video antes de subirlo.

### Detecciones incorrectas

- Ajusta `YOLO_CONFIDENCE_THRESHOLD` (aumentar para menos detecciones)
- Verifica la configuración de `LANE_LINES` para tu video específico

## Tecnologías Utilizadas

- **FastAPI**: Framework web moderno y rápido
- **YOLOv26s**: Modelo de detección y segmentación
- **OpenCV**: Procesamiento de video
- **Ultralytics**: Biblioteca YOLO
- **NumPy**: Operaciones numéricas

## Licencia

MIT

## Autor

Desarrollado usando YOLOv26s y buenas prácticas de desarrollo.
