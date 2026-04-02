"""
Servicio de detección y seguimiento de vehículos usando YOLOv26s.
"""
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Optional
from ultralytics import YOLO

from app.config import settings
from app.models import VehicleDetection, VehicleTrack, TrafficStats

logger = logging.getLogger(__name__)


class VehicleDetector:
    """Detector de vehículos usando YOLO con segmentación."""

    def __init__(self, model_path: Optional[Path] = None):
        """
        Inicializa el detector de vehículos.

        Args:
            model_path: Ruta al modelo YOLO. Si es None, usa el de configuración.
        """
        self.model_path = model_path or settings.YOLO_MODEL_PATH

        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {self.model_path}")

        # Cargar modelo YOLO
        self.model = YOLO(str(self.model_path))

        # Diccionario de tracks activos
        self.tracks: Dict[int, VehicleTrack] = {}

        # Estadísticas
        self.stats = TrafficStats()

        # Contador de IDs únicos
        self._next_track_id = 0

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = None,
        iou_threshold: float = None
    ) -> List[VehicleDetection]:
        """
        Detecta vehículos en un frame.

        Args:
            frame: Frame de video
            conf_threshold: Umbral de confianza
            iou_threshold: Umbral de IoU

        Returns:
            Lista de detecciones de vehículos
        """
        conf = conf_threshold or settings.YOLO_CONFIDENCE_THRESHOLD
        iou = iou_threshold or settings.YOLO_IOU_THRESHOLD

        # Ejecutar detección con tracking
        results = self.model.track(
            frame,
            conf=conf,
            iou=iou,
            persist=True,
            classes=list(settings.VEHICLE_CLASSES.keys()),
            verbose=False
        )

        detections = []

        if results and len(results) > 0:
            result = results[0]

            # Obtener boxes, máscaras y IDs de tracking
            boxes = result.boxes
            masks = result.masks if hasattr(result, 'masks') and result.masks is not None else None

            if boxes is not None and len(boxes) > 0:
                for i, box in enumerate(boxes):
                    # Extraer información de la detección
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    track_id = int(box.id[0]) if box.id is not None else self._next_track_id
                    self._next_track_id = max(self._next_track_id, track_id + 1)

                    # Obtener nombre de clase
                    class_name = settings.VEHICLE_CLASSES.get(cls, "unknown")

                    # Extraer máscara si está disponible
                    mask_points = None
                    if masks is not None and i < len(masks):
                        mask = masks[i].xy[0]  # Puntos del contorno de la máscara
                        if len(mask) > 0:
                            mask_points = mask.astype(int).tolist()

                    # Crear detección
                    detection = VehicleDetection(
                        track_id=track_id,
                        class_name=class_name,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        mask=mask_points
                    )

                    detections.append(detection)

        return detections

    def update_tracks(self, detections: List[VehicleDetection]):
        """
        Actualiza los tracks de vehículos con las nuevas detecciones.

        Args:
            detections: Lista de detecciones
        """
        detected_ids = set()

        for detection in detections:
            detected_ids.add(detection.track_id)

            if detection.track_id not in self.tracks:
                # Crear nuevo track
                self.tracks[detection.track_id] = VehicleTrack(
                    track_id=detection.track_id,
                    class_name=detection.class_name
                )

            # Actualizar posición del track con la clase actual para estabilización
            self.tracks[detection.track_id].update_position(
                detection.centroid, 
                class_name=detection.class_name
            )

        # Incrementar contador de frames perdidos para tracks no detectados
        for track_id in list(self.tracks.keys()):
            if track_id not in detected_ids:
                self.tracks[track_id].increment_missed_frames()

                # Eliminar tracks que llevan mucho tiempo sin detección
                if self.tracks[track_id].frames_since_last_detection > settings.TRACKING_MAX_AGE:
                    del self.tracks[track_id]

    def count_vehicle_crossing(
        self,
        track: VehicleTrack,
        lane_lines: dict,
    ) -> bool:
        """
        Verifica si un vehículo ha cruzado una línea de carril y lo cuenta.

        Args:
            track: Track del vehículo
            lane_lines: Configuración de líneas de carriles

        Returns:
            True si el vehículo fue contado
        """
        if track.counted or len(track.positions) < 2:
            return False

        current_pos = track.positions[-1]
        prev_pos = track.positions[-2]

        # Verificar cruces horizontales (línea REFERENCIA)
        for i, lane in enumerate(lane_lines.get("horizontal", [])):
            y_line = lane["y"]

            # Determinar dirección del cruce
            direction = None

            # Factor que ayuda a contar vehículos que se encuentran en la línea de referencia
            epsilon = 0.5

            # Sur -> Norte (de abajo hacia arriba, Y disminuye)
            if prev_pos[1] > y_line + epsilon > current_pos[1]:
                direction = "Sur->Norte"

            # Norte -> Sur (de arriba hacia abajo, Y aumenta)
            elif prev_pos[1] < y_line + epsilon < current_pos[1]:
                direction = "Norte->Sur"

            if direction:
                track.counted = True
                track.lane = i
                
                # Obtener la clase predominante del historial con pesos lineales
                track.class_name = track.get_stable_class()
                
                self._update_stats(track, direction)
                logger.info(f"Vehículo contado - ID: {track.track_id}, Clase: {track.class_name}, Dirección: {direction}, Posición previa Y: {prev_pos[1]}, Posición actual Y: {current_pos[1]}, Línea Y: {y_line}")
                return True

        return False

    def _update_stats(self, track: VehicleTrack, direction: str):
        """
        Actualiza las estadísticas de tráfico.

        Args:
            track: Track del vehículo
            direction: Dirección del movimiento
        """
        self.stats.total_vehicles += 1

        # Por clase
        if track.class_name not in self.stats.by_class:
            self.stats.by_class[track.class_name] = 0
        self.stats.by_class[track.class_name] += 1

        # Por dirección
        if direction not in self.stats.by_direction:
            self.stats.by_direction[direction] = 0
        self.stats.by_direction[direction] += 1

        # Por clase y dirección (combinado)
        class_dir_key = f"{track.class_name}_{direction}"
        if "by_class_direction" not in self.stats.__dict__:
            self.stats.__dict__["by_class_direction"] = {}
        if class_dir_key not in self.stats.__dict__["by_class_direction"]:
            self.stats.__dict__["by_class_direction"][class_dir_key] = 0
        self.stats.__dict__["by_class_direction"][class_dir_key] += 1

    def get_stats(self) -> TrafficStats:
        """Retorna las estadísticas actuales."""
        return self.stats

    def reset_stats(self):
        """Reinicia las estadísticas."""
        self.stats = TrafficStats()
        self.tracks = {}
