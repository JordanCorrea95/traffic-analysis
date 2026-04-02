"""
Servicio de procesamiento de video para análisis de tráfico.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Callable
import logging

from app.config import settings
from app.services.vehicle_detector import VehicleDetector
from app.utils import (
    get_video_info,
    create_video_writer,
    draw_text_with_background,
    draw_lane_lines
)

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Procesador de video para detección y análisis de tráfico vehicular."""

    def __init__(self, detector: Optional[VehicleDetector] = None):
        """
        Inicializa el procesador de video.

        Args:
            detector: Detector de vehículos. Si es None, crea uno nuevo.
        """
        self.detector = detector or VehicleDetector()

    def process_video(
        self,
        input_path: Path,
        output_path: Path,
        lane_config: Optional[dict] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> dict:
        """
        Procesa un video completo detectando y contando vehículos.

        Args:
            input_path: Ruta del video de entrada
            output_path: Ruta del video de salida
            lane_config: Configuración de carriles (usa default si es None)
            progress_callback: Función de callback para reportar progreso

        Returns:
            Diccionario con estadísticas del procesamiento
        """
        logger.info(f"Iniciando procesamiento de video: {input_path}")

        # Validar que existe el video
        if not input_path.exists():
            raise FileNotFoundError(f"Video no encontrado: {input_path}")

        # Usar configuración de carriles por defecto si no se proporciona
        lane_config = lane_config or settings.LANE_LINES

        # Resetear estadísticas
        self.detector.reset_stats()

        # Obtener información del video
        video_info = get_video_info(input_path)
        logger.info(f"Info del video: {video_info}")

        # Abrir video de entrada
        cap = cv2.VideoCapture(str(input_path))

        # Crear video de salida
        writer = create_video_writer(
            output_path,
            video_info["width"],
            video_info["height"],
            video_info["fps"]
        )

        frame_count = 0
        total_frames = video_info["frame_count"]

        try:
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                frame_count += 1

                # Detectar vehículos en el frame
                detections = self.detector.detect(frame)

                # Actualizar tracks
                self.detector.update_tracks(detections)

                # Verificar cruces de línea y contar vehículos
                for track in self.detector.tracks.values():
                    self.detector.count_vehicle_crossing(track, lane_config)

                # Dibujar visualizaciones en el frame
                annotated_frame = self._annotate_frame(
                    frame,
                    detections,
                    lane_config
                )

                # Escribir frame procesado
                writer.write(annotated_frame)

                # Callback de progreso
                if progress_callback and frame_count % 30 == 0:
                    progress_callback(frame_count, total_frames)

                # Log de progreso cada 10%
                if frame_count % (total_frames // 10 or 1) == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Progreso: {progress:.1f}% ({frame_count}/{total_frames})")

        finally:
            # Liberar recursos
            cap.release()
            writer.release()

        logger.info(f"Procesamiento completado: {frame_count} frames procesados")

        # Retornar estadísticas
        stats = self.detector.get_stats()
        return {
            "video_info": video_info,
            "frames_processed": frame_count,
            "statistics": stats.to_dict(),
            "output_path": str(output_path)
        }

    def _annotate_frame(
        self,
        frame: np.ndarray,
        detections: list,
        lane_config: dict
    ) -> np.ndarray:
        """
        Anota el frame con detecciones, segmentaciones y estadísticas.

        Args:
            frame: Frame original
            detections: Lista de detecciones
            lane_config: Configuración de carriles

        Returns:
            Frame anotado
        """
        annotated = frame.copy()

        # Dibujar líneas de carriles
        annotated = draw_lane_lines(annotated, lane_config)

        # Dibujar detecciones
        for detection in detections:
            color = settings.COLORS.get(detection.class_name, (0, 255, 255))

            # Dibujar máscara de segmentación si existe
            if detection.mask and len(detection.mask) > 0:
                mask_overlay = annotated.copy()
                points = np.array(detection.mask, dtype=np.int32)
                cv2.fillPoly(mask_overlay, [points], color)
                # Aplicar overlay con transparencia
                cv2.addWeighted(mask_overlay, 0.3, annotated, 0.7, 0, annotated)

                # Dibujar contorno de la máscara
                cv2.polylines(annotated, [points], True, color, 2)

            # Dibujar bounding box
            x1, y1, x2, y2 = detection.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Preparar etiqueta
            label = f"{detection.class_name.upper()} {detection.confidence:.2f}"

            # Dibujar etiqueta con fondo
            draw_text_with_background(
                annotated,
                label,
                (x1, y1 - 10),
                font_scale=0.5,
                thickness=1,
                text_color=(255, 255, 255),
                bg_color=color,
                padding=3
            )

            # Dibujar ID de tracking
            track_label = f"ID:{detection.track_id}"
            draw_text_with_background(
                annotated,
                track_label,
                (x1, y2 + 20),
                font_scale=0.4,
                thickness=1,
                text_color=(255, 255, 255),
                bg_color=color,
                padding=2
            )

            # Dibujar centroide
            # Verificar si este vehículo ya fue contado
            track = self.detector.tracks.get(detection.track_id)
            if track and track.counted:
                # Centroide de vehículo contado: círculo blanco con borde verde
                cv2.circle(annotated, detection.centroid, 6, (0, 255, 0), 2)
                cv2.circle(annotated, detection.centroid, 3, (255, 255, 255), -1)
            else:
                # Centroide normal
                cv2.circle(annotated, detection.centroid, 4, color, -1)

        # Dibujar estadísticas en el frame
        self._draw_statistics(annotated)

        return annotated

    def _draw_statistics(self, frame: np.ndarray):
        """
        Dibuja las estadísticas actuales en el frame.

        Args:
            frame: Frame a anotar
        """
        stats = self.detector.get_stats()

        # Traducción de nombres de clases
        class_translations = {
            "car": "Autos",
            "motorcycle": "Motos",
            "bus": "Autobuses",
            "truck": "Camiones"
        }

        # Posición inicial para las estadísticas
        x, y = 10, 30
        line_height = 25

        # Calcular altura del fondo según contenido
        num_lines = 3  # Total + 2 títulos
        if stats.by_direction:
            num_lines += len(stats.by_direction)
        if stats.by_class and stats.by_direction:
            # Mostrar por clase y dirección
            by_class_dir = getattr(stats, "by_class_direction", {})
            if by_class_dir:
                num_lines += len(by_class_dir)

        bg_height = num_lines * line_height + 20

        # Fondo semi-transparente para las estadísticas
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (400, bg_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Total de vehículos
        draw_text_with_background(
            frame,
            f"Total Vehiculos: {stats.total_vehicles}",
            (x, y),
            font_scale=0.6,
            thickness=2,
            text_color=(0, 255, 0),
            bg_color=(0, 0, 0),
            padding=2
        )
        y += line_height

        # Por dirección
        if stats.by_direction:
            draw_text_with_background(
                frame,
                "Por Direccion:",
                (x, y),
                font_scale=0.5,
                thickness=1,
                text_color=(255, 255, 255),
                bg_color=(0, 0, 0),
                padding=2
            )
            y += line_height

            for direction, count in stats.by_direction.items():
                draw_text_with_background(
                    frame,
                    f"  {direction}: {count}",
                    (x + 10, y),
                    font_scale=0.5,
                    thickness=1,
                    text_color=(255, 255, 0),
                    bg_color=(0, 0, 0),
                    padding=2
                )
                y += line_height

        # Por clase y dirección
        by_class_dir = getattr(stats, "by_class_direction", {})
        if by_class_dir:
            draw_text_with_background(
                frame,
                "Por Tipo y Direccion:",
                (x, y),
                font_scale=0.5,
                thickness=1,
                text_color=(255, 255, 255),
                bg_color=(0, 0, 0),
                padding=2
            )
            y += line_height

            for key, count in sorted(by_class_dir.items()):
                class_name, direction = key.split("_", 1)
                class_spanish = class_translations.get(class_name, class_name.capitalize())
                color = settings.COLORS.get(class_name, (255, 255, 255))
                draw_text_with_background(
                    frame,
                    f"  {class_spanish} ({direction}): {count}",
                    (x + 10, y),
                    font_scale=0.45,
                    thickness=1,
                    text_color=color,
                    bg_color=(0, 0, 0),
                    padding=2
                )
                y += line_height
