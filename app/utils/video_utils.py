"""
Utilidades para procesamiento de video.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def get_video_info(video_path: Path) -> dict:
    """
    Obtiene información del video.

    Args:
        video_path: Ruta al archivo de video

    Returns:
        Diccionario con información del video
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video: {video_path}")

    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": int(cap.get(cv2.CAP_PROP_FPS)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
    }

    cap.release()
    return info


def create_video_writer(
    output_path: Path,
    width: int,
    height: int,
    fps: int,
    codec: str = "mp4v"
) -> cv2.VideoWriter:
    """
    Crea un objeto VideoWriter para guardar el video procesado.

    Args:
        output_path: Ruta de salida del video
        width: Ancho del frame
        height: Alto del frame
        fps: Frames por segundo
        codec: Codec de video (default: mp4v)

    Returns:
        Objeto VideoWriter
    """
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        raise ValueError(f"No se pudo crear el video writer: {output_path}")

    return writer


def draw_text_with_background(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_scale: float = 0.6,
    thickness: int = 2,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    padding: int = 5
) -> np.ndarray:
    """
    Dibuja texto con fondo en el frame.

    Args:
        frame: Frame de video
        text: Texto a dibujar
        position: Posición (x, y)
        font_scale: Escala de la fuente
        thickness: Grosor del texto
        text_color: Color del texto (BGR)
        bg_color: Color de fondo (BGR)
        padding: Padding alrededor del texto

    Returns:
        Frame modificado
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Obtener tamaño del texto
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    x, y = position

    # Dibujar rectángulo de fondo
    cv2.rectangle(
        frame,
        (x - padding, y - text_height - padding),
        (x + text_width + padding, y + baseline + padding),
        bg_color,
        -1
    )

    # Dibujar texto
    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA
    )

    return frame


def draw_lane_lines(
    frame: np.ndarray,
    lane_config: dict,
    color: Tuple[int, int, int] = (255, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Dibuja las líneas de carriles en el frame.

    Args:
        frame: Frame de video
        lane_config: Configuración de carriles
        color: Color de las líneas (BGR)
        thickness: Grosor de las líneas

    Returns:
        Frame modificado
    """
    height, width = frame.shape[:2]

    # Dibujar líneas horizontales
    for lane in lane_config.get("horizontal", []):
        y = lane["y"]
        cv2.line(frame, (0, y), (width, y), color, thickness)
        # Etiqueta de nombre
        label = lane.get("name", lane.get("direction", "")).upper()
        draw_text_with_background(
            frame,
            label,
            (10, y - 10),
            font_scale=0.5,
            text_color=(255, 255, 255),
            bg_color=color
        )

    # Dibujar líneas verticales
    for lane in lane_config.get("vertical", []):
        x = lane["x"]
        cv2.line(frame, (x, 0), (x, height), color, thickness)
        # Etiqueta de nombre
        label = lane.get("name", lane.get("direction", "")).upper()
        draw_text_with_background(
            frame,
            label,
            (x + 10, 30),
            font_scale=0.5,
            text_color=(255, 255, 255),
            bg_color=color
        )

    return frame


def resize_frame(
    frame: np.ndarray,
    max_width: int = 1920,
    max_height: int = 1080
) -> np.ndarray:
    """
    Redimensiona el frame si excede las dimensiones máximas.

    Args:
        frame: Frame a redimensionar
        max_width: Ancho máximo
        max_height: Alto máximo

    Returns:
        Frame redimensionado
    """
    height, width = frame.shape[:2]

    if width <= max_width and height <= max_height:
        return frame

    # Calcular ratio de escala
    scale = min(max_width / width, max_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)

    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
