"""
Modelos de datos para vehículos y tracking.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum


class Direction(str, Enum):
    """Direcciones de movimiento."""
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"
    UNKNOWN = "unknown"


@dataclass
class VehicleDetection:
    """Representa una detección de vehículo."""
    track_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    mask: Optional[List] = None  # Puntos de segmentación
    centroid: Tuple[int, int] = field(default=(0, 0))

    def __post_init__(self):
        """Calcula el centroide (centro del bounding box) si no se proporciona."""
        if self.centroid == (0, 0) and self.bbox:
            x1, y1, x2, y2 = self.bbox
            self.centroid = ((x1 + x2) // 2, (y1 + y2) // 2)


@dataclass
class VehicleTrack:
    """Representa el seguimiento de un vehículo a lo largo del tiempo."""
    track_id: int
    class_name: str
    positions: List[Tuple[int, int]] = field(default_factory=list)
    class_history: List[str] = field(default_factory=list)
    direction: Direction = Direction.UNKNOWN
    lane: Optional[int] = None
    counted: bool = False
    frames_since_last_detection: int = 0

    def update_position(self, position: Tuple[int, int], class_name: Optional[str] = None):
        """Actualiza la posición y el historial de clases del vehículo."""
        self.positions.append(position)
        self.frames_since_last_detection = 0

        # Actualizar historial de clases si se proporciona
        if class_name:
            self.class_history.append(class_name)
            # Mantener ventana de estabilidad (últimas N detecciones)
            if len(self.class_history) > 30:
                self.class_history = self.class_history[-30:]

        # Mantener solo las últimas N posiciones para calcular dirección
        if len(self.positions) > 30:
            self.positions = self.positions[-30:]

        self._calculate_direction()

    def _calculate_direction(self):
        """Calcula la dirección de movimiento basado en el historial de posiciones."""
        if len(self.positions) < 5:
            return

        # Comparar posición actual con posición hace 5 frames
        current_pos = self.positions[-1]
        past_pos = self.positions[-5]

        # Cambio vertical
        dy = current_pos[1] - past_pos[1]

        # Determinar dirección predominante (sólo consideraremos eje y)
        if dy > 0:
            self.direction = Direction.SOUTH
        elif dy < 0:
            self.direction = Direction.NORTH
        else:
            # dy == 0 (No hay movimiento)
            self.direction = None

    def increment_missed_frames(self):
        """Incrementa el contador de frames sin detección."""
        self.frames_since_last_detection += 1

    def get_stable_class(self) -> str:
        """
        Calcula la clase más probable basándose en el historial de detecciones
        utilizando una ponderación lineal (más peso a los frames más recientes).
        
        Returns:
            str: Clase con mayor peso acumulado.
        """
        if not self.class_history:
            return self.class_name

        # Mapeo para acumular pesos por clase
        class_weights = {}
        
        # Longitud del historial
        n = len(self.class_history)
        
        # Asignar pesos lineales: i=0 (más antiguo) -> peso 1, i=n-1 (más reciente) -> peso n
        for i, cls in enumerate(self.class_history):
            weight = i + 1
            class_weights[cls] = class_weights.get(cls, 0) + weight
            
        # Retornar la clase con el peso máximo
        return max(class_weights, key=class_weights.get)


@dataclass
class TrafficStats:
    """Estadísticas de tráfico."""
    total_vehicles: int = 0
    by_class: dict = field(default_factory=dict)
    by_direction: dict = field(default_factory=dict)
    by_lane: dict = field(default_factory=dict)

    def to_dict(self):
        """Convierte las estadísticas a diccionario."""
        return {
            "total_vehicles": self.total_vehicles,
            "by_class": self.by_class,
            "by_direction": self.by_direction,
            "by_lane": self.by_lane
        }
