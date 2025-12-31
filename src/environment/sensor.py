"""
Ray-casting sensor system for the self-driving car.
"""

import math
from typing import List, Tuple, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .car import Car
    from .track import Track
    try:
        import pygame
    except ImportError:
        pass


class RaySensor:
    """
    A single ray-casting sensor that detects distance to obstacles.
    """

    def __init__(
        self,
        angle_offset: float,
        max_range: float = 500.0,
    ):
        """
        Initialize a ray sensor.

        Args:
            angle_offset: Angle offset from car heading in degrees
            max_range: Maximum detection range in pixels
        """
        self.angle_offset = math.radians(angle_offset)  # Convert to radians
        self.max_range = max_range
        self.current_distance = max_range  # Current reading
        self.hit_point: Optional[Tuple[float, float]] = None  # Where ray hit

    def cast(
        self,
        car_x: float,
        car_y: float,
        car_angle: float,
        track: "Track",
        step_size: float = 2.0,
    ) -> float:
        """
        Cast a ray from car position and find distance to track boundary.

        Args:
            car_x: Car x position
            car_y: Car y position
            car_angle: Car heading angle in radians
            track: Track to check collision with
            step_size: Step size for ray marching in pixels

        Returns:
            Normalized distance [0.0 = at wall, 1.0 = no obstacle in range]
        """
        # Calculate absolute ray angle
        ray_angle = car_angle + self.angle_offset

        # Direction vector
        dx = math.cos(ray_angle)
        dy = math.sin(ray_angle)

        # Start from car center
        x, y = car_x, car_y

        # March along ray until hitting boundary or max range
        distance = 0.0
        self.hit_point = None

        while distance < self.max_range:
            # Move along ray
            x += dx * step_size
            y += dy * step_size
            distance += step_size

            # Check if point is off track (hit boundary)
            if not track.is_point_on_track(x, y):
                self.hit_point = (x, y)
                self.current_distance = distance
                # Normalize: 0.0 = at wall, 1.0 = max range
                return 1.0 - (distance / self.max_range)

        # No hit within range
        self.current_distance = self.max_range
        self.hit_point = (x, y)
        return 0.0  # Maximum distance = 0.0 (safest)

    def get_normalized_reading(self) -> float:
        """
        Get current normalized distance reading.

        Returns:
            Normalized distance [0.0 = max range, 1.0 = at wall]
        """
        return 1.0 - (self.current_distance / self.max_range)

    def get_raw_distance(self) -> float:
        """
        Get current raw distance in pixels.

        Returns:
            Distance in pixels
        """
        return self.current_distance


class SensorArray:
    """
    Array of multiple ray sensors for the car.
    """

    def __init__(
        self,
        num_sensors: int = 5,
        angles: Optional[List[float]] = None,
        max_range: float = 200.0,
    ):
        """
        Initialize sensor array.

        Args:
            num_sensors: Number of sensors
            angles: List of angle offsets in degrees (if None, use default spread)
            max_range: Maximum detection range in pixels
        """
        self.num_sensors = num_sensors
        self.max_range = max_range

        # Default angles: evenly spread from -60 to +60 degrees
        if angles is None:
            if num_sensors == 1:
                angles = [0]
            else:
                # Spread from -60 to +60
                angles = [
                    -60 + (120 / (num_sensors - 1)) * i
                    for i in range(num_sensors)
                ]

        self.angles = angles

        # Create sensors
        self.sensors = [
            RaySensor(angle, max_range) for angle in angles
        ]

    def read_all(
        self,
        car_x: float,
        car_y: float,
        car_angle: float,
        track: "Track",
    ) -> np.ndarray:
        """
        Read all sensors and return normalized distances.

        Args:
            car_x: Car x position
            car_y: Car y position
            car_angle: Car heading angle in radians
            track: Track to check collision with

        Returns:
            NumPy array of normalized distances [0.0-1.0]
        """
        readings = []
        for sensor in self.sensors:
            reading = sensor.cast(car_x, car_y, car_angle, track)
            readings.append(reading)

        return np.array(readings, dtype=np.float32)

    def render(
        self,
        screen: "pygame.Surface",
        car_x: float,
        car_y: float,
    ) -> None:
        """
        Render sensor rays on screen with color coding.

        Args:
            screen: Pygame surface to draw on
            car_x: Car x position
            car_y: Car y position
        """
        try:
            import pygame
        except ImportError:
            return

        for sensor in self.sensors:
            if sensor.hit_point is None:
                continue

            # Color based on distance (6-level gradient)
            # normalized: 0.0 = far from wall (safe), 1.0 = close to wall (danger)
            normalized = sensor.get_normalized_reading()

            # Gradient: Green (safe/far) → Red (danger/close)
            if normalized > 0.85:
                # CRITICAL - Very close to wall
                color = (211, 47, 47)  # Dark Red
                line_width = 3
            elif normalized > 0.7:
                # DANGER - Close to wall
                color = (244, 67, 54)  # Red
                line_width = 3
            elif normalized > 0.55:
                # WARNING - Getting close
                color = (255, 152, 0)  # Orange
                line_width = 2
            elif normalized > 0.4:
                # CAUTION - Moderate distance
                color = (255, 193, 7)  # Yellow
                line_width = 2
            elif normalized > 0.2:
                # SAFE - Good distance
                color = (139, 195, 74)  # Light Green
                line_width = 1
            else:
                # CLEAR - Far from wall
                color = (76, 175, 80)  # Green
                line_width = 1

            # Draw line from car to hit point (thicker when closer to obstacles)
            pygame.draw.line(
                screen,
                color,
                (int(car_x), int(car_y)),
                (int(sensor.hit_point[0]), int(sensor.hit_point[1])),
                line_width,
            )

            # Draw small circle at hit point
            pygame.draw.circle(
                screen,
                color,
                (int(sensor.hit_point[0]), int(sensor.hit_point[1])),
                4,
            )

    def get_readings_array(self) -> np.ndarray:
        """
        Get current normalized readings from all sensors.

        Returns:
            NumPy array of normalized distances
        """
        return np.array(
            [sensor.get_normalized_reading() for sensor in self.sensors],
            dtype=np.float32,
        )

    def __repr__(self) -> str:
        readings = self.get_readings_array()
        return f"SensorArray({self.num_sensors} sensors, readings={readings})"


# Helper function for easy sensor creation
def create_default_sensors(
    num_sensors: int = 7,
    angles: list = None,
    max_range: float = 1000.0
) -> SensorArray:
    """
    Create sensor array with configurable parameters.

    Args:
        num_sensors: Number of sensors
        angles: List of sensor angles in degrees (if None, uses default 7-sensor config)
        max_range: Maximum detection range

    Returns:
        SensorArray with specified configuration
    """
    if angles is None:
        angles = [-90, -60, -30, 0, 30, 60, 90]

    return SensorArray(
        num_sensors=num_sensors,
        angles=angles,
        max_range=max_range,
    )
