"""
Track class for defining racing circuits with collision detection.
"""

import json
import math
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING
import numpy as np

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    if TYPE_CHECKING:
        import pygame


class Line:
    """Represents a line segment (used for checkpoints)."""

    def __init__(self, start: Tuple[float, float], end: Tuple[float, float]):
        """
        Initialize a line segment.

        Args:
            start: (x, y) start point
            end: (x, y) end point
        """
        self.start = start
        self.end = end

    def intersects(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> bool:
        """
        Check if line segment (p1, p2) intersects with this line.
        Uses cross product method.

        Args:
            p1: First point of segment to check
            p2: Second point of segment to check

        Returns:
            True if segments intersect
        """
        def ccw(A, B, C):
            """Check if three points are in counter-clockwise order."""
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        A, B = self.start, self.end
        C, D = p1, p2

        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    def __repr__(self) -> str:
        return f"Line({self.start} -> {self.end})"


class Track:
    """
    Represents a racing track with inner and outer boundaries.
    Uses polygon-based collision detection.
    """

    def __init__(
        self,
        name: str,
        outer_boundary: List[Tuple[float, float]],
        inner_boundary: List[Tuple[float, float]],
        checkpoints: List[Dict[str, Tuple[float, float]]],
        start_pos: Tuple[float, float],
        start_angle: float = 0.0,
    ):
        """
        Initialize a track.

        Args:
            name: Track name
            outer_boundary: List of (x, y) points defining outer wall
            inner_boundary: List of (x, y) points defining inner wall
            checkpoints: List of dicts with 'start' and 'end' keys
            start_pos: (x, y) starting position for car
            start_angle: Starting angle in radians
        """
        self.name = name
        self.outer_boundary = outer_boundary
        self.inner_boundary = inner_boundary
        self.start_pos = start_pos
        self.start_angle = start_angle

        # Convert checkpoint dicts to Line objects
        self.checkpoints = [
            Line(cp["start"], cp["end"]) for cp in checkpoints
        ]

        # Track dimensions for rendering
        self._calculate_bounds()

        # Dual-level optimization for performance with large point counts:
        # 1. Rendering points: Very simplified (~500 points) for display only
        self._render_outer = self._simplify_for_rendering(outer_boundary, max_points=500)
        self._render_inner = self._simplify_for_rendering(inner_boundary, max_points=500)

        # 2. Collision points: Moderately simplified (~1500 points) for physics
        #    Balances accuracy vs performance for collision detection
        self._collision_outer = self._simplify_for_rendering(outer_boundary, max_points=1500)
        self._collision_inner = self._simplify_for_rendering(inner_boundary, max_points=1500)

        # Rendering cache (pre-rendered track surface for performance)
        self._cached_surface: Optional["pygame.Surface"] = None
        self._cache_size: Optional[Tuple[int, int]] = None

    def _simplify_for_rendering(
        self,
        points: List[Tuple[float, float]],
        max_points: int = 500,
        tolerance: float = 2.0
    ) -> List[Tuple[float, float]]:
        """
        Simplify points for rendering while preserving shape.
        Uses fast uniform decimation for huge point counts.
        Keeps original points for collision detection.

        Args:
            points: Original boundary points
            max_points: Target maximum number of points for rendering
            tolerance: Simplification tolerance (lower = more detail)

        Returns:
            Simplified points for rendering only
        """
        if not points or len(points) <= max_points:
            return points

        original_count = len(points)

        # Check if closed loop
        is_closed = (len(points) > 2 and
                     abs(points[0][0] - points[-1][0]) < 0.1 and
                     abs(points[0][1] - points[-1][1]) < 0.1)

        working = list(points[:-1] if is_closed else points)

        # FAST PATH: For huge point counts (>5000), use uniform decimation
        # This is O(n) and avoids recursion issues with Douglas-Peucker
        if len(working) > 5000:
            # Calculate step size to get roughly max_points
            step = max(1, len(working) // max_points)
            simplified = [working[i] for i in range(0, len(working), step)]

            # Ensure we got close to max_points (add last point if needed)
            if len(simplified) < max_points // 2:
                step = max(1, step // 2)
                simplified = [working[i] for i in range(0, len(working), step)]

            # Always include the last point
            if simplified[-1] != working[-1]:
                simplified.append(working[-1])
        else:
            # For smaller point counts, use Douglas-Peucker for better shape preservation
            def perpendicular_distance(point, line_start, line_end):
                """Calculate perpendicular distance from point to line."""
                x0, y0 = point
                x1, y1 = line_start
                x2, y2 = line_end
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0 and dy == 0:
                    return math.sqrt((x0 - x1)**2 + (y0 - y1)**2)
                return abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1) / math.sqrt(dx**2 + dy**2)

            def douglas_peucker_iterative(pts, epsilon):
                """Iterative Douglas-Peucker (avoids recursion limit)."""
                if len(pts) < 3:
                    return pts

                # Stack-based iterative approach
                stack = [(0, len(pts) - 1)]
                keep_indices = {0, len(pts) - 1}

                while stack:
                    start_idx, end_idx = stack.pop()

                    if end_idx - start_idx <= 1:
                        continue

                    # Find point with max distance
                    max_dist = 0
                    max_idx = start_idx
                    for i in range(start_idx + 1, end_idx):
                        dist = perpendicular_distance(pts[i], pts[start_idx], pts[end_idx])
                        if dist > max_dist:
                            max_dist = dist
                            max_idx = i

                    # If max distance is greater than epsilon, keep this point
                    if max_dist > epsilon:
                        keep_indices.add(max_idx)
                        stack.append((start_idx, max_idx))
                        stack.append((max_idx, end_idx))

                # Build result from kept indices
                result = [pts[i] for i in sorted(keep_indices)]
                return result

            simplified = douglas_peucker_iterative(working, tolerance)

            # If still too many points, increase tolerance
            while len(simplified) > max_points and tolerance < 20.0:
                tolerance *= 1.5
                simplified = douglas_peucker_iterative(working, tolerance)

        # Re-close if was closed
        if is_closed and simplified[0] != simplified[-1]:
            simplified.append(simplified[0])

        simplified_count = len(simplified)
        if original_count > max_points:
            # Determine what this simplification is for based on max_points
            if max_points <= 500:
                purpose = "rendering"
            elif max_points <= 1500:
                purpose = "collision"
            else:
                purpose = "optimization"
            print(f"   ⚡ {purpose.capitalize()}: {original_count} → {simplified_count} points")

        return simplified

    def _calculate_bounds(self) -> None:
        """Calculate bounding box of the track."""
        all_points = self.outer_boundary + self.inner_boundary

        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]

        self.min_x = min(xs)
        self.max_x = max(xs)
        self.min_y = min(ys)
        self.max_y = max(ys)
        self.width = self.max_x - self.min_x
        self.height = self.max_y - self.min_y

    def is_point_on_track(self, x: float, y: float) -> bool:
        """
        Check if a point is on the track (between inner and outer boundaries).
        Uses simplified collision boundaries for performance.
        Uses ray-casting algorithm for point-in-polygon test.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            True if point is on track (inside outer AND outside inner)
        """
        # Use simplified collision boundaries for speed (still accurate with 1500 points)
        inside_outer = self._point_in_polygon(x, y, self._collision_outer)

        # If no inner boundary, just check outer
        if not self._collision_inner:
            return inside_outer

        # Must also be outside inner boundary
        inside_inner = self._point_in_polygon(x, y, self._collision_inner)
        return inside_outer and not inside_inner

    def _point_in_polygon(self, x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
        """
        Ray-casting algorithm to check if point is inside polygon.

        Args:
            x: X coordinate
            y: Y coordinate
            polygon: List of (x, y) points defining polygon

        Returns:
            True if point is inside polygon
        """
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]

            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside

            p1x, p1y = p2x, p2y

        return inside

    def check_checkpoint(
        self,
        prev_pos: Tuple[float, float],
        curr_pos: Tuple[float, float],
        checkpoint_idx: int,
    ) -> bool:
        """
        Check if car crossed a checkpoint.

        Args:
            prev_pos: Previous car position
            curr_pos: Current car position
            checkpoint_idx: Index of checkpoint to check

        Returns:
            True if car crossed the checkpoint
        """
        if checkpoint_idx >= len(self.checkpoints):
            return False

        checkpoint = self.checkpoints[checkpoint_idx]
        return checkpoint.intersects(prev_pos, curr_pos)

    def get_next_checkpoint_idx(self, current_idx: int) -> int:
        """
        Get index of next checkpoint.

        Args:
            current_idx: Current checkpoint index

        Returns:
            Next checkpoint index (wraps around)
        """
        return (current_idx + 1) % len(self.checkpoints)

    def invalidate_cache(self) -> None:
        """
        Invalidate the rendering and collision cache.
        Call this if track boundaries are modified after initialization.
        """
        # Regenerate simplified rendering points
        self._render_outer = self._simplify_for_rendering(self.outer_boundary, max_points=500)
        self._render_inner = self._simplify_for_rendering(self.inner_boundary, max_points=500)

        # Regenerate simplified collision points
        self._collision_outer = self._simplify_for_rendering(self.outer_boundary, max_points=1500)
        self._collision_inner = self._simplify_for_rendering(self.inner_boundary, max_points=1500)

        # Clear rendering cache
        self._cached_surface = None
        self._cache_size = None

    def _render_to_cache(self, screen_size: Tuple[int, int]) -> "pygame.Surface":
        """
        Pre-render the track to a cached surface for performance.
        Uses simplified rendering points for speed (collision uses full detail).

        Args:
            screen_size: (width, height) of the screen

        Returns:
            Cached pygame Surface with pre-rendered track
        """
        if not PYGAME_AVAILABLE:
            return None

        # Create a new surface for caching
        cache_surface = pygame.Surface(screen_size)
        cache_surface.fill((34, 34, 34))  # Background color

        # Draw outer boundary (dark gray) - using simplified points for speed
        pygame.draw.polygon(cache_surface, (50, 50, 50), self._render_outer)

        # Draw inner boundary (background color - creates "hole") if exists
        if self._render_inner and len(self._render_inner) > 2:
            pygame.draw.polygon(cache_surface, (34, 34, 34), self._render_inner)

        # Draw boundary lines - using simplified points
        pygame.draw.lines(cache_surface, (200, 200, 200), True, self._render_outer, 3)
        if self._render_inner and len(self._render_inner) > 2:
            pygame.draw.lines(cache_surface, (200, 200, 200), True, self._render_inner, 3)

        return cache_surface

    def render(
        self,
        screen: Optional["pygame.Surface"] = None,
        show_checkpoints: bool = True,
    ) -> None:
        """
        Render the track on a pygame surface.
        Uses cached pre-rendered surface for performance (prevents lag with many points).

        Args:
            screen: Pygame surface to draw on
            show_checkpoints: Whether to draw checkpoint lines
        """
        if not PYGAME_AVAILABLE or screen is None:
            return

        screen_size = screen.get_size()

        # Check if we need to (re)create the cache
        if self._cached_surface is None or self._cache_size != screen_size:
            # Pre-render the track to cache (only happens once or on resize)
            self._cached_surface = self._render_to_cache(screen_size)
            self._cache_size = screen_size

        # Blit the cached track (MUCH faster than drawing thousands of polygon points!)
        screen.blit(self._cached_surface, (0, 0))

        # Draw dynamic elements on top (these change during rendering)
        # Draw checkpoints
        if show_checkpoints:
            for i, checkpoint in enumerate(self.checkpoints):
                color = (156, 39, 176)  # Purple
                pygame.draw.line(
                    screen,
                    color,
                    checkpoint.start,
                    checkpoint.end,
                    2,
                )
                # Draw checkpoint number
                if hasattr(pygame, 'font') and pygame.font.get_init():
                    font = pygame.font.Font(None, 24)
                    text = font.render(str(i), True, color)
                    mid_x = (checkpoint.start[0] + checkpoint.end[0]) / 2
                    mid_y = (checkpoint.start[1] + checkpoint.end[1]) / 2
                    screen.blit(text, (mid_x - 10, mid_y - 10))

        # Draw start position (green circle)
        pygame.draw.circle(screen, (76, 175, 80), self.start_pos, 10)

    @classmethod
    def load(cls, filepath: str) -> "Track":
        """
        Load a track from a JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            Track instance

        Example JSON format:
        {
            "name": "Oval Easy",
            "outer_boundary": [[x1, y1], [x2, y2], ...],
            "inner_boundary": [[x1, y1], [x2, y2], ...],
            "checkpoints": [
                {"start": [x1, y1], "end": [x2, y2]},
                ...
            ],
            "start_pos": [x, y],
            "start_angle": 0.0
        }
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        return cls(
            name=data["name"],
            outer_boundary=[tuple(p) for p in data["outer_boundary"]],
            inner_boundary=[tuple(p) for p in data["inner_boundary"]],
            checkpoints=data["checkpoints"],
            start_pos=tuple(data["start_pos"]),
            start_angle=data.get("start_angle", 0.0),
        )

    def save(self, filepath: str) -> None:
        """
        Save track to a JSON file.

        Args:
            filepath: Path to save JSON file
        """
        data = {
            "name": self.name,
            "outer_boundary": self.outer_boundary,
            "inner_boundary": self.inner_boundary,
            "checkpoints": [
                {"start": list(cp.start), "end": list(cp.end)}
                for cp in self.checkpoints
            ],
            "start_pos": list(self.start_pos),
            "start_angle": self.start_angle,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def __repr__(self) -> str:
        return (
            f"Track(name='{self.name}', "
            f"outer_points={len(self.outer_boundary)}, "
            f"inner_points={len(self.inner_boundary)}, "
            f"checkpoints={len(self.checkpoints)})"
        )
