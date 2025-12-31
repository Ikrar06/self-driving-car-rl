"""
Camera system for game view transformation and control.
"""


class Camera:
    """Camera system with zoom and pan controls."""

    def __init__(self, screen_width, screen_height):
        """
        Initialize camera.

        Args:
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.x = 0  # Camera offset x (world coordinates)
        self.y = 0  # Camera offset y (world coordinates)
        self.zoom = 1.0  # Zoom level (0.3x - 3.0x)
        self.follow_car = True  # Follow car mode
        self.drag_start = None  # For mouse dragging

    def screen_to_world(self, screen_x, screen_y):
        """
        Convert screen coordinates to world coordinates.

        Args:
            screen_x: Screen x coordinate
            screen_y: Screen y coordinate

        Returns:
            (world_x, world_y)
        """
        world_x = (screen_x - self.screen_width / 2) / self.zoom + self.x
        world_y = (screen_y - self.screen_height / 2) / self.zoom + self.y
        return world_x, world_y

    def world_to_screen(self, world_x, world_y):
        """
        Convert world coordinates to screen coordinates.

        Args:
            world_x: World x coordinate
            world_y: World y coordinate

        Returns:
            (screen_x, screen_y) as integers
        """
        screen_x = (world_x - self.x) * self.zoom + self.screen_width / 2
        screen_y = (world_y - self.y) * self.zoom + self.screen_height / 2
        return int(screen_x), int(screen_y)

    def update_follow_car(self, car_x, car_y):
        """
        Update camera to follow car position.

        Args:
            car_x: Car x position
            car_y: Car y position
        """
        if self.follow_car:
            self.x = car_x
            self.y = car_y

    def zoom_in(self):
        """Zoom in (max 3.0x)."""
        self.zoom = min(3.0, self.zoom * 1.1)

    def zoom_out(self):
        """Zoom out (min 0.3x)."""
        self.zoom = max(0.3, self.zoom / 1.1)

    def pan(self, dx, dy):
        """
        Pan camera by offset.

        Args:
            dx: Delta x in pixels
            dy: Delta y in pixels
        """
        self.x += dx / self.zoom
        self.y += dy / self.zoom

    def reset(self):
        """Reset camera to default state."""
        self.x = 0
        self.y = 0
        self.zoom = 1.0
        self.follow_car = True

    def set_follow(self, follow: bool):
        """
        Enable or disable follow mode.

        Args:
            follow: True to enable follow mode
        """
        self.follow_car = follow


class DualCamera:
    """
    Manages two independent cameras for split-screen rendering.
    """

    def __init__(self, screen_width, screen_height):
        """
        Initialize dual camera system.

        Args:
            screen_width: Width per camera viewport
            screen_height: Height per camera viewport
        """
        self.camera_left = Camera(screen_width, screen_height)
        self.camera_right = Camera(screen_width, screen_height)
        self.active_camera = "left"  # "left", "right", or "both"

    def get_camera(self, side: str) -> Camera:
        """
        Get camera for specified side.

        Args:
            side: "left" or "right"

        Returns:
            Camera instance
        """
        if side == "left":
            return self.camera_left
        elif side == "right":
            return self.camera_right
        else:
            raise ValueError(f"Invalid side: {side}. Must be 'left' or 'right'")

    def get_active_camera(self) -> Camera:
        """
        Get currently active camera.

        Returns:
            Active camera instance
        """
        if self.active_camera == "left":
            return self.camera_left
        elif self.active_camera == "right":
            return self.camera_right
        else:  # both
            return self.camera_left  # Default to left

    def set_active(self, side: str):
        """
        Set active camera for controls.

        Args:
            side: "left", "right", or "both"
        """
        if side not in ["left", "right", "both"]:
            raise ValueError(f"Invalid side: {side}")
        self.active_camera = side

    def switch_active(self):
        """Toggle active camera between left and right."""
        if self.active_camera == "left":
            self.active_camera = "right"
        elif self.active_camera == "right":
            self.active_camera = "left"
        # If "both", switch to "left"
        else:
            self.active_camera = "left"

    def sync_cameras(self):
        """Sync right camera to match left camera settings."""
        self.camera_right.x = self.camera_left.x
        self.camera_right.y = self.camera_left.y
        self.camera_right.zoom = self.camera_left.zoom
        self.camera_right.follow_car = self.camera_left.follow_car

    def zoom_in(self):
        """Zoom in active camera(s)."""
        if self.active_camera == "both":
            self.camera_left.zoom_in()
            self.camera_right.zoom_in()
        else:
            self.get_active_camera().zoom_in()

    def zoom_out(self):
        """Zoom out active camera(s)."""
        if self.active_camera == "both":
            self.camera_left.zoom_out()
            self.camera_right.zoom_out()
        else:
            self.get_active_camera().zoom_out()

    def pan(self, dx, dy):
        """Pan active camera(s)."""
        if self.active_camera == "both":
            self.camera_left.pan(dx, dy)
            self.camera_right.pan(dx, dy)
        else:
            self.get_active_camera().pan(dx, dy)

    def reset(self):
        """Reset all cameras."""
        self.camera_left.reset()
        self.camera_right.reset()

    def toggle_follow(self):
        """Toggle follow mode for active camera."""
        camera = self.get_active_camera()
        camera.set_follow(not camera.follow_car)
