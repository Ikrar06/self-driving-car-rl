"""
Interactive F1 Track Builder
Create tracks visually with mouse clicks!
"""

import pygame
import json
import math
import subprocess
import sys
from pathlib import Path

# Check if we can use native file dialogs
NATIVE_DIALOG_AVAILABLE = sys.platform == 'darwin'  # macOS supports osascript
if not NATIVE_DIALOG_AVAILABLE:
    print("⚠️  Native file dialog not available - will use text-based track selection")


def show_native_file_picker(initial_dir):
    """
    Show native macOS file picker using osascript (AppleScript).
    This avoids tkinter conflicts with pygame/SDL.

    Returns:
        str: Selected file path, or None if cancelled
    """
    if not NATIVE_DIALOG_AVAILABLE:
        return None

    # AppleScript to show file picker
    # Note: Removed "of type" filter to allow all JSON files to be visible
    applescript = f'''
    tell application "System Events"
        activate
        set theFile to choose file with prompt "Select Track to Load (.json)" default location POSIX file "{initial_dir}"
        return POSIX path of theFile
    end tell
    '''

    try:
        result = subprocess.run(
            ['osascript', '-e', applescript],
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout
        )

        if result.returncode == 0:
            # Success - return the file path (strip newline)
            filepath = result.stdout.strip()

            # Validate it's a JSON file
            if not filepath.lower().endswith('.json'):
                print(f"⚠️  Selected file is not a JSON file: {filepath}")
                return None

            return filepath
        else:
            # User cancelled or error
            return None
    except Exception as e:
        print(f"⚠️  Error showing file picker: {e}")
        return None


class TrackBuilder:
    """Visual track builder with mouse controls."""

    def __init__(self):
        pygame.init()
        self.width = 1400
        self.height = 1000
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption("F1 Track Builder - Click to Build!")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.title_font = pygame.font.Font(None, 36)
        self.is_fullscreen = False

        # Track data
        self.outer_boundary = []
        self.inner_boundary = []
        self.checkpoints = []
        self.start_pos = None
        self.start_angle = -1.5708  # Default: facing up

        # UI state
        self.mode = "outer"  # "outer", "inner", "checkpoint", "start"
        self.checkpoint_start = None
        self.track_name = "My Custom F1 Track"

        # Grid
        self.grid_size = 50
        self.snap_to_grid = True

        # Camera controls
        self.camera_x = 0  # Camera offset x
        self.camera_y = 0  # Camera offset y
        self.zoom = 1.0    # Zoom level (1.0 = normal)
        self.pan_speed = 20

        # Car reference size (for scale comparison)
        self.car_width = 20
        self.car_height = 15
        self.show_car_reference = True

        # Colors
        self.bg_color = (30, 30, 30)
        self.grid_color = (50, 50, 50)
        self.outer_color = (100, 200, 100)
        self.inner_color = (200, 100, 100)
        self.checkpoint_color = (100, 200, 255)
        self.start_color = (255, 255, 0)
        self.car_color = (255, 255, 100)

    def world_to_screen(self, world_pos):
        """Convert world coordinates to screen coordinates with camera transform."""
        x, y = world_pos
        screen_x = (x - self.camera_x) * self.zoom + self.width // 2
        screen_y = (y - self.camera_y) * self.zoom + self.height // 2
        return (int(screen_x), int(screen_y))

    def screen_to_world(self, screen_pos):
        """Convert screen coordinates to world coordinates."""
        screen_x, screen_y = screen_pos
        world_x = (screen_x - self.width // 2) / self.zoom + self.camera_x
        world_y = (screen_y - self.height // 2) / self.zoom + self.camera_y
        return (world_x, world_y)

    def snap(self, pos):
        """Snap position to grid if enabled."""
        if self.snap_to_grid:
            x = round(pos[0] / self.grid_size) * self.grid_size
            y = round(pos[1] / self.grid_size) * self.grid_size
            return (x, y)
        return pos

    def zoom_in(self):
        """Zoom in (max 5x)."""
        self.zoom = min(5.0, self.zoom * 1.2)
        print(f"Zoom: {self.zoom:.2f}x")

    def zoom_out(self):
        """Zoom out (min 0.2x)."""
        self.zoom = max(0.2, self.zoom / 1.2)
        print(f"Zoom: {self.zoom:.2f}x")

    def pan_camera(self, dx, dy):
        """Pan camera by dx, dy in world coordinates."""
        self.camera_x += dx / self.zoom
        self.camera_y += dy / self.zoom

    def toggle_fullscreen(self):
        """Toggle between fullscreen and windowed mode."""
        self.is_fullscreen = not self.is_fullscreen
        if self.is_fullscreen:
            # Get desktop resolution
            info = pygame.display.Info()
            self.width = info.current_w
            self.height = info.current_h
            self.screen = pygame.display.set_mode((self.width, self.height), pygame.FULLSCREEN)
            print(f"Fullscreen: {self.width}x{self.height}")
        else:
            self.width = 1400
            self.height = 1000
            self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
            print(f"Windowed: {self.width}x{self.height}")

    def handle_resize(self, new_width, new_height):
        """Handle window resize event."""
        self.width = new_width
        self.height = new_height
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        print(f"Window resized: {self.width}x{self.height}")

    def draw_grid(self):
        """Draw background grid with camera transform."""
        # Calculate visible grid range
        top_left_world = self.screen_to_world((0, 0))
        bottom_right_world = self.screen_to_world((self.width, self.height))

        # Draw vertical lines
        start_x = int(top_left_world[0] / self.grid_size) * self.grid_size
        end_x = int(bottom_right_world[0] / self.grid_size) * self.grid_size + self.grid_size
        for x in range(start_x, end_x, self.grid_size):
            screen_start = self.world_to_screen((x, top_left_world[1]))
            screen_end = self.world_to_screen((x, bottom_right_world[1]))
            if 0 <= screen_start[0] <= self.width:
                pygame.draw.line(self.screen, self.grid_color, screen_start, screen_end, 1)

        # Draw horizontal lines
        start_y = int(top_left_world[1] / self.grid_size) * self.grid_size
        end_y = int(bottom_right_world[1] / self.grid_size) * self.grid_size + self.grid_size
        for y in range(start_y, end_y, self.grid_size):
            screen_start = self.world_to_screen((top_left_world[0], y))
            screen_end = self.world_to_screen((bottom_right_world[0], y))
            if 0 <= screen_start[1] <= self.height:
                pygame.draw.line(self.screen, self.grid_color, screen_start, screen_end, 1)

    def draw_track(self):
        """Draw the current track with camera transform."""
        # Draw outer boundary
        if len(self.outer_boundary) > 1:
            screen_points = [self.world_to_screen(p) for p in self.outer_boundary]
            pygame.draw.lines(self.screen, self.outer_color, False, screen_points, 3)
            # Draw points
            for point in screen_points:
                pygame.draw.circle(self.screen, self.outer_color, point, 5)

        # Draw inner boundary
        if len(self.inner_boundary) > 1:
            screen_points = [self.world_to_screen(p) for p in self.inner_boundary]
            pygame.draw.lines(self.screen, self.inner_color, False, screen_points, 3)
            # Draw points
            for point in screen_points:
                pygame.draw.circle(self.screen, self.inner_color, point, 5)

        # Draw checkpoints
        for i, checkpoint in enumerate(self.checkpoints):
            start_world, end_world = checkpoint
            start_screen = self.world_to_screen(start_world)
            end_screen = self.world_to_screen(end_world)
            pygame.draw.line(self.screen, self.checkpoint_color, start_screen, end_screen, 4)
            # Draw checkpoint number
            mid_screen = ((start_screen[0] + end_screen[0]) // 2,
                         (start_screen[1] + end_screen[1]) // 2)
            text = self.font.render(str(i + 1), True, (255, 255, 255))
            self.screen.blit(text, mid_screen)

        # Draw start position
        if self.start_pos:
            start_screen = self.world_to_screen(self.start_pos)
            pygame.draw.circle(self.screen, self.start_color, start_screen, 10)
            # Draw direction indicator
            end_world_x = self.start_pos[0] + 30 * math.cos(self.start_angle)
            end_world_y = self.start_pos[1] + 30 * math.sin(self.start_angle)
            end_screen = self.world_to_screen((end_world_x, end_world_y))
            pygame.draw.line(self.screen, self.start_color, start_screen, end_screen, 3)

    def draw_car_reference(self, mouse_pos):
        """Draw car size reference at mouse cursor."""
        if not self.show_car_reference:
            return

        # Get world position of mouse
        world_pos = self.screen_to_world(mouse_pos)

        # Calculate car corners in world space
        hw = self.car_width / 2
        hh = self.car_height / 2
        corners_world = [
            (world_pos[0] - hw, world_pos[1] - hh),
            (world_pos[0] + hw, world_pos[1] - hh),
            (world_pos[0] + hw, world_pos[1] + hh),
            (world_pos[0] - hw, world_pos[1] + hh),
        ]

        # Convert to screen space
        corners_screen = [self.world_to_screen(c) for c in corners_world]

        # Draw car outline
        pygame.draw.polygon(self.screen, self.car_color, corners_screen, 2)

        # Draw size label
        label = self.font.render(f"Car: {self.car_width}x{self.car_height}px", True, self.car_color)
        self.screen.blit(label, (mouse_pos[0] + 15, mouse_pos[1] + 15))

    def draw_ui(self):
        """Draw UI instructions."""
        instructions = [
            "=== F1 TRACK BUILDER ===",
            "",
            f"MODE: {self.mode.upper()}",
            f"ZOOM: {self.zoom:.2f}x",
            f"CAR: {self.car_width}x{self.car_height}px",
            "",
            "CONTROLS:",
            "  [1] Outer Boundary Mode",
            "  [2] Inner Boundary Mode",
            "  [3] Checkpoint Mode",
            "  [4] Start Position Mode",
            "  [Mouse Wheel] Zoom In/Out",
            "  [W/A/S/D] Pan Camera",
            "  [Arrow Keys] Pan Camera",
            "  [G] Toggle Grid Snap: " + ("ON" if self.snap_to_grid else "OFF"),
            "  [V] Toggle Car Reference: " + ("ON" if self.show_car_reference else "OFF"),
            "  [C] Close Current Boundary",
            "  [Z] Undo Last Point",
            "  [M] Smooth Track (Shift+M = Stronger)",
            "  [R] Rotate Start Angle",
            "  [Cmd+S] Save Track",
            "  [Cmd+L] Load Track",
            "  [Cmd+F / F11] Fullscreen",
            "  [DELETE] Clear All",
            "",
            f"Outer: {len(self.outer_boundary)} points",
            f"Inner: {len(self.inner_boundary)} points",
            f"Checkpoints: {len(self.checkpoints)}",
            f"Start: {'Set' if self.start_pos else 'Not Set'}",
        ]

        y_offset = 10
        for i, text in enumerate(instructions):
            if i == 0:
                surface = self.title_font.render(text, True, (255, 255, 100))
            else:
                surface = self.font.render(text, True, (255, 255, 255))

            # Background
            bg_rect = surface.get_rect()
            bg_rect.topleft = (10, y_offset)
            bg_rect.inflate_ip(10, 5)
            pygame.draw.rect(self.screen, (40, 40, 40, 200), bg_rect)

            self.screen.blit(surface, (10, y_offset))
            y_offset += 25 if i == 0 else 22

    def handle_click(self, screen_pos):
        """Handle mouse click - convert screen to world coordinates."""
        world_pos = self.screen_to_world(screen_pos)
        snapped_pos = self.snap(world_pos)

        if self.mode == "outer":
            self.outer_boundary.append(snapped_pos)
            print(f"Added outer point: {snapped_pos}")

        elif self.mode == "inner":
            self.inner_boundary.append(snapped_pos)
            print(f"Added inner point: {snapped_pos}")

        elif self.mode == "checkpoint":
            if self.checkpoint_start is None:
                self.checkpoint_start = snapped_pos
                print(f"Checkpoint start: {snapped_pos}")
            else:
                self.checkpoints.append((self.checkpoint_start, snapped_pos))
                print(f"Checkpoint added: {self.checkpoint_start} -> {snapped_pos}")
                self.checkpoint_start = None

        elif self.mode == "start":
            self.start_pos = snapped_pos
            print(f"Start position set: {snapped_pos}")

    def undo(self):
        """Undo last action."""
        if self.mode == "outer" and self.outer_boundary:
            removed = self.outer_boundary.pop()
            print(f"Undid outer point: {removed}")
        elif self.mode == "inner" and self.inner_boundary:
            removed = self.inner_boundary.pop()
            print(f"Undid inner point: {removed}")
        elif self.mode == "checkpoint" and self.checkpoints:
            removed = self.checkpoints.pop()
            print(f"Undid checkpoint: {removed}")

    def simplify_points(self, points, tolerance=5.0):
        """
        Reduce number of points while preserving shape (Douglas-Peucker algorithm).

        Args:
            points: List of (x, y) tuples
            tolerance: Maximum distance threshold (pixels)

        Returns:
            Simplified list of points
        """
        if len(points) < 3:
            return points

        # Check if closed
        is_closed = (abs(points[0][0] - points[-1][0]) < 0.1 and
                     abs(points[0][1] - points[-1][1]) < 0.1)

        # Remove duplicate last point if closed
        working_points = list(points[:-1] if is_closed else points)

        def perpendicular_distance(point, line_start, line_end):
            """Calculate perpendicular distance from point to line."""
            x0, y0 = point
            x1, y1 = line_start
            x2, y2 = line_end

            dx = x2 - x1
            dy = y2 - y1

            # Avoid division by zero
            if dx == 0 and dy == 0:
                return math.sqrt((x0 - x1)**2 + (y0 - y1)**2)

            # Calculate perpendicular distance
            return abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1) / math.sqrt(dx**2 + dy**2)

        def douglas_peucker(points_list, epsilon):
            """Recursive Douglas-Peucker algorithm."""
            if len(points_list) < 3:
                return points_list

            # Find point with maximum distance
            max_dist = 0
            max_index = 0

            for i in range(1, len(points_list) - 1):
                dist = perpendicular_distance(points_list[i], points_list[0], points_list[-1])
                if dist > max_dist:
                    max_dist = dist
                    max_index = i

            # If max distance is greater than epsilon, recursively simplify
            if max_dist > epsilon:
                # Recursive call
                left = douglas_peucker(points_list[:max_index + 1], epsilon)
                right = douglas_peucker(points_list[max_index:], epsilon)

                # Combine results (avoid duplicate middle point)
                return left[:-1] + right
            else:
                # Return only endpoints
                return [points_list[0], points_list[-1]]

        # Apply algorithm
        simplified = douglas_peucker(working_points, tolerance)

        # Re-close if was closed
        if is_closed:
            simplified.append(simplified[0])

        return simplified

    def smooth_boundary(self, points, iterations=1, simplify=True, tolerance=2.0):
        """
        Smooth boundary using Chaikin's corner cutting algorithm.
        This actually rounds corners by subdividing edges.
        Preserves closed loops (when first point == last point).
        Automatically reduces points after smoothing to prevent lag.

        Args:
            points: List of (x, y) tuples
            iterations: Number of subdivision iterations (1-2 recommended)
            simplify: Whether to reduce points after smoothing (default: True)
            tolerance: Simplification tolerance in pixels (default: 2.0 = very smooth)

        Returns:
            Smoothed list of points with rounded corners
        """
        if len(points) < 3:
            return points

        # Check if boundary is closed (first point == last point)
        is_closed = (len(points) > 2 and
                     abs(points[0][0] - points[-1][0]) < 0.1 and
                     abs(points[0][1] - points[-1][1]) < 0.1)

        # If closed, remove duplicate last point before smoothing
        smoothed = list(points[:-1] if is_closed else points)

        # Apply Chaikin's algorithm
        for _ in range(iterations):
            new_points = []
            n = len(smoothed)

            for i in range(n):
                # Current point and next point (wrap around for closed loop)
                p0 = smoothed[i]
                p1 = smoothed[(i + 1) % n]

                # Chaikin's algorithm: cut each edge at 1/4 and 3/4
                # This creates two new points per edge, rounding corners
                q = (
                    0.75 * p0[0] + 0.25 * p1[0],
                    0.75 * p0[1] + 0.25 * p1[1]
                )
                r = (
                    0.25 * p0[0] + 0.75 * p1[0],
                    0.25 * p0[1] + 0.75 * p1[1]
                )

                new_points.append(q)
                new_points.append(r)

            smoothed = new_points

        # If was closed, re-close by adding duplicate first point at end
        if is_closed:
            smoothed.append(smoothed[0])

        # Note: No point reduction needed!
        # Training uses dual-level optimization:
        # - Collision: Full detail (all points)
        # - Rendering: Auto-simplified to ~500 points (no lag!)
        print(f"   ✅ Track has {len(smoothed)} points - full detail for physics, auto-simplified for rendering!")

        return smoothed

    def smooth_track(self, iterations=1):
        """
        Smooth both boundaries using Chaikin's corner cutting.
        This truly rounds sharp corners!

        Args:
            iterations: Number of smoothing passes
                1 = gentle rounding (doubles points)
                2 = very smooth (4x points)
        """
        if len(self.outer_boundary) >= 3:
            original_count = len(self.outer_boundary)
            self.outer_boundary = self.smooth_boundary(self.outer_boundary, iterations=iterations)
            print(f"✨ Smoothed outer boundary: {original_count} → {len(self.outer_boundary)} points")

        if len(self.inner_boundary) >= 3:
            original_count = len(self.inner_boundary)
            self.inner_boundary = self.smooth_boundary(self.inner_boundary, iterations=iterations)
            print(f"✨ Smoothed inner boundary: {original_count} → {len(self.inner_boundary)} points")

    def close_boundary(self):
        """Close current boundary by connecting to first point."""
        if self.mode == "outer" and len(self.outer_boundary) > 2:
            if self.outer_boundary[0] != self.outer_boundary[-1]:
                self.outer_boundary.append(self.outer_boundary[0])
                print("Outer boundary closed!")
        elif self.mode == "inner" and len(self.inner_boundary) > 2:
            if self.inner_boundary[0] != self.inner_boundary[-1]:
                self.inner_boundary.append(self.inner_boundary[0])
                print("Inner boundary closed!")

    def rotate_start_angle(self):
        """Rotate start angle by 45 degrees."""
        self.start_angle += math.pi / 4
        if self.start_angle > math.pi:
            self.start_angle -= 2 * math.pi
        print(f"Start angle: {math.degrees(self.start_angle):.1f}°")

    def save_track(self):
        """Save track to JSON file."""
        if not self.outer_boundary or not self.inner_boundary:
            print("ERROR: Need both outer and inner boundaries!")
            return

        if not self.start_pos:
            print("ERROR: Need start position!")
            return

        if not self.checkpoints:
            print("ERROR: Need at least one checkpoint!")
            return

        track_data = {
            "name": self.track_name,
            "outer_boundary": self.outer_boundary,
            "inner_boundary": self.inner_boundary,
            "checkpoints": [
                {
                    "start": list(checkpoint[0]),
                    "end": list(checkpoint[1]),
                    "name": f"Checkpoint {i + 1}"
                }
                for i, checkpoint in enumerate(self.checkpoints)
            ],
            "start_pos": list(self.start_pos),
            "start_angle": self.start_angle
        }

        # Save to tracks folder
        filename = f"tracks/custom_track_{len(list(Path('tracks').glob('custom_track_*.json')))}.json"
        with open(filename, 'w') as f:
            json.dump(track_data, f, indent=2)

        print(f"✅ Track saved to: {filename}")
        return filename

    def load_track(self):
        """Load existing track for editing using native file dialog or text menu."""
        tracks_dir = Path('tracks').absolute()
        tracks_dir.mkdir(exist_ok=True)

        filepath = None

        if NATIVE_DIALOG_AVAILABLE:
            # Use native macOS file picker (no pygame/SDL conflict!)
            print("📂 Opening native file picker...")
            filepath = show_native_file_picker(str(tracks_dir))

            # Check if user cancelled
            if not filepath:
                print("❌ Load cancelled")
                return
        else:
            # Use text-based selection
            tracks = sorted(list(tracks_dir.glob('*.json')))
            if not tracks:
                print("❌ No tracks found in tracks/ folder")
                return

            print("\n" + "=" * 60)
            print("📁 AVAILABLE TRACKS")
            print("=" * 60)
            for i, track in enumerate(tracks):
                print(f"  {i + 1}. {track.name}")
            print("=" * 60)

            try:
                choice = int(input("Enter track number to load (0 to cancel): "))
                if choice == 0:
                    print("❌ Load cancelled")
                    return
                if 1 <= choice <= len(tracks):
                    filepath = str(tracks[choice - 1])
                else:
                    print("❌ Invalid choice")
                    return
            except ValueError:
                print("❌ Invalid input")
                return

        # Load the track
        if filepath:
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                self.outer_boundary = [tuple(p) for p in data['outer_boundary']]
                self.inner_boundary = [tuple(p) for p in data['inner_boundary']]
                self.checkpoints = [(tuple(cp['start']), tuple(cp['end'])) for cp in data['checkpoints']]
                self.start_pos = tuple(data['start_pos'])
                self.start_angle = data['start_angle']
                self.track_name = data.get('name', 'Imported Track')

                print(f"\n✅ Loaded track: {Path(filepath).name}")
                print(f"   • Outer boundary: {len(self.outer_boundary)} points")
                print(f"   • Inner boundary: {len(self.inner_boundary)} points")
                print(f"   • Checkpoints: {len(self.checkpoints)}\n")
            except Exception as e:
                print(f"❌ Error loading track: {e}")

    def clear_all(self):
        """Clear all track data."""
        self.outer_boundary = []
        self.inner_boundary = []
        self.checkpoints = []
        self.start_pos = None
        self.checkpoint_start = None
        print("Cleared all track data!")

    def run(self):
        """Main loop."""
        print("\n" + "=" * 60)
        print("F1 TRACK BUILDER - INTERACTIVE MODE")
        print("=" * 60)
        print("\nClick to place points, follow on-screen instructions!")
        print("Create smooth curves by placing many points.\n")

        running = True
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.VIDEORESIZE:
                    # Handle window resize
                    self.handle_resize(event.w, event.h)

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self.handle_click(event.pos)

                elif event.type == pygame.MOUSEWHEEL:
                    # Zoom in/out with mouse wheel
                    if event.y > 0:
                        self.zoom_in()
                    else:
                        self.zoom_out()

                elif event.type == pygame.KEYDOWN:
                    # Check modifiers
                    mods = pygame.key.get_mods()
                    has_modifier = mods & (pygame.KMOD_CTRL | pygame.KMOD_META)

                    # Pan camera with WASD or arrow keys (only if no Cmd/Ctrl pressed)
                    if not has_modifier and (event.key == pygame.K_w or event.key == pygame.K_UP):
                        self.pan_camera(0, -self.pan_speed)
                    elif not has_modifier and (event.key == pygame.K_s or event.key == pygame.K_DOWN):
                        self.pan_camera(0, self.pan_speed)
                    elif not has_modifier and (event.key == pygame.K_a or event.key == pygame.K_LEFT):
                        self.pan_camera(-self.pan_speed, 0)
                    elif not has_modifier and (event.key == pygame.K_d or event.key == pygame.K_RIGHT):
                        self.pan_camera(self.pan_speed, 0)

                    # Mode selection
                    elif event.key == pygame.K_1:
                        self.mode = "outer"
                        print("Mode: Outer Boundary")
                    elif event.key == pygame.K_2:
                        self.mode = "inner"
                        print("Mode: Inner Boundary")
                    elif event.key == pygame.K_3:
                        self.mode = "checkpoint"
                        self.checkpoint_start = None
                        print("Mode: Checkpoint (click start, then end)")
                    elif event.key == pygame.K_4:
                        self.mode = "start"
                        print("Mode: Start Position")

                    # Toggles
                    elif event.key == pygame.K_g:
                        self.snap_to_grid = not self.snap_to_grid
                        print(f"Grid snap: {'ON' if self.snap_to_grid else 'OFF'}")
                    elif event.key == pygame.K_v:
                        self.show_car_reference = not self.show_car_reference
                        print(f"Car reference: {'ON' if self.show_car_reference else 'OFF'}")

                    # Actions
                    elif event.key == pygame.K_c:
                        self.close_boundary()
                    elif event.key == pygame.K_z:
                        self.undo()
                    elif event.key == pygame.K_m:
                        # Smooth track - press once for gentle, hold Shift for stronger
                        iterations = 2 if (mods & pygame.KMOD_SHIFT) else 1
                        self.smooth_track(iterations=iterations)
                    elif event.key == pygame.K_r:
                        self.rotate_start_angle()
                    elif has_modifier and event.key == pygame.K_s:
                        # Cmd+S (Mac) or Ctrl+S (Windows/Linux) to save
                        filename = self.save_track()
                        if filename:
                            print(f"\n✅ Track saved!")
                            print(f"Test it with: venv/bin/python3 train_with_camera.py --track {filename}\n")
                    elif has_modifier and event.key == pygame.K_l:
                        # Cmd+L (Mac) or Ctrl+L (Windows/Linux) to load
                        self.load_track()
                    elif has_modifier and event.key == pygame.K_f:
                        # Cmd+F (Mac) or Ctrl+F (Windows/Linux) to toggle fullscreen
                        self.toggle_fullscreen()
                    elif event.key == pygame.K_F11:
                        # F11 to toggle fullscreen
                        self.toggle_fullscreen()
                    elif event.key == pygame.K_DELETE or event.key == pygame.K_BACKSPACE:
                        self.clear_all()
                    elif event.key == pygame.K_ESCAPE:
                        running = False

            # Draw everything
            self.screen.fill(self.bg_color)
            self.draw_grid()
            self.draw_track()
            self.draw_ui()

            # Draw checkpoint preview
            if self.mode == "checkpoint" and self.checkpoint_start:
                mouse_pos = pygame.mouse.get_pos()
                start_screen = self.world_to_screen(self.checkpoint_start)
                pygame.draw.line(self.screen, (100, 200, 255, 128),
                               start_screen, mouse_pos, 2)

            # Draw car reference at mouse cursor
            mouse_pos = pygame.mouse.get_pos()
            self.draw_car_reference(mouse_pos)

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        print("\n✅ Track Builder closed!")


if __name__ == "__main__":
    builder = TrackBuilder()
    builder.run()
