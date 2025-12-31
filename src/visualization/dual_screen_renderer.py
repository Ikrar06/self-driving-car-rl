"""
Dual-screen renderer for side-by-side agent comparison.
Supports fullscreen on Mac (Cmd+F, F11) and resizable windows.
"""

import pygame
import sys
from pathlib import Path
from typing import Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.track import Track
from .camera import DualCamera


class DualScreenRenderer:
    """
    Split-screen renderer for comparing two agents side-by-side.

    Screen Layout (default 2000x700):
    ┌────────────────────┬────────────────────┐
    │  Agent 1 (1000x700)│  Agent 2 (1000x700)│
    │  ┌──────────────┐  │  ┌──────────────┐  │
    │  │ Track + Car  │  │  │ Track + Car  │  │
    │  │ + Sensors    │  │  │ + Sensors    │  │
    │  └──────────────┘  │  └──────────────┘  │
    │  Info Panel        │  Info Panel        │
    └────────────────────┴────────────────────┘
    """

    def __init__(
        self,
        track: Track,
        agent1_name: str = "DQN",
        agent2_name: str = "PPO",
        total_width: int = 2000,
        total_height: int = 700,
        fps: int = 30,
    ):
        """
        Initialize dual screen renderer.

        Args:
            track: Track object to render
            agent1_name: Name of first agent (for UI)
            agent2_name: Name of second agent (for UI)
            total_width: Total window width
            total_height: Total window height
            fps: Target frames per second
        """
        pygame.init()

        self.track = track
        self.agent1_name = agent1_name
        self.agent2_name = agent2_name
        self.total_width = total_width
        self.total_height = total_height
        self.agent_width = total_width // 2
        self.fps = fps

        # Window state
        self.is_fullscreen = False

        # Create resizable window
        self.screen = pygame.display.set_mode(
            (total_width, total_height),
            pygame.RESIZABLE
        )
        pygame.display.set_caption(
            f"Training Comparison: {agent1_name} vs {agent2_name}"
        )

        # Create surfaces for each agent
        self.left_surface = pygame.Surface((self.agent_width, total_height))
        self.right_surface = pygame.Surface((self.agent_width, total_height))

        # Dual camera system
        self.cameras = DualCamera(self.agent_width, total_height)

        # Fonts
        pygame.font.init()
        self.font = pygame.font.Font(None, 24)
        self.title_font = pygame.font.Font(None, 32)
        self.small_font = pygame.font.Font(None, 20)

        # Clock for FPS control
        self.clock = pygame.time.Clock()

        # Agent states (to be updated from shared memory)
        self.agent1_state = {
            'car': {'x': 0, 'y': 0, 'angle': 0, 'velocity': 0},
            'sensors': [0] * 7,
            'stats': {'episode': 0, 'step': 0, 'reward': 0, 'checkpoint': 0},
        }
        self.agent2_state = {
            'car': {'x': 0, 'y': 0, 'angle': 0, 'velocity': 0},
            'sensors': [0] * 7,
            'stats': {'episode': 0, 'step': 0, 'reward': 0, 'checkpoint': 0},
        }

        # Colors
        self.BG_COLOR = (34, 34, 34)
        self.SEPARATOR_COLOR = (100, 100, 100)
        self.AGENT1_COLOR = (66, 135, 245)  # Blue for DQN
        self.AGENT2_COLOR = (76, 175, 80)   # Green for PPO

        print(f"🎨 Dual Screen Renderer initialized ({total_width}x{total_height})")

    def update_agent_state(
        self,
        agent_id: int,
        car_state: dict,
        sensors: list,
        stats: dict,
    ):
        """
        Update state for specified agent.

        Args:
            agent_id: 1 or 2
            car_state: Dict with x, y, angle, velocity
            sensors: List of 7 sensor readings
            stats: Dict with episode, step, reward, checkpoint
        """
        if agent_id == 1:
            self.agent1_state['car'] = car_state
            self.agent1_state['sensors'] = sensors
            self.agent1_state['stats'] = stats
        else:
            self.agent2_state['car'] = car_state
            self.agent2_state['sensors'] = sensors
            self.agent2_state['stats'] = stats

    def render_frame(self):
        """Render complete split-screen frame."""
        # Clear screen
        self.screen.fill(self.BG_COLOR)

        # Render left half (Agent 1)
        self._render_agent_half(
            surface=self.left_surface,
            camera=self.cameras.camera_left,
            agent_state=self.agent1_state,
            agent_name=self.agent1_name,
            agent_color=self.AGENT1_COLOR,
        )

        # Render right half (Agent 2)
        self._render_agent_half(
            surface=self.right_surface,
            camera=self.cameras.camera_right,
            agent_state=self.agent2_state,
            agent_name=self.agent2_name,
            agent_color=self.AGENT2_COLOR,
        )

        # Blit to main screen
        self.screen.blit(self.left_surface, (0, 0))
        self.screen.blit(self.right_surface, (self.agent_width, 0))

        # Draw separator line
        pygame.draw.line(
            self.screen,
            self.SEPARATOR_COLOR,
            (self.agent_width, 0),
            (self.agent_width, self.total_height),
            3
        )

        # Render comparison panel at bottom
        self._render_comparison_panel()

        # Update display
        pygame.display.flip()
        self.clock.tick(self.fps)

    def _render_agent_half(
        self,
        surface: pygame.Surface,
        camera,
        agent_state: dict,
        agent_name: str,
        agent_color: tuple,
    ):
        """Render one agent's viewport."""
        # Clear surface
        surface.fill(self.BG_COLOR)

        # Update camera to follow car
        car = agent_state['car']
        camera.update_follow_car(car['x'], car['y'])

        # Render track
        self._render_track_transformed(surface, camera)

        # Render car
        self._render_car_transformed(surface, camera, car, agent_color)

        # Render sensors
        self._render_sensors_transformed(
            surface, camera, car, agent_state['sensors']
        )

        # Render info panel
        self._render_info_panel(surface, agent_state, agent_name)

    def _render_track_transformed(self, surface, camera):
        """Render track with camera transformation."""
        # Render outer boundary
        if len(self.track.outer_boundary) > 2:
            points = [camera.world_to_screen(x, y) for x, y in self.track.outer_boundary]
            pygame.draw.polygon(surface, (50, 50, 50), points, 2)

        # Render inner boundary
        if len(self.track.inner_boundary) > 2:
            points = [camera.world_to_screen(x, y) for x, y in self.track.inner_boundary]
            pygame.draw.polygon(surface, (50, 50, 50), points, 2)

        # Render checkpoints
        for checkpoint in self.track.checkpoints:
            p1_screen = camera.world_to_screen(checkpoint.start[0], checkpoint.start[1])
            p2_screen = camera.world_to_screen(checkpoint.end[0], checkpoint.end[1])
            pygame.draw.line(surface, (156, 39, 176), p1_screen, p2_screen, 2)

        # Render start position
        start_screen = camera.world_to_screen(
            self.track.start_pos[0], self.track.start_pos[1]
        )
        pygame.draw.circle(surface, (76, 175, 80), start_screen, 8)

    def _render_car_transformed(self, surface, camera, car_state, color):
        """Render car with camera transformation."""
        import math

        # Car dimensions (scaled by zoom)
        car_width = 20 * camera.zoom
        car_height = 15 * camera.zoom

        # Get car position in screen coordinates
        car_screen = camera.world_to_screen(car_state['x'], car_state['y'])

        # Create car rectangle (rotated)
        car_rect = pygame.Surface((int(car_width), int(car_height)), pygame.SRCALPHA)
        pygame.draw.rect(car_rect, color, (0, 0, int(car_width), int(car_height)))

        # Rotate
        rotated = pygame.transform.rotate(car_rect, -math.degrees(car_state['angle']))
        rotated_rect = rotated.get_rect(center=car_screen)

        # Blit
        surface.blit(rotated, rotated_rect)

        # Draw front indicator
        front_x = car_state['x'] + math.cos(car_state['angle']) * 15
        front_y = car_state['y'] + math.sin(car_state['angle']) * 15
        front_screen = camera.world_to_screen(front_x, front_y)
        pygame.draw.line(surface, (255, 255, 0), car_screen, front_screen, 2)

    def _render_sensors_transformed(self, surface, camera, car_state, sensors):
        """Render sensor rays with gradient colors."""
        import math

        sensor_angles = [-90, -60, -30, 0, 30, 60, 90]
        sensor_range = 200

        for i, (angle_deg, distance) in enumerate(zip(sensor_angles, sensors)):
            angle_rad = math.radians(angle_deg) + car_state['angle']

            # End point
            end_x = car_state['x'] + math.cos(angle_rad) * distance * sensor_range
            end_y = car_state['y'] + math.sin(angle_rad) * distance * sensor_range

            # Screen coordinates
            car_screen = camera.world_to_screen(car_state['x'], car_state['y'])
            end_screen = camera.world_to_screen(end_x, end_y)

            # Color gradient based on distance (6 levels)
            # distance: 0.0 = far from wall (safe), 1.0 = close to wall (danger)
            if distance > 0.85:
                # CRITICAL - Very close
                color = (211, 47, 47)  # Dark Red
                line_width = 2
            elif distance > 0.7:
                # DANGER - Close
                color = (244, 67, 54)  # Red
                line_width = 2
            elif distance > 0.55:
                # WARNING - Approaching
                color = (255, 152, 0)  # Orange
                line_width = 1
            elif distance > 0.4:
                # CAUTION - Medium
                color = (255, 193, 7)  # Yellow
                line_width = 1
            elif distance > 0.2:
                # SAFE - Good
                color = (139, 195, 74)  # Light Green
                line_width = 1
            else:
                # CLEAR - Far
                color = (76, 175, 80)  # Green
                line_width = 1

            pygame.draw.line(surface, color, car_screen, end_screen, line_width)

    def _render_info_panel(self, surface, agent_state, agent_name):
        """Render info panel for one agent."""
        stats = agent_state['stats']
        car = agent_state['car']

        info_texts = [
            f"{agent_name}",
            f"Episode: {stats['episode']}",
            f"Step: {stats['step']}",
            f"Reward: {stats['reward']:.1f}",
            f"Checkpoint: {stats['checkpoint']}/{len(self.track.checkpoints)}",
            f"Speed: {car['velocity']:.2f}",
        ]

        # Semi-transparent background (taller for speed bar)
        panel_height = len(info_texts) * 25 + 50
        panel_surface = pygame.Surface((250, panel_height))
        panel_surface.set_alpha(200)
        panel_surface.fill((40, 40, 40))
        surface.blit(panel_surface, (10, 10))

        # Render text
        y_offset = 15
        for i, text in enumerate(info_texts):
            if i == 0:  # Agent name
                text_surface = self.title_font.render(text, True, (255, 255, 255))
            else:
                text_surface = self.font.render(text, True, (255, 255, 255))
            surface.blit(text_surface, (15, y_offset + i * 25))

        # Render speed bar
        speed_bar_y = y_offset + len(info_texts) * 25 + 5
        speed_bar_width = 220
        speed_bar_height = 20

        # Assume max velocity is around 10.0 (from environment config)
        max_velocity = 10.0
        speed_ratio = min(car['velocity'] / max_velocity, 1.0)

        # Background
        pygame.draw.rect(surface, (60, 60, 60),
                        (15, speed_bar_y, speed_bar_width, speed_bar_height))

        # Speed bar with color based on speed
        if speed_ratio > 0.7:
            bar_color = (76, 175, 80)  # Green - fast
        elif speed_ratio > 0.4:
            bar_color = (255, 193, 7)  # Yellow - medium
        else:
            bar_color = (244, 67, 54)  # Red - slow

        bar_width = int(speed_bar_width * speed_ratio)
        if bar_width > 0:
            pygame.draw.rect(surface, bar_color,
                           (15, speed_bar_y, bar_width, speed_bar_height))

        # Border
        pygame.draw.rect(surface, (150, 150, 150),
                        (15, speed_bar_y, speed_bar_width, speed_bar_height), 2)

    def _render_comparison_panel(self):
        """Render comparison panel at bottom."""
        # Create semi-transparent panel
        panel_height = 80
        panel_y = self.total_height - panel_height

        panel_surface = pygame.Surface((self.total_width, panel_height))
        panel_surface.set_alpha(220)
        panel_surface.fill((30, 30, 30))
        self.screen.blit(panel_surface, (0, panel_y))

        # Comparison stats
        stats1 = self.agent1_state['stats']
        stats2 = self.agent2_state['stats']

        reward_diff = stats2['reward'] - stats1['reward']
        checkpoint_diff = stats2['checkpoint'] - stats1['checkpoint']

        # Title
        title = self.title_font.render(
            f"{self.agent1_name} vs {self.agent2_name} - Comparison",
            True,
            (255, 255, 255)
        )
        self.screen.blit(title, (self.total_width // 2 - title.get_width() // 2, panel_y + 10))

        # Stats
        comp_texts = [
            f"Reward: {self.agent1_name} {stats1['reward']:.1f} | {self.agent2_name} {stats2['reward']:.1f} ({reward_diff:+.1f})",
            f"Checkpoint: {self.agent1_name} {stats1['checkpoint']}/{len(self.track.checkpoints)} | {self.agent2_name} {stats2['checkpoint']}/{len(self.track.checkpoints)} ({checkpoint_diff:+d})",
        ]

        y_offset = panel_y + 40
        for i, text in enumerate(comp_texts):
            text_surface = self.small_font.render(text, True, (200, 200, 200))
            self.screen.blit(text_surface, (20, y_offset + i * 20))

    def handle_events(self) -> bool:
        """
        Handle pygame events.

        Returns:
            False if should quit, True otherwise
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.VIDEORESIZE:
                self._handle_resize(event.w, event.h)

            elif event.type == pygame.MOUSEWHEEL:
                if event.y > 0:
                    self.cameras.zoom_in()
                else:
                    self.cameras.zoom_out()

            elif event.type == pygame.KEYDOWN:
                mods = pygame.key.get_mods()
                has_cmd = mods & (pygame.KMOD_META | pygame.KMOD_CTRL)

                # Fullscreen toggle (Cmd+F or F11)
                if (has_cmd and event.key == pygame.K_f) or event.key == pygame.K_F11:
                    self.toggle_fullscreen()

                # Camera controls (without modifier)
                elif not has_cmd:
                    if event.key in [pygame.K_w, pygame.K_UP]:
                        self.cameras.pan(0, -50)
                    elif event.key in [pygame.K_s, pygame.K_DOWN]:
                        self.cameras.pan(0, 50)
                    elif event.key in [pygame.K_a, pygame.K_LEFT]:
                        self.cameras.pan(-50, 0)
                    elif event.key in [pygame.K_d, pygame.K_RIGHT]:
                        self.cameras.pan(50, 0)

                    # Switch active camera
                    elif event.key == pygame.K_TAB:
                        self.cameras.switch_active()
                        print(f"Active camera: {self.cameras.active_camera}")

                    # Focus specific camera
                    elif event.key == pygame.K_1:
                        self.cameras.set_active("left")
                        print(f"Active camera: left ({self.agent1_name})")
                    elif event.key == pygame.K_2:
                        self.cameras.set_active("right")
                        print(f"Active camera: right ({self.agent2_name})")
                    elif event.key == pygame.K_3:
                        self.cameras.set_active("both")
                        print("Active camera: both (synchronized)")

                    # Toggle follow mode
                    elif event.key == pygame.K_f:
                        self.cameras.toggle_follow()
                        print(f"Follow mode: {self.cameras.get_active_camera().follow_car}")

                    # Reset camera
                    elif event.key == pygame.K_r:
                        self.cameras.reset()
                        print("Camera reset")

                    # Sync cameras
                    elif event.key == pygame.K_EQUALS:  # + key
                        self.cameras.sync_cameras()
                        print("Cameras synchronized")

                # Quit
                elif event.key == pygame.K_ESCAPE:
                    return False

        return True

    def toggle_fullscreen(self):
        """Toggle between fullscreen and windowed mode (Mac compatible)."""
        self.is_fullscreen = not self.is_fullscreen

        if self.is_fullscreen:
            # Get desktop resolution
            info = pygame.display.Info()
            self.total_width = info.current_w
            self.total_height = info.current_h
            self.agent_width = self.total_width // 2

            # Set fullscreen
            self.screen = pygame.display.set_mode(
                (self.total_width, self.total_height),
                pygame.FULLSCREEN
            )
            print(f"Fullscreen: {self.total_width}x{self.total_height}")
        else:
            # Restore windowed mode
            self.total_width = 2000
            self.total_height = 700
            self.agent_width = 1000

            self.screen = pygame.display.set_mode(
                (self.total_width, self.total_height),
                pygame.RESIZABLE
            )
            print(f"Windowed: {self.total_width}x{self.total_height}")

        # Recreate surfaces
        self.left_surface = pygame.Surface((self.agent_width, self.total_height))
        self.right_surface = pygame.Surface((self.agent_width, self.total_height))

        # Update camera dimensions
        self.cameras.camera_left.screen_width = self.agent_width
        self.cameras.camera_left.screen_height = self.total_height
        self.cameras.camera_right.screen_width = self.agent_width
        self.cameras.camera_right.screen_height = self.total_height

    def _handle_resize(self, new_width, new_height):
        """Handle window resize event."""
        if not self.is_fullscreen:
            self.total_width = new_width
            self.total_height = new_height
            self.agent_width = new_width // 2

            self.screen = pygame.display.set_mode(
                (new_width, new_height),
                pygame.RESIZABLE
            )

            # Recreate surfaces
            self.left_surface = pygame.Surface((self.agent_width, new_height))
            self.right_surface = pygame.Surface((self.agent_width, new_height))

            # Update camera dimensions
            self.cameras.camera_left.screen_width = self.agent_width
            self.cameras.camera_left.screen_height = new_height
            self.cameras.camera_right.screen_width = self.agent_width
            self.cameras.camera_right.screen_height = new_height

            print(f"Window resized: {new_width}x{new_height}")

    def close(self):
        """Clean up and close renderer."""
        pygame.quit()
        print("🎨 Dual Screen Renderer closed")
