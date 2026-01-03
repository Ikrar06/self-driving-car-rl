"""
Training with Camera Controls - Zoom, Pan, Follow Car
"""

import sys
sys.path.insert(0, 'src')

import torch
from pathlib import Path
from environment.simulation import CarEnvironment
from algorithms.dqn.agent import DQNAgent
from visualization.camera import Camera
from utils.config_loader import (
    load_environment_config,
    get_state_dim_from_config,
    get_action_dim_from_config,
)
import yaml
import pygame
import math


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def render_with_camera(env, camera, ghost_renderer=None):
    """Render environment with camera transformation."""
    # Clear screen
    env.screen.fill((34, 34, 34))

    # Create transformed surface
    transform_surface = pygame.Surface((env.window_width, env.window_height))
    transform_surface.fill((34, 34, 34))

    # Render track
    render_track_transformed(env.track, transform_surface, camera)

    # Render ghost if available
    if ghost_renderer:
        render_ghosts_transformed(ghost_renderer, transform_surface, camera)

    # Render car
    render_car_transformed(env.car, transform_surface, camera)

    # Blit to screen
    env.screen.blit(transform_surface, (0, 0))

    # Render UI (not transformed)
    render_camera_ui(env, camera)


def render_track_transformed(track, surface, camera):
    """Render track with camera transformation."""
    # Transform outer boundary
    outer_points = [camera.world_to_screen(x, y) for x, y in track.outer_boundary]
    if len(outer_points) > 2:
        pygame.draw.polygon(surface, (50, 50, 50), outer_points)

    # Transform inner boundary
    if track.inner_boundary and len(track.inner_boundary) > 2:
        inner_points = [camera.world_to_screen(x, y) for x, y in track.inner_boundary]
        pygame.draw.polygon(surface, (34, 34, 34), inner_points)

    # Render checkpoints
    for checkpoint in track.checkpoints:
        start_screen = camera.world_to_screen(*checkpoint.start)
        end_screen = camera.world_to_screen(*checkpoint.end)
        pygame.draw.line(surface, (100, 200, 255), start_screen, end_screen, 2)


def render_car_transformed(car, surface, camera):
    """Render car with camera transformation."""
    if not car.alive:
        color = (150, 150, 150)
    else:
        color = (66, 135, 245)

    # Get car corners and transform
    corners = car.get_corners()
    screen_corners = [camera.world_to_screen(x, y) for x, y in corners]
    pygame.draw.polygon(surface, color, screen_corners)

    # Render sensors
    if car.sensors:
        render_sensors_transformed(car, surface, camera)

    # Front indicator
    front_x, front_y = car.get_front_center()
    front_screen = camera.world_to_screen(front_x, front_y)
    car_screen = camera.world_to_screen(car.x, car.y)
    pygame.draw.line(surface, (255, 255, 0), car_screen, front_screen, 3)


def render_sensors_transformed(car, surface, camera):
    """Render sensors with camera transformation and gradient colors."""
    for sensor in car.sensors.sensors:
        if sensor.hit_point is None:
            continue

        # Color gradient based on distance (6 levels for better visualization)
        # normalized: 0.0 = far from wall (safe), 1.0 = close to wall (danger)
        normalized = sensor.get_normalized_reading()
        if normalized > 0.85:
            # CRITICAL - Very close to wall
            color = (211, 47, 47)  # Dark Red
        elif normalized > 0.7:
            # DANGER - Close to wall
            color = (244, 67, 54)  # Red
        elif normalized > 0.55:
            # WARNING - Approaching wall
            color = (255, 152, 0)  # Orange
        elif normalized > 0.4:
            # CAUTION - Medium distance
            color = (255, 193, 7)  # Yellow
        elif normalized > 0.2:
            # SAFE - Good distance
            color = (139, 195, 74)  # Light Green
        else:
            # CLEAR - Far from wall
            color = (76, 175, 80)  # Green

        # Transform points
        car_screen = camera.world_to_screen(car.x, car.y)
        hit_screen = camera.world_to_screen(*sensor.hit_point)

        # Thicker line for closer obstacles
        line_width = 3 if normalized > 0.7 else 2
        pygame.draw.line(surface, color, car_screen, hit_screen, line_width)
        pygame.draw.circle(surface, color, hit_screen, 5 if normalized > 0.7 else 4)


def render_ghosts_transformed(ghost_renderer, surface, camera):
    """Render ghost cars with camera transformation."""
    if not ghost_renderer.previous_episode_path:
        return

    for i in range(0, len(ghost_renderer.previous_episode_path), 5):
        x, y, angle = ghost_renderer.previous_episode_path[i]
        opacity = int(30 + (i / len(ghost_renderer.previous_episode_path)) * 120)

        # Calculate ghost car corners
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        hw, hh = ghost_renderer.car_width / 2, ghost_renderer.car_height / 2

        corners = [
            (x + cos_a * hw - sin_a * hh, y + sin_a * hw + cos_a * hh),
            (x + cos_a * hw + sin_a * hh, y + sin_a * hw - cos_a * hh),
            (x - cos_a * hw + sin_a * hh, y - sin_a * hw - cos_a * hh),
            (x - cos_a * hw - sin_a * hh, y - sin_a * hw + cos_a * hh),
        ]

        # Transform corners
        screen_corners = [camera.world_to_screen(cx, cy) for cx, cy in corners]
        color = (255, 100, 100)
        pygame.draw.polygon(surface, color, screen_corners, 0)


def render_speedometer(env):
    """Render speedometer with gas/brake indicator."""
    # Speedometer panel dimensions
    panel_width = 300
    panel_height = 140
    panel_x = env.window_width - panel_width - 10
    panel_y = env.window_height - panel_height - 10

    # Semi-transparent background
    panel_surface = pygame.Surface((panel_width, panel_height))
    panel_surface.set_alpha(220)
    panel_surface.fill((40, 40, 40))
    env.screen.blit(panel_surface, (panel_x, panel_y))

    # Get car velocity
    velocity = env.car.velocity
    max_velocity = env.car.max_velocity

    # Title
    title_font = pygame.font.Font(None, 32)
    title_surface = title_font.render("SPEEDOMETER", True, (255, 255, 255))
    env.screen.blit(title_surface, (panel_x + 10, panel_y + 10))

    # Speed value with large font
    speed_font = pygame.font.Font(None, 48)
    speed_text = f"{abs(velocity):.1f}"
    speed_color = (76, 175, 80) if velocity > 0 else (244, 67, 54)
    speed_surface = speed_font.render(speed_text, True, speed_color)
    env.screen.blit(speed_surface, (panel_x + 20, panel_y + 45))

    # Unit
    unit_surface = env.font.render("m/s", True, (200, 200, 200))
    env.screen.blit(unit_surface, (panel_x + 100, panel_y + 65))

    # Speed bar
    bar_x = panel_x + 10
    bar_y = panel_y + 95
    bar_width = 280
    bar_height = 30

    # Background
    pygame.draw.rect(env.screen, (60, 60, 60), (bar_x, bar_y, bar_width, bar_height))

    # Calculate speed ratio
    speed_ratio = min(abs(velocity) / max_velocity, 1.0)

    # Bar color based on speed
    if speed_ratio > 0.8:
        bar_color = (244, 67, 54)  # Red - very fast
    elif speed_ratio > 0.6:
        bar_color = (255, 193, 7)  # Yellow - fast
    elif speed_ratio > 0.3:
        bar_color = (76, 175, 80)  # Green - optimal
    else:
        bar_color = (100, 100, 255)  # Blue - slow

    # Draw speed bar
    fill_width = int(bar_width * speed_ratio)
    if fill_width > 0:
        pygame.draw.rect(env.screen, bar_color, (bar_x, bar_y, fill_width, bar_height))

    # Border
    pygame.draw.rect(env.screen, (150, 150, 150), (bar_x, bar_y, bar_width, bar_height), 2)

    # Gas/Brake indicators
    # Check if car is accelerating or braking based on sensor readings
    if env.car.sensors:
        readings = env.car.sensors.get_readings_array()
        min_reading = float(readings.min()) if len(readings) > 0 else 1.0
    else:
        min_reading = 1.0

    # Determine state
    if min_reading < 0.3:
        state_text = "🔴 BRAKE!"
        state_color = (244, 67, 54)
    elif min_reading < 0.5:
        state_text = "🟡 SLOW"
        state_color = (255, 193, 7)
    elif velocity > max_velocity * 0.7:
        state_text = "🟢 FAST"
        state_color = (76, 175, 80)
    else:
        state_text = "🔵 CRUISE"
        state_color = (100, 100, 255)

    state_surface = env.font.render(state_text, True, state_color)
    env.screen.blit(state_surface, (panel_x + 150, panel_y + 45))


def render_camera_ui(env, camera):
    """Render camera controls UI with speedometer."""
    ui_texts = [
        f"Camera Controls:",
        f"  [Mouse Wheel] Zoom: {camera.zoom:.2f}x",
        f"  [Arrow Keys] Pan",
        f"  [F] Follow Car: {'ON' if camera.follow_car else 'OFF'}",
        f"  [R] Reset Camera",
        f"  [Space] Toggle Follow",
        f"  [F11 / Cmd+F] Fullscreen",
        "",
        f"Episode Info:",
        f"  Step: {env.current_step}/{env.max_steps}",
        f"  Checkpoint: {env.current_checkpoint}/{len(env.track.checkpoints)}",
        f"  Reward: {env.total_reward:.1f}",
    ]

    # Main UI panel
    y_offset = 10
    for text in ui_texts:
        text_surface = env.font.render(text, True, (255, 255, 255))
        bg_rect = text_surface.get_rect()
        bg_rect.topleft = (10, y_offset)
        bg_rect.inflate_ip(10, 5)
        pygame.draw.rect(env.screen, (40, 40, 40, 180), bg_rect)
        env.screen.blit(text_surface, (10, y_offset))
        y_offset += 25

    # Speedometer panel (bottom right)
    render_speedometer(env)


class GhostRenderer:
    """Renders ghost cars from previous episode."""

    def __init__(self, car_width, car_height):
        self.car_width = car_width
        self.car_height = car_height
        self.previous_episode_path = []

    def update_previous_path(self, new_path):
        """Store the path from the episode that just finished."""
        self.previous_episode_path = new_path.copy()


def toggle_fullscreen(env):
    """Toggle fullscreen mode for the environment window."""
    import pygame
    import sys

    # Get current display info
    display_info = pygame.display.Info()

    if env.screen.get_flags() & pygame.FULLSCREEN:
        # Exit fullscreen - restore to windowed mode
        env.screen = pygame.display.set_mode(
            (env.window_width, env.window_height),
            pygame.RESIZABLE
        )
        print("📺 Switched to windowed mode")
    else:
        # Save current window size before going fullscreen
        if not (env.screen.get_flags() & pygame.FULLSCREEN):
            env.window_width = env.screen.get_width()
            env.window_height = env.screen.get_height()

        # Enter fullscreen
        # On macOS, use FULLSCREEN with SCALED for better compatibility
        if sys.platform == 'darwin':  # macOS
            flags = pygame.FULLSCREEN | pygame.SCALED
        else:
            flags = pygame.FULLSCREEN

        env.screen = pygame.display.set_mode(
            (display_info.current_w, display_info.current_h),
            flags
        )
        print(f"📺 Switched to fullscreen ({display_info.current_w}x{display_info.current_h})")


def train_with_camera(
    track_path: str = "tracks/f1_spa_style_long.json",
    num_episodes: int = 200,
    fps: int = 30,
):
    """Train with camera controls."""
    print("=" * 70)
    print("🎥 TRAINING WITH CAMERA CONTROLS")
    print("=" * 70)

    # Load config
    modes_config = load_config("config/training_modes.yaml")
    mode = modes_config['dqn_camera']  # Camera mode config
    shared = modes_config['shared']   # Shared config
    env_config = load_config("config/environment.yaml")

    # Create environment
    print(f"\n📺 Creating environment...")
    env = CarEnvironment(
        track_path=track_path,
        render_mode="human",
        max_steps=shared['training']['max_steps_per_episode'],
        car_width=env_config['car']['width'],
        car_height=env_config['car']['height'],
        car_max_velocity=env_config['car']['max_velocity'],
        car_min_velocity=env_config['car']['min_velocity'],
        car_acceleration=env_config['car']['acceleration'],
        car_friction=env_config['car']['friction'],
        car_turn_rate=env_config['car']['turn_rate'],
        reward_checkpoint=env_config['rewards']['checkpoint'],
        reward_survival=env_config['rewards']['survival'],
        reward_crash=env_config['rewards']['crash'],
    )
    env.fps = fps

    # Create camera
    camera = Camera(env.window_width, env.window_height)

    # Create ghost renderer
    ghost_renderer = GhostRenderer(env.car.width, env.car.height)

    # Create agent
    print("\n🤖 Creating DQN agent...")

    agent = DQNAgent(
        state_dim=shared['network']['state_dim'],
        action_dim=shared['network']['action_dim'],
        hidden_dims=shared['network']['hidden_dims'],
        learning_rate=shared['training']['learning_rate'],
        gamma=shared['training']['gamma'],
        epsilon_start=mode['exploration']['epsilon_start'],  # From config: 0.3
        epsilon_end=mode['exploration']['epsilon_end'],      # From config: 0.05
        epsilon_decay=mode['exploration']['epsilon_decay'],  # From config: 0.995
        buffer_capacity=shared['replay']['buffer_size'],
        batch_size=shared['training']['batch_size'],
        target_update_freq=shared['target_network']['update_freq'],
        device=shared['training'].get('device'),
        double_dqn=shared['training'].get('double_dqn', False),
    )

    print(f"✓ Track: {env.track.name}")
    print(f"✓ Car: {env.car.width}x{env.car.height}")
    print(f"✓ FPS: {fps}")
    print(f"✓ Camera: Zoom={camera.zoom}x, Follow={camera.follow_car}")
    print(f"✓ Mode: {mode['name']}")

    print("\n" + "=" * 70)
    print("🚀 STARTING TRAINING")
    print("=" * 70)

    best_reward = -float('inf')
    best_length = 0
    best_finish_time = float('inf')  # Track fastest lap time
    checkpoint_dir = Path(mode['checkpoint']['save_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        episode_length = 0
        episode_finish_time = None  # Will be set if agent finishes
        current_episode_path = []
        max_checkpoint = 0
        done = False

        while not done:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        env.close()
                        return
                    elif event.key == pygame.K_F11:
                        # F11 for fullscreen (Windows/Linux style)
                        toggle_fullscreen(env)
                    elif event.key == pygame.K_f and (event.mod & pygame.KMOD_META or event.mod & pygame.KMOD_CTRL):
                        # Cmd+F (Mac) or Ctrl+F for fullscreen
                        toggle_fullscreen(env)
                    elif event.key == pygame.K_f or event.key == pygame.K_SPACE:
                        camera.follow_car = not camera.follow_car
                    elif event.key == pygame.K_r:
                        camera.reset()
                    elif event.key == pygame.K_UP:
                        camera.pan(0, -50)
                    elif event.key == pygame.K_DOWN:
                        camera.pan(0, 50)
                    elif event.key == pygame.K_LEFT:
                        camera.pan(-50, 0)
                    elif event.key == pygame.K_RIGHT:
                        camera.pan(50, 0)
                elif event.type == pygame.MOUSEWHEEL:
                    if event.y > 0:
                        camera.zoom_in()
                    else:
                        camera.zoom_out()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 2:  # Middle mouse button
                        camera.drag_start = event.pos
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 2:
                        camera.drag_start = None
                elif event.type == pygame.MOUSEMOTION:
                    if camera.drag_start:
                        dx = event.pos[0] - camera.drag_start[0]
                        dy = event.pos[1] - camera.drag_start[1]
                        camera.pan(-dx, -dy)
                        camera.drag_start = event.pos
                elif event.type == pygame.VIDEORESIZE:
                    # Handle window resize
                    env.window_width = event.w
                    env.window_height = event.h
                    env.screen = pygame.display.set_mode(
                        (event.w, event.h),
                        pygame.RESIZABLE
                    )
                    camera.screen_width = event.w
                    camera.screen_height = event.h

            # Select action
            action = agent.select_action(state, training=True)

            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store position
            current_episode_path.append((env.car.x, env.car.y, env.car.angle))

            # Update camera to follow car
            camera.update_follow_car(env.car.x, env.car.y)

            # Render with camera
            render_with_camera(env, camera, ghost_renderer)
            pygame.display.flip()
            env.clock.tick(env.fps)

            # Track progress
            if info['checkpoint'] > max_checkpoint:
                max_checkpoint = info['checkpoint']

            # Track finish time if agent completed the lap
            if info.get('finish_time') is not None:
                episode_finish_time = info['finish_time']

            # Store and train
            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            episode_reward += reward
            episode_length += 1

        # Update ghost
        ghost_renderer.update_previous_path(current_episode_path)

        # Decay epsilon
        agent.decay_epsilon()

        # Print summary
        improvement = "📈" if episode_length > best_length else ""
        finish_info = ""
        if episode_finish_time is not None:
            finish_info = f", ⏱️ Time={episode_finish_time:.2f}s"
            if episode_finish_time < best_finish_time:
                best_finish_time = episode_finish_time
                finish_info += " 🏆"  # New record!

        print(f"Ep {episode+1:3d}: Length={episode_length:3d}, "
              f"Reward={episode_reward:7.2f}, "
              f"Checkpoints={max_checkpoint}/{len(env.track.checkpoints)}, "
              f"Epsilon={agent.epsilon:.3f}{finish_info} {improvement}")

        # Track best
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_length = episode_length
            save_path = checkpoint_dir / "best_model_camera.pt"
            agent.save(str(save_path))

        # Periodic save
        if (episode + 1) % 50 == 0:
            save_path = checkpoint_dir / f"camera_ep{episode + 1}.pt"
            agent.save(str(save_path))
            print(f"  💾 Checkpoint saved")

    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Best Reward: {best_reward:.2f}")
    print(f"Best Length: {best_length} steps")
    if best_finish_time < float('inf'):
        print(f"🏆 Best Lap Time: {best_finish_time:.2f}s")
    print("=" * 70)

    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--track", type=str, default="tracks/f1_spa_style_long.json")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    train_with_camera(
        track_path=args.track,
        num_episodes=args.episodes,
        fps=args.fps,
    )
