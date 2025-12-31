"""
Training with Camera Controls - Zoom, Pan, Follow Car
"""

import sys
sys.path.insert(0, 'src')

import torch
from pathlib import Path
from environment.simulation import CarEnvironment
from algorithms.dqn.agent import DQNAgent
import yaml
import pygame
import math


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class Camera:
    """Camera system with zoom and pan controls."""

    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.x = 0  # Camera offset x
        self.y = 0  # Camera offset y
        self.zoom = 1.0  # Zoom level
        self.follow_car = True  # Follow car mode
        self.drag_start = None  # For mouse dragging

    def screen_to_world(self, screen_x, screen_y):
        """Convert screen coordinates to world coordinates."""
        world_x = (screen_x - self.screen_width / 2) / self.zoom + self.x
        world_y = (screen_y - self.screen_height / 2) / self.zoom + self.y
        return world_x, world_y

    def world_to_screen(self, world_x, world_y):
        """Convert world coordinates to screen coordinates."""
        screen_x = (world_x - self.x) * self.zoom + self.screen_width / 2
        screen_y = (world_y - self.y) * self.zoom + self.screen_height / 2
        return int(screen_x), int(screen_y)

    def update_follow_car(self, car_x, car_y):
        """Update camera to follow car."""
        if self.follow_car:
            self.x = car_x
            self.y = car_y

    def zoom_in(self):
        """Zoom in (max 3x)."""
        self.zoom = min(3.0, self.zoom * 1.1)

    def zoom_out(self):
        """Zoom out (min 0.3x)."""
        self.zoom = max(0.3, self.zoom / 1.1)

    def pan(self, dx, dy):
        """Pan camera by dx, dy."""
        self.x += dx / self.zoom
        self.y += dy / self.zoom

    def reset(self):
        """Reset camera to default."""
        self.x = 0
        self.y = 0
        self.zoom = 1.0
        self.follow_car = True


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
    """Render sensors with camera transformation."""
    for sensor in car.sensors.sensors:
        if sensor.hit_point is None:
            continue

        # Color based on distance
        normalized = sensor.get_normalized_reading()
        if normalized < 0.2:
            color = (76, 175, 80)  # Green
        elif normalized < 0.8:
            color = (255, 193, 7)  # Yellow
        else:
            color = (244, 67, 54)  # Red

        # Transform points
        car_screen = camera.world_to_screen(car.x, car.y)
        hit_screen = camera.world_to_screen(*sensor.hit_point)

        pygame.draw.line(surface, color, car_screen, hit_screen, 2)
        pygame.draw.circle(surface, color, hit_screen, 4)


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


def render_camera_ui(env, camera):
    """Render camera controls UI."""
    ui_texts = [
        f"Camera Controls:",
        f"  [Mouse Wheel] Zoom: {camera.zoom:.2f}x",
        f"  [Arrow Keys] Pan",
        f"  [F] Follow Car: {'ON' if camera.follow_car else 'OFF'}",
        f"  [R] Reset Camera",
        f"  [Space] Toggle Follow",
        "",
        f"Episode Info:",
        f"  Step: {env.current_step}/{env.max_steps}",
        f"  Checkpoint: {env.current_checkpoint}/{len(env.track.checkpoints)}",
        f"  Reward: {env.total_reward:.1f}",
    ]

    y_offset = 10
    for text in ui_texts:
        text_surface = env.font.render(text, True, (255, 255, 255))
        bg_rect = text_surface.get_rect()
        bg_rect.topleft = (10, y_offset)
        bg_rect.inflate_ip(10, 5)
        pygame.draw.rect(env.screen, (40, 40, 40, 180), bg_rect)
        env.screen.blit(text_surface, (10, y_offset))
        y_offset += 25


class GhostRenderer:
    """Renders ghost cars from previous episode."""

    def __init__(self, car_width, car_height):
        self.car_width = car_width
        self.car_height = car_height
        self.previous_episode_path = []

    def update_previous_path(self, new_path):
        """Store the path from the episode that just finished."""
        self.previous_episode_path = new_path.copy()


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
    config = load_config("config/dqn_config.yaml")
    env_config = load_config("config/environment.yaml")

    # Create environment
    print(f"\n📺 Creating environment...")
    env = CarEnvironment(
        track_path=track_path,
        render_mode="human",
        max_steps=config['training']['max_steps_per_episode'],
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
        state_dim=config['network']['state_dim'],
        action_dim=config['network']['action_dim'],
        hidden_dims=config['network']['hidden_dims'],
        learning_rate=config['training']['learning_rate'],
        gamma=config['training']['gamma'],
        epsilon_start=0.3,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        buffer_capacity=config['replay']['buffer_size'],
        batch_size=config['training']['batch_size'],
        target_update_freq=config['target_network']['update_freq'],
        device=config['training'].get('device'),
        double_dqn=config['training'].get('double_dqn', False),
    )

    print(f"✓ Track: {env.track.name}")
    print(f"✓ Car: {env.car.width}x{env.car.height}")
    print(f"✓ FPS: {fps}")
    print(f"✓ Camera: Zoom={camera.zoom}x, Follow={camera.follow_car}")

    print("\n" + "=" * 70)
    print("🚀 STARTING TRAINING")
    print("=" * 70)

    best_reward = -float('inf')
    best_length = 0
    checkpoint_dir = Path(config['checkpoint']['save_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        episode_length = 0
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
        print(f"Ep {episode+1:3d}: Length={episode_length:3d}, "
              f"Reward={episode_reward:7.2f}, "
              f"Checkpoints={max_checkpoint}/{len(env.track.checkpoints)}, "
              f"Epsilon={agent.epsilon:.3f} {improvement}")

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
