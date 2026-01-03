"""
Training with GHOST visualization - shows previous episode's path.
This lets you see if the agent is improving compared to last episode.
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


class GhostRenderer:
    """Renders ghost cars from previous episode."""

    def __init__(self, screen, car_width, car_height):
        self.screen = screen
        self.car_width = car_width
        self.car_height = car_height
        self.previous_episode_path = []

    def update_previous_path(self, new_path):
        """Store the path from the episode that just finished."""
        self.previous_episode_path = new_path.copy()

    def render_ghosts(self):
        """Render ghost cars from previous episode."""
        if not self.previous_episode_path:
            return

        # Draw every 5th position to avoid clutter
        for i in range(0, len(self.previous_episode_path), 5):
            x, y, angle = self.previous_episode_path[i]

            # Calculate opacity based on position (earlier = more transparent)
            opacity = int(30 + (i / len(self.previous_episode_path)) * 120)

            # Draw ghost car
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            hw, hh = self.car_width / 2, self.car_height / 2

            corners = [
                (x + cos_a * hw - sin_a * hh, y + sin_a * hw + cos_a * hh),
                (x + cos_a * hw + sin_a * hh, y + sin_a * hw - cos_a * hh),
                (x - cos_a * hw + sin_a * hh, y - sin_a * hw - cos_a * hh),
                (x - cos_a * hw - sin_a * hh, y - sin_a * hw + cos_a * hh),
            ]

            # Red/orange ghost color
            color = (255, 100, 100, opacity)
            pygame.draw.polygon(self.screen, color[:3], corners, 0)


def train_with_ghost_comparison(
    track_path: str = "tracks/simple_straight.json",
    num_episodes: int = 200,
    fps: int = 30,
):
    """
    Train agent with ghost visualization showing previous episode.

    Args:
        track_path: Path to track
        num_episodes: Number of training episodes
        fps: Frames per second
    """
    print("=" * 70)
    print("👻 TRAINING WITH GHOST COMPARISON")
    print("=" * 70)
    print("Current episode: BRIGHT car")
    print("Previous episode: RED/ORANGE GHOST")
    print("Watch to see if agent improves each episode!")
    print("=" * 70)

    # Load config
    modes_config = load_config("config/training_modes.yaml")
    mode = modes_config['dqn_ghost']  # Ghost mode config
    shared = modes_config['shared']   # Shared config
    env_config = load_config("config/environment.yaml")

    # Create environment with car physics AND rewards from environment.yaml
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
    # Disable built-in trail
    env.max_trail_length = 0

    print(f"✓ Track: {env.track.name}")
    print(f"✓ FPS: {fps}")
    print(f"✓ Mode: {mode['name']}")

    # Create ghost renderer
    ghost_renderer = GhostRenderer(env.screen, env.car.width, env.car.height)

    # Create agent
    print("\n🤖 Creating DQN agent...")
    agent = DQNAgent(
        state_dim=shared['network']['state_dim'],
        action_dim=shared['network']['action_dim'],
        hidden_dims=shared['network']['hidden_dims'],
        learning_rate=shared['training']['learning_rate'],
        gamma=shared['training']['gamma'],
        epsilon_start=mode['exploration']['epsilon_start'],
        epsilon_end=mode['exploration']['epsilon_end'],
        epsilon_decay=mode['exploration']['epsilon_decay'],
        buffer_capacity=shared['replay']['buffer_size'],
        batch_size=shared['training']['batch_size'],
        target_update_freq=shared['target_network']['update_freq'],
        device=shared['training'].get('device'),
        double_dqn=shared['training'].get('double_dqn', False),
    )
    print(f"✓ Device: {agent.device}")

    print("\n" + "=" * 70)
    print("🚀 STARTING TRAINING")
    print("=" * 70)
    print(f"Episodes: {num_episodes}")
    print(f"Epsilon: {agent.epsilon:.3f} → {agent.epsilon_end}")
    print("\n💡 VISUALIZATION:")
    print("   - RED GHOSTS = Previous episode's path")
    print("   - BRIGHT CAR = Current episode")
    print("   - Agent should learn to go FURTHER than ghosts!")
    print("=" * 70 + "\n")

    best_reward = -float('inf')
    best_length = 0
    checkpoint_dir = Path(mode['checkpoint']['save_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for episode in range(num_episodes):
        # Reset environment
        state = env.reset()
        episode_reward = 0.0
        episode_length = 0
        current_episode_path = []
        max_checkpoint = 0

        # Episode loop
        done = False

        while not done:
            # Check for quit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    env.close()
                    return

            # Select action
            action = agent.select_action(state, training=True)

            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store current position for ghost
            current_episode_path.append((env.car.x, env.car.y, env.car.angle))

            # Custom rendering with ghost
            env.screen.fill((34, 34, 34))
            env.track.render(env.screen, show_checkpoints=True)

            # Render ghosts FIRST (behind current car)
            ghost_renderer.render_ghosts()

            # Render current car
            env.car.render(env.screen, show_sensors=True)

            # Render info panel with comparison
            info_texts = [
                f"Episode: {episode + 1}/{num_episodes}",
                f"Current Length: {episode_length}",
                f"Best Length: {best_length}",
                f"Reward: {episode_reward:.1f}",
                f"Epsilon: {agent.epsilon:.3f}",
                f"Checkpoints: {info['checkpoint']}/{len(env.track.checkpoints)}",
            ]

            y_offset = 10
            for i, text in enumerate(info_texts):
                # Background
                text_surface = env.font.render(text, True, (255, 255, 255))
                bg_rect = text_surface.get_rect()
                bg_rect.topleft = (10, y_offset + i * 30)
                bg_rect.inflate_ip(10, 5)
                pygame.draw.rect(env.screen, (40, 40, 40), bg_rect)
                # Text
                env.screen.blit(text_surface, (10, y_offset + i * 30))

            pygame.display.flip()
            env.clock.tick(env.fps)

            # Track checkpoint progress
            if info['checkpoint'] > max_checkpoint:
                max_checkpoint = info['checkpoint']

            # Store transition and train
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train_step()

            # Update
            state = next_state
            episode_reward += reward
            episode_length += 1

        # Episode finished - update ghost with current path
        ghost_renderer.update_previous_path(current_episode_path)

        # Decay epsilon
        agent.decay_epsilon()

        # Print summary
        improvement = "📈 IMPROVED!" if episode_length > best_length else ""
        finished_flag = "🏁 FINISHED LAP!" if info.get('finished', False) else ""
        print(f"Ep {episode+1:3d}: Length={episode_length:3d}, "
              f"Reward={episode_reward:7.2f}, "
              f"Checkpoints={max_checkpoint}/{len(env.track.checkpoints)}, "
              f"Epsilon={agent.epsilon:.3f} {improvement} {finished_flag}")

        # Track best
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_length = episode_length
            save_path = checkpoint_dir / "best_model_ghost.pt"
            agent.save(str(save_path))

        # Periodic checkpoint
        if (episode + 1) % 50 == 0:
            save_path = checkpoint_dir / f"ghost_ep{episode + 1}.pt"
            agent.save(str(save_path))
            print(f"  💾 Checkpoint saved")

    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Best Reward: {best_reward:.2f}")
    print(f"Best Length: {best_length} steps")
    print(f"Final Epsilon: {agent.epsilon:.3f}")
    print("=" * 70)

    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--track", type=str, default="tracks/simple_straight.json")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    train_with_ghost_comparison(
        track_path=args.track,
        num_episodes=args.episodes,
        fps=args.fps,
    )
