"""
Model Evaluation Script - Test trained RL agents with visualization.
Supports DQN and PPO models with deterministic inference.
"""

import sys
import argparse
import json
import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from environment.track import Track
from environment.simulation import CarEnvironment
from algorithms.dqn.agent import DQNAgent
from algorithms.ppo.agent import PPOAgent
from visualization.camera import Camera
from utils.config_loader import (
    load_environment_config,
    get_state_dim_from_config,
    get_action_dim_from_config,
)

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


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
    # Get car corners and transform
    corners = car.get_corners()
    screen_corners = [camera.world_to_screen(x, y) for x, y in corners]
    color = (76, 175, 80)  # Green
    pygame.draw.polygon(surface, color, screen_corners)

    # Front indicator
    front_x, front_y = car.get_front_center()
    front_screen = camera.world_to_screen(front_x, front_y)
    car_screen = camera.world_to_screen(car.x, car.y)
    pygame.draw.line(surface, (255, 255, 0), car_screen, front_screen, 3)


def render_sensors_transformed(env, surface, camera):
    """Render sensors with camera transformation and color gradient."""
    car = env.car
    sensors = env.sensors

    for sensor in sensors.sensors:
        if sensor.hit_point is None:
            continue

        # Normalized distance for color (0 = far, 1 = close)
        # get_normalized_reading() returns 1.0 when at wall, 0.0 when at max range
        normalized = sensor.get_normalized_reading()

        # 6-level color gradient: green (far) → red (close)
        if normalized > 0.85:
            color = (211, 47, 47)  # Dark Red (critical)
            line_width = 3
        elif normalized > 0.7:
            color = (244, 67, 54)  # Red (danger)
            line_width = 3
        elif normalized > 0.55:
            color = (255, 152, 0)  # Orange (warning)
            line_width = 2
        elif normalized > 0.4:
            color = (255, 193, 7)  # Yellow (caution)
            line_width = 2
        elif normalized > 0.2:
            color = (139, 195, 74)  # Light Green (safe)
            line_width = 1
        else:
            color = (76, 175, 80)  # Green (clear)
            line_width = 1

        # Transform points
        car_screen = camera.world_to_screen(car.x, car.y)
        hit_screen = camera.world_to_screen(*sensor.hit_point)

        pygame.draw.line(surface, color, car_screen, hit_screen, line_width)

        # Draw small circle at hit point
        pygame.draw.circle(surface, color, hit_screen, 4)


class ModelEvaluator:
    """Evaluates trained RL models on racing tracks."""

    def __init__(
        self,
        model_path: str,
        track_path: str,
        agent_type: str = "auto",
        render: bool = True,
        fps: int = 60,
    ):
        """
        Initialize evaluator.

        Args:
            model_path: Path to .pt checkpoint file
            track_path: Path to track JSON file
            agent_type: "DQN", "PPO", or "auto" (auto-detect from checkpoint)
            render: Enable visualization
            fps: Frames per second for rendering
        """
        self.model_path = model_path
        self.track_path = track_path
        self.render = render and PYGAME_AVAILABLE
        self.fps = fps

        # Load environment config
        self.env_config = load_environment_config()
        state_dim = get_state_dim_from_config(self.env_config)
        action_dim = get_action_dim_from_config(self.env_config)

        # Create environment with correct parameters
        # Note: CarEnvironment loads its own config internally
        self.env = CarEnvironment(track_path, render_mode=None)

        # Get track reference from environment
        self.track = self.env.track
        print(f"✅ Track loaded: {self.track.name}")

        # Load agent
        self.agent_type = agent_type
        self.agent = self._load_agent(model_path, state_dim, action_dim)

        # Initialize pygame if rendering
        if self.render:
            pygame.init()
            self.screen_width = 1000
            self.screen_height = 700
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption(f"Evaluation: {Path(model_path).stem}")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)

            # Create camera
            self.camera = Camera(self.screen_width, self.screen_height)
            print("✅ Camera initialized (zoom={:.2f}x, follow={})".format(
                self.camera.zoom, self.camera.follow_car
            ))

            # Initial render to show window (macOS fix)
            self.screen.fill((34, 34, 34))
            pygame.display.flip()
            pygame.event.pump()  # Process events to show window
            time.sleep(0.1)  # Small delay to ensure window appears
            print("✅ Pygame window created and initialized")
        else:
            self.screen = None
            self.clock = None
            self.font = None
            self.camera = None

        # Metrics storage
        self.episode_results: List[Dict] = []

    def _load_agent(self, model_path: str, state_dim: int, action_dim: int):
        """Load trained agent from checkpoint."""
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')

        # Auto-detect agent type from checkpoint
        if self.agent_type == "auto":
            # Check checkpoint structure to determine type
            if "target_network" in checkpoint:
                self.agent_type = "DQN"
            elif "actor" in checkpoint or "critic" in checkpoint:
                self.agent_type = "PPO"
            else:
                raise ValueError("Cannot auto-detect agent type. Please specify --agent-type")

        print(f"📦 Loading {self.agent_type} model from: {model_path}")

        # Create agent
        if self.agent_type == "DQN":
            agent = DQNAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dims=[128, 128],
                learning_rate=0.0005,
                gamma=0.99,
                epsilon_start=0.0,  # Deterministic (no exploration)
                epsilon_end=0.0,
                epsilon_decay=1.0,
                buffer_capacity=1000,  # Fixed: was buffer_size
                batch_size=64,
                target_update_freq=200,
            )
            agent.q_network.load_state_dict(checkpoint["q_network"])  # Fixed: was policy_network
            agent.epsilon = 0.0  # Force deterministic
        elif self.agent_type == "PPO":
            agent = PPOAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dims=[128, 128],
                learning_rate=0.0003,
                gamma=0.99,
                gae_lambda=0.95,
                clip_epsilon=0.2,
                value_coef=0.5,
                entropy_coef=0.01,
                max_grad_norm=0.5,
                update_epochs=10,
                batch_size=64,
                continuous_actions=False,
            )
            agent.actor_critic.load_state_dict(checkpoint["actor_critic"])
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")

        # Set to evaluation mode
        if self.agent_type == "DQN":
            agent.q_network.eval()
        else:
            agent.actor_critic.eval()

        print(f"✅ Model loaded successfully (deterministic mode)")

        return agent

    def evaluate_episode(self, episode_num: int) -> Dict:
        """
        Run one evaluation episode.

        Returns:
            Dictionary with episode metrics
        """
        state = self.env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        checkpoints_collected = 0
        lap_completed = False
        lap_time = None
        prev_checkpoint = 0  # Track previous checkpoint to detect collection

        while not done:
            # Handle pygame events
            if self.render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return None  # Signal to stop
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            return None
                        elif event.key == pygame.K_SPACE:
                            # Toggle follow car
                            self.camera.set_follow(not self.camera.follow_car)
                        elif event.key == pygame.K_r:
                            # Reset camera
                            self.camera.reset()
                    elif event.type == pygame.MOUSEWHEEL:
                        # Zoom with mouse wheel
                        if event.y > 0:
                            self.camera.zoom_in()
                        else:
                            self.camera.zoom_out()

                # Handle arrow keys for panning
                keys = pygame.key.get_pressed()
                pan_speed = 10
                if keys[pygame.K_LEFT]:
                    self.camera.pan(-pan_speed, 0)
                    self.camera.set_follow(False)  # Disable follow when manually panning
                if keys[pygame.K_RIGHT]:
                    self.camera.pan(pan_speed, 0)
                    self.camera.set_follow(False)
                if keys[pygame.K_UP]:
                    self.camera.pan(0, -pan_speed)
                    self.camera.set_follow(False)
                if keys[pygame.K_DOWN]:
                    self.camera.pan(0, pan_speed)
                    self.camera.set_follow(False)

            # Select action (deterministic)
            if self.agent_type == "DQN":
                action = self.agent.select_action(state, training=False)
            else:  # PPO
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action_probs, _ = self.agent.actor_critic(state_tensor)
                    # Take argmax for deterministic (no sampling)
                    action = torch.argmax(action_probs, dim=1).item()

            # Step environment (Gymnasium API returns 5 values)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            total_reward += reward
            steps += 1

            # Track checkpoints (detect when checkpoint index changes)
            current_checkpoint = info.get('checkpoint', 0)
            if current_checkpoint != prev_checkpoint:
                checkpoints_collected += 1
                prev_checkpoint = current_checkpoint

            # Check if lap completed (environment uses 'finished' key)
            if info.get('finished', False):
                lap_completed = True
                lap_time = info.get('finish_time', None)

            # Update camera to follow car
            if self.render:
                self.camera.update_follow_car(self.env.car.x, self.env.car.y)

            # Render if enabled
            if self.render:
                self._render_frame(episode_num, steps, total_reward, checkpoints_collected, lap_time)

            state = next_state

        # Return episode metrics
        return {
            'episode': episode_num,
            'success': lap_completed,
            'total_reward': total_reward,
            'steps': steps,
            'checkpoints': checkpoints_collected,
            'lap_time': lap_time,
        }

    def _render_frame(
        self,
        episode: int,
        steps: int,
        reward: float,
        checkpoints: int,
        lap_time: float = None,
    ):
        """Render the current frame with metrics and camera transformation."""
        # Clear screen
        self.screen.fill((34, 34, 34))

        # Render track with camera transformation
        render_track_transformed(self.track, self.screen, self.camera)

        # Render car with camera transformation
        render_car_transformed(self.env.car, self.screen, self.camera)

        # Render sensors with camera transformation
        render_sensors_transformed(self.env, self.screen, self.camera)

        # Draw metrics panel
        self._draw_metrics_panel(episode, steps, reward, checkpoints, lap_time)

        # Draw camera controls UI
        self._draw_camera_controls()

        # Update display
        pygame.display.flip()
        self.clock.tick(self.fps)

    def _draw_metrics_panel(
        self,
        episode: int,
        steps: int,
        reward: float,
        checkpoints: int,
        lap_time: float = None,
    ):
        """Draw metrics overlay."""
        y_offset = 10
        x_offset = 10

        # Background panel
        panel_rect = pygame.Rect(x_offset - 5, y_offset - 5, 350, 180)
        pygame.draw.rect(self.screen, (0, 0, 0, 180), panel_rect)
        pygame.draw.rect(self.screen, (100, 100, 100), panel_rect, 2)

        # Title
        title = self.font.render("🎯 EVALUATION MODE", True, (76, 175, 80))
        self.screen.blit(title, (x_offset, y_offset))
        y_offset += 30

        # Metrics
        metrics = [
            f"Episode: {episode}",
            f"Steps: {steps}",
            f"Reward: {reward:.1f}",
            f"Checkpoints: {checkpoints}/{len(self.track.checkpoints)}",
        ]

        if lap_time is not None:
            metrics.append(f"Lap Time: {lap_time:.2f}s")

        for metric in metrics:
            text = self.font.render(metric, True, (255, 255, 255))
            self.screen.blit(text, (x_offset, y_offset))
            y_offset += 25

        # Model info
        y_offset += 10
        model_name = Path(self.model_path).stem
        model_text = self.font.render(f"Model: {model_name}", True, (200, 200, 200))
        self.screen.blit(model_text, (x_offset, y_offset))

    def _draw_camera_controls(self):
        """Draw camera controls UI."""
        y_offset = self.screen_height - 120
        x_offset = 10

        # Background panel
        panel_rect = pygame.Rect(x_offset - 5, y_offset - 5, 280, 115)
        pygame.draw.rect(self.screen, (0, 0, 0, 180), panel_rect)
        pygame.draw.rect(self.screen, (100, 100, 100), panel_rect, 2)

        # Title
        title = self.font.render("Camera Controls", True, (100, 200, 255))
        self.screen.blit(title, (x_offset, y_offset))
        y_offset += 25

        # Controls
        controls = [
            f"[Wheel] Zoom: {self.camera.zoom:.2f}x",
            f"[Arrows] Pan",
            f"[Space] Follow: {'ON' if self.camera.follow_car else 'OFF'}",
            f"[R] Reset Camera",
        ]

        for control in controls:
            text = self.font.render(control, True, (200, 200, 200))
            self.screen.blit(text, (x_offset, y_offset))
            y_offset += 20

    def run_evaluation(self, num_episodes: int = 10) -> Dict:
        """
        Run multiple evaluation episodes.

        Args:
            num_episodes: Number of episodes to evaluate

        Returns:
            Dictionary with aggregate statistics
        """
        print("\n" + "=" * 70)
        print("🎯 STARTING MODEL EVALUATION")
        print("=" * 70)
        print(f"Model: {Path(self.model_path).name}")
        print(f"Track: {self.track.name}")
        print(f"Episodes: {num_episodes}")
        print(f"Agent Type: {self.agent_type}")
        print(f"Rendering: {'Enabled' if self.render else 'Disabled'}")
        print("=" * 70 + "\n")

        for ep in range(1, num_episodes + 1):
            print(f"\n📊 Episode {ep}/{num_episodes}")
            print("-" * 50)

            result = self.evaluate_episode(ep)

            if result is None:  # User quit
                print("\n⚠️  Evaluation stopped by user")
                break

            self.episode_results.append(result)

            # Print episode summary
            print(f"  Success: {'✅' if result['success'] else '❌'}")
            print(f"  Reward: {result['total_reward']:.1f}")
            print(f"  Steps: {result['steps']}")
            print(f"  Checkpoints: {result['checkpoints']}/{len(self.track.checkpoints)}")
            if result['lap_time'] is not None:
                print(f"  Lap Time: {result['lap_time']:.2f}s")

        # Calculate statistics
        stats = self._calculate_statistics()
        self._print_summary(stats)

        return stats

    def _calculate_statistics(self) -> Dict:
        """Calculate aggregate statistics from all episodes."""
        if not self.episode_results:
            return {}

        successes = [r for r in self.episode_results if r['success']]
        completed_laps = [r for r in self.episode_results if r['lap_time'] is not None]

        stats = {
            'total_episodes': len(self.episode_results),
            'success_rate': len(successes) / len(self.episode_results) * 100,
            'avg_reward': np.mean([r['total_reward'] for r in self.episode_results]),
            'std_reward': np.std([r['total_reward'] for r in self.episode_results]),
            'avg_steps': np.mean([r['steps'] for r in self.episode_results]),
            'avg_checkpoints': np.mean([r['checkpoints'] for r in self.episode_results]),
            'checkpoint_rate': np.mean([r['checkpoints'] / len(self.track.checkpoints) * 100
                                       for r in self.episode_results]),
        }

        if completed_laps:
            stats['avg_lap_time'] = np.mean([r['lap_time'] for r in completed_laps])
            stats['best_lap_time'] = min([r['lap_time'] for r in completed_laps])
        else:
            stats['avg_lap_time'] = None
            stats['best_lap_time'] = None

        return stats

    def _print_summary(self, stats: Dict):
        """Print evaluation summary."""
        print("\n" + "=" * 70)
        print("📈 EVALUATION SUMMARY")
        print("=" * 70)
        print(f"Total Episodes: {stats['total_episodes']}")
        print(f"Success Rate: {stats['success_rate']:.1f}%")
        print(f"Average Reward: {stats['avg_reward']:.1f} ± {stats['std_reward']:.1f}")
        print(f"Average Steps: {stats['avg_steps']:.1f}")
        print(f"Checkpoint Collection Rate: {stats['checkpoint_rate']:.1f}%")

        if stats['avg_lap_time'] is not None:
            print(f"Average Lap Time: {stats['avg_lap_time']:.2f}s")
            print(f"Best Lap Time: {stats['best_lap_time']:.2f}s")
        else:
            print("No laps completed")

        print("=" * 70 + "\n")

    def export_results(self, output_dir: str = "evaluation_results"):
        """Export evaluation results to CSV and JSON."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        model_name = Path(self.model_path).stem
        track_name = Path(self.track_path).stem

        # Export episode results to CSV
        csv_path = output_path / f"eval_{model_name}_{track_name}.csv"
        with open(csv_path, 'w', newline='') as f:
            if self.episode_results:
                writer = csv.DictWriter(f, fieldnames=self.episode_results[0].keys())
                writer.writeheader()
                writer.writerows(self.episode_results)

        print(f"📄 Episode results exported to: {csv_path}")

        # Export summary statistics to JSON
        stats = self._calculate_statistics()
        stats['model_path'] = self.model_path
        stats['track_path'] = self.track_path
        stats['agent_type'] = self.agent_type

        json_path = output_path / f"eval_summary_{model_name}_{track_name}.json"
        with open(json_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"📊 Summary statistics exported to: {json_path}")

    def close(self):
        """Cleanup resources."""
        if self.render and PYGAME_AVAILABLE:
            pygame.quit()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained RL models on racing tracks"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--track",
        type=str,
        required=True,
        help="Path to track JSON file",
    )
    parser.add_argument(
        "--agent-type",
        type=str,
        default="auto",
        choices=["DQN", "PPO", "auto"],
        help="Agent type (auto-detect by default)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Frames per second for rendering",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable visualization (headless mode)",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export results to CSV/JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory for exported results",
    )

    args = parser.parse_args()

    # Create evaluator
    evaluator = ModelEvaluator(
        model_path=args.model,
        track_path=args.track,
        agent_type=args.agent_type,
        render=not args.no_render,
        fps=args.fps,
    )

    try:
        # Run evaluation
        stats = evaluator.run_evaluation(num_episodes=args.episodes)

        # Export results if requested
        if args.export:
            evaluator.export_results(output_dir=args.output_dir)

    finally:
        evaluator.close()


if __name__ == "__main__":
    main()
