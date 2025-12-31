"""
Training worker for multi-process comparison.
Runs DQN or PPO agent in separate process.
"""

import sys
from pathlib import Path
import multiprocessing as mp
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.simulation import CarEnvironment
from algorithms.dqn.agent import DQNAgent
from algorithms.ppo.agent import PPOAgent
from .shared_state import RenderingState, MetricsMessage, ControlEvents
from utils.config_loader import (
    load_environment_config,
    get_state_dim_from_config,
    get_action_dim_from_config,
)


class TrainingWorker:
    """
    Generic training worker that runs in a separate process.
    Supports both DQN and PPO agents.
    """

    def __init__(
        self,
        agent_id: int,
        agent_type: str,
        agent_config: dict,
        track_path: str,
        rendering_state: RenderingState,
        metrics_queue: mp.Queue,
        control_events: ControlEvents,
        num_episodes: int = 1000,
        render_update_freq: int = 5,
    ):
        """
        Initialize training worker.

        Args:
            agent_id: Agent ID (1 or 2)
            agent_type: "DQN" or "PPO"
            agent_config: Agent configuration dict
            track_path: Path to track JSON file
            rendering_state: Shared rendering state
            metrics_queue: Queue for sending metrics
            control_events: Control events (shutdown, pause)
            num_episodes: Number of episodes to train
            render_update_freq: Update rendering state every N steps
        """
        self.agent_id = agent_id
        self.agent_type = agent_type.upper()
        self.agent_config = agent_config
        self.track_path = track_path
        self.rendering_state = rendering_state
        self.metrics_queue = metrics_queue
        self.control_events = control_events
        self.num_episodes = num_episodes
        self.render_update_freq = render_update_freq

    def run(self):
        """Main training loop (runs in separate process)."""
        try:
            # Create environment
            env = CarEnvironment(track_path=self.track_path, render_mode=None)

            # Create agent
            if self.agent_type == "DQN":
                agent = self._create_dqn_agent()
            elif self.agent_type == "PPO":
                agent = self._create_ppo_agent()
            else:
                raise ValueError(f"Unknown agent type: {self.agent_type}")

            print(f"[Worker {self.agent_id}] {self.agent_type} agent started")

            # Training loop
            for episode in range(self.num_episodes):
                # Check for shutdown
                if self.control_events.is_shutdown():
                    print(f"[Worker {self.agent_id}] Shutdown requested")
                    break

                # Wait if paused
                while self.control_events.is_paused():
                    if self.control_events.is_shutdown():
                        break
                    mp.sleep(0.1)

                # Run episode
                episode_reward, episode_length, checkpoints_passed, loss = (
                    self._run_episode(env, agent, episode)
                )

                # Send metrics to collector
                metrics_msg = MetricsMessage(
                    agent_type=self.agent_type,
                    episode=episode,
                    reward=episode_reward,
                    length=episode_length,
                    checkpoints=checkpoints_passed,
                    loss=loss,
                )
                self.metrics_queue.put(metrics_msg)

                # Decay epsilon (DQN only) or increment episode (PPO)
                if self.agent_type == "DQN":
                    agent.decay_epsilon()
                else:  # PPO
                    agent.increment_episode()

            print(f"[Worker {self.agent_id}] Training complete!")

        except Exception as e:
            print(f"[Worker {self.agent_id}] Error: {e}")
            import traceback
            traceback.print_exc()

    def _run_episode(self, env, agent, episode: int) -> tuple:
        """
        Run single training episode.

        Args:
            env: Environment
            agent: Agent (DQN or PPO)
            episode: Episode number

        Returns:
            (episode_reward, episode_length, checkpoints_passed, avg_loss)
        """
        state = env.reset()
        episode_reward = 0.0
        episode_length = 0
        losses = []
        step_count = 0

        done = False
        while not done:
            # Select action
            action = agent.select_action(state, training=True)

            # Execute action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            agent.store_transition(state, action, reward, next_state, done)

            # Train agent
            train_result = agent.train_step()
            if train_result is not None:
                if isinstance(train_result, dict):  # PPO returns dict
                    losses.append(train_result.get('policy_loss', 0.0))
                else:  # DQN returns float
                    losses.append(train_result)

            # Update rendering state periodically
            if step_count % self.render_update_freq == 0:
                self._update_rendering_state(
                    env, episode, episode_length, episode_reward
                )

            # Update state
            state = next_state
            episode_reward += reward
            episode_length += 1
            step_count += 1

        # Final rendering update
        self._update_rendering_state(env, episode, episode_length, episode_reward)

        # Calculate average loss
        avg_loss = np.mean(losses) if losses else None

        return episode_reward, episode_length, info['checkpoint'], avg_loss

    def _update_rendering_state(
        self, env, episode: int, step: int, reward: float
    ):
        """Update shared rendering state."""
        # Get car state
        car = env.car

        # Get sensor readings
        observation = car.get_observation(env.track)
        sensors = observation[:7]  # First 7 values are sensors

        # Update shared state
        self.rendering_state.update_agent_state(
            agent_id=self.agent_id,
            car_x=car.x,
            car_y=car.y,
            car_angle=car.angle,
            car_velocity=car.velocity,
            sensors=sensors,
            episode=episode,
            step=step,
            reward=reward,
            checkpoint=env.current_checkpoint,
        )

    def _create_dqn_agent(self) -> DQNAgent:
        """Create DQN agent from config."""
        # Load environment config to get state/action dimensions
        env_config = load_environment_config()
        state_dim = self.agent_config.get('state_dim', get_state_dim_from_config(env_config))
        action_dim = self.agent_config.get('action_dim', get_action_dim_from_config(env_config))

        return DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=self.agent_config.get('hidden_dims', [128, 128]),
            learning_rate=self.agent_config.get('learning_rate', 0.0005),
            gamma=self.agent_config.get('gamma', 0.99),
            epsilon_start=self.agent_config.get('epsilon_start', 1.0),
            epsilon_end=self.agent_config.get('epsilon_end', 0.1),
            epsilon_decay=self.agent_config.get('epsilon_decay', 0.9985),
            buffer_capacity=self.agent_config.get('buffer_size', 20000),
            batch_size=self.agent_config.get('batch_size', 64),
            target_update_freq=self.agent_config.get('target_update_freq', 200),
            device=self.agent_config.get('device', None),
            double_dqn=self.agent_config.get('double_dqn', True),
        )

    def _create_ppo_agent(self) -> PPOAgent:
        """Create PPO agent from config."""
        # Load environment config to get state/action dimensions
        env_config = load_environment_config()
        state_dim = self.agent_config.get('state_dim', get_state_dim_from_config(env_config))
        action_dim = self.agent_config.get('action_dim', get_action_dim_from_config(env_config))

        return PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=self.agent_config.get('hidden_dims', [128, 128]),
            learning_rate=self.agent_config.get('learning_rate', 0.0003),
            gamma=self.agent_config.get('gamma', 0.99),
            gae_lambda=self.agent_config.get('gae_lambda', 0.95),
            clip_epsilon=self.agent_config.get('clip_epsilon', 0.2),
            value_coef=self.agent_config.get('value_coef', 0.5),
            entropy_coef=self.agent_config.get('entropy_coef', 0.01),
            max_grad_norm=self.agent_config.get('max_grad_norm', 0.5),
            update_epochs=self.agent_config.get('update_epochs', 10),
            batch_size=self.agent_config.get('batch_size', 64),
            trajectory_length=self.agent_config.get('trajectory_length', 2048),
            device=self.agent_config.get('device', None),
            continuous_actions=self.agent_config.get('continuous_actions', False),
        )


def worker_process(
    agent_id: int,
    agent_type: str,
    agent_config: dict,
    track_path: str,
    rendering_state: RenderingState,
    metrics_queue: mp.Queue,
    control_events: ControlEvents,
    num_episodes: int = 1000,
    render_update_freq: int = 5,
):
    """
    Worker process entry point.

    This function is called by multiprocessing.Process.
    """
    worker = TrainingWorker(
        agent_id=agent_id,
        agent_type=agent_type,
        agent_config=agent_config,
        track_path=track_path,
        rendering_state=rendering_state,
        metrics_queue=metrics_queue,
        control_events=control_events,
        num_episodes=num_episodes,
        render_update_freq=render_update_freq,
    )
    worker.run()
