"""
Shared state management for multi-process training comparison.
Uses multiprocessing.Array for lock-free shared memory.
"""

import multiprocessing as mp
import numpy as np
from typing import Tuple, Optional
import ctypes


class RenderingState:
    """
    Manages shared memory for rendering two agents simultaneously.

    Layout:
    - Agent 1 (DQN): car state (4) + sensors (7) + stats (4) = 15 floats
    - Agent 2 (PPO): car state (4) + sensors (7) + stats (4) = 15 floats
    - Frame ready flags: 2 bools

    Total: 30 floats + 2 bools (~130 bytes)
    """

    # State indices
    CAR_X = 0
    CAR_Y = 1
    CAR_ANGLE = 2
    CAR_VELOCITY = 3
    SENSORS_START = 4  # 7 sensors: indices 4-10
    STAT_EPISODE = 11
    STAT_STEP = 12
    STAT_REWARD = 13
    STAT_CHECKPOINT = 14

    STATE_SIZE = 15  # Size per agent

    def __init__(self):
        """Initialize shared memory arrays."""
        # Shared arrays for both agents (lock-free)
        self.agent1_state = mp.Array('f', self.STATE_SIZE)
        self.agent2_state = mp.Array('f', self.STATE_SIZE)

        # Frame ready flags
        self.agent1_ready = mp.Value(ctypes.c_bool, False)
        self.agent2_ready = mp.Value(ctypes.c_bool, False)

        # Initialize to zeros
        self._clear_agent_state(self.agent1_state)
        self._clear_agent_state(self.agent2_state)

    def _clear_agent_state(self, state_array):
        """Clear agent state array."""
        for i in range(self.STATE_SIZE):
            state_array[i] = 0.0

    def update_agent_state(
        self,
        agent_id: int,
        car_x: float,
        car_y: float,
        car_angle: float,
        car_velocity: float,
        sensors: np.ndarray,
        episode: int,
        step: int,
        reward: float,
        checkpoint: int,
    ) -> None:
        """
        Update state for specified agent (lock-free writes).

        Args:
            agent_id: 1 for DQN, 2 for PPO
            car_x: Car x position
            car_y: Car y position
            car_angle: Car angle in radians
            car_velocity: Car velocity
            sensors: Array of 7 sensor readings
            episode: Current episode number
            step: Current step number
            reward: Current episode reward
            checkpoint: Current checkpoint index
        """
        # Select appropriate state array
        state = self.agent1_state if agent_id == 1 else self.agent2_state

        # Update car state
        state[self.CAR_X] = car_x
        state[self.CAR_Y] = car_y
        state[self.CAR_ANGLE] = car_angle
        state[self.CAR_VELOCITY] = car_velocity

        # Update sensors (7 values)
        for i, sensor_val in enumerate(sensors[:7]):
            state[self.SENSORS_START + i] = sensor_val

        # Update stats
        state[self.STAT_EPISODE] = float(episode)
        state[self.STAT_STEP] = float(step)
        state[self.STAT_REWARD] = reward
        state[self.STAT_CHECKPOINT] = float(checkpoint)

        # Set frame ready flag
        if agent_id == 1:
            self.agent1_ready.value = True
        else:
            self.agent2_ready.value = True

    def get_agent_state(self, agent_id: int) -> Tuple:
        """
        Get current state for specified agent.

        Args:
            agent_id: 1 for DQN, 2 for PPO

        Returns:
            Tuple of (car_state, sensors, stats, ready)
        """
        # Select appropriate state array
        state = self.agent1_state if agent_id == 1 else self.agent2_state
        ready_flag = self.agent1_ready if agent_id == 1 else self.agent2_ready

        # Extract car state
        car_state = {
            'x': state[self.CAR_X],
            'y': state[self.CAR_Y],
            'angle': state[self.CAR_ANGLE],
            'velocity': state[self.CAR_VELOCITY],
        }

        # Extract sensors
        sensors = [state[self.SENSORS_START + i] for i in range(7)]

        # Extract stats
        stats = {
            'episode': int(state[self.STAT_EPISODE]),
            'step': int(state[self.STAT_STEP]),
            'reward': state[self.STAT_REWARD],
            'checkpoint': int(state[self.STAT_CHECKPOINT]),
        }

        # Get and clear ready flag
        ready = ready_flag.value
        if ready:
            ready_flag.value = False

        return car_state, sensors, stats, ready

    def clear_ready_flags(self):
        """Clear both ready flags."""
        self.agent1_ready.value = False
        self.agent2_ready.value = False

    def is_any_ready(self) -> bool:
        """Check if any agent has new data."""
        return self.agent1_ready.value or self.agent2_ready.value


class MetricsMessage:
    """
    Message format for metrics queue.
    """

    def __init__(
        self,
        agent_type: str,
        episode: int,
        reward: float,
        length: int,
        checkpoints: int,
        loss: Optional[float] = None,
        additional_metrics: Optional[dict] = None,
    ):
        """
        Initialize metrics message.

        Args:
            agent_type: "DQN" or "PPO"
            episode: Episode number
            reward: Episode reward
            length: Episode length (steps)
            checkpoints: Checkpoints passed
            loss: Training loss (optional)
            additional_metrics: Additional agent-specific metrics
        """
        self.agent_type = agent_type
        self.episode = episode
        self.reward = reward
        self.length = length
        self.checkpoints = checkpoints
        self.loss = loss
        self.additional_metrics = additional_metrics or {}

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'agent_type': self.agent_type,
            'episode': self.episode,
            'reward': self.reward,
            'length': self.length,
            'checkpoints': self.checkpoints,
            'loss': self.loss,
            **self.additional_metrics,
        }

    def __repr__(self):
        return (
            f"MetricsMessage({self.agent_type}, "
            f"ep={self.episode}, r={self.reward:.1f}, "
            f"len={self.length}, cp={self.checkpoints})"
        )


class ControlEvents:
    """
    Control events for coordinating workers.
    """

    def __init__(self):
        """Initialize control events."""
        self.shutdown_event = mp.Event()
        self.pause_event = mp.Event()

    def shutdown(self):
        """Signal shutdown to all workers."""
        self.shutdown_event.set()

    def is_shutdown(self) -> bool:
        """Check if shutdown was requested."""
        return self.shutdown_event.is_set()

    def pause(self):
        """Pause training."""
        self.pause_event.set()

    def resume(self):
        """Resume training."""
        self.pause_event.clear()

    def is_paused(self) -> bool:
        """Check if paused."""
        return self.pause_event.is_set()
