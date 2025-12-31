"""
Trajectory Buffer for PPO with GAE computation.
"""

import numpy as np
import torch
from typing import Tuple, List


class TrajectoryBuffer:
    """
    Stores trajectories for on-policy PPO training.
    Computes advantages using Generalized Advantage Estimation (GAE).
    """

    def __init__(self, capacity: int = 2048, device: str = "cpu"):
        """
        Initialize Trajectory Buffer.

        Args:
            capacity: Maximum number of transitions to store
            device: Device for tensor conversion ('cpu', 'cuda', 'mps')
        """
        self.capacity = capacity
        self.device = device

        # Storage
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.dones: List[bool] = []

        # Computed advantages and returns
        self.advantages: Optional[np.ndarray] = None
        self.returns: Optional[np.ndarray] = None

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        """
        Add transition to buffer.

        Args:
            state: State observation
            action: Action taken
            reward: Reward received
            value: Value estimate V(s) from critic
            log_prob: Log probability of action log π(a|s)
            done: Terminal flag
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def compute_gae(
        self,
        next_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """
        Compute Generalized Advantage Estimation (GAE).

        GAE(λ) formula:
            δt = rt + γV(st+1) - V(st)
            At = Σ(γλ)^k δt+k

        Args:
            next_value: Value estimate for next state V(s_{T+1})
            gamma: Discount factor
            gae_lambda: GAE lambda parameter (bias-variance tradeoff)
        """
        # Convert lists to numpy arrays
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        # Compute advantages using GAE
        advantages = np.zeros_like(rewards)
        last_gae_lambda = 0.0

        # Backward pass to compute advantages
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value_t = values[t + 1]
                next_non_terminal = 1.0 - dones[t]

            # TD error: δt = rt + γV(st+1) - V(st)
            delta = rewards[t] + gamma * next_value_t * next_non_terminal - values[t]

            # GAE: At = δt + γλδt+1 + (γλ)^2δt+2 + ...
            advantages[t] = last_gae_lambda = (
                delta + gamma * gae_lambda * next_non_terminal * last_gae_lambda
            )

        # Compute returns: Rt = At + V(st)
        returns = advantages + values

        # Store computed advantages and returns
        self.advantages = advantages
        self.returns = returns

    def get(self) -> Tuple[torch.Tensor, ...]:
        """
        Get all stored transitions as tensors.

        Returns:
            Tuple of (states, actions, old_log_probs, returns, advantages)
        """
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        returns = torch.FloatTensor(self.returns).to(self.device)
        advantages = torch.FloatTensor(self.advantages).to(self.device)

        # Normalize advantages (improves stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return states, actions, old_log_probs, returns, advantages

    def clear(self) -> None:
        """Clear buffer after policy update."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = None
        self.returns = None

    def is_ready(self, min_size: int = 1) -> bool:
        """
        Check if buffer has enough samples.

        Args:
            min_size: Minimum number of samples required

        Returns:
            True if buffer size >= min_size
        """
        return len(self.states) >= min_size

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.states)

    def get_batch_iterator(
        self,
        batch_size: int,
        shuffle: bool = True,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Generator that yields mini-batches for PPO updates.

        Args:
            batch_size: Size of mini-batches
            shuffle: Whether to shuffle indices

        Yields:
            Mini-batches of (states, actions, old_log_probs, returns, advantages)
        """
        # Get all data
        states, actions, old_log_probs, returns, advantages = self.get()

        # Total number of samples
        n_samples = len(states)

        # Generate indices
        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)

        # Yield mini-batches
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]

            yield (
                states[batch_indices],
                actions[batch_indices],
                old_log_probs[batch_indices],
                returns[batch_indices],
                advantages[batch_indices],
            )


class RolloutBuffer:
    """
    Alternative rollout buffer implementation with episode tracking.
    Useful for environments with variable episode lengths.
    """

    def __init__(self, capacity: int = 2048, device: str = "cpu"):
        """
        Initialize Rollout Buffer.

        Args:
            capacity: Maximum buffer size
            device: Device for tensors
        """
        self.capacity = capacity
        self.device = device

        # Current episode storage
        self.current_episode = {
            "states": [],
            "actions": [],
            "rewards": [],
            "values": [],
            "log_probs": [],
            "dones": [],
        }

        # Completed episodes
        self.episodes: List[dict] = []

        # Flattened buffer (for training)
        self.buffer = TrajectoryBuffer(capacity, device)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        """Add transition to current episode."""
        self.current_episode["states"].append(state)
        self.current_episode["actions"].append(action)
        self.current_episode["rewards"].append(reward)
        self.current_episode["values"].append(value)
        self.current_episode["log_probs"].append(log_prob)
        self.current_episode["dones"].append(done)

        # If episode is done, store it
        if done:
            self.episodes.append(self.current_episode.copy())
            # Reset current episode
            self.current_episode = {
                "states": [],
                "actions": [],
                "rewards": [],
                "values": [],
                "log_probs": [],
                "dones": [],
            }

    def finalize(self, next_value: float, gamma: float = 0.99, gae_lambda: float = 0.95):
        """
        Finalize buffer by computing GAE for all episodes.

        Args:
            next_value: Value of next state (for incomplete episode)
            gamma: Discount factor
            gae_lambda: GAE lambda
        """
        # Add current incomplete episode to episodes list
        if len(self.current_episode["states"]) > 0:
            self.episodes.append(self.current_episode.copy())

        # Flatten all episodes into buffer
        for episode in self.episodes:
            for i in range(len(episode["states"])):
                self.buffer.push(
                    episode["states"][i],
                    episode["actions"][i],
                    episode["rewards"][i],
                    episode["values"][i],
                    episode["log_probs"][i],
                    episode["dones"][i],
                )

        # Compute GAE on flattened buffer
        self.buffer.compute_gae(next_value, gamma, gae_lambda)

    def get(self) -> Tuple[torch.Tensor, ...]:
        """Get all data from buffer."""
        return self.buffer.get()

    def get_batch_iterator(self, batch_size: int, shuffle: bool = True):
        """Get mini-batch iterator."""
        return self.buffer.get_batch_iterator(batch_size, shuffle)

    def clear(self):
        """Clear all stored data."""
        self.current_episode = {
            "states": [],
            "actions": [],
            "rewards": [],
            "values": [],
            "log_probs": [],
            "dones": [],
        }
        self.episodes = []
        self.buffer.clear()

    def __len__(self):
        """Return total number of transitions."""
        return len(self.buffer)
