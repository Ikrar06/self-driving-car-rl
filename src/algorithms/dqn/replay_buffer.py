"""
Experience Replay Buffer for DQN.
"""

import numpy as np
import torch
from collections import deque
from typing import Tuple, List
import random


class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.
    Implements experience replay to break correlation between consecutive samples.
    """

    def __init__(self, capacity: int = 10000, device: str = "cpu"):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of experiences to store
            device: PyTorch device ('cpu', 'cuda', or 'mps')
        """
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Add experience to buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a random batch of experiences.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
            All as PyTorch tensors on the specified device
        """
        # Sample random experiences
        experiences = random.sample(self.buffer, batch_size)

        # Unzip into separate lists
        states, actions, rewards, next_states, dones = zip(*experiences)

        # Convert to PyTorch tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Return current size of buffer."""
        return len(self.buffer)

    def is_ready(self, min_size: int) -> bool:
        """
        Check if buffer has enough samples for training.

        Args:
            min_size: Minimum number of samples required

        Returns:
            True if buffer size >= min_size
        """
        return len(self.buffer) >= min_size

    def clear(self) -> None:
        """Clear all experiences from buffer."""
        self.buffer.clear()


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay (PER) - optional advanced version.
    Samples experiences based on TD error magnitude.
    """

    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
        device: str = "cpu",
    ):
        """
        Initialize prioritized replay buffer.

        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0=uniform, 1=full prioritization)
            beta: Importance sampling exponent (0=no correction, 1=full correction)
            beta_increment: Beta annealing rate
            epsilon: Small constant to prevent zero priorities
            device: PyTorch device
        """
        super().__init__(capacity, device)

        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon

        # Store priorities separately
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add experience with maximum priority."""
        super().push(state, action, reward, next_state, done)
        self.priorities.append(self.max_priority)

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """
        Sample batch based on priorities.

        Returns:
            states, actions, rewards, next_states, dones, weights, indices
        """
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)

        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*experiences)

        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize

        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices: List[int], td_errors: np.ndarray) -> None:
        """
        Update priorities based on TD errors.

        Args:
            indices: Indices of sampled experiences
            td_errors: TD errors for those experiences
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
