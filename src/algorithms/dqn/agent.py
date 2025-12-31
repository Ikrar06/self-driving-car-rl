"""
DQN Agent implementation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path

from .network import QNetwork, create_qnetwork
from .replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Deep Q-Network Agent with experience replay and target network.
    """

    def __init__(
        self,
        state_dim: int = 6,
        action_dim: int = 3,
        hidden_dims: list = [64, 64],
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        device: Optional[str] = None,
        double_dqn: bool = False,
        dueling: bool = False,
    ):
        """
        Initialize DQN Agent.

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            hidden_dims: Hidden layer sizes
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Epsilon decay rate per episode
            buffer_capacity: Replay buffer size
            batch_size: Minibatch size for training
            target_update_freq: Steps between target network updates
            device: 'cpu', 'cuda', or 'mps' (None for auto-detect)
            double_dqn: Use Double DQN
            dueling: Use Dueling architecture
        """
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        print(f"🔧 DQN Agent using device: {self.device}")

        # Hyperparameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.double_dqn = double_dqn

        # Epsilon-greedy parameters
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Q-Networks
        self.q_network = create_qnetwork(
            state_dim, action_dim, hidden_dims, dueling
        ).to(self.device)

        self.target_network = create_qnetwork(
            state_dim, action_dim, hidden_dims, dueling
        ).to(self.device)

        # Initialize target network with same weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Always in eval mode

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity, device=str(self.device))

        # Training stats
        self.steps = 0
        self.episodes = 0

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            training: If True, use epsilon-greedy; else greedy

        Returns:
            Selected action
        """
        state_tensor = torch.FloatTensor(state).to(self.device)

        if training:
            action = self.q_network.get_action(state_tensor, self.epsilon)
        else:
            action = self.q_network.get_action(state_tensor, epsilon=0.0)

        return action

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        """
        Perform one training step (sample batch and update Q-network).

        Returns:
            Loss value if training occurred, None otherwise
        """
        # Check if enough samples in buffer
        if not self.replay_buffer.is_ready(self.batch_size):
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Compute current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Compute target Q-values
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: use online network to select action, target network to evaluate
                next_actions = self.q_network(next_states).argmax(1, keepdim=True)
                next_q_values = self.target_network(next_states).gather(1, next_actions)
            else:
                # Standard DQN: use target network for both selection and evaluation
                next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]

            target_q_values = rewards.unsqueeze(1) + self.gamma * next_q_values * (
                1 - dones.unsqueeze(1)
            )

        # Compute loss
        loss = nn.functional.mse_loss(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update step counter
        self.steps += 1

        # Update target network periodically
        if self.steps % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()

    def update_target_network(self) -> None:
        """Update target network with current Q-network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self) -> None:
        """Decay epsilon for exploration."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.episodes += 1

    def save(self, path: str) -> None:
        """
        Save agent state.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps": self.steps,
            "episodes": self.episodes,
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        print(f"💾 Saved checkpoint to {path}")

    def load(self, path: str) -> None:
        """
        Load agent state.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.steps = checkpoint["steps"]
        self.episodes = checkpoint["episodes"]

        print(f"📂 Loaded checkpoint from {path}")

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "epsilon": self.epsilon,
            "steps": self.steps,
            "episodes": self.episodes,
            "buffer_size": len(self.replay_buffer),
        }

    def __repr__(self) -> str:
        return (
            f"DQNAgent(episodes={self.episodes}, steps={self.steps}, "
            f"epsilon={self.epsilon:.3f}, buffer_size={len(self.replay_buffer)})"
        )
