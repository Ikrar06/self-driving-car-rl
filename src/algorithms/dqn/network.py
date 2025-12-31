"""
Q-Network architecture for DQN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class QNetwork(nn.Module):
    """
    Q-Network: estimates Q-values for each action given a state.

    Architecture: MLP with 2 hidden layers
    Input: state (6 values: 5 sensors + velocity)
    Output: Q-values for 3 actions (LEFT, STRAIGHT, RIGHT)
    """

    def __init__(
        self,
        state_dim: int = 6,
        action_dim: int = 3,
        hidden_dims: List[int] = [64, 64],
    ):
        """
        Initialize Q-Network.

        Args:
            state_dim: Dimension of state space (default: 6)
            action_dim: Number of actions (default: 3)
            hidden_dims: List of hidden layer sizes
        """
        super(QNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims

        # Build layers
        layers = []
        input_dim = state_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        # Output layer (Q-values for each action)
        layers.append(nn.Linear(input_dim, action_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            state: Batch of states, shape (batch_size, state_dim)

        Returns:
            Q-values for each action, shape (batch_size, action_dim)
        """
        return self.network(state)

    def get_action(self, state: torch.Tensor, epsilon: float = 0.0) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Single state, shape (state_dim,)
            epsilon: Exploration probability [0-1]

        Returns:
            Selected action index
        """
        # Exploration: random action
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.action_dim, (1,)).item()

        # Exploitation: greedy action
        with torch.no_grad():
            # Add batch dimension
            state = state.unsqueeze(0)
            q_values = self.forward(state)
            action = q_values.argmax(dim=1).item()

        return action


class DuelingQNetwork(nn.Module):
    """
    Dueling Q-Network architecture (optional advanced version).
    Separates value and advantage streams for better learning.
    """

    def __init__(
        self,
        state_dim: int = 6,
        action_dim: int = 3,
        hidden_dims: List[int] = [64, 64],
    ):
        """
        Initialize Dueling Q-Network.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            hidden_dims: List of hidden layer sizes
        """
        super(DuelingQNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared feature extraction
        self.features = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
        )

        # Value stream (estimates state value)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),
        )

        # Advantage stream (estimates action advantages)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], action_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dueling architecture.

        Args:
            state: Batch of states, shape (batch_size, state_dim)

        Returns:
            Q-values, shape (batch_size, action_dim)
        """
        # Shared features
        features = self.features(state)

        # Value and advantage streams
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values

    def get_action(self, state: torch.Tensor, epsilon: float = 0.0) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Single state, shape (state_dim,)
            epsilon: Exploration probability

        Returns:
            Selected action index
        """
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.action_dim, (1,)).item()

        with torch.no_grad():
            state = state.unsqueeze(0)
            q_values = self.forward(state)
            action = q_values.argmax(dim=1).item()

        return action


def create_qnetwork(
    state_dim: int = 6,
    action_dim: int = 3,
    hidden_dims: List[int] = [64, 64],
    dueling: bool = False,
) -> nn.Module:
    """
    Factory function to create Q-Network.

    Args:
        state_dim: State dimension
        action_dim: Action dimension
        hidden_dims: Hidden layer sizes
        dueling: Use Dueling architecture (default: False)

    Returns:
        Q-Network instance
    """
    if dueling:
        return DuelingQNetwork(state_dim, action_dim, hidden_dims)
    else:
        return QNetwork(state_dim, action_dim, hidden_dims)
