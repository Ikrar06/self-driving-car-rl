"""
PPO Actor-Critic Network implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Tuple, Optional


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic Network for PPO with shared feature extractor.

    Architecture:
        Input (state_dim) → Shared Features [hidden_dims]
            ├─→ Actor Head → Action Distribution (action_dim)
            └─→ Critic Head → Value Estimate (scalar)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [128, 128],
    ):
        """
        Initialize Actor-Critic Network.

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            hidden_dims: Hidden layer sizes for shared feature extractor
        """
        super(ActorCriticNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared feature extractor
        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        self.shared_features = nn.Sequential(*layers)

        # Actor head (policy network)
        self.actor = nn.Linear(hidden_dims[-1], action_dim)

        # Critic head (value network)
        self.critic = nn.Linear(hidden_dims[-1], 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

        # Special initialization for actor head (smaller weights for exploration)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            state: State tensor (batch_size, state_dim) or (state_dim,)

        Returns:
            action_logits: Logits for action distribution (batch_size, action_dim)
            value: Value estimate (batch_size, 1) or (1,)
        """
        # Extract shared features
        features = self.shared_features(state)

        # Get action logits and value
        action_logits = self.actor(features)
        value = self.critic(features)

        return action_logits, value

    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Select action from policy.

        Args:
            state: State tensor (state_dim,)
            deterministic: If True, select argmax; else sample from distribution

        Returns:
            action: Selected action (int)
            log_prob: Log probability of the action
            value: Value estimate for the state
        """
        with torch.no_grad():
            # Add batch dimension if needed
            if state.dim() == 1:
                state = state.unsqueeze(0)

            # Get action logits and value
            action_logits, value = self.forward(state)

            # Create categorical distribution
            dist = Categorical(logits=action_logits)

            # Select action
            if deterministic:
                action = torch.argmax(action_logits, dim=-1)
            else:
                action = dist.sample()

            # Get log probability
            log_prob = dist.log_prob(action)

            # Remove batch dimension
            action = action.item()
            log_prob = log_prob.squeeze()
            value = value.squeeze()

        return action, log_prob, value

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.

        Args:
            states: State batch (batch_size, state_dim)
            actions: Action batch (batch_size,)

        Returns:
            log_probs: Log probabilities of actions (batch_size,)
            values: Value estimates (batch_size,)
            entropy: Entropy of the policy distribution (batch_size,)
        """
        # Get action logits and values
        action_logits, values = self.forward(states)

        # Create categorical distribution
        dist = Categorical(logits=action_logits)

        # Get log probabilities
        log_probs = dist.log_prob(actions)

        # Get entropy
        entropy = dist.entropy()

        # Remove extra dimensions from values
        values = values.squeeze(-1)

        return log_probs, values, entropy


class ContinuousActorCriticNetwork(nn.Module):
    """
    Actor-Critic Network for continuous action spaces (optional extension).

    Uses Gaussian policy with state-dependent standard deviation.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [128, 128],
        log_std_init: float = 0.0,
    ):
        """
        Initialize Continuous Actor-Critic Network.

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension (continuous)
            hidden_dims: Hidden layer sizes
            log_std_init: Initial value for log std
        """
        super(ContinuousActorCriticNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared feature extractor
        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        self.shared_features = nn.Sequential(*layers)

        # Actor head (mean)
        self.actor_mean = nn.Linear(hidden_dims[-1], action_dim)

        # Actor head (log std) - learnable or fixed
        self.actor_log_std = nn.Parameter(
            torch.ones(action_dim) * log_std_init
        )

        # Critic head
        self.critic = nn.Linear(hidden_dims[-1], 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            state: State tensor

        Returns:
            mean: Mean of action distribution
            std: Standard deviation of action distribution
            value: Value estimate
        """
        features = self.shared_features(state)

        mean = self.actor_mean(features)
        std = torch.exp(self.actor_log_std)
        value = self.critic(features)

        return mean, std, value


def create_actor_critic(
    state_dim: int,
    action_dim: int,
    hidden_dims: list = [128, 128],
    continuous: bool = False,
) -> nn.Module:
    """
    Factory function to create Actor-Critic network.

    Args:
        state_dim: State space dimension
        action_dim: Action space dimension
        hidden_dims: Hidden layer sizes
        continuous: If True, create continuous action network

    Returns:
        Actor-Critic network
    """
    if continuous:
        return ContinuousActorCriticNetwork(state_dim, action_dim, hidden_dims)
    else:
        return ActorCriticNetwork(state_dim, action_dim, hidden_dims)
