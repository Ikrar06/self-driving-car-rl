"""
PPO Agent implementation with DQN-compatible API.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path

from .network import ActorCriticNetwork, create_actor_critic
from .trajectory_buffer import TrajectoryBuffer


class PPOAgent:
    """
    Proximal Policy Optimization Agent.
    API compatible with DQNAgent for easy comparison.
    """

    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 3,
        hidden_dims: list = [128, 128],
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        batch_size: int = 64,
        trajectory_length: int = 2048,
        device: Optional[str] = None,
        continuous_actions: bool = False,
    ):
        """
        Initialize PPO Agent.

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            hidden_dims: Hidden layer sizes
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            update_epochs: Number of epochs per PPO update
            batch_size: Mini-batch size for updates
            trajectory_length: Steps before policy update
            device: 'cpu', 'cuda', or 'mps' (None for auto-detect)
            continuous_actions: Use continuous action space
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
        print(f"🔧 PPO Agent using device: {self.device}")

        # Hyperparameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.trajectory_length = trajectory_length
        self.continuous_actions = continuous_actions

        # Actor-Critic Network
        self.actor_critic = create_actor_critic(
            state_dim, action_dim, hidden_dims, continuous_actions
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(), lr=learning_rate, eps=1e-5
        )

        # Trajectory buffer
        self.trajectory_buffer = TrajectoryBuffer(
            capacity=trajectory_length, device=str(self.device)
        )

        # Training stats
        self.steps = 0
        self.episodes = 0
        self.updates = 0

        # Temporary storage for current step
        self._last_value = None

    def select_action(
        self, state: np.ndarray, training: bool = True
    ) -> int:
        """
        Select action using current policy.

        Args:
            state: Current state
            training: If True, sample from policy; else use deterministic

        Returns:
            Selected action
        """
        state_tensor = torch.FloatTensor(state).to(self.device)

        # Get action from policy
        action, log_prob, value = self.actor_critic.get_action(
            state_tensor, deterministic=not training
        )

        # Store value for later use in store_transition
        self._last_value = value.item()

        return action

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store transition in trajectory buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Terminal flag
        """
        # Need to get log_prob and value for this transition
        # We already have value from select_action
        state_tensor = torch.FloatTensor(state).to(self.device)

        with torch.no_grad():
            action_logits, _ = self.actor_critic(state_tensor)
            dist = torch.distributions.Categorical(logits=action_logits)
            action_tensor = torch.tensor(action, device=self.device)
            log_prob = dist.log_prob(action_tensor).item()

        # Store in buffer
        self.trajectory_buffer.push(
            state=state,
            action=action,
            reward=reward,
            value=self._last_value if self._last_value is not None else 0.0,
            log_prob=log_prob,
            done=done,
        )

    def train_step(self) -> Optional[Dict[str, float]]:
        """
        Perform PPO update when trajectory buffer is full.

        Returns:
            Dictionary with loss metrics if update occurred, None otherwise
        """
        # Check if trajectory buffer is ready
        if not self.trajectory_buffer.is_ready(self.trajectory_length):
            return None

        # Get next value for GAE computation
        # Use zero if last transition was terminal
        if self.trajectory_buffer.dones[-1]:
            next_value = 0.0
        else:
            # Get value of last state
            last_state = torch.FloatTensor(self.trajectory_buffer.states[-1]).to(
                self.device
            )
            with torch.no_grad():
                _, next_value_tensor = self.actor_critic(last_state)
                next_value = next_value_tensor.item()

        # Compute GAE
        self.trajectory_buffer.compute_gae(next_value, self.gamma, self.gae_lambda)

        # Perform multiple epochs of PPO updates
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl_div = 0.0
        update_count = 0

        for epoch in range(self.update_epochs):
            # Iterate over mini-batches
            for batch in self.trajectory_buffer.get_batch_iterator(
                self.batch_size, shuffle=True
            ):
                states, actions, old_log_probs, returns, advantages = batch

                # Evaluate actions with current policy
                log_probs, values, entropy = self.actor_critic.evaluate_actions(
                    states, actions
                )

                # Compute policy loss (clipped surrogate objective)
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                    * advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Compute value loss
                value_loss = nn.functional.mse_loss(values, returns)

                # Compute entropy bonus (for exploration)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()

                # Approximate KL divergence (for monitoring)
                approx_kl = (old_log_probs - log_probs).mean().item()
                total_kl_div += approx_kl

                update_count += 1

        # Clear trajectory buffer after update
        self.trajectory_buffer.clear()

        # Update counters
        self.steps += self.trajectory_length
        self.updates += 1

        # Return average metrics
        return {
            "policy_loss": total_policy_loss / update_count,
            "value_loss": total_value_loss / update_count,
            "entropy": total_entropy / update_count,
            "kl_div": total_kl_div / update_count,
        }

    def save(self, path: str) -> None:
        """
        Save agent state.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "actor_critic": self.actor_critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
            "episodes": self.episodes,
            "updates": self.updates,
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        print(f"💾 Saved PPO checkpoint to {path}")

    def load(self, path: str) -> None:
        """
        Load agent state.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.actor_critic.load_state_dict(checkpoint["actor_critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps = checkpoint["steps"]
        self.episodes = checkpoint["episodes"]
        self.updates = checkpoint.get("updates", 0)

        print(f"📂 Loaded PPO checkpoint from {path}")

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "steps": self.steps,
            "episodes": self.episodes,
            "updates": self.updates,
            "buffer_size": len(self.trajectory_buffer),
        }

    def increment_episode(self) -> None:
        """Increment episode counter (call at end of episode)."""
        self.episodes += 1

    def get_buffer_size(self) -> int:
        """Get current trajectory buffer size."""
        return len(self.trajectory_buffer)
