"""
Training script for PPO agent (headless).
"""

import sys
import argparse
from pathlib import Path
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from environment.simulation import CarEnvironment
from algorithms.ppo.agent import PPOAgent


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train(
    env: CarEnvironment,
    agent: PPOAgent,
    num_episodes: int,
    writer: SummaryWriter,
    save_freq: int = 50,
    eval_freq: int = 50,
    checkpoint_dir: str = "models/checkpoints/ppo",
):
    """
    Train PPO agent.

    Args:
        env: Training environment
        agent: PPO agent
        num_episodes: Number of training episodes
        writer: TensorBoard writer
        save_freq: Save checkpoint every N episodes
        eval_freq: Evaluate agent every N episodes
        checkpoint_dir: Directory to save checkpoints
    """
    best_reward = -float('inf')
    episode_rewards = []
    episode_lengths = []

    print("\n" + "=" * 60)
    print("🚀 STARTING PPO TRAINING")
    print("=" * 60)
    print(f"Episodes: {num_episodes}")
    print(f"Device: {agent.device}")
    print(f"Trajectory Length: {agent.trajectory_length}")
    print(f"Update Epochs: {agent.update_epochs}")
    print("=" * 60 + "\n")

    # Track policy update metrics
    recent_policy_losses = []
    recent_value_losses = []
    recent_entropies = []

    for episode in tqdm(range(num_episodes), desc="Training"):
        # Reset environment
        state = env.reset()
        episode_reward = 0.0
        episode_length = 0

        # Episode loop
        done = False
        while not done:
            # Select and execute action
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            agent.store_transition(state, action, reward, next_state, done)

            # Train agent (PPO updates when trajectory buffer is full)
            metrics = agent.train_step()
            if metrics is not None:
                recent_policy_losses.append(metrics['policy_loss'])
                recent_value_losses.append(metrics['value_loss'])
                recent_entropies.append(metrics['entropy'])

                # Log to TensorBoard
                writer.add_scalar("train/policy_loss", metrics['policy_loss'], agent.updates)
                writer.add_scalar("train/value_loss", metrics['value_loss'], agent.updates)
                writer.add_scalar("train/entropy", metrics['entropy'], agent.updates)
                writer.add_scalar("train/kl_div", metrics['kl_div'], agent.updates)

            # Update state and stats
            state = next_state
            episode_reward += reward
            episode_length += 1

        # Increment episode counter
        agent.increment_episode()

        # Store episode stats
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Log to TensorBoard
        writer.add_scalar("train/episode_reward", episode_reward, episode)
        writer.add_scalar("train/episode_length", episode_length, episode)
        writer.add_scalar("train/checkpoints_passed", info['checkpoint'], episode)
        writer.add_scalar("train/buffer_size", agent.get_buffer_size(), episode)

        # Moving average reward
        if len(episode_rewards) >= 100:
            avg_reward_100 = np.mean(episode_rewards[-100:])
            writer.add_scalar("train/avg_reward_100", avg_reward_100, episode)

        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"  Avg Reward (10): {avg_reward:.2f}")
            print(f"  Avg Length (10): {avg_length:.1f}")
            print(f"  Updates: {agent.updates}")
            print(f"  Buffer Size: {agent.get_buffer_size()}")

            if recent_policy_losses:
                print(f"  Policy Loss: {np.mean(recent_policy_losses[-5:]):.4f}")
                print(f"  Value Loss: {np.mean(recent_value_losses[-5:]):.4f}")
                print(f"  Entropy: {np.mean(recent_entropies[-5:]):.4f}")

        # Save checkpoint
        if (episode + 1) % save_freq == 0:
            checkpoint_path = f"{checkpoint_dir}/checkpoint_ep{episode+1}.pt"
            agent.save(checkpoint_path)

            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_path = f"{checkpoint_dir}/best_model.pt"
                agent.save(best_path)
                print(f"  🏆 New best reward: {best_reward:.2f}")

        # Evaluate agent
        if (episode + 1) % eval_freq == 0:
            eval_reward, eval_length, eval_checkpoints = evaluate_agent(
                env, agent, num_episodes=5
            )
            writer.add_scalar("eval/reward", eval_reward, episode)
            writer.add_scalar("eval/length", eval_length, episode)
            writer.add_scalar("eval/checkpoints", eval_checkpoints, episode)

            print(f"  📊 Eval - Reward: {eval_reward:.2f}, Length: {eval_length:.1f}, Checkpoints: {eval_checkpoints:.1f}")

    print("\n" + "=" * 60)
    print("✅ PPO TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Best Reward: {best_reward:.2f}")
    print(f"Total Updates: {agent.updates}")
    print(f"Total Steps: {agent.steps}")
    print("=" * 60 + "\n")


def evaluate_agent(
    env: CarEnvironment,
    agent: PPOAgent,
    num_episodes: int = 5,
) -> tuple:
    """
    Evaluate agent performance (deterministic policy).

    Args:
        env: Environment
        agent: Agent to evaluate
        num_episodes: Number of evaluation episodes

    Returns:
        (avg_reward, avg_length, avg_checkpoints)
    """
    rewards = []
    lengths = []
    checkpoints = []

    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False

        while not done:
            # Deterministic action (no sampling)
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            state = next_state
            episode_reward += reward
            episode_length += 1

        rewards.append(episode_reward)
        lengths.append(episode_length)
        checkpoints.append(info['checkpoint'])

    return np.mean(rewards), np.mean(lengths), np.mean(checkpoints)


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent")
    parser.add_argument(
        "--track",
        type=str,
        default=None,
        help="Path to track file (overrides config)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of episodes (overrides config)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    # Load configuration
    modes_config = load_config("config/training_modes.yaml")
    mode = modes_config['ppo']  # PPO mode config

    # Use track from args or config
    track = args.track if args.track else mode['track']

    # Create environment
    env = CarEnvironment(track_path=track, render_mode=None)

    # Create PPO agent
    agent = PPOAgent(
        state_dim=mode['network']['state_dim'],
        action_dim=mode['network']['action_dim'],
        hidden_dims=mode['network']['hidden_dims'],
        learning_rate=mode['training']['learning_rate'],
        gamma=mode['training']['gamma'],
        gae_lambda=mode['ppo']['gae_lambda'],
        clip_epsilon=mode['ppo']['clip_epsilon'],
        value_coef=mode['ppo']['value_coef'],
        entropy_coef=mode['ppo']['entropy_coef'],
        max_grad_norm=mode['ppo']['max_grad_norm'],
        update_epochs=mode['training']['update_epochs'],
        batch_size=mode['training']['batch_size'],
        trajectory_length=mode['training']['trajectory_length'],
        device=mode['training']['device'],
        continuous_actions=mode['network']['continuous_actions'],
    )

    # Resume from checkpoint if specified
    if args.resume:
        agent.load(args.resume)
        print(f"📂 Resumed from {args.resume}")

    # Create TensorBoard writer
    tensorboard_dir = mode['logging']['tensorboard_dir']
    writer = SummaryWriter(tensorboard_dir)

    # Get training parameters
    num_episodes = args.episodes if args.episodes else mode['training']['num_episodes']
    save_freq = mode['checkpoint']['save_freq']
    eval_freq = mode['evaluation']['eval_freq']
    checkpoint_dir = mode['checkpoint']['save_dir']

    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Train agent
    train(
        env=env,
        agent=agent,
        num_episodes=num_episodes,
        writer=writer,
        save_freq=save_freq,
        eval_freq=eval_freq,
        checkpoint_dir=checkpoint_dir,
    )

    # Close writer
    writer.close()

    # Final save
    final_path = f"{checkpoint_dir}/final_model.pt"
    agent.save(final_path)
    print(f"💾 Final model saved to {final_path}")


if __name__ == "__main__":
    main()
