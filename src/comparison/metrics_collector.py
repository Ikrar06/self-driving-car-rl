"""
Metrics collection and aggregation for comparison training.
"""

import multiprocessing as mp
from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np
import csv
from pathlib import Path


class MetricsCollector:
    """
    Collects and aggregates metrics from multiple training workers.
    """

    def __init__(self, metrics_queue: mp.Queue):
        """
        Initialize metrics collector.

        Args:
            metrics_queue: Queue to receive metrics from workers
        """
        self.metrics_queue = metrics_queue

        # Storage for each agent
        self.agent_metrics = defaultdict(lambda: {
            'episode_rewards': [],
            'episode_lengths': [],
            'checkpoints_passed': [],
            'losses': [],
            'episodes': [],
        })

    def collect(self, timeout: float = 0.1) -> Optional[Dict]:
        """
        Collect metrics from queue (non-blocking).

        Args:
            timeout: Timeout for queue.get()

        Returns:
            Metrics dict if available, None otherwise
        """
        try:
            msg = self.metrics_queue.get(timeout=timeout)
            self._store_metrics(msg)
            return msg.to_dict()
        except:
            return None

    def collect_all(self, timeout: float = 0.1):
        """Collect all pending metrics from queue."""
        while True:
            msg_dict = self.collect(timeout=timeout)
            if msg_dict is None:
                break

    def _store_metrics(self, msg):
        """Store metrics from message."""
        agent_type = msg.agent_type
        metrics = self.agent_metrics[agent_type]

        metrics['episodes'].append(msg.episode)
        metrics['episode_rewards'].append(msg.reward)
        metrics['episode_lengths'].append(msg.length)
        metrics['checkpoints_passed'].append(msg.checkpoints)

        if msg.loss is not None:
            metrics['losses'].append(msg.loss)

    def get_agent_stats(self, agent_type: str) -> Dict:
        """
        Get statistics for specific agent.

        Args:
            agent_type: "DQN" or "PPO"

        Returns:
            Dictionary with statistics
        """
        metrics = self.agent_metrics[agent_type]

        if not metrics['episode_rewards']:
            return {}

        rewards = metrics['episode_rewards']
        lengths = metrics['episode_lengths']
        checkpoints = metrics['checkpoints_passed']

        # Calculate stats
        stats = {
            'total_episodes': len(rewards),
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            'mean_length': np.mean(lengths),
            'mean_checkpoints': np.mean(checkpoints),
            'max_checkpoints': np.max(checkpoints),
        }

        # Recent performance (last 100 episodes)
        if len(rewards) >= 100:
            stats['recent_mean_reward'] = np.mean(rewards[-100:])
            stats['recent_mean_length'] = np.mean(lengths[-100:])
            stats['recent_mean_checkpoints'] = np.mean(checkpoints[-100:])

        # Recent performance (last 10 episodes)
        if len(rewards) >= 10:
            stats['last10_mean_reward'] = np.mean(rewards[-10:])

        return stats

    def get_comparison_stats(self) -> Dict:
        """
        Get comparative statistics between agents.

        Returns:
            Dictionary with comparison metrics
        """
        agent_types = list(self.agent_metrics.keys())

        if len(agent_types) < 2:
            return {}

        agent1_type = agent_types[0]
        agent2_type = agent_types[1]

        agent1_stats = self.get_agent_stats(agent1_type)
        agent2_stats = self.get_agent_stats(agent2_type)

        if not agent1_stats or not agent2_stats:
            return {}

        comparison = {
            f'{agent1_type}_vs_{agent2_type}': {
                'reward_difference': agent1_stats['mean_reward'] - agent2_stats['mean_reward'],
                'reward_ratio': agent1_stats['mean_reward'] / agent2_stats['mean_reward'] if agent2_stats['mean_reward'] != 0 else 0,
                'length_difference': agent1_stats['mean_length'] - agent2_stats['mean_length'],
                'checkpoint_difference': agent1_stats['mean_checkpoints'] - agent2_stats['mean_checkpoints'],
            },
            agent1_type: agent1_stats,
            agent2_type: agent2_stats,
        }

        return comparison

    def export_to_csv(self, output_dir: str):
        """
        Export all metrics to CSV files.

        Args:
            output_dir: Directory to save CSV files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for agent_type, metrics in self.agent_metrics.items():
            csv_path = output_path / f"{agent_type.lower()}_metrics.csv"

            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)

                # Header
                writer.writerow([
                    'episode',
                    'reward',
                    'length',
                    'checkpoints',
                    'loss',
                ])

                # Data rows
                for i in range(len(metrics['episodes'])):
                    episode = metrics['episodes'][i]
                    reward = metrics['episode_rewards'][i]
                    length = metrics['episode_lengths'][i]
                    checkpoint = metrics['checkpoints_passed'][i]
                    loss = metrics['losses'][i] if i < len(metrics['losses']) else ''

                    writer.writerow([episode, reward, length, checkpoint, loss])

            print(f"📊 Exported {agent_type} metrics to {csv_path}")

    def print_summary(self):
        """Print summary of collected metrics."""
        print("\n" + "=" * 60)
        print("📊 TRAINING METRICS SUMMARY")
        print("=" * 60)

        for agent_type in self.agent_metrics.keys():
            stats = self.get_agent_stats(agent_type)

            if not stats:
                print(f"\n{agent_type}: No data collected")
                continue

            print(f"\n{agent_type}:")
            print(f"  Episodes: {stats['total_episodes']}")
            print(f"  Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
            print(f"  Max Reward: {stats['max_reward']:.2f}")
            print(f"  Mean Length: {stats['mean_length']:.1f}")
            print(f"  Mean Checkpoints: {stats['mean_checkpoints']:.1f} / {stats['max_checkpoints']}")

            if 'recent_mean_reward' in stats:
                print(f"  Recent (100 ep): {stats['recent_mean_reward']:.2f}")

            if 'last10_mean_reward' in stats:
                print(f"  Last 10 ep: {stats['last10_mean_reward']:.2f}")

        # Comparison
        comparison = self.get_comparison_stats()
        if comparison and len(self.agent_metrics) >= 2:
            agent_types = list(self.agent_metrics.keys())
            comp_key = f"{agent_types[0]}_vs_{agent_types[1]}"

            if comp_key in comparison:
                print(f"\n{'=' * 60}")
                print(f"COMPARISON: {agent_types[0]} vs {agent_types[1]}")
                print(f"{'=' * 60}")

                comp_stats = comparison[comp_key]
                print(f"  Reward Difference: {comp_stats['reward_difference']:+.2f}")
                print(f"  Reward Ratio: {comp_stats['reward_ratio']:.2f}x")
                print(f"  Length Difference: {comp_stats['length_difference']:+.1f}")
                print(f"  Checkpoint Difference: {comp_stats['checkpoint_difference']:+.1f}")

        print("=" * 60 + "\n")

    def get_raw_metrics(self, agent_type: str) -> Dict[str, List]:
        """
        Get raw metrics arrays for specified agent.

        Args:
            agent_type: "DQN" or "PPO"

        Returns:
            Dictionary with raw metric lists
        """
        return dict(self.agent_metrics[agent_type])
