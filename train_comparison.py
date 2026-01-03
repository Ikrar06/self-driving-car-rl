"""
Side-by-side comparison training: DQN vs PPO
Runs both agents in parallel with split-screen visualization.
"""

import sys
import argparse
from pathlib import Path
import yaml
import multiprocessing as mp
import time
from torch.utils.tensorboard import SummaryWriter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from environment.track import Track
from comparison.coordinator import DualTrainingCoordinator
from comparison.metrics_collector import MetricsCollector
from visualization.dual_screen_renderer import DualScreenRenderer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Side-by-side comparison training: DQN vs PPO"
    )
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
        "--fps",
        type=int,
        default=None,
        help="Target FPS for visualization (overrides config)",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable visualization (headless mode)",
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="Start in fullscreen mode",
    )
    args = parser.parse_args()

    # Load configuration
    modes_config = load_config("config/training_modes.yaml")
    config = modes_config['comparison']  # Comparison mode config

    # Override config with command-line args
    track_path = args.track if args.track else config['track']
    num_episodes = args.episodes if args.episodes else config['num_episodes']
    fps = args.fps if args.fps else config.get('fps', 60)
    enable_viz = not args.no_viz and config['visualization']['enabled']

    # Set multiprocessing start method for Mac compatibility
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    print("\n" + "=" * 70)
    print("🏁 SIDE-BY-SIDE COMPARISON TRAINING")
    print("=" * 70)
    print(f"Agent 1: {config['agent1']['name']} ({config['agent1']['type']})")
    print(f"Agent 2: {config['agent2']['name']} ({config['agent2']['type']})")
    print(f"Track: {track_path}")
    print(f"Episodes: {num_episodes}")
    print(f"FPS: {fps}")
    print(f"Visualization: {'Enabled' if enable_viz else 'Disabled (Headless)'}")
    print("=" * 70 + "\n")

    # Load track
    track = Track.load(track_path)
    print(f"✅ Track loaded: {track.name}")

    # Create coordinator
    coordinator = DualTrainingCoordinator(
        agent1_type=config['agent1']['type'],
        agent2_type=config['agent2']['type'],
        agent1_config=config['agent1']['config'],
        agent2_config=config['agent2']['config'],
        track_path=track_path,
        num_episodes=num_episodes,
        render_update_freq=config.get('render_update_freq', 5),
    )

    # Create metrics collector
    metrics_collector = MetricsCollector(coordinator.get_metrics_queue())

    # Create TensorBoard writers
    tensorboard_dir = config['logging']['tensorboard_dir']
    writer_agent1 = SummaryWriter(f"{tensorboard_dir}/{config['agent1']['name'].lower()}")
    writer_agent2 = SummaryWriter(f"{tensorboard_dir}/{config['agent2']['name'].lower()}")
    writer_comparison = SummaryWriter(f"{tensorboard_dir}/comparison")

    print(f"📊 TensorBoard logging to: {tensorboard_dir}")

    # Create renderer if visualization enabled
    renderer = None
    if enable_viz:
        renderer = DualScreenRenderer(
            track=track,
            agent1_name=config['agent1']['name'],
            agent2_name=config['agent2']['name'],
            total_width=config['visualization']['window_width'],
            total_height=config['visualization']['window_height'],
            fps=fps,
        )
        print(f"🎨 Visualization enabled: {config['visualization']['window_width']}x{config['visualization']['window_height']}")

        # Start fullscreen if requested
        if args.fullscreen or config['visualization']['fullscreen']:
            renderer.toggle_fullscreen()

    # Start training workers
    coordinator.start()
    print("\n⏳ Training in progress...\n")

    # Main loop
    try:
        running = True
        last_metrics_log = 0

        while coordinator.is_running() and running:
            # Collect metrics from queue
            msg_dict = metrics_collector.collect(timeout=0.01)
            if msg_dict:
                # Log to TensorBoard
                agent_type = msg_dict['agent_type']
                episode = msg_dict['episode']
                reward = msg_dict['reward']
                length = msg_dict['length']
                checkpoints = msg_dict['checkpoints']

                # Select appropriate writer
                if agent_type == config['agent1']['type']:
                    writer = writer_agent1
                else:
                    writer = writer_agent2

                # Log metrics
                writer.add_scalar("train/episode_reward", reward, episode)
                writer.add_scalar("train/episode_length", length, episode)
                writer.add_scalar("train/checkpoints", checkpoints, episode)

                if msg_dict.get('loss') is not None:
                    writer.add_scalar("train/loss", msg_dict['loss'], episode)

                # Print progress
                if config['logging']['verbose'] and episode % config['logging']['log_freq'] == 0:
                    print(f"[{agent_type}] Episode {episode}: Reward={reward:.1f}, Length={length}, CP={checkpoints}")

            # Update visualization if enabled
            if enable_viz and renderer:
                # Get rendering state from shared memory
                rendering_state = coordinator.get_rendering_state()

                # Get state for both agents
                car1, sensors1, stats1, ready1 = rendering_state.get_agent_state(1)
                car2, sensors2, stats2, ready2 = rendering_state.get_agent_state(2)

                # Update renderer
                if ready1 or ready2:
                    if ready1:
                        renderer.update_agent_state(1, car1, sensors1, stats1)
                    if ready2:
                        renderer.update_agent_state(2, car2, sensors2, stats2)

                    # Render frame
                    renderer.render_frame()

                # Handle events
                running = renderer.handle_events()

            # Small sleep to prevent busy waiting
            time.sleep(0.001)

        # Wait for workers to complete
        if coordinator.is_running():
            coordinator.wait_for_completion()

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")

    finally:
        # Shutdown coordinator
        coordinator.shutdown()

        # Close renderer
        if renderer:
            renderer.close()

        # Collect remaining metrics
        print("\n📊 Collecting final metrics...")
        metrics_collector.collect_all(timeout=0.5)

        # Print summary
        metrics_collector.print_summary()

        # Log comparison metrics to TensorBoard
        comparison_stats = metrics_collector.get_comparison_stats()
        if comparison_stats:
            agent_types = list(metrics_collector.agent_metrics.keys())
            if len(agent_types) >= 2:
                comp_key = f"{agent_types[0]}_vs_{agent_types[1]}"
                if comp_key in comparison_stats:
                    comp = comparison_stats[comp_key]
                    writer_comparison.add_scalar("reward_difference", comp['reward_difference'], 0)
                    writer_comparison.add_scalar("reward_ratio", comp['reward_ratio'], 0)

        # Export metrics to CSV if configured
        if config['logging']['save_metrics_csv']:
            csv_dir = f"{tensorboard_dir}/metrics_csv"
            metrics_collector.export_to_csv(csv_dir)

        # Close TensorBoard writers
        writer_agent1.close()
        writer_agent2.close()
        writer_comparison.close()

        print("\n" + "=" * 70)
        print("✅ COMPARISON TRAINING COMPLETE!")
        print("=" * 70)
        print(f"View results in TensorBoard:")
        print(f"  tensorboard --logdir {tensorboard_dir}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
