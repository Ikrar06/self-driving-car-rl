"""
Dual training coordinator for side-by-side agent comparison.
Manages two training processes (DQN vs PPO).
"""

import multiprocessing as mp
import time
from typing import Optional
from .shared_state import RenderingState, ControlEvents
from .worker import worker_process


class DualTrainingCoordinator:
    """
    Coordinates two training workers for side-by-side comparison.
    Manages process lifecycle, shared memory, and communication.
    """

    def __init__(
        self,
        agent1_type: str,
        agent2_type: str,
        agent1_config: dict,
        agent2_config: dict,
        track_path: str,
        num_episodes: int = 1000,
        render_update_freq: int = 5,
    ):
        """
        Initialize dual training coordinator.

        Args:
            agent1_type: Type of first agent ("DQN" or "PPO")
            agent2_type: Type of second agent ("DQN" or "PPO")
            agent1_config: Configuration for first agent
            agent2_config: Configuration for second agent
            track_path: Path to track JSON file
            num_episodes: Number of episodes to train
            render_update_freq: Update rendering every N steps
        """
        self.agent1_type = agent1_type
        self.agent2_type = agent2_type
        self.agent1_config = agent1_config
        self.agent2_config = agent2_config
        self.track_path = track_path
        self.num_episodes = num_episodes
        self.render_update_freq = render_update_freq

        # Shared state for rendering
        self.rendering_state = RenderingState()

        # Metrics queue
        self.metrics_queue = mp.Queue()

        # Control events
        self.control_events = ControlEvents()

        # Worker processes
        self.worker1: Optional[mp.Process] = None
        self.worker2: Optional[mp.Process] = None

        print("🎮 Dual Training Coordinator initialized")
        print(f"   Agent 1: {agent1_type}")
        print(f"   Agent 2: {agent2_type}")
        print(f"   Track: {track_path}")
        print(f"   Episodes: {num_episodes}")

    def start(self):
        """Start both training workers."""
        print("\n🚀 Starting training workers...")

        # Create worker processes
        self.worker1 = mp.Process(
            target=worker_process,
            args=(
                1,  # agent_id
                self.agent1_type,
                self.agent1_config,
                self.track_path,
                self.rendering_state,
                self.metrics_queue,
                self.control_events,
                self.num_episodes,
                self.render_update_freq,
            ),
            name=f"Worker-{self.agent1_type}",
        )

        self.worker2 = mp.Process(
            target=worker_process,
            args=(
                2,  # agent_id
                self.agent2_type,
                self.agent2_config,
                self.track_path,
                self.rendering_state,
                self.metrics_queue,
                self.control_events,
                self.num_episodes,
                self.render_update_freq,
            ),
            name=f"Worker-{self.agent2_type}",
        )

        # Start processes
        self.worker1.start()
        self.worker2.start()

        print(f"✅ Worker 1 ({self.agent1_type}) started (PID: {self.worker1.pid})")
        print(f"✅ Worker 2 ({self.agent2_type}) started (PID: {self.worker2.pid})")

    def pause(self):
        """Pause both workers."""
        self.control_events.pause()
        print("⏸️  Training paused")

    def resume(self):
        """Resume both workers."""
        self.control_events.resume()
        print("▶️  Training resumed")

    def shutdown(self):
        """Gracefully shutdown both workers."""
        print("\n🛑 Shutting down workers...")

        # Signal shutdown
        self.control_events.shutdown()

        # Wait for workers to finish (with timeout)
        timeout = 10.0
        start_time = time.time()

        if self.worker1 and self.worker1.is_alive():
            remaining_time = max(0, timeout - (time.time() - start_time))
            self.worker1.join(timeout=remaining_time)
            if self.worker1.is_alive():
                print(f"⚠️  Worker 1 did not stop, terminating...")
                self.worker1.terminate()
                self.worker1.join()

        if self.worker2 and self.worker2.is_alive():
            remaining_time = max(0, timeout - (time.time() - start_time))
            self.worker2.join(timeout=remaining_time)
            if self.worker2.is_alive():
                print(f"⚠️  Worker 2 did not stop, terminating...")
                self.worker2.terminate()
                self.worker2.join()

        print("✅ All workers stopped")

    def is_running(self) -> bool:
        """Check if any worker is still running."""
        worker1_alive = self.worker1 is not None and self.worker1.is_alive()
        worker2_alive = self.worker2 is not None and self.worker2.is_alive()
        return worker1_alive or worker2_alive

    def get_rendering_state(self) -> RenderingState:
        """Get shared rendering state."""
        return self.rendering_state

    def get_metrics_queue(self) -> mp.Queue:
        """Get metrics queue."""
        return self.metrics_queue

    def wait_for_completion(self):
        """Wait for both workers to complete training."""
        print("\n⏳ Waiting for training to complete...")

        if self.worker1:
            self.worker1.join()
            print(f"✅ Worker 1 ({self.agent1_type}) completed")

        if self.worker2:
            self.worker2.join()
            print(f"✅ Worker 2 ({self.agent2_type}) completed")

        print("🎉 All training complete!")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure cleanup."""
        self.shutdown()
