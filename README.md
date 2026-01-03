# AI Racing Trainer - Deep Reinforcement Learning Racing Simulator

> Train autonomous racing cars using DQN & PPO with custom physics engine, side-by-side comparison training, and interactive track builder

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Pygame](https://img.shields.io/badge/Pygame-2.5+-green.svg)](https://www.pygame.org/)

## Overview

A complete reinforcement learning racing simulator featuring:
- **Dual RL Algorithms** - DQN and PPO with side-by-side comparison training
- **Realistic Physics** - Smooth steering with angular damping, 9-action discrete control
- **7-Sensor Ray-Casting** - Color-coded distance gradient (green=far, red=close)
- **Interactive Track Builder** - Visual editor with smoothing, file dialog, and optimization
- **Split-Screen Training** - Compare DQN vs PPO performance in real-time
- **Performance Optimized** - Triple-level caching for lag-free training with complex tracks
- **Finish Time Tracking** - Speed-based rewards for faster lap completion

## Features

### Dual RL Algorithms
- **DQN (Deep Q-Network)** - Off-policy value-based learning
  - Experience replay with 20k buffer
  - Double DQN for stable Q-values
  - Epsilon-greedy exploration (1.0 → 0.1)
- **PPO (Proximal Policy Optimization)** - On-policy actor-critic
  - GAE (λ=0.95) for advantage estimation
  - Clipped surrogate loss (ε=0.2)
  - Entropy bonus for exploration
- **Side-by-Side Comparison** - Train both agents simultaneously with split-screen visualization

### Advanced Car Physics
- **Smooth Steering** - Angular velocity with damping (not instant snapping)
- **9 Discrete Actions** - 3 steering directions × 3 speed levels
  - LEFT/STRAIGHT/RIGHT × SLOW/NORMAL/FAST
  - Agent learns speed control, steering, and braking
- **Realistic Dynamics** - Acceleration, friction, angular damping
- **Finish Time Rewards** - Bonus for faster lap completion

### Enhanced Sensors
- **7 ray-cast sensors** at -90°, -60°, -30°, 0°, +30°, +60°, +90°
- **6-level color gradient** - Green (safe) → Yellow → Orange → Red (danger)
- **Dynamic line width** - Thicker lines when closer to walls
- Normalized readings for neural network input

### Track Builder Pro
- **Interactive visual editor** with mouse controls
- **Chaikin's smoothing algorithm** - Round sharp corners (press M)
- **File dialog picker** - GUI for loading tracks (with fallback)
- **Camera system** - Zoom (0.2x-5.0x), pan, fullscreen
- **Car size reference** - 30×15px overlay at cursor
- **Grid snapping** - Precise placement
- **Performance optimized** - Handles 40k+ point tracks without lag

### Training Features
- **Split-screen comparison** - DQN vs PPO side-by-side (2000×700)
- **Real-time visualization** - Camera follow with independent controls
- **Finish time tracking** - Speed-based or target-based rewards
- **Episode statistics** - Steps, rewards, checkpoints, lap times
- **Model checkpointing** - Best + periodic saves
- **Fullscreen support** - Mac-optimized with resize handling

### Performance Optimization
- **Triple-level track caching**:
  - Rendering: ~500 points (display)
  - Collision: ~1500 points (physics)
  - Original: Full detail preserved
- **Fast decimation** - O(n) uniform sampling for 5k+ points
- **Douglas-Peucker** - Iterative shape preservation for <5k points
- **Result**: Lag-free training even with 39k+ point tracks!

## Quick Start

### Prerequisites

**macOS (M1/M2/M3/M4):**
```bash
# Install SDL2 dependencies
brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf
```

**Linux:**
```bash
sudo apt-get install python3-dev libsdl2-dev libsdl2-image-dev \
                     libsdl2-mixer-dev libsdl2-ttf-dev
```

### Installation

```bash
# Clone repository
git clone https://github.com/Ikrar06/self-driving-car-rl.git
cd self-driving-car-rl

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### Build Custom Tracks

```bash
# Launch track builder
python3 track_builder.py
```

**Track Builder Controls:**
- **[1-4]** - Switch mode (Outer/Inner/Checkpoint/Start)
- **[M]** - Smooth selected boundary (Chaikin's algorithm)
- **[Mouse Wheel]** - Zoom in/out
- **[W/A/S/D]** or **[Arrow Keys]** - Pan camera
- **[Click]** - Place points
- **[C]** - Close boundary
- **[Z]** - Undo last point
- **[R]** - Rotate start angle
- **[Cmd+S]** - Save track
- **[Cmd+L]** - Load track (GUI dialog)
- **[Cmd+F]** or **[F11]** - Toggle fullscreen
- **[V]** - Toggle car reference
- **[G]** - Toggle grid snap

#### Train Single Agent

**DQN:**
```bash
python3 train_with_camera.py --track tracks/oval_easy.json --episodes 1000
```

**PPO:**
```bash
python3 train_ppo.py --track tracks/oval_easy.json --episodes 1000
```

#### Compare DQN vs PPO

```bash
# Side-by-side comparison with split-screen (2000×700)
python3 train_comparison.py

# Custom track and episodes
python3 train_comparison.py --track tracks/oval_easy.json --episodes 300

# Adjust FPS for performance
python3 train_comparison.py --fps 30

# Combine all options
python3 train_comparison.py --track tracks/megacool_track.json --episodes 500 --fps 60

# Headless mode (no visualization)
python3 train_comparison.py --no-viz

# Start in fullscreen
python3 train_comparison.py --fullscreen

# All options can be configured in config/comparison_config.yaml
```

**Comparison Training Controls:**
- **[Tab]** - Switch active camera
- **[1/2]** - Focus left/right agent
- **[Space]** - Toggle camera follow
- **[Arrow Keys]** - Pan active camera
- **[Mouse Wheel]** - Zoom active camera
- **[F11]** - Toggle fullscreen
- **[ESC]** - Exit training

#### Evaluate Trained Models

```bash
# Test a trained model with visualization
python3 evaluate_model.py --model models/checkpoints/dqn_best.pt --track tracks/oval_easy.json --episodes 10

# Test on different track (generalization test)
python3 evaluate_model.py --model models/checkpoints/ppo_best.pt --track tracks/megacool_track.json --episodes 20

# Headless evaluation (no visualization, faster)
python3 evaluate_model.py --model models/checkpoints/dqn_best.pt --track tracks/oval_easy.json --episodes 100 --no-render

# Export results to CSV and JSON
python3 evaluate_model.py --model models/checkpoints/ppo_best.pt --track tracks/oval_easy.json --episodes 50 --export

# Specify agent type manually (if auto-detect fails)
python3 evaluate_model.py --model path/to/model.pt --track tracks/oval_easy.json --agent-type DQN

# Adjust FPS for smoother visualization
python3 evaluate_model.py --model models/checkpoints/dqn_best.pt --track tracks/oval_easy.json --fps 30
```

**Evaluation Metrics:**
- **Success Rate** - Percentage of completed laps
- **Average Reward** - Mean episode reward
- **Lap Time** - Average and best lap times
- **Checkpoint Rate** - Percentage of checkpoints collected
- **Steps per Episode** - Efficiency metric

**Camera Controls:**
- **[Mouse Wheel]** - Zoom in/out (0.3x - 3.0x)
- **[Arrow Keys]** - Pan camera
- **[Space]** - Toggle follow car mode
- **[R]** - Reset camera to default
- **[ESC]** - Exit evaluation

## Training Scripts & Tools Guide

This project provides multiple training scripts and tools for different use cases. Here's when to use each:

### Training Scripts Comparison

| Script | Algorithm | Visualization | Best For | Key Features |
|--------|-----------|---------------|----------|--------------|
| [train_ppo.py](train_ppo.py) | PPO | None (headless) | Fast training, production runs | TensorBoard logging, periodic evaluation, checkpoints |
| [train_with_camera.py](train_with_camera.py) | DQN | Advanced camera | Debugging, track exploration | Zoom, pan, follow car, speedometer, ghost trail |
| [train_comparison.py](train_comparison.py) | DQN + PPO | Split-screen | Algorithm comparison | Side-by-side DQN vs PPO, dual cameras, real-time metrics |
| [train_with_ghost.py](train_with_ghost.py) | DQN | Ghost trail | Visualizing improvement | Shows previous episode path, progress comparison |
| [train_progressive_learning.py](train_progressive_learning.py) | DQN | Ghost trail | Consistent learning | Low epsilon (0.3→0.05), less exploration, stable policy |

### Detailed Script Descriptions

#### 1. train_ppo.py - PPO Headless Training
**Purpose**: Fast, efficient PPO training without visualization overhead.

**When to use**:
- Production training runs for best performance
- Long training sessions (500+ episodes)
- When you need TensorBoard metrics
- Server/remote training without display

**Features**:
- Pure console output with progress bars
- Automatic evaluation every N episodes
- TensorBoard logging (policy loss, value loss, entropy, KL divergence)
- Checkpoint management (best + periodic saves)
- Trajectory-based learning (2048 steps buffer)

**Example**:
```bash
python3 train_ppo.py --track tracks/oval_easy.json --episodes 1000
# Monitor with: tensorboard --logdir logs/tensorboard/ppo
```

---

#### 2. train_with_camera.py - DQN with Advanced Camera
**Purpose**: Interactive training with professional camera controls for detailed observation.

**When to use**:
- Debugging agent behavior in specific track sections
- Understanding why agent crashes at certain corners
- Creating demo videos or screenshots
- Exploring complex tracks while training

**Features**:
- **Camera Controls**: Zoom (0.3x-3.0x), pan, follow car mode
- **Speedometer Panel**: Real-time speed, gas/brake indicators, speed bar
- **Ghost Trail**: See previous episode's path
- **Fullscreen Support**: F11 or Cmd+F for fullscreen
- **Keyboard Controls**: Arrow keys for pan, Space for follow toggle, R for reset

**Example**:
```bash
python3 train_with_camera.py --track tracks/f1_spa_style_long.json --episodes 200 --fps 30
```

**Controls**:
- Mouse Wheel: Zoom in/out
- Arrow Keys: Pan camera
- Space / F: Toggle follow car
- R: Reset camera
- F11 / Cmd+F: Fullscreen

---

#### 3. train_comparison.py - DQN vs PPO Split-Screen
**Purpose**: Direct algorithm comparison with side-by-side visualization.

**When to use**:
- Comparing DQN vs PPO sample efficiency
- Testing hyperparameters side-by-side
- Research and analysis
- Demonstrating algorithm differences

**Features**:
- **Split-Screen**: 2000×700 window (1000px per agent)
- **Independent Cameras**: Separate zoom/pan for each agent
- **Real-time Metrics**: Episode rewards, checkpoints, steps
- **Synchronized Environment**: Same track, same conditions
- **Dual TensorBoard Logs**: Separate logs for each agent

**Example**:
```bash
python3 train_comparison.py --track tracks/oval_easy.json --episodes 300 --fps 60
```

**Controls**:
- Tab: Switch active camera (left/right/both)
- 1/2: Focus left/right agent
- Space: Toggle camera follow
- Arrow Keys: Pan active camera
- Mouse Wheel: Zoom active camera

---

#### 4. train_with_ghost.py - DQN with Ghost Visualization
**Purpose**: Visualize episode-to-episode improvement by showing previous attempt.

**When to use**:
- Seeing if agent is making progress each episode
- Visual confirmation of learning
- Understanding where agent improves vs repeats mistakes
- Teaching/demonstration purposes

**Features**:
- **Red Ghost Cars**: Previous episode path rendered as semi-transparent red cars
- **Current Episode**: Bright colored car for current attempt
- **Improvement Tracking**: Compare current length vs previous
- **Visual Learning**: Immediately see if agent goes further than last time

**Example**:
```bash
python3 train_with_ghost.py --track tracks/simple_straight.json --episodes 200 --fps 30
```

**Visual Cues**:
- Red/Orange Ghosts = Where you crashed last episode
- Bright Car = Current attempt
- Goal: Go further than the ghosts!

---

#### 5. train_progressive_learning.py - DQN with Low Epsilon
**Purpose**: Progressive learning with consistent behavior (low exploration).

**When to use**:
- Agent already learned basics, needs fine-tuning
- Want consistent behavior with minimal randomness
- Incremental improvement on complex tracks
- After initial exploration phase

**Features**:
- **Low Epsilon**: Starts at 0.3 (vs normal 1.0)
- **Slow Decay**: 0.995 decay rate for gradual reduction
- **Exploitation Focus**: 70% exploitation from start
- **Ghost Visualization**: Shows previous episode for comparison
- **Stable Policy**: Less random actions, more predictable behavior

**Example**:
```bash
python3 train_progressive_learning.py --track tracks/f1_grand_circuit.json --episodes 200 --fps 20
```

**Key Difference**: Normal training explores 100% → 10%, this explores 30% → 5%

---

### Utility Tools

#### track_builder.py - Interactive Track Editor
**Purpose**: Create custom racing tracks with visual editor.

**Features**:
- Draw outer/inner boundaries with mouse
- Chaikin's smoothing algorithm for smooth curves
- Place checkpoints and start position
- Camera controls (zoom, pan)
- File save/load with GUI dialog
- Grid snapping for precision

**Usage**:
```bash
python3 track_builder.py
```

**Full controls documented in**: [Build Custom Tracks](#build-custom-tracks) section above

---

#### view_track.py - Track Layout Viewer
**Purpose**: Preview track layouts without starting training.

**When to use**:
- Inspecting track before training
- Checking checkpoint placement
- Verifying track validity
- Getting track coordinates for debugging

**Features**:
- Static track visualization
- Checkpoint numbering (CP1, CP2, ...)
- Start position marker (green circle + cross)
- Click-to-get-coordinates
- Mouse position tracking
- Color legend

**Example**:
```bash
python3 view_track.py --track tracks/f1_grand_circuit.json
```

**Controls**:
- ESC or Q: Quit
- Click: Print coordinates to console

---

#### evaluate_model.py - Model Evaluation
**Purpose**: Test trained models with comprehensive metrics and visualization.

**When to use**:
- After training completes
- Testing generalization on different tracks
- Generating performance reports
- Comparing model checkpoints

**Features**:
- **Auto-detection**: Automatically detects DQN vs PPO from checkpoint
- **Visualization**: Full camera controls (zoom, pan, follow)
- **Comprehensive Metrics**: Success rate, avg reward, lap times, checkpoints
- **Export Options**: CSV and JSON output
- **Deterministic Evaluation**: Pure exploitation (epsilon=0)

**Full usage documented in**: [Evaluate Trained Models](#evaluate-trained-models) section above

---

### Quick Selection Guide

**Choose your script based on your goal:**

**I want the fastest training** → `train_ppo.py` (headless, TensorBoard)

**I want to watch and debug** → `train_with_camera.py` (zoom, pan, speedometer)

**I want to compare algorithms** → `train_comparison.py` (DQN vs PPO split-screen)

**I want to see improvement visually** → `train_with_ghost.py` (ghost trails)

**I want consistent learning** → `train_progressive_learning.py` (low epsilon)

**I want to create tracks** → `track_builder.py` (interactive editor)

**I want to preview a track** → `view_track.py` (static viewer)

**I want to test a trained model** → `evaluate_model.py` (metrics + export)

## Project Structure

```
self-driving-car-rl/
├── src/
│   ├── environment/
│   │   ├── car.py              # Smooth steering physics
│   │   ├── track.py            # Triple-level caching
│   │   ├── sensor.py           # 6-level color gradient
│   │   └── simulation.py       # RL env + finish time tracking
│   ├── algorithms/
│   │   ├── dqn/
│   │   │   ├── agent.py        # DQN with Double DQN
│   │   │   ├── network.py      # Q-Network [128, 128]
│   │   │   └── replay_buffer.py
│   │   └── ppo/
│   │       ├── agent.py        # PPO discrete actions
│   │       ├── network.py      # Actor-Critic [128, 128]
│   │       └── trajectory_buffer.py  # GAE computation
│   ├── comparison/
│   │   ├── coordinator.py      # Multi-process training
│   │   ├── worker.py           # Training workers
│   │   └── shared_state.py     # IPC for rendering
│   ├── visualization/
│   │   ├── dual_screen_renderer.py  # Split-screen
│   │   ├── camera.py           # Independent cameras
│   │   └── ui_components.py    # Reusable UI
│   └── utils/
│       └── config_loader.py    # Centralized config
├── config/
│   ├── environment.yaml        # Shared: physics, sensors, rewards
│   └── training_modes.yaml     # All training modes (DQN, PPO, comparison)
├── tracks/
│   ├── oval_easy.json
│   ├── megacool_track.json
│   └── *.json                 # Your custom tracks
├── track_builder.py           # Interactive track editor
├── view_track.py              # Track layout viewer
├── train_ppo.py               # PPO headless training
├── train_with_camera.py       # DQN + camera controls
├── train_comparison.py        # DQN vs PPO side-by-side
├── train_with_ghost.py        # DQN + ghost visualization
├── train_progressive_learning.py  # DQN with low epsilon
├── evaluate_model.py          # Model evaluation with metrics
└── requirements.txt
```

## Configuration

### Environment Settings (`config/environment.yaml`)

```yaml
car:
  width: 30
  height: 15
  max_velocity: 40.0
  min_velocity: -1.0
  acceleration: 0.25
  friction: 0.95
  turn_rate: 0.08
  angular_damping: 0.85  # Smooth steering

sensors:
  num_sensors: 7
  angles: [-90, -60, -30, 0, 30, 60, 90]
  max_range: 350

actions:
  type: "discrete"
  num_actions: 9  # 3 steering × 3 speed

rewards:
  survival: 2.0
  checkpoint: 500.0
  crash: -2000.0
  finish: 1000.0

finish_time:
  enabled: true
  mode: "speed_based"  # or "target_based"
  max_time: 60.0
  speed_multiplier: 50.0
```

### Training Modes (`config/training_modes.yaml`)

**Universal configuration file** with all training modes in one place. Each script loads its specific section:

```yaml
# Shared configuration (used by all modes)
shared:
  network:
    state_dim: 8
    action_dim: 9
    hidden_dims: [128, 128]
  training:
    batch_size: 64
    learning_rate: 0.0005
    gamma: 0.99
    double_dqn: true

# DQN Standard Mode (epsilon: 1.0 → 0.1)
dqn_standard:
  track: "tracks/oval_easy.json"
  num_episodes: 1000
  exploration:
    epsilon_start: 1.0
    epsilon_end: 0.1
    epsilon_decay: 0.9985

# DQN Ghost Mode (epsilon: 1.0 → 0.1)
dqn_ghost:
  track: "tracks/simple_straight.json"
  num_episodes: 200
  exploration:
    epsilon_start: 1.0
    epsilon_end: 0.1

# DQN Progressive Mode (epsilon: 0.3 → 0.05 - LOW!)
dqn_progressive:
  track: "tracks/f1_grand_circuit.json"
  num_episodes: 200
  exploration:
    epsilon_start: 0.3  # LOW for consistent behavior
    epsilon_end: 0.05
    epsilon_decay: 0.995

# DQN Camera Mode (epsilon: 0.3 → 0.05)
dqn_camera:
  track: "tracks/f1_spa_style_long.json"
  num_episodes: 200
  exploration:
    epsilon_start: 0.3
    epsilon_end: 0.05

# PPO Mode
ppo:
  track: "tracks/oval_easy.json"
  num_episodes: 1000
  ppo:
    gae_lambda: 0.95
    clip_epsilon: 0.2
    trajectory_length: 2048

# Comparison Mode (DQN vs PPO)
comparison:
  track: "tracks/supercool_track.json"
  num_episodes: 300
  visualization:
    window_width: 2000
    window_height: 700
```

---

### Training Modes Summary

| Mode | Script | Epsilon | Track | Purpose |
|------|--------|---------|-------|---------|
| `dqn_standard` | - | 1.0 → 0.1 | oval_easy.json | Standard DQN training |
| `dqn_ghost` | train_with_ghost.py | 1.0 → 0.1 | simple_straight.json | Ghost trail visualization |
| `dqn_progressive` | train_progressive_learning.py | **0.3 → 0.05** | f1_grand_circuit.json | Low epsilon, consistent learning |
| `dqn_camera` | train_with_camera.py | **0.3 → 0.05** | f1_spa_style_long.json | Camera controls + speedometer |
| `ppo` | train_ppo.py | - | oval_easy.json | PPO headless training |
| `comparison` | train_comparison.py | Both | supercool_track.json | DQN vs PPO split-screen |

**How scripts load config:**
```python
modes_config = load_config("config/training_modes.yaml")
mode = modes_config['dqn_ghost']  # or 'dqn_progressive', 'ppo', etc.
shared = modes_config['shared']   # Shared parameters
env_config = load_config("config/environment.yaml")
```

**Note**: All training scripts accept CLI arguments (`--track`, `--episodes`, `--fps`) that override config values.

## State & Action Space

### State (8-dimensional vector)
```python
[
  sensor_1,    # -90° (left)
  sensor_2,    # -60°
  sensor_3,    # -30°
  sensor_4,    # 0° (forward)
  sensor_5,    # +30°
  sensor_6,    # +60°
  sensor_7,    # +90° (right)
  velocity     # Current speed (normalized)
]
```

### Actions (9 Discrete Actions)
- **0**: LEFT_SLOW - Turn left + slow speed
- **1**: LEFT_NORMAL - Turn left + normal speed
- **2**: LEFT_FAST - Turn left + fast speed
- **3**: STRAIGHT_SLOW - Straight + slow speed
- **4**: STRAIGHT_NORMAL - Straight + normal speed
- **5**: STRAIGHT_FAST - Straight + fast speed
- **6**: RIGHT_SLOW - Turn right + slow speed
- **7**: RIGHT_NORMAL - Turn right + normal speed
- **8**: RIGHT_FAST - Turn right + fast speed

## Tech Stack

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.8+ |
| **Deep Learning** | PyTorch 2.0+ (MPS support) |
| **Visualization** | Pygame 2.5+ |
| **Math** | NumPy |
| **Config** | PyYAML |
| **Multiprocessing** | Python multiprocessing |

## Training Results

Trained agents achieve:
- **Lap completion**: 80-90% on complex tracks
- **Checkpoint rate**: 90%+ collection
- **Collision avoidance**: Smooth cornering
- **Speed optimization**: Learns when to slow/fast
- **Finish time**: Progressive improvement with time-based rewards

## Development

### Run Tests
```bash
pytest tests/ -v
```

### Create Smooth Track
1. Launch: `python3 track_builder.py`
2. Draw outer boundary (press [1])
3. Close boundary (press [C])
4. **Smooth it** (press [M] multiple times)
5. Draw inner boundary (press [2])
6. **Smooth it** (press [M])
7. Add checkpoints (press [3])
8. Set start position (press [4])
9. Save (press [Cmd+S])

### Compare Algorithms
```bash
# Use command-line options (overrides config file)
python3 train_comparison.py --track tracks/oval_easy.json --episodes 300 --fps 60

# Or edit config/comparison_config.yaml for defaults
```

## Performance Tips

### For Smooth Tracks (5k-40k points)
- Track builder automatically optimizes on save
- Triple-level caching handles rendering + collision
- No manual point reduction needed!

### For Faster Training
- Use `--fps 30` for lower CPU usage
- Disable visualization: `train_ppo.py --no-render`
- Increase `batch_size` for GPU utilization

## Troubleshooting

**Pygame fails to install:**
```bash
# macOS
brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf

# Linux
sudo apt-get install libsdl2-dev
```

**PyTorch MPS not working:**
```python
import torch
print(torch.backends.mps.is_available())  # Should be True
```

**Track builder file dialog not showing:**
- Install tkinter: `brew install python-tk`
- Or use fallback text menu

**Lag with many track points:**
- Track caching is automatic
- For 40k+ points: Rendering uses ~500, collision uses ~1500
- Original full detail preserved

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'feat: add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

Use conventional commits: `feat:`, `fix:`, `perf:`, `docs:`, `chore:`

## Contact

**Project Link:** https://github.com/Ikrar06/self-driving-car-rl

---

⭐ Star this repo if you find it useful!
