# AI Racing Trainer - Deep Reinforcement Learning Racing Simulator

> Train autonomous racing cars using Deep Q-Network (DQN) with custom physics engine and interactive track builder

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Pygame](https://img.shields.io/badge/Pygame-2.5+-green.svg)](https://www.pygame.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

A complete reinforcement learning racing simulator featuring:
- **Custom Physics Engine** - Realistic car dynamics with acceleration, friction, and steering
- **7-Sensor Ray-Casting System** - Wide-angle coverage for obstacle detection
- **Interactive Track Builder** - Visual tool with zoom, pan, and fullscreen support
- **DQN Implementation** - Deep Q-Network with experience replay and target network
- **Real-time Training Visualization** - Camera controls, ghost cars, and live metrics
- **F1-Style Circuits** - Multiple challenging tracks with checkpoints

## Features

### Car Physics
- Realistic acceleration and friction
- Configurable max/min velocity
- Smooth steering mechanics
- Collision detection

### Advanced Sensors
- **7 ray-cast sensors** at -90°, -60°, -30°, 0°, +30°, +60°, +90°
- Color-coded distance visualization (green → yellow → red)
- Normalized readings for neural network input
- Configurable range and angles

### Track Builder
- **Interactive visual editor** with mouse controls
- **Camera system** with zoom (0.2x-5.0x) and pan
- **Car size reference** (20x15px) overlay at cursor
- **Grid snapping** for precise placement
- **Fullscreen mode** (Cmd+F or F11)
- Save/load tracks in JSON format
- Real-time boundary preview

### DQN Agent
- Deep Q-Network with configurable architecture
- Experience replay buffer (100k transitions)
- Target network for stable learning
- Double DQN support
- Epsilon-greedy exploration (0.3 → 0.05)
- PyTorch with MPS (Metal) acceleration on Mac

### Training Features
- Real-time visualization with camera follow
- Ghost car showing previous episode path
- Episode statistics (steps, reward, checkpoints)
- Model checkpointing (best + periodic)
- Configurable reward system

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
git clone https://github.com/yourusername/ai-racing-trainer.git
cd ai-racing-trainer

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
- **[Mouse Wheel]** - Zoom in/out
- **[W/A/S/D]** or **[Arrow Keys]** - Pan camera
- **[Click]** - Place points
- **[C]** - Close boundary
- **[Z]** - Undo last point
- **[R]** - Rotate start angle
- **[Cmd+S]** - Save track
- **[Cmd+L]** - Load track
- **[Cmd+F]** or **[F11]** - Toggle fullscreen
- **[V]** - Toggle car reference
- **[G]** - Toggle grid snap

#### Train Agent

```bash
# Train on default track
python3 train_with_camera.py

# Train on custom track
python3 train_with_camera.py --track tracks/f1_professional_circuit.json

# Adjust training parameters
python3 train_with_camera.py --episodes 500 --fps 60
```

**Training Controls:**
- **[Space]** or **[F]** - Toggle camera follow
- **[Mouse Wheel]** - Zoom
- **[Arrow Keys]** - Pan camera
- **[R]** - Reset camera
- **[ESC]** - Exit training

## Project Structure

```
ai-racing-trainer/
├── src/
│   ├── environment/
│   │   ├── car.py              # Car physics & dynamics
│   │   ├── track.py            # Track loading & collision
│   │   ├── sensor.py           # 7-sensor ray-casting
│   │   └── simulation.py       # Main RL environment
│   └── algorithms/
│       └── dqn/
│           ├── agent.py        # DQN agent
│           ├── network.py      # Q-Network
│           └── replay_buffer.py # Experience replay
├── config/
│   ├── environment.yaml        # Car physics, sensors, rewards
│   └── dqn_config.yaml        # Network architecture, training
├── tracks/
│   ├── f1_professional_circuit.json
│   ├── f1_spa_style_long.json
│   └── *.json                 # Your custom tracks
├── track_builder.py           # Interactive track editor
├── train_with_camera.py       # Training with visualization
└── requirements.txt           # Python dependencies
```

## Configuration

### Environment Settings (`config/environment.yaml`)

```yaml
car:
  width: 5
  height: 2
  max_velocity: 8.0
  min_velocity: 2.0
  acceleration: 0.3
  friction: 0.05
  turn_rate: 0.12

sensors:
  num_sensors: 7
  angles: [-90, -60, -30, 0, 30, 60, 90]
  max_range: 200

rewards:
  checkpoint: 100
  survival: 0.1
  crash: -50
```

### DQN Hyperparameters (`config/dqn_config.yaml`)

```yaml
network:
  state_dim: 8        # 7 sensors + 1 velocity
  action_dim: 3       # LEFT, STRAIGHT, RIGHT
  hidden_dims: [128, 128]

training:
  learning_rate: 0.0003
  gamma: 0.99
  epsilon_start: 0.3
  epsilon_end: 0.05
  epsilon_decay: 0.995
  batch_size: 64

replay:
  buffer_size: 100000
```

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

### Actions (Discrete)
- **0**: Turn LEFT
- **1**: Go STRAIGHT
- **2**: Turn RIGHT

## Tech Stack

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.8+ |
| **Deep Learning** | PyTorch 2.0+ (with MPS support) |
| **Visualization** | Pygame 2.5+ |
| **Math** | NumPy |
| **Config** | PyYAML |

## Training Results

Trained agents achieve:
- Successful lap completion on oval tracks
- Checkpoint collection rate: 80%+
- Collision avoidance in tight corners
- Smooth steering behavior
- F1-style circuit mastery (in progress)

## Development

### Run Tests
```bash
pytest tests/ -v
```

### Create New Track
1. Launch track builder: `python3 track_builder.py`
2. Press **[1]** for Outer Boundary mode
3. Click to place outer wall points
4. Press **[C]** to close boundary
5. Press **[2]** for Inner Boundary mode
6. Repeat for inner wall
7. Press **[3]** for Checkpoint mode
8. Click start and end for each checkpoint
9. Press **[4]** for Start Position
10. Click starting position
11. Press **[Cmd+S]** to save

### Adjust Car Physics
Edit `config/environment.yaml`:
- Increase `max_velocity` for faster racing
- Increase `turn_rate` for sharper turns
- Adjust `friction` for drift behavior

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
print(torch.backends.mps.is_available())  # Should be True on Apple Silicon
```

**ModuleNotFoundError:**
```bash
# Ensure venv is activated
source venv/bin/activate
which python  # Should show ./venv/bin/python
```

## Roadmap

- [x] Custom physics engine
- [x] 7-sensor ray-casting
- [x] DQN implementation
- [x] Interactive track builder
- [x] Camera controls & visualization
- [ ] PPO algorithm
- [ ] NEAT (genetic algorithm)
- [ ] Multi-agent racing
- [ ] Curriculum learning

## License

MIT License - see [LICENSE](LICENSE) file

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'feat: add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## Contact

**Project Link:** https://github.com/yourusername/ai-racing-trainer

---

Star this repo if you find it useful!
