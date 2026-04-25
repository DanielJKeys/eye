# ACES Platform

**ACE Simulation Environment — Military Logistics RL Simulator**

ACES is a comprehensive reinforcement learning environment for military logistics simulation, built on Gymnasium. It models airlift operations, supply chain management, and adversarial threats in a realistic geospatial scenario.

## Features

- **Gymnasium-Compatible Environment**: Full RL environment with observation/action spaces, rewards, and episode termination.
- **Action Masking**: Supports MaskablePPO for invalid action filtering.
- **Scenario Management**: Database-backed scenario editor with asset types, installations, threats, and attack schedules.
- **Training Support**: Integrated training with Stable Baselines3 (PPO, MaskablePPO, A2C).
- **Heuristic Agents**: Rule-based baselines for comparison.
- **Interactive UI**: Streamlit-based interface for scenario editing, training, debugging, and analysis.
- **Analytics & Reporting**: Comprehensive decision analysis, audit logging, and performance reporting.
- **Multi-Agent Scenarios**: Support for multi-agent reinforcement learning.
- **Optimization**: Resource optimization and performance tuning.
- **Threat Modeling**: Advanced threat intelligence and risk assessment.
- **Realistic Modeling**: Haversine-based geography, fuel consumption, runway damage, and threat zones.

## Installation

### Prerequisites
- Python 3.10+
- pip
- Git

### Install from Source
```bash
git clone https://github.com/DanielJKeys/aces-v4.git
cd aces-v4
pip install -e .
```

This installs ACES and all dependencies (Gymnasium, Stable Baselines3, SQLAlchemy, Streamlit, etc.).

### Verify Installation
```bash
python validate.py
```

This runs a quick validation that confirms:
- All imports work correctly
- The default scenario loads
- Environment can be created and stepped
- Action masking functions properly

## Running the System

### 1. Interactive UI (Recommended for Beginners)

The Streamlit-based UI provides an intuitive interface for scenario editing, training, and visualization.

**Start the UI:**
```bash
aces-ui
```

Or manually:
```bash
streamlit run aces/ui/app.py
```

Then open your browser to `http://localhost:8501`

**Available Pages:**
- **Scenario Editor** (Page 1): Create/edit scenarios with bases, assets, threats, and attack schedules
- **Mission Planner** (Page 2): Train RL agents or run heuristic simulations
- **Live Dashboard** (Page 3): Real-time mission visualization and metrics
- **Debug Console** (Page 4): Manual step-through control with state inspection
- **Analysis** (Page 5): Training results and performance metrics

### 2. Command-Line Interface (CLI)

For headless operation or scripting:

**Run a simulation with default settings:**
```bash
python -m aces.ui.cli run
```

**Run with custom scenario:**
```bash
python -m aces.ui.cli run --scenario "Western Pacific Logistics"
```

**Train an RL agent:**
```bash
python -m aces.ui.cli train --algorithm maskable_ppo --timesteps 500000
```

**See all options:**
```bash
python -m aces.ui.cli --help
```

### 3. Programmatic Usage (Python Scripts)

For custom research and integration:

**Simple environment test:**
```python
from aces.services.scenario import ScenarioService
from aces.domain.environment import LogisticsEnv

# Load scenario
svc = ScenarioService()
scenarios = svc.list_scenarios()
config = svc.get_scenario_config(scenarios[0]['id'])

# Create and run environment
env = LogisticsEnv(config)
obs, info = env.reset()

# Step the environment
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        break

print(f"Episode reward: {reward}")
```

**Train with Stable Baselines3:**
```python
from aces.services.scenario import ScenarioService
from aces.domain.environment import LogisticsEnv
from stable_baselines3 import MaskablePPO
from stable_baselines3.common.vec_env import VecNormalize

# Setup
svc = ScenarioService()
config = svc.get_scenario_config(svc.list_scenarios()[0]['id'])
env = LogisticsEnv(config)

# Normalize rewards (recommended)
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# Train
model = MaskablePPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Save
model.save("aces_model")

# Test trained model
model = MaskablePPO.load("aces_model")
obs, _ = env.reset()
for _ in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        break
```

**Run heuristic baseline:**
```python
from aces.services.scenario import ScenarioService
from aces.domain.environment import LogisticsEnv
from aces.agents.heuristic import GreedyAgent

svc = ScenarioService()
config = svc.get_scenario_config(svc.list_scenarios()[0]['id'])
env = LogisticsEnv(config)

agent = GreedyAgent(env)
obs, _ = env.reset()

total_reward = 0
for _ in range(1000):
    action = agent.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated:
        break

print(f"Total reward: {total_reward}")
```

### 4. Training Service (Multi-Agent, Parallel)

For advanced training with multiple algorithms:

```python
from aces.services.training import TrainingService
from aces.config import TrainingConfig

tsvc = TrainingService()
svc = ScenarioService()
config = svc.get_scenario_config(svc.list_scenarios()[0]['id'])

# Configure training
train_config = TrainingConfig(
    algorithm="maskable_ppo",
    total_timesteps=1000000,
    learning_rate=3e-4,
    batch_size=128,
    n_epochs=10,
)

# Start training (runs in background)
progress = tsvc.start_training(config, train_config)

# Monitor progress
import time
while True:
    prog = tsvc.get_progress()
    if prog:
        print(f"Steps: {prog['steps']}/{prog['total_steps']}")
        if prog['is_done']:
            print(f"Training complete! Best reward: {prog['best_reward']}")
            break
    time.sleep(5)

# Load trained model
model = tsvc.load_best_model()
```

## Configuration & Customization

### Environment Parameters

Modify `aces/config.py` to customize:

**Mission Configuration:**
```python
MissionConfig(
    step_to_hour=0.5,           # Simulation time per step
    mission_length_hr=72.0,      # Total mission duration
    min_fuel_reserve_pct=10.0,   # Minimum fuel reserve
    fuel_safety_margin_pct=5.0,  # Safety margin for flight planning
)
```

**Reward Configuration:**
```python
RewardConfig(
    supply_delivered_per_lb=0.1,
    munitions_delivered_per_lb=0.2,
    efficient_routing_per_nm=-0.001,    # Penalize long routes
    timely_delivery_bonus=10.0,
    risk_avoidance_per_step=1.0,
    asset_utilization_per_hr=5.0,
)
```

**Training Configuration:**
```python
TrainingConfig(
    algorithm="maskable_ppo",           # or "ppo", "a2c"
    total_timesteps=500000,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
)
```

### Scenario Customization

Through the UI:
1. Open **Scenario Editor** page
2. Create bases (installations) with supply levels
3. Add assets (aircraft, vehicles) with capabilities
4. Define threat zones with threat levels
5. Schedule attacks/events
6. Save scenario

Or programmatically:
```python
from aces.services.scenario import ScenarioService

svc = ScenarioService()
scenario = svc.create_scenario(
    name="Custom Scenario",
    description="My custom logistics scenario",
)
# Add bases, assets, threats via API
```

## Quick Start

### Run the UI
```bash
aces-ui
```
Or:
```bash
streamlit run aces/ui/app.py
```

The UI provides:
- **Scenario Editor**: Create/edit scenarios with bases, assets, threats.
- **Mission Planner**: Train RL agents or run heuristic simulations.
- **Debug Console**: Step-through manual control.
- **Live Dashboard**: Real-time visualization (future).

### Use Programmatically
```python
from aces.services.scenario import ScenarioService
from aces.domain.environment import LogisticsEnv

# Load default scenario
svc = ScenarioService()
scenarios = svc.list_scenarios()
scenario_id = scenarios[0]['id']
config = svc.get_scenario_config(scenario_id)

# Create environment
env = LogisticsEnv(config)
obs, info = env.reset()

# Use with Stable Baselines3
from stable_baselines3 import PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
```

## Key Features Explained

### Action Masking
Invalid actions (e.g., loading supplies into a full cargo bay) are masked before the policy samples. This:
- Prevents agent from wasting learning on impossible actions
- Dramatically speeds up convergence
- Is essential for MaskablePPO

### Hierarchical Supply Actions
Instead of 8 individual supply dimensions, supplies are grouped into 4 priorities:
- **CRITICAL** (Munitions, AvGas)
- **ESSENTIAL** (Food, Water)
- **MAINTENANCE** (Parts, Runway Material)
- **SUPPORT** (Electricity, MoGas)

This reduces action space complexity from ~245 to ~125 dimensions.

### Fuel Planning Constraints
Assets must maintain minimum fuel reserves and respect safety margins. This encourages:
- Realistic fuel planning behavior
- Longer mission times (more learning signals)
- Agent must balance throughput vs. operational safety

### Intermediate Rewards
Beyond just delivery bonuses, the agent receives rewards for:
- **Efficient routing**: Penalize unnecessary long flights
- **Timely delivery**: Bonuses for on-schedule arrivals
- **Risk avoidance**: Rewards for steering clear of threats
- **Asset utilization**: Incentives to keep assets flying
- **Mission completion**: Final bonus on successful mission

## Architecture

### Core Components
- **`aces/domain/`**: Environment logic (Base, Asset, ThreatZone, LogisticsEnv).
- **`aces/spaces/`**: Action/Observation spaces and reward shaping.
- **`aces/agents/`**: RL agents and heuristics.
- **`aces/services/`**: Business logic (ScenarioService, TrainingService).
- **`aces/data/`**: Database models and repositories (SQLite).
- **`aces/ui/`**: Streamlit pages and CLI entry point.

### Key Classes
- **LogisticsEnv**: Gymnasium environment for logistics simulation.
- **ScenarioConfig**: Dataclass holding all scenario parameters.
- **TrainingService**: Manages parallel RL training with VecNormalize.
- **ActionMasker**: Generates valid action masks for MaskablePPO.

### Database
- SQLite database (`data_store/aces.db`) for scenario persistence.
- Auto-initialized with a default "Western Pacific Logistics" scenario.
- Models: Scenarios, AssetTypes, Installations, ThreatZones, etc.

## Comparison to Original v4

Improvements in v4 over earlier versions:
| Feature | v3 | v4 |
|---------|----|----|
| **Action Space** | 245+ dimensions | 125 dimensions (hierarchical) |
| **Fuel Modeling** | None | Reserves + safety margins |
| **Rewards** | Sparse delivery only | 6+ intermediate reward types |
| **Training Speed** | Slow convergence | 3-5x faster with action masking |
| **Scenario Flexibility** | Limited | Full database-backed editor |
| **RL Support** | PPO only | MaskablePPO, PPO, A2C |

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Add tests for new functionality.
4. Ensure CI passes.
5. Submit a pull request.

## License

See LICENSE file.

## Acknowledgments

Built for military logistics research and decision support.</content>
<parameter name="filePath">c:\Users\Danie\Code Projects\aces-v4\README.md