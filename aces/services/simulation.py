"""SimulationService — owns the runtime environment for manual/debug runs.

The service holds a persistent LogisticsEnv instance across Streamlit reruns
via session state. The UI calls step() or run_to_event() and receives full
state snapshots for rendering.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from aces.config import ScenarioConfig, TrainingConfig
from aces.domain.environment import LogisticsEnv
from aces.agents.heuristic import HeuristicAgent
from aces.spaces.action import ActionBuilder


class SimulationService:
    def __init__(self) -> None:
        self.env: Optional[LogisticsEnv] = None
        self._scenario: Optional[ScenarioConfig] = None
        self._obs: Optional[np.ndarray] = None
        self._last_reward: float = 0.0
        self._cumulative_reward: float = 0.0
        self._terminated: bool = False
        self._agent: Optional[object] = None  # loaded RL model or HeuristicAgent
        self._action_builder: Optional[ActionBuilder] = None

    # ------------------------------------------------------------------ lifecycle

    def initialize(self, scenario: ScenarioConfig, seed: Optional[int] = None) -> dict:
        """Create a fresh environment and return the initial state snapshot."""
        self.env = LogisticsEnv(scenario, render_mode="console")
        self._scenario = scenario
        self._action_builder = ActionBuilder(scenario)
        obs, info = self.env.reset(seed=seed)
        self._obs = obs
        self._last_reward = 0.0
        self._cumulative_reward = 0.0
        self._terminated = False
        return self._state_snapshot(info)

    def load_agent(self, model_path: str, algorithm: str, vecnormalize_path: str = "") -> None:
        """Load a trained RL model for agent-assisted manual runs."""
        from aces.agents.registry import get_algorithm_class, requires_masking
        cls = get_algorithm_class(algorithm)
        if cls is None:
            self._agent = HeuristicAgent(self._action_builder)
        else:
            self._agent = cls.load(model_path)
            if vecnormalize_path:
                from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
                dummy = DummyVecEnv([lambda: self.env])
                self._agent._vec_normalize_env = VecNormalize.load(vecnormalize_path, dummy)

    def set_heuristic_agent(self) -> None:
        self._agent = HeuristicAgent(self._action_builder)

    # ------------------------------------------------------------------ step control

    def step(self, action: Optional[np.ndarray] = None) -> dict:
        """Execute one step. If action is None, query the loaded agent."""
        if self.env is None or self._terminated:
            return self._state_snapshot({})

        if action is None:
            action = self._get_agent_action()

        obs, reward, terminated, truncated, info = self.env.step(action)
        self._obs = obs
        self._last_reward = reward
        self._cumulative_reward += reward
        self._terminated = terminated or truncated
        return self._state_snapshot(info)

    def run_to_event(self) -> list[dict]:
        """Run until a missile strike, asset destruction, or mission end. Returns step snapshots."""
        snapshots = []
        while not self._terminated:
            snap = self.step()
            snapshots.append(snap)
            info = snap.get("info", {})
            # Pause on destruction events or mission end
            if snap.get("terminated") or self._check_destruction_occurred():
                break
            if len(snapshots) > 500:  # safety guard
                break
        return snapshots

    def run_to_completion(self) -> list[dict]:
        """Run the full episode. Returns all step snapshots."""
        snapshots = []
        while not self._terminated:
            snap = self.step()
            snapshots.append(snap)
        return snapshots

    def get_agent_suggestion(self) -> np.ndarray:
        """Return the agent's recommended action without executing it."""
        return self._get_agent_action()

    # ------------------------------------------------------------------ state access

    def get_state(self) -> dict:
        if self.env is None:
            return {}
        info = {"time_hr": self.env.current_time, "step": self.env._step_count,
                "history": self.env._history}
        return self._state_snapshot(info)

    def is_initialized(self) -> bool:
        return self.env is not None

    def is_terminated(self) -> bool:
        return self._terminated

    # ------------------------------------------------------------------ internal

    def _get_agent_action(self) -> np.ndarray:
        if self._agent is None:
            return self.env.action_space.sample()
        if isinstance(self._agent, HeuristicAgent):
            return self._agent.predict(
                self._obs, self.env.bases, self.env.assets, self.env.scenario.installations
            )
        # SB3 model
        action, _ = self._agent.predict(self._obs, deterministic=True)
        return action

    def _check_destruction_occurred(self) -> bool:
        if self.env is None:
            return False
        return any(a.is_destroyed for a in self.env.assets)

    def _state_snapshot(self, info: dict) -> dict:
        if self.env is None:
            return {}
        return {
            "time_hr": self.env.current_time,
            "step": self.env._step_count,
            "terminated": self._terminated,
            "last_reward": self._last_reward,
            "cumulative_reward": self._cumulative_reward,
            "bases": [b.status() for b in self.env.bases],
            "assets": [a.status() for a in self.env.assets],
            "threat_zones": [tz.as_dict() for tz in self.env.threat_zones],
            "info": info,
        }
