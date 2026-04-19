"""AgentRegistry — maps algorithm name strings to SB3 / custom agent classes.

Add new algorithms here. UI dropdowns pull from AVAILABLE_ALGORITHMS.
"""
from __future__ import annotations

from typing import Any

AVAILABLE_ALGORITHMS: dict[str, str] = {
    "maskable_ppo": "MaskablePPO (recommended — action masking)",
    "ppo": "PPO (no masking, baseline)",
    "a2c": "A2C (faster wall-clock, lower sample efficiency)",
    "heuristic_greedy": "Greedy Heuristic (rule-based, no training)",
}


def get_algorithm_class(name: str) -> Any:
    """Return the agent class for the given algorithm key."""
    if name == "maskable_ppo":
        from sb3_contrib import MaskablePPO
        return MaskablePPO
    if name == "ppo":
        from stable_baselines3 import PPO
        return PPO
    if name == "a2c":
        from stable_baselines3 import A2C
        return A2C
    if name == "heuristic_greedy":
        return None  # handled separately by HeuristicAgent
    raise ValueError(f"Unknown algorithm: {name!r}. Valid: {list(AVAILABLE_ALGORITHMS)}")


def is_heuristic(name: str) -> bool:
    return name.startswith("heuristic_")


def requires_masking(name: str) -> bool:
    return name == "maskable_ppo"
