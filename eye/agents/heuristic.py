"""Rule-based heuristic agents for baseline comparison and demo runs.

These are callable objects that accept (obs, bases, assets, installations) and
return an action array in the same format as an RL policy.
"""
from __future__ import annotations

import numpy as np

from eye.config import FUEL_LEVELS, SUPPLY_LEVELS, SUPPLY_TYPES, SupplyType
from eye.spaces.action import ActionBuilder


class HeuristicAgent:
    """
    Greedy priority heuristic:
    - Always top-up fuel at the current base (load to 100%)
    - If current base has munitions and there is a target installation, load up and fly there
    - Otherwise, fly to the highest-stocked hub base and load supplies
    """

    def __init__(self, action_builder: ActionBuilder) -> None:
        self.ab = action_builder

    def predict(self, obs: np.ndarray, bases, assets, installations) -> np.ndarray:
        from eye.config import ScenarioConfig
        action = np.zeros(self.ab.total_dims, dtype=np.int64)
        dim = 0

        target_indices = [i for i, b in enumerate(bases) if b.config.is_target]
        hub_indices = [i for i, b in enumerate(bases) if not b.config.is_target]

        for i, asset in enumerate(assets):
            if asset.is_destroyed or asset.in_transit:
                # Keep current destination, no loading
                action[dim] = 0; dim += 1  # dest (irrelevant when in transit)
                action[dim] = 0; dim += 1  # fuel 0%
                for _ in SUPPLY_TYPES:
                    action[dim] = 0; dim += 1
                continue

            current_base = next(
                (j for j, b in enumerate(bases) if b.config.id == asset.current_installation_id), 0
            )

            # Decide destination
            if target_indices and asset.supplies.get(SupplyType.MUNITIONS.value, 0.0) > 1000.0:
                chosen_dest = target_indices[0]
            elif hub_indices:
                best_hub = max(
                    hub_indices,
                    key=lambda j: bases[j].supplies.get(SupplyType.MUNITIONS.value, 0.0),
                )
                chosen_dest = best_hub
            else:
                chosen_dest = current_base

            action[dim] = chosen_dest; dim += 1

            # Fuel: always 100% if staying at base; 0% if about to deliver
            if chosen_dest != current_base:
                action[dim] = len(FUEL_LEVELS) - 1; dim += 1  # 100%
            else:
                action[dim] = 0; dim += 1  # 0% (no need)

            # Supplies: load munitions to 100%, others at 25%
            for st in SUPPLY_TYPES:
                if st == SupplyType.MUNITIONS and chosen_dest in target_indices:
                    action[dim] = len(SUPPLY_LEVELS) - 1  # 100%
                elif chosen_dest not in target_indices and bases[current_base].supplies.get(st.value, 0.0) > 0:
                    action[dim] = 1  # 25%
                else:
                    action[dim] = 0
                dim += 1

        return action
