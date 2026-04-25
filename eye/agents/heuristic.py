"""Rule-based heuristic agents for baseline comparison and demo runs.

These are callable objects that accept (obs, bases, assets, installations) and
return an action array in the same format as an RL policy.
"""
from __future__ import annotations

import numpy as np

from eye.config import FUEL_LEVELS, SUPPLY_LEVELS, SUPPLY_PRIORITIES, SupplyType
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
        action = np.zeros(self.ab.total_dims, dtype=np.int64)
        dim = 0

        target_indices = [i for i, b in enumerate(bases) if b.config.is_target]
        hub_indices = [i for i, b in enumerate(bases) if not b.config.is_target]

        for i, asset in enumerate(assets):
            if asset.is_destroyed or asset.in_transit:
                # Keep current destination, no loading
                action[dim] = 0; dim += 1  # dest (irrelevant when in transit)
                action[dim] = 0; dim += 1  # fuel 0%
                for _ in SUPPLY_PRIORITIES:
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

            # Supply priorities: 4 dims matching the action space (CRITICAL/ESSENTIAL/MAINTENANCE/SUPPORT)
            for priority_name in SUPPLY_PRIORITIES.keys():
                if priority_name == "CRITICAL" and chosen_dest in target_indices:
                    action[dim] = len(SUPPLY_LEVELS) - 1  # 100% — fill munitions + fuel for attack run
                elif chosen_dest != current_base:
                    action[dim] = 1  # 25% general resupply when flying somewhere
                else:
                    action[dim] = 0  # no load if staying put
                dim += 1

        return action
