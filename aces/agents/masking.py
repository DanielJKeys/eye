"""ActionMasker — generates per-step boolean validity masks for MaskablePPO.

Masks invalid actions before the policy samples, so:
  - No gradients are wasted on impossible actions
  - Convergence is dramatically faster than the v4 supply_fractionalizer approach
  - Physical constraints are hard-enforced, not penalized
"""
from __future__ import annotations

import numpy as np

from aces.config import FUEL_LEVELS, SUPPLY_LEVELS, SUPPLY_PRIORITIES
from aces.spaces.action import ActionBuilder


class ActionMasker:
    def __init__(self, action_builder: ActionBuilder) -> None:
        self.ab = action_builder

    def compute_mask(self, bases, assets) -> np.ndarray:
        """Compute full boolean mask. True = action is valid."""
        mask = np.ones(self.ab.mask_size, dtype=bool)

        for i, asset in enumerate(assets):
            offsets = self.ab.mask_offsets_for_asset(i)

            if asset.is_destroyed:
                # Mask everything for a destroyed asset — keep only action[0] (stay)
                start, end = self.ab.zero_mask_for_asset(i)
                mask[start:end] = False
                mask[offsets["dest_start"]] = True  # must choose *something*
                continue

            if asset.in_transit:
                # Can redirect destination, but cannot load supplies
                mask[offsets["fuel_start"]:offsets["fuel_end"]] = False
                mask[offsets["fuel_start"]] = True  # 0% (no load) must be valid
                for p_start in offsets["priority_starts"]:
                    mask[p_start:p_start + len(SUPPLY_LEVELS)] = False
                    mask[p_start] = True  # 0% must be valid
            else:
                # At base — mask destinations with runway too short for this asset type
                for j, base in enumerate(bases):
                    dest_idx = offsets["dest_start"] + j
                    if not base.can_operate(asset.type.runway_required_ft):
                        mask[dest_idx] = False

                # If no valid destination, keep current one available
                dest_slice = mask[offsets["dest_start"]:offsets["dest_end"]]
                if not dest_slice.any():
                    current_idx = next(
                        (k for k, b in enumerate(bases) if b.config.id == asset.current_installation_id), 0
                    )
                    mask[offsets["dest_start"] + current_idx] = True

                # Mask fuel levels beyond remaining capacity and reserve requirements
                max_fuel_to_load = asset.type.fuel_capacity_lbs - asset.fuel_lbs
                min_reserve = asset.type.fuel_capacity_lbs * self.ab.scenario.mission_config.min_fuel_reserve_pct / 100.0
                effective_capacity = max(0.0, asset.type.fuel_capacity_lbs - min_reserve - asset.fuel_lbs)
                
                for level_idx, pct in enumerate(FUEL_LEVELS):
                    requested = asset.type.fuel_capacity_lbs * pct / 100.0
                    if requested > effective_capacity + 1.0:  # +1 tolerance
                        mask[offsets["fuel_start"] + level_idx] = False
                if not mask[offsets["fuel_start"]:offsets["fuel_end"]].any():
                    mask[offsets["fuel_start"]] = True  # 0% always allowed

                # Mask priority levels beyond cargo capacity or base availability
                current_base = next(
                    (b for b in bases if b.config.id == asset.current_installation_id), None
                )
                for p_idx, priority_name in enumerate(SUPPLY_PRIORITIES.keys()):
                    p_start = offsets["priority_starts"][p_idx]
                    # Calculate total available supplies in this priority group
                    priority_supplies = SUPPLY_PRIORITIES[priority_name]
                    total_available = sum(current_base.supplies.get(st.value, 0.0) for st in priority_supplies) if current_base else 0.0
                    cargo_space = max(0.0, asset.type.max_cargo_weight_lbs - asset.cargo_weight_lbs)
                    
                    for level_idx, pct in enumerate(SUPPLY_LEVELS):
                        requested = total_available * pct / 100.0
                        if requested > cargo_space + 1.0:
                            mask[p_start + level_idx] = False
                    if not mask[p_start:p_start + len(SUPPLY_LEVELS)].any():
                        mask[p_start] = True  # 0% always allowed

        return mask
