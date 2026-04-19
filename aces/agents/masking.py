"""ActionMasker — generates per-step boolean validity masks for MaskablePPO.

Masks invalid actions before the policy samples, so:
  - No gradients are wasted on impossible actions
  - Convergence is dramatically faster than the v4 supply_fractionalizer approach
  - Physical constraints are hard-enforced, not penalized
"""
from __future__ import annotations

import numpy as np

from aces.config import FUEL_LEVELS, SUPPLY_LEVELS, SUPPLY_TYPES
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
                for s_start in offsets["supply_starts"]:
                    mask[s_start:s_start + len(SUPPLY_LEVELS)] = False
                    mask[s_start] = True  # 0% must be valid
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

                # Mask fuel levels beyond remaining capacity
                space = max(0.0, asset.type.fuel_capacity_lbs - asset.fuel_lbs)
                for level_idx, pct in enumerate(FUEL_LEVELS):
                    requested = asset.type.fuel_capacity_lbs * pct / 100.0
                    if requested > space + 1.0:  # +1 tolerance
                        mask[offsets["fuel_start"] + level_idx] = False
                if not mask[offsets["fuel_start"]:offsets["fuel_end"]].any():
                    mask[offsets["fuel_start"]] = True  # 0% always allowed

                # Mask supply levels beyond cargo capacity or base availability
                current_base = next(
                    (b for b in bases if b.config.id == asset.current_installation_id), None
                )
                for s_idx, st in enumerate(SUPPLY_TYPES):
                    s_start = offsets["supply_starts"][s_idx]
                    available_at_base = current_base.supplies.get(st.value, 0.0) if current_base else 0.0
                    cargo_space = max(0.0, asset.type.max_cargo_weight_lbs - asset.cargo_weight_lbs)
                    for level_idx, pct in enumerate(SUPPLY_LEVELS):
                        requested = available_at_base * pct / 100.0
                        if requested > cargo_space + 1.0:
                            mask[s_start + level_idx] = False
                    if not mask[s_start:s_start + len(SUPPLY_LEVELS)].any():
                        mask[s_start] = True  # 0% always allowed

        return mask
