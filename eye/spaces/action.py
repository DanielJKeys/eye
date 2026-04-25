"""ActionBuilder — defines the MultiDiscrete action space.

Each asset gets:
  - 1 dimension  : destination (n_bases choices, index 0..n_bases-1)
  - 1 dimension  : fuel load level (5 choices: 0/25/50/75/100 % of capacity)
  - 4 dimensions : supply priority load levels (4 priorities × 3 amounts each)

Action levels map:  0→0%, 1→25%, 2→50%, 3→75%, 4→100%

The ActionMasker uses this class to build per-step validity masks.
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces

from eye.config import FUEL_LEVELS, N_SUPPLY_PRIORITIES, SUPPLY_LEVELS, ScenarioConfig


class ActionBuilder:
    def __init__(self, scenario: ScenarioConfig) -> None:
        self.scenario = scenario
        self.n_bases = len(scenario.installations)
        self.n_assets = len(scenario.mission_assets)
        self.n_priorities = N_SUPPLY_PRIORITIES

        # dims_per_asset = 1 (dest) + 1 (fuel) + n_priorities
        self.dims_per_asset = 2 + self.n_priorities

        # nvec: number of choices per action dimension
        self.nvec: list[int] = []
        for _ in range(self.n_assets):
            self.nvec.append(self.n_bases)  # destination
            self.nvec.append(len(FUEL_LEVELS))  # fuel level
            for _ in range(self.n_priorities):
                self.nvec.append(len(SUPPLY_LEVELS))  # priority level

        self.total_dims = len(self.nvec)
        self.mask_size = sum(self.nvec)

    @property
    def space(self) -> spaces.MultiDiscrete:
        return spaces.MultiDiscrete(self.nvec, dtype=np.int64)

    def decode(self, action: np.ndarray) -> list[dict]:
        """Convert flat action array into per-asset action dictionaries."""
        from eye.config import SUPPLY_PRIORITIES
        result = []
        dim = 0
        for _ in range(self.n_assets):
            dest_idx = int(action[dim]); dim += 1
            fuel_level = FUEL_LEVELS[int(action[dim])]; dim += 1
            priority_pcts: dict[str, int] = {}
            for priority_name in SUPPLY_PRIORITIES.keys():
                priority_pcts[priority_name] = SUPPLY_LEVELS[int(action[dim])]; dim += 1
            result.append({
                "destination_idx": dest_idx,
                "fuel_pct": fuel_level,
                "priority_pcts": priority_pcts,
            })
        return result

    def full_mask(self) -> np.ndarray:
        """All actions valid — used when no constraints are active."""
        return np.ones(self.mask_size, dtype=bool)

    def zero_mask_for_asset(self, asset_idx: int) -> tuple[int, int]:
        """Return (start, end) offsets into the flat mask for asset_idx."""
        offset = 0
        for i in range(asset_idx):
            offset += self.nvec[i * self.dims_per_asset]      # dest
            offset += self.nvec[i * self.dims_per_asset + 1]  # fuel
            for j in range(self.n_priorities):
                offset += self.nvec[i * self.dims_per_asset + 2 + j]
        end = offset
        for j in range(self.dims_per_asset):
            end += self.nvec[asset_idx * self.dims_per_asset + j]
        return offset, end

    def mask_offsets_for_asset(self, asset_idx: int) -> dict:
        """Return byte offsets for each sub-dimension of an asset's action block."""
        base_dim = asset_idx * self.dims_per_asset
        offset = sum(self.nvec[:base_dim])
        return {
            "dest_start": offset,
            "dest_end": offset + self.n_bases,
            "fuel_start": offset + self.n_bases,
            "fuel_end": offset + self.n_bases + len(FUEL_LEVELS),
            "priority_starts": [
                offset + self.n_bases + len(FUEL_LEVELS) + i * len(SUPPLY_LEVELS)
                for i in range(self.n_priorities)
            ],
        }
