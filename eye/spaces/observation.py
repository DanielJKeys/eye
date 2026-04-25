"""ObservationBuilder — constructs and normalizes the flat observation vector.

Structure (all values normalized to [0, 1]):
  Per base:   supply_levels (N_SUPPLY_TYPES), runway_health, missile_alert
  Per asset:  lat_norm, lon_norm, fuel_pct, cargo_pct, ready_timer_norm,
              in_transit, is_destroyed, supply_levels (N_SUPPLY_TYPES)
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces

from eye.config import N_SUPPLY_TYPES, SUPPLY_TYPES, ScenarioConfig


class ObservationBuilder:
    def __init__(self, scenario: ScenarioConfig) -> None:
        self._n_bases = len(scenario.installations)
        self._n_assets = len(scenario.mission_assets)
        self._n_supply = N_SUPPLY_TYPES

        # Reference capacities for normalization
        self._base_supply_caps: list[dict[str, float]] = [
            {
                st.value: max(inst.supply_inventory.get(st.value, 0.0), 1.0)
                for st in SUPPLY_TYPES
            }
            for inst in scenario.installations
        ]
        self._asset_fuel_caps: list[float] = []
        self._asset_cargo_caps: list[float] = []
        at_map = {at.id: at for at in scenario.asset_types}
        for ma in scenario.mission_assets:
            at = at_map[ma.asset_type_id]
            self._asset_fuel_caps.append(max(at.fuel_capacity_lbs, 1.0))
            self._asset_cargo_caps.append(max(at.max_cargo_weight_lbs, 1.0))

        self._max_ready_timer = scenario.mission_config.load_time_hr * 2

        dims_per_base = self._n_supply + 2  # supplies + runway_health + missile_alert
        dims_per_asset = 7 + self._n_supply  # lat, lon, fuel, cargo, timer, in_transit, destroyed + supplies
        self.obs_dim = self._n_bases * dims_per_base + self._n_assets * dims_per_asset

    @property
    def space(self) -> spaces.Box:
        return spaces.Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)

    def build(self, bases, assets) -> np.ndarray:
        vec: list[float] = []

        for i, base in enumerate(bases):
            caps = self._base_supply_caps[i]
            for st in SUPPLY_TYPES:
                raw = base.supplies.get(st.value, 0.0)
                vec.append(float(np.clip(raw / caps[st.value], 0.0, 1.0)))
            vec.append(float(np.clip(base.runway_health_pct / 100.0, 0.0, 1.0)))
            vec.append(float(np.clip(base.missile_alert, 0.0, 1.0)))

        for i, asset in enumerate(assets):
            vec.append(float(np.clip((asset.latitude + 90.0) / 180.0, 0.0, 1.0)))
            vec.append(float(np.clip((asset.longitude + 180.0) / 360.0, 0.0, 1.0)))
            vec.append(float(np.clip(asset.fuel_lbs / self._asset_fuel_caps[i], 0.0, 1.0)))
            vec.append(float(np.clip(asset.cargo_weight_lbs / self._asset_cargo_caps[i], 0.0, 1.0)))
            vec.append(float(np.clip(asset.ready_timer_hr / self._max_ready_timer, 0.0, 1.0)))
            vec.append(1.0 if asset.in_transit else 0.0)
            vec.append(1.0 if asset.is_destroyed else 0.0)
            for st in SUPPLY_TYPES:
                raw = asset.supplies.get(st.value, 0.0)
                cap = self._asset_cargo_caps[i]
                vec.append(float(np.clip(raw / cap, 0.0, 1.0)))

        return np.array(vec, dtype=np.float32)
