"""Base — military installation with supply inventory and crater-authoritative runway.

Bug fix from v4: runway_length is a computed property derived from the crater list.
Repair removes craters from the list; length follows automatically. Desync is impossible.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Optional

import numpy as np

from eye.config import InstallationConfig, MissionConfig, SupplyType


class Base:
    def __init__(self, config: InstallationConfig, mission_config: MissionConfig) -> None:
        self.config = config
        self.mc = mission_config

        self.supplies: dict[str, float] = deepcopy(config.supply_inventory)
        self.personnel_count: int = config.initial_personnel

        # Crater list is the single source of truth for runway state.
        # Each entry represents one crater on the runway surface.
        self.craters: list[float] = []

        # Normalized time until next missile strike [0, 1]. Set by LogisticsEnv.
        self.missile_alert: float = 0.0

        self._destruction_events: list[str] = []

    # ------------------------------------------------------------------ runway (computed)

    @property
    def runway_length_ft(self) -> float:
        return max(0.0, self.config.runway_length_max_ft - len(self.craters) * self.mc.small_crater_size_ft)

    @property
    def runway_health_pct(self) -> float:
        if self.config.runway_length_max_ft == 0:
            return 0.0
        return self.runway_length_ft / self.config.runway_length_max_ft * 100.0

    def can_operate(self, runway_required_ft: float) -> bool:
        return self.runway_length_ft >= runway_required_ft

    # ------------------------------------------------------------------ supply

    def resupply(self, supply_type: str, amount_lbs: float) -> float:
        """Add supplies. Returns actual amount added."""
        self.supplies[supply_type] = self.supplies.get(supply_type, 0.0) + amount_lbs
        return amount_lbs

    def pickup(self, supply_type: str, requested_lbs: float) -> float:
        """Remove up to requested_lbs from inventory. Returns actual amount removed."""
        available = self.supplies.get(supply_type, 0.0)
        actual = min(available, requested_lbs)
        self.supplies[supply_type] = available - actual
        return actual

    def supply_pct(self, supply_type: str, max_capacity: float) -> float:
        if max_capacity <= 0:
            return 0.0
        return self.supplies.get(supply_type, 0.0) / max_capacity * 100.0

    # ------------------------------------------------------------------ per-step operations

    def consume(self, step_hours: float) -> None:
        """Burn food and water proportional to personnel headcount."""
        food_burn = self.personnel_count * self.mc.food_rate_lbs_per_person_hr * step_hours
        water_burn = self.personnel_count * self.mc.water_rate_lbs_per_person_hr * step_hours
        self.supplies[SupplyType.FOOD.value] = max(0.0, self.supplies.get(SupplyType.FOOD.value, 0.0) - food_burn)
        self.supplies[SupplyType.WATER.value] = max(0.0, self.supplies.get(SupplyType.WATER.value, 0.0) - water_burn)

        # Consume electricity at configured rate
        elec_rate = self.config.consumption_rates.get(SupplyType.ELECTRICITY.value, 0.0)
        if elec_rate > 0:
            self.supplies[SupplyType.ELECTRICITY.value] = max(
                0.0, self.supplies.get(SupplyType.ELECTRICITY.value, 0.0) - elec_rate * step_hours
            )

    def repair_runway(self, step_hours: float, rng: np.random.Generator) -> int:
        """Remove craters proportional to repair rate. Returns number of craters repaired."""
        if not self.craters or self.personnel_count < self.mc.__dict__.get("min_personnel_for_repair", 0):
            return 0
        feet_repairable = self.config.runway_repair_rate_ft_per_hr * step_hours

        # Consume runway material proportionally
        material_needed = feet_repairable / self.mc.material_to_ft_ratio
        available_material = self.supplies.get(SupplyType.RUNWAY_MATERIAL.value, 0.0)
        material_used = min(material_needed, available_material)
        feet_actually = (material_used / material_needed) * feet_repairable if material_needed > 0 else 0.0

        craters_to_repair = int(feet_actually / self.mc.small_crater_size_ft)
        craters_to_repair = min(craters_to_repair, len(self.craters))

        if craters_to_repair > 0:
            self.craters = self.craters[craters_to_repair:]
            self.supplies[SupplyType.RUNWAY_MATERIAL.value] = max(0.0, available_material - material_used)

        return craters_to_repair

    # ------------------------------------------------------------------ strike damage

    def apply_runway_strike(self, num_craters: int, rng: np.random.Generator) -> None:
        """Add craters at random positions. Runway length recomputed automatically."""
        for _ in range(num_craters):
            position = rng.uniform(0.0, self.config.runway_length_max_ft)
            self.craters.append(position)
        if self.runway_length_ft == 0:
            self._destruction_events.append(f"{self.config.name}: runway destroyed")

    def apply_supply_strike(self, supplies_hit_pct: float, rng: np.random.Generator) -> None:
        """Destroy a percentage of each supply type at random."""
        for st in self.supplies:
            if rng.random() < 0.5:  # not every supply type is hit each strike
                loss = self.supplies[st] * (supplies_hit_pct / 100.0) * rng.uniform(0.5, 1.0)
                self.supplies[st] = max(0.0, self.supplies[st] - loss)

    def pop_destruction_events(self) -> list[str]:
        events = self._destruction_events.copy()
        self._destruction_events.clear()
        return events

    # ------------------------------------------------------------------ serialization

    def status(self) -> dict:
        return {
            "name": self.config.name,
            "latitude": self.config.latitude,
            "longitude": self.config.longitude,
            "runway_length_ft": self.runway_length_ft,
            "runway_health_pct": self.runway_health_pct,
            "num_craters": len(self.craters),
            "personnel_count": self.personnel_count,
            "missile_alert": self.missile_alert,
            "supplies": {k: round(v, 1) for k, v in self.supplies.items()},
            "is_target": self.config.is_target,
        }
