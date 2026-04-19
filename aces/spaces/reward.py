"""RewardShaper — computes scalar reward from state deltas.

All coefficients live in RewardConfig so researchers can tune via the UI
without touching code. VecNormalize handles scale during training.
"""
from __future__ import annotations

from aces.config import RewardConfig, SupplyType


class RewardShaper:
    def __init__(self, config: RewardConfig) -> None:
        self.cfg = config

    def delivery(self, supply_type: str, lbs: float) -> float:
        if supply_type == SupplyType.MUNITIONS.value:
            return lbs * self.cfg.munitions_delivered_per_lb
        return lbs * self.cfg.supply_delivered_per_lb

    def asset_destroyed(self) -> float:
        return self.cfg.asset_destroyed

    def runway_destroyed(self) -> float:
        return self.cfg.base_runway_destroyed

    def threat_exposure(self, threat_level: float) -> float:
        return threat_level * self.cfg.threat_exposure_per_step

    def mission_complete(self) -> float:
        return self.cfg.mission_complete_bonus
