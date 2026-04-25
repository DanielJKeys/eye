"""RewardShaper — computes scalar reward from state deltas.

All coefficients live in RewardConfig so researchers can tune via the UI
without touching code. VecNormalize handles scale during training.
"""
from __future__ import annotations

from eye.config import RewardConfig, SupplyType


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

    def efficient_routing(self, distance_nm: float) -> float:
        """Reward for selecting optimal flight paths."""
        return distance_nm * self.cfg.efficient_routing_per_nm

    def timely_delivery(self) -> float:
        """Bonus for deliveries completed on schedule."""
        return self.cfg.timely_delivery_bonus

    def risk_avoidance(self) -> float:
        """Reward for avoiding threat zones."""
        return self.cfg.risk_avoidance_per_step

    def asset_utilization(self, flight_hours: float) -> float:
        """Reward for keeping assets productively active."""
        return flight_hours * self.cfg.asset_utilization_per_hr

    def mission_complete(self) -> float:
        return self.cfg.mission_complete_bonus
