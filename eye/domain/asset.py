"""Asset — aircraft or vehicle with physics, state machine, and seeded RNG.

Personnel is tracked as an integer headcount (not a supply weight),
eliminating the unit confusion present in the original v4 implementation.
All random draws use the env's np_random for deterministic replay.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from eye.config import (
    AssetTypeConfig, MissionAssetConfig, MissionConfig, SUPPLY_TYPES, SupplyType,
)
from eye.utils.geometry import bearing_deg, haversine_nm, in_radius_nm, move_point


class Asset:
    def __init__(
        self,
        asset_config: MissionAssetConfig,
        type_config: AssetTypeConfig,
        start_lat: float,
        start_lon: float,
        mission_config: MissionConfig,
    ) -> None:
        self.config = asset_config
        self.type = type_config
        self.mc = mission_config

        # Position
        self.latitude: float = start_lat
        self.longitude: float = start_lon

        # State
        self.current_installation_id: str | None = asset_config.starting_installation_id
        self.destination_installation_id: str | None = None
        self.destination_lat: float = start_lat
        self.destination_lon: float = start_lon

        self.in_transit: bool = False
        self.is_destroyed: bool = False

        # Cargo
        self.fuel_lbs: float = asset_config.starting_fuel_lbs
        self.supplies: dict[str, float] = {st.value: 0.0 for st in SUPPLY_TYPES}
        self.supplies.update(asset_config.starting_supplies)
        self.personnel_count: int = asset_config.starting_personnel
        self.broken_parts_lbs: float = 0.0

        # Countdown before next action (loading / unloading)
        self.ready_timer_hr: float = 0.0

        self._destruction_events: list[str] = []
        self._delivery_events: list[dict] = []

    # ------------------------------------------------------------------ properties

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def is_ready(self) -> bool:
        return self.ready_timer_hr <= 0.0

    @property
    def cargo_weight_lbs(self) -> float:
        return sum(self.supplies.values()) + self.personnel_count * self.mc.weight_per_person_lbs

    # ------------------------------------------------------------------ loading

    def load_fuel(self, amount_lbs: float) -> float:
        """Load fuel; returns actual amount loaded."""
        capacity = self.type.fuel_capacity_lbs
        space = max(0.0, capacity - self.fuel_lbs)
        actual = min(amount_lbs, space)
        self.fuel_lbs += actual
        return actual

    def load_supply(self, supply_type: str, amount_lbs: float) -> float:
        """Load supply respecting cargo weight limit; returns actual amount loaded."""
        if supply_type not in [st.value for st in SUPPLY_TYPES]:
            return 0.0
        weight_space = max(0.0, self.type.max_cargo_weight_lbs - self.cargo_weight_lbs)
        actual = min(amount_lbs, weight_space)
        self.supplies[supply_type] = self.supplies.get(supply_type, 0.0) + actual
        return actual

    def unload_all(self) -> dict[str, float]:
        """Remove all cargo and return it for the base to receive."""
        delivered = {k: v for k, v in self.supplies.items() if v > 0}
        self.supplies = {st.value: 0.0 for st in SUPPLY_TYPES}
        self.personnel_count = 0
        return delivered

    # ------------------------------------------------------------------ flight

    def consume_fuel(self, hours: float) -> float:
        """Burn fuel proportional to hours flown. Returns fuel consumed."""
        burn = self.type.fuel_burn_lbs_per_hr * hours
        actual = min(burn, self.fuel_lbs)
        self.fuel_lbs = max(0.0, self.fuel_lbs - actual)
        return actual

    def travel_step(self, dest_lat: float, dest_lon: float, hours: float) -> None:
        """Move toward destination for one time step."""
        dist_available = self.type.speed_kts * hours  # nautical miles
        dist_to_dest = haversine_nm(self.latitude, self.longitude, dest_lat, dest_lon)

        if dist_to_dest <= dist_available:
            self.latitude, self.longitude = dest_lat, dest_lon
        else:
            brng = bearing_deg(self.latitude, self.longitude, dest_lat, dest_lon)
            self.latitude, self.longitude = move_point(self.latitude, self.longitude, brng, dist_available)

        self.consume_fuel(hours)

    def at_destination(self, dest_lat: float, dest_lon: float) -> bool:
        return in_radius_nm(self.latitude, self.longitude, dest_lat, dest_lon, self.mc.arrival_radius_nm)

    # ------------------------------------------------------------------ ground operations

    def takeoff(self, rng: np.random.Generator) -> None:
        """Transition to in-transit. Apply parts break check."""
        self.in_transit = True
        self.current_installation_id = None
        self.parts_break(rng)

    def land(self, installation_id: str, install_lat: float, install_lon: float) -> dict[str, float]:
        """Land at installation. Return cargo for base to receive."""
        self.in_transit = False
        self.current_installation_id = installation_id
        self.latitude = install_lat
        self.longitude = install_lon
        self.destination_installation_id = None
        self.ready_timer_hr = self.mc.load_time_hr  # unloading time
        return self.unload_all()

    def record_delivery(self, installation_id: str, supplies: dict[str, float]) -> None:
        self._delivery_events.append({"installation_id": installation_id, "supplies": supplies})

    def tick_ready_timer(self, hours: float) -> None:
        self.ready_timer_hr = max(0.0, self.ready_timer_hr - hours)

    def parts_break(self, rng: np.random.Generator) -> float:
        """Random parts failure on takeoff. Returns damage in lbs."""
        if rng.random() < self.type.part_break_chance_pct / 100.0:
            damage = rng.uniform(self.type.part_break_min_lbs, self.type.part_break_max_lbs)
            self.broken_parts_lbs += damage
            return damage
        return 0.0

    def repair_parts(self, parts_lbs: float) -> float:
        """Consume parts supply to fix broken components. Returns parts used."""
        used = min(parts_lbs, self.broken_parts_lbs)
        self.broken_parts_lbs = max(0.0, self.broken_parts_lbs - used)
        return used

    # ------------------------------------------------------------------ threat / damage

    def apply_strike(self, asset_hit_pct: float, rng: np.random.Generator) -> bool:
        """Returns True if asset is destroyed."""
        if rng.random() < asset_hit_pct / 100.0:
            self.is_destroyed = True
            self._destruction_events.append(f"{self.name} destroyed by missile strike")
            return True
        return False

    def threat_exposure(self, threat_zones) -> float:
        """Max threat level at current position across all threat zones."""
        return max((tz.threat_level(self.latitude, self.longitude) for tz in threat_zones), default=0.0)

    # ------------------------------------------------------------------ events

    def pop_destruction_events(self) -> list[str]:
        events = self._destruction_events.copy()
        self._destruction_events.clear()
        return events

    def pop_delivery_events(self) -> list[dict]:
        events = self._delivery_events.copy()
        self._delivery_events.clear()
        return events

    # ------------------------------------------------------------------ serialization

    def status(self) -> dict:
        return {
            "name": self.name,
            "type": self.type.name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "fuel_lbs": round(self.fuel_lbs, 1),
            "fuel_pct": round(self.fuel_lbs / self.type.fuel_capacity_lbs * 100, 1) if self.type.fuel_capacity_lbs else 0.0,
            "cargo_weight_lbs": round(self.cargo_weight_lbs, 1),
            "cargo_pct": round(self.cargo_weight_lbs / self.type.max_cargo_weight_lbs * 100, 1) if self.type.max_cargo_weight_lbs else 0.0,
            "supplies": {k: round(v, 1) for k, v in self.supplies.items() if v > 0},
            "personnel_count": self.personnel_count,
            "broken_parts_lbs": round(self.broken_parts_lbs, 1),
            "in_transit": self.in_transit,
            "is_destroyed": self.is_destroyed,
            "ready_timer_hr": round(self.ready_timer_hr, 2),
            "current_installation_id": self.current_installation_id,
            "destination_installation_id": self.destination_installation_id,
        }
