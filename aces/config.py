"""Scenario and simulation configuration dataclasses.

All tunable parameters live here. No magic numbers anywhere else in the codebase.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SupplyType(str, Enum):
    AVGAS = "AvGas"
    MUNITIONS = "Munitions"
    FOOD = "Food"
    WATER = "Water"
    PARTS = "Parts"
    MOGAS = "MoGas"
    ELECTRICITY = "Electricity"
    RUNWAY_MATERIAL = "RunwayMaterial"


SUPPLY_TYPES: list[SupplyType] = list(SupplyType)
N_SUPPLY_TYPES: int = len(SUPPLY_TYPES)

# Action space encoding: percentage of capacity/available to load
FUEL_LEVELS: list[int] = [0, 25, 50, 75, 100]
SUPPLY_LEVELS: list[int] = [0, 25, 50, 75, 100]
N_LEVELS: int = len(FUEL_LEVELS)


# ---------------------------------------------------------------------------
# Mission-level tunable parameters
# ---------------------------------------------------------------------------

@dataclass
class MissionConfig:
    step_to_hour: float = 0.5           # simulation hours per env step
    mission_length_hr: float = 120.0    # total mission duration
    alert_window_hr: float = 1.0        # warning horizon before a strike
    load_time_hr: float = 1.0           # hours to load/unload an asset
    small_crater_size_ft: float = 10.0  # feet of runway destroyed per crater
    food_rate_lbs_per_person_hr: float = 0.01
    water_rate_lbs_per_person_hr: float = 0.01
    weight_per_person_lbs: float = 220.0   # cargo weight per transported person
    material_to_ft_ratio: float = 8000.0  # lbs of runway material per foot repaired
    arrival_radius_nm: float = 10.0        # nautical miles — counts as "arrived"


@dataclass
class RewardConfig:
    munitions_delivered_per_lb: float = 1.0
    supply_delivered_per_lb: float = 0.1
    asset_destroyed: float = -50.0
    base_runway_destroyed: float = -25.0       # when runway hits 0
    threat_exposure_per_step: float = -0.5     # per step inside a threat zone
    mission_complete_bonus: float = 100.0


@dataclass
class TrainingConfig:
    algorithm: str = "maskable_ppo"
    n_envs: int = 4
    total_timesteps: int = 500_000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    norm_obs: bool = True
    norm_reward: bool = True
    seed: Optional[int] = None
    model_save_dir: str = "models"


# ---------------------------------------------------------------------------
# Scenario configuration dataclasses (passed to LogisticsEnv)
# These are created by ScenarioService from ORM records.
# ---------------------------------------------------------------------------

@dataclass
class AssetTypeConfig:
    id: str
    name: str
    cost: float
    fuel_capacity_lbs: float
    fuel_burn_lbs_per_hr: float
    speed_kts: float
    signature_pct: float                # radar signature [0–100]
    max_cargo_weight_lbs: float
    max_cargo_volume_ft3: float
    runway_required_ft: float
    personnel_seats: int
    part_break_chance_pct: float        # probability of failure on takeoff
    part_break_min_lbs: float
    part_break_max_lbs: float
    cargo_types: list[str] = field(default_factory=lambda: [s.value for s in SUPPLY_TYPES])


@dataclass
class InstallationConfig:
    id: str
    name: str
    latitude: float
    longitude: float
    runway_length_max_ft: float
    runway_repair_rate_ft_per_hr: float
    is_target: bool
    access_type: str                    # "air", "sea", "land"
    supply_inventory: dict[str, float]  # SupplyType.value -> initial lbs
    consumption_rates: dict[str, float] # SupplyType.value -> lbs/hr (0 for most)
    alert_thresholds: dict[str, float]  # SupplyType.value -> pct for UI warning
    initial_personnel: int


@dataclass
class ThreatZoneConfig:
    id: str
    latitude: float
    longitude: float
    radius_nm: float                    # nautical miles — haversine checked
    ground_threat_pct: float
    sea_threat_pct: float
    air_threat_pct: float


@dataclass
class MissileTypeConfig:
    id: str
    name: str
    impact_chance_pct: float
    supplies_hit_pct: float
    runway_hit_pct: float
    asset_hit_pct: float
    craters_created: int


@dataclass
class AttackEvent:
    time_hr: float
    target_installation_id: str
    num_missiles: int
    missile_type_id: str


@dataclass
class MissionAssetConfig:
    id: str
    name: str
    asset_type_id: str
    starting_installation_id: str
    starting_fuel_lbs: float
    starting_personnel: int
    starting_supplies: dict[str, float]  # SupplyType.value -> lbs


@dataclass
class ScenarioConfig:
    id: str
    name: str
    description: str
    asset_types: list[AssetTypeConfig]
    installations: list[InstallationConfig]
    threat_zones: list[ThreatZoneConfig]
    missile_types: list[MissileTypeConfig]
    attack_schedule: list[AttackEvent]
    mission_assets: list[MissionAssetConfig]
    mission_config: MissionConfig = field(default_factory=MissionConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
