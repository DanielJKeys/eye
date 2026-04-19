"""Data access layer — all SQL goes here, nothing above this layer touches ORM directly."""
from __future__ import annotations

from sqlalchemy.orm import Session

from aces.config import (
    AssetTypeConfig, AttackEvent, InstallationConfig, MissionAssetConfig,
    MissileTypeConfig, ScenarioConfig, ThreatZoneConfig,
)
from .models import (
    AssetType, AttackSchedule, Installation, MissionAsset,
    MissileType, SavedModel, Scenario, ThreatZone,
)


class ScenarioRepository:
    def __init__(self, session: Session) -> None:
        self._s = session

    # ------------------------------------------------------------------ query

    def list_scenarios(self) -> list[Scenario]:
        return self._s.query(Scenario).order_by(Scenario.updated_at.desc()).all()

    def get_scenario(self, scenario_id: str) -> Scenario | None:
        return self._s.get(Scenario, scenario_id)

    def get_scenario_by_name(self, name: str) -> Scenario | None:
        return self._s.query(Scenario).filter_by(name=name).first()

    # ------------------------------------------------------------------ create

    def create_scenario(self, name: str, description: str = "") -> Scenario:
        sc = Scenario(name=name, description=description)
        self._s.add(sc)
        self._s.flush()
        return sc

    def add_asset_type(self, scenario_id: str, **kwargs) -> AssetType:
        rec = AssetType(scenario_id=scenario_id, **kwargs)
        self._s.add(rec)
        self._s.flush()
        return rec

    def add_installation(self, scenario_id: str, **kwargs) -> Installation:
        rec = Installation(scenario_id=scenario_id, **kwargs)
        self._s.add(rec)
        self._s.flush()
        return rec

    def add_threat_zone(self, scenario_id: str, **kwargs) -> ThreatZone:
        rec = ThreatZone(scenario_id=scenario_id, **kwargs)
        self._s.add(rec)
        self._s.flush()
        return rec

    def add_missile_type(self, scenario_id: str, **kwargs) -> MissileType:
        rec = MissileType(scenario_id=scenario_id, **kwargs)
        self._s.add(rec)
        self._s.flush()
        return rec

    def add_attack_event(self, scenario_id: str, **kwargs) -> AttackSchedule:
        rec = AttackSchedule(scenario_id=scenario_id, **kwargs)
        self._s.add(rec)
        self._s.flush()
        return rec

    def add_mission_asset(self, scenario_id: str, **kwargs) -> MissionAsset:
        rec = MissionAsset(scenario_id=scenario_id, **kwargs)
        self._s.add(rec)
        self._s.flush()
        return rec

    # ------------------------------------------------------------------ update

    def update_record(self, record, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(record, k, v)
        self._s.flush()

    # ------------------------------------------------------------------ delete

    def delete_record(self, record) -> None:
        self._s.delete(record)
        self._s.flush()

    def delete_scenario(self, scenario_id: str) -> None:
        sc = self.get_scenario(scenario_id)
        if sc:
            self._s.delete(sc)
            self._s.flush()

    # ------------------------------------------------------------------ export to config

    def to_scenario_config(self, scenario_id: str) -> ScenarioConfig:
        sc = self.get_scenario(scenario_id)
        if sc is None:
            raise ValueError(f"Scenario {scenario_id!r} not found")

        asset_types = [
            AssetTypeConfig(
                id=at.id, name=at.name, cost=at.cost,
                fuel_capacity_lbs=at.fuel_capacity_lbs,
                fuel_burn_lbs_per_hr=at.fuel_burn_lbs_per_hr,
                speed_kts=at.speed_kts, signature_pct=at.signature_pct,
                max_cargo_weight_lbs=at.max_cargo_weight_lbs,
                max_cargo_volume_ft3=at.max_cargo_volume_ft3,
                runway_required_ft=at.runway_required_ft,
                personnel_seats=at.personnel_seats,
                part_break_chance_pct=at.part_break_chance_pct,
                part_break_min_lbs=at.part_break_min_lbs,
                part_break_max_lbs=at.part_break_max_lbs,
                cargo_types=at.cargo_types or [],
            )
            for at in sc.asset_types
        ]

        installations = [
            InstallationConfig(
                id=i.id, name=i.name, latitude=i.latitude, longitude=i.longitude,
                runway_length_max_ft=i.runway_length_max_ft,
                runway_repair_rate_ft_per_hr=i.runway_repair_rate_ft_per_hr,
                is_target=i.is_target, access_type=i.access_type,
                supply_inventory=i.supply_inventory or {},
                consumption_rates=i.consumption_rates or {},
                alert_thresholds=i.alert_thresholds or {},
                initial_personnel=i.initial_personnel,
            )
            for i in sc.installations
        ]

        threat_zones = [
            ThreatZoneConfig(
                id=t.id, latitude=t.latitude, longitude=t.longitude,
                radius_nm=t.radius_nm,
                ground_threat_pct=t.ground_threat_pct,
                sea_threat_pct=t.sea_threat_pct,
                air_threat_pct=t.air_threat_pct,
            )
            for t in sc.threat_zones
        ]

        missile_types = [
            MissileTypeConfig(
                id=m.id, name=m.name,
                impact_chance_pct=m.impact_chance_pct,
                supplies_hit_pct=m.supplies_hit_pct,
                runway_hit_pct=m.runway_hit_pct,
                asset_hit_pct=m.asset_hit_pct,
                craters_created=m.craters_created,
            )
            for m in sc.missile_types
        ]

        attack_schedule = [
            AttackEvent(
                time_hr=a.time_hr,
                target_installation_id=a.target_installation_id,
                num_missiles=a.num_missiles,
                missile_type_id=a.missile_type_id,
            )
            for a in sorted(sc.attack_schedule, key=lambda x: x.time_hr)
        ]

        mission_assets = [
            MissionAssetConfig(
                id=ma.id, name=ma.name,
                asset_type_id=ma.asset_type_id,
                starting_installation_id=ma.starting_installation_id,
                starting_fuel_lbs=ma.starting_fuel_lbs,
                starting_personnel=ma.starting_personnel,
                starting_supplies=ma.starting_supplies or {},
            )
            for ma in sc.mission_assets
        ]

        return ScenarioConfig(
            id=sc.id, name=sc.name, description=sc.description,
            asset_types=asset_types,
            installations=installations,
            threat_zones=threat_zones,
            missile_types=missile_types,
            attack_schedule=attack_schedule,
            mission_assets=mission_assets,
        )


class ModelRepository:
    def __init__(self, session: Session) -> None:
        self._s = session

    def list_models(self, scenario_id: str) -> list[SavedModel]:
        return self._s.query(SavedModel).filter_by(scenario_id=scenario_id).order_by(SavedModel.created_at.desc()).all()

    def save_model(self, scenario_id: str, name: str, algorithm: str,
                   file_path: str, vecnormalize_path: str,
                   training_steps: int, mean_reward: float) -> SavedModel:
        rec = SavedModel(
            scenario_id=scenario_id, name=name, algorithm=algorithm,
            file_path=file_path, vecnormalize_path=vecnormalize_path,
            training_steps=training_steps, mean_reward=mean_reward,
        )
        self._s.add(rec)
        self._s.flush()
        return rec

    def delete_model(self, model_id: str) -> None:
        rec = self._s.get(SavedModel, model_id)
        if rec:
            self._s.delete(rec)
            self._s.flush()
