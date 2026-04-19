"""ScenarioService — CRUD operations for scenario data.

This is the only layer the UI talks to for database operations.
It never exposes ORM objects to higher layers — only config dataclasses and plain dicts.
"""
from __future__ import annotations

from aces.config import ScenarioConfig
from aces.data.database import get_session, init_db
from aces.data.repositories import ScenarioRepository, ModelRepository
from aces.data.seed_data import seed_default_scenario


class ScenarioService:
    def __init__(self) -> None:
        init_db()

    # ------------------------------------------------------------------ scenarios

    def list_scenarios(self) -> list[dict]:
        with get_session() as session:
            repo = ScenarioRepository(session)
            return [
                {"id": s.id, "name": s.name, "description": s.description,
                 "updated_at": s.updated_at.isoformat()}
                for s in repo.list_scenarios()
            ]

    def get_scenario_config(self, scenario_id: str) -> ScenarioConfig:
        with get_session() as session:
            return ScenarioRepository(session).to_scenario_config(scenario_id)

    def create_scenario(self, name: str, description: str = "") -> str:
        with get_session() as session:
            return ScenarioRepository(session).create_scenario(name, description).id

    def delete_scenario(self, scenario_id: str) -> None:
        with get_session() as session:
            ScenarioRepository(session).delete_scenario(scenario_id)

    def ensure_seed_data(self) -> str:
        """Seed the default scenario if the DB is empty. Returns seed scenario id."""
        scenarios = self.list_scenarios()
        if not scenarios:
            return seed_default_scenario()
        return scenarios[0]["id"]

    # ------------------------------------------------------------------ asset types

    def list_asset_types(self, scenario_id: str) -> list[dict]:
        with get_session() as session:
            sc = ScenarioRepository(session).get_scenario(scenario_id)
            return [
                {
                    "id": at.id, "name": at.name, "speed_kts": at.speed_kts,
                    "fuel_capacity_lbs": at.fuel_capacity_lbs,
                    "fuel_burn_lbs_per_hr": at.fuel_burn_lbs_per_hr,
                    "max_cargo_weight_lbs": at.max_cargo_weight_lbs,
                    "runway_required_ft": at.runway_required_ft,
                    "personnel_seats": at.personnel_seats,
                    "part_break_chance_pct": at.part_break_chance_pct,
                    "cost": at.cost,
                    "signature_pct": at.signature_pct,
                    "cargo_types": at.cargo_types,
                }
                for at in sc.asset_types
            ] if sc else []

    def add_asset_type(self, scenario_id: str, **kwargs) -> str:
        with get_session() as session:
            return ScenarioRepository(session).add_asset_type(scenario_id, **kwargs).id

    def update_asset_type(self, scenario_id: str, asset_type_id: str, **kwargs) -> None:
        with get_session() as session:
            repo = ScenarioRepository(session)
            sc = repo.get_scenario(scenario_id)
            rec = next((at for at in sc.asset_types if at.id == asset_type_id), None)
            if rec:
                repo.update_record(rec, **kwargs)

    def delete_asset_type(self, scenario_id: str, asset_type_id: str) -> None:
        with get_session() as session:
            repo = ScenarioRepository(session)
            sc = repo.get_scenario(scenario_id)
            rec = next((at for at in sc.asset_types if at.id == asset_type_id), None)
            if rec:
                repo.delete_record(rec)

    # ------------------------------------------------------------------ installations

    def list_installations(self, scenario_id: str) -> list[dict]:
        with get_session() as session:
            sc = ScenarioRepository(session).get_scenario(scenario_id)
            return [
                {
                    "id": i.id, "name": i.name, "latitude": i.latitude,
                    "longitude": i.longitude,
                    "runway_length_max_ft": i.runway_length_max_ft,
                    "runway_repair_rate_ft_per_hr": i.runway_repair_rate_ft_per_hr,
                    "is_target": i.is_target, "access_type": i.access_type,
                    "initial_personnel": i.initial_personnel,
                    "supply_inventory": i.supply_inventory,
                    "consumption_rates": i.consumption_rates,
                }
                for i in sc.installations
            ] if sc else []

    def add_installation(self, scenario_id: str, **kwargs) -> str:
        with get_session() as session:
            return ScenarioRepository(session).add_installation(scenario_id, **kwargs).id

    def update_installation(self, scenario_id: str, install_id: str, **kwargs) -> None:
        with get_session() as session:
            repo = ScenarioRepository(session)
            sc = repo.get_scenario(scenario_id)
            rec = next((i for i in sc.installations if i.id == install_id), None)
            if rec:
                repo.update_record(rec, **kwargs)

    def delete_installation(self, scenario_id: str, install_id: str) -> None:
        with get_session() as session:
            repo = ScenarioRepository(session)
            sc = repo.get_scenario(scenario_id)
            rec = next((i for i in sc.installations if i.id == install_id), None)
            if rec:
                repo.delete_record(rec)

    # ------------------------------------------------------------------ threat zones

    def list_threat_zones(self, scenario_id: str) -> list[dict]:
        with get_session() as session:
            sc = ScenarioRepository(session).get_scenario(scenario_id)
            return [
                {
                    "id": t.id, "latitude": t.latitude, "longitude": t.longitude,
                    "radius_nm": t.radius_nm,
                    "ground_threat_pct": t.ground_threat_pct,
                    "sea_threat_pct": t.sea_threat_pct,
                    "air_threat_pct": t.air_threat_pct,
                }
                for t in sc.threat_zones
            ] if sc else []

    def add_threat_zone(self, scenario_id: str, **kwargs) -> str:
        with get_session() as session:
            return ScenarioRepository(session).add_threat_zone(scenario_id, **kwargs).id

    def delete_threat_zone(self, scenario_id: str, zone_id: str) -> None:
        with get_session() as session:
            repo = ScenarioRepository(session)
            sc = repo.get_scenario(scenario_id)
            rec = next((t for t in sc.threat_zones if t.id == zone_id), None)
            if rec:
                repo.delete_record(rec)

    # ------------------------------------------------------------------ attack schedule

    def list_attacks(self, scenario_id: str) -> list[dict]:
        with get_session() as session:
            sc = ScenarioRepository(session).get_scenario(scenario_id)
            return [
                {
                    "id": a.id, "time_hr": a.time_hr,
                    "target_installation_id": a.target_installation_id,
                    "num_missiles": a.num_missiles,
                    "missile_type_id": a.missile_type_id,
                }
                for a in sorted(sc.attack_schedule, key=lambda x: x.time_hr)
            ] if sc else []

    def add_attack(self, scenario_id: str, **kwargs) -> str:
        with get_session() as session:
            return ScenarioRepository(session).add_attack_event(scenario_id, **kwargs).id

    def delete_attack(self, scenario_id: str, attack_id: str) -> None:
        with get_session() as session:
            repo = ScenarioRepository(session)
            sc = repo.get_scenario(scenario_id)
            rec = next((a for a in sc.attack_schedule if a.id == attack_id), None)
            if rec:
                repo.delete_record(rec)

    # ------------------------------------------------------------------ missile types

    def list_missile_types(self, scenario_id: str) -> list[dict]:
        with get_session() as session:
            sc = ScenarioRepository(session).get_scenario(scenario_id)
            return [
                {
                    "id": m.id, "name": m.name,
                    "impact_chance_pct": m.impact_chance_pct,
                    "supplies_hit_pct": m.supplies_hit_pct,
                    "runway_hit_pct": m.runway_hit_pct,
                    "asset_hit_pct": m.asset_hit_pct,
                    "craters_created": m.craters_created,
                }
                for m in sc.missile_types
            ] if sc else []

    def add_missile_type(self, scenario_id: str, **kwargs) -> str:
        with get_session() as session:
            return ScenarioRepository(session).add_missile_type(scenario_id, **kwargs).id

    def delete_missile_type(self, scenario_id: str, missile_id: str) -> None:
        with get_session() as session:
            repo = ScenarioRepository(session)
            sc = repo.get_scenario(scenario_id)
            rec = next((m for m in sc.missile_types if m.id == missile_id), None)
            if rec:
                repo.delete_record(rec)

    # ------------------------------------------------------------------ mission assets

    def list_mission_assets(self, scenario_id: str) -> list[dict]:
        with get_session() as session:
            sc = ScenarioRepository(session).get_scenario(scenario_id)
            return [
                {
                    "id": ma.id, "name": ma.name,
                    "asset_type_id": ma.asset_type_id,
                    "starting_installation_id": ma.starting_installation_id,
                    "starting_fuel_lbs": ma.starting_fuel_lbs,
                    "starting_personnel": ma.starting_personnel,
                    "starting_supplies": ma.starting_supplies,
                }
                for ma in sc.mission_assets
            ] if sc else []

    def add_mission_asset(self, scenario_id: str, **kwargs) -> str:
        with get_session() as session:
            return ScenarioRepository(session).add_mission_asset(scenario_id, **kwargs).id

    def delete_mission_asset(self, scenario_id: str, asset_id: str) -> None:
        with get_session() as session:
            repo = ScenarioRepository(session)
            sc = repo.get_scenario(scenario_id)
            rec = next((ma for ma in sc.mission_assets if ma.id == asset_id), None)
            if rec:
                repo.delete_record(rec)

    # ------------------------------------------------------------------ saved models

    def list_saved_models(self, scenario_id: str) -> list[dict]:
        with get_session() as session:
            return [
                {
                    "id": m.id, "name": m.name, "algorithm": m.algorithm,
                    "file_path": m.file_path, "vecnormalize_path": m.vecnormalize_path,
                    "training_steps": m.training_steps, "mean_reward": m.mean_reward,
                    "created_at": m.created_at.isoformat(),
                }
                for m in ModelRepository(session).list_models(scenario_id)
            ]

    def register_saved_model(self, scenario_id: str, **kwargs) -> str:
        with get_session() as session:
            return ModelRepository(session).save_model(scenario_id=scenario_id, **kwargs).id

    def delete_saved_model(self, scenario_id: str, model_id: str) -> None:
        with get_session() as session:
            ModelRepository(session).delete_model(model_id)
