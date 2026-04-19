"""SQLAlchemy ORM models.

OrmBase is the declarative base (named to avoid collision with the domain Base/Installation class).
All models belong to a Scenario so that multiple independent scenarios can coexist in the same DB.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class OrmBase(DeclarativeBase):
    pass


def _uuid() -> str:
    return str(uuid.uuid4())


class Scenario(OrmBase):
    __tablename__ = "scenarios"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    asset_types: Mapped[list[AssetType]] = relationship(back_populates="scenario", cascade="all, delete-orphan")
    installations: Mapped[list[Installation]] = relationship(back_populates="scenario", cascade="all, delete-orphan")
    threat_zones: Mapped[list[ThreatZone]] = relationship(back_populates="scenario", cascade="all, delete-orphan")
    missile_types: Mapped[list[MissileType]] = relationship(back_populates="scenario", cascade="all, delete-orphan")
    attack_schedule: Mapped[list[AttackSchedule]] = relationship(back_populates="scenario", cascade="all, delete-orphan")
    mission_assets: Mapped[list[MissionAsset]] = relationship(back_populates="scenario", cascade="all, delete-orphan")
    saved_models: Mapped[list[SavedModel]] = relationship(back_populates="scenario", cascade="all, delete-orphan")


class AssetType(OrmBase):
    __tablename__ = "asset_types"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    scenario_id: Mapped[str] = mapped_column(ForeignKey("scenarios.id"))
    name: Mapped[str] = mapped_column(String, nullable=False)
    cost: Mapped[float] = mapped_column(Float, default=0.0)
    fuel_capacity_lbs: Mapped[float] = mapped_column(Float, default=0.0)
    fuel_burn_lbs_per_hr: Mapped[float] = mapped_column(Float, default=0.0)
    speed_kts: Mapped[float] = mapped_column(Float, default=0.0)
    signature_pct: Mapped[float] = mapped_column(Float, default=0.0)
    max_cargo_weight_lbs: Mapped[float] = mapped_column(Float, default=0.0)
    max_cargo_volume_ft3: Mapped[float] = mapped_column(Float, default=0.0)
    runway_required_ft: Mapped[float] = mapped_column(Float, default=3000.0)
    personnel_seats: Mapped[int] = mapped_column(Integer, default=0)
    part_break_chance_pct: Mapped[float] = mapped_column(Float, default=0.0)
    part_break_min_lbs: Mapped[float] = mapped_column(Float, default=0.0)
    part_break_max_lbs: Mapped[float] = mapped_column(Float, default=0.0)
    cargo_types: Mapped[list] = mapped_column(JSON, default=list)

    scenario: Mapped[Scenario] = relationship(back_populates="asset_types")


class Installation(OrmBase):
    __tablename__ = "installations"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    scenario_id: Mapped[str] = mapped_column(ForeignKey("scenarios.id"))
    name: Mapped[str] = mapped_column(String, nullable=False)
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    runway_length_max_ft: Mapped[float] = mapped_column(Float, default=10000.0)
    runway_repair_rate_ft_per_hr: Mapped[float] = mapped_column(Float, default=50.0)
    is_target: Mapped[bool] = mapped_column(Boolean, default=False)
    access_type: Mapped[str] = mapped_column(String, default="air")
    initial_personnel: Mapped[int] = mapped_column(Integer, default=0)
    supply_inventory: Mapped[dict] = mapped_column(JSON, default=dict)
    consumption_rates: Mapped[dict] = mapped_column(JSON, default=dict)
    alert_thresholds: Mapped[dict] = mapped_column(JSON, default=dict)

    scenario: Mapped[Scenario] = relationship(back_populates="installations")


class ThreatZone(OrmBase):
    __tablename__ = "threat_zones"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    scenario_id: Mapped[str] = mapped_column(ForeignKey("scenarios.id"))
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    radius_nm: Mapped[float] = mapped_column(Float, default=100.0)
    ground_threat_pct: Mapped[float] = mapped_column(Float, default=0.0)
    sea_threat_pct: Mapped[float] = mapped_column(Float, default=0.0)
    air_threat_pct: Mapped[float] = mapped_column(Float, default=0.0)

    scenario: Mapped[Scenario] = relationship(back_populates="threat_zones")


class MissileType(OrmBase):
    __tablename__ = "missile_types"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    scenario_id: Mapped[str] = mapped_column(ForeignKey("scenarios.id"))
    name: Mapped[str] = mapped_column(String, nullable=False)
    impact_chance_pct: Mapped[float] = mapped_column(Float, default=0.7)
    supplies_hit_pct: Mapped[float] = mapped_column(Float, default=0.1)
    runway_hit_pct: Mapped[float] = mapped_column(Float, default=0.3)
    asset_hit_pct: Mapped[float] = mapped_column(Float, default=0.1)
    craters_created: Mapped[int] = mapped_column(Integer, default=3)

    scenario: Mapped[Scenario] = relationship(back_populates="missile_types")


class AttackSchedule(OrmBase):
    __tablename__ = "attack_schedule"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    scenario_id: Mapped[str] = mapped_column(ForeignKey("scenarios.id"))
    time_hr: Mapped[float] = mapped_column(Float)
    target_installation_id: Mapped[str] = mapped_column(String)
    num_missiles: Mapped[int] = mapped_column(Integer, default=1)
    missile_type_id: Mapped[str] = mapped_column(String)

    scenario: Mapped[Scenario] = relationship(back_populates="attack_schedule")


class MissionAsset(OrmBase):
    __tablename__ = "mission_assets"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    scenario_id: Mapped[str] = mapped_column(ForeignKey("scenarios.id"))
    name: Mapped[str] = mapped_column(String, nullable=False)
    asset_type_id: Mapped[str] = mapped_column(String)
    starting_installation_id: Mapped[str] = mapped_column(String)
    starting_fuel_lbs: Mapped[float] = mapped_column(Float, default=0.0)
    starting_personnel: Mapped[int] = mapped_column(Integer, default=0)
    starting_supplies: Mapped[dict] = mapped_column(JSON, default=dict)

    scenario: Mapped[Scenario] = relationship(back_populates="mission_assets")


class SavedModel(OrmBase):
    __tablename__ = "saved_models"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    scenario_id: Mapped[str] = mapped_column(ForeignKey("scenarios.id"))
    name: Mapped[str] = mapped_column(String, nullable=False)
    algorithm: Mapped[str] = mapped_column(String)
    file_path: Mapped[str] = mapped_column(String)
    vecnormalize_path: Mapped[str] = mapped_column(String, default="")
    training_steps: Mapped[int] = mapped_column(Integer, default=0)
    mean_reward: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    scenario: Mapped[Scenario] = relationship(back_populates="saved_models")
