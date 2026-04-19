"""Default seed scenario: Western Pacific Logistics.

Populates the database with a realistic example scenario so users can explore
the platform immediately after installation without manual data entry.
"""
from __future__ import annotations

from aces.config import SUPPLY_TYPES, SupplyType
from .database import get_session, init_db
from .repositories import ScenarioRepository


def _default_supply_inventory(**overrides) -> dict:
    base = {st.value: 0.0 for st in SUPPLY_TYPES}
    base.update(overrides)
    return base


def _default_consumption_rates(**overrides) -> dict:
    base = {st.value: 0.0 for st in SUPPLY_TYPES}
    base.update(overrides)
    return base


def _default_alert_thresholds(pct: float = 25.0) -> dict:
    return {st.value: pct for st in SUPPLY_TYPES}


def seed_default_scenario() -> str:
    """Create the Western Pacific scenario if it does not already exist. Returns scenario id."""
    init_db()
    with get_session() as session:
        repo = ScenarioRepository(session)
        existing = repo.get_scenario_by_name("Western Pacific Logistics")
        if existing:
            return existing.id

        sc = repo.create_scenario(
            name="Western Pacific Logistics",
            description=(
                "Multi-base logistics scenario across the Western Pacific. "
                "Forward Operating Base requires munitions resupply while "
                "adversarial missile strikes threaten hub installations."
            ),
        )
        sid = sc.id

        # ------------------------------------------------------------------ asset types
        c17 = repo.add_asset_type(
            sid,
            name="C-17 Globemaster III",
            cost=218_000_000.0,
            fuel_capacity_lbs=181_054.0,
            fuel_burn_lbs_per_hr=18_000.0,
            speed_kts=450.0,
            signature_pct=30.0,
            max_cargo_weight_lbs=170_900.0,
            max_cargo_volume_ft3=20_900.0,
            runway_required_ft=3_500.0,
            personnel_seats=102,
            part_break_chance_pct=2.0,
            part_break_min_lbs=100.0,
            part_break_max_lbs=500.0,
            cargo_types=[st.value for st in SUPPLY_TYPES],
        )
        c130 = repo.add_asset_type(
            sid,
            name="C-130J Hercules",
            cost=67_000_000.0,
            fuel_capacity_lbs=62_755.0,
            fuel_burn_lbs_per_hr=8_500.0,
            speed_kts=340.0,
            signature_pct=25.0,
            max_cargo_weight_lbs=42_000.0,
            max_cargo_volume_ft3=6_057.0,
            runway_required_ft=2_500.0,
            personnel_seats=92,
            part_break_chance_pct=3.0,
            part_break_min_lbs=50.0,
            part_break_max_lbs=300.0,
            cargo_types=[st.value for st in SUPPLY_TYPES],
        )
        f15 = repo.add_asset_type(
            sid,
            name="F-15E Strike Eagle",
            cost=87_700_000.0,
            fuel_capacity_lbs=35_550.0,
            fuel_burn_lbs_per_hr=12_000.0,
            speed_kts=1_650.0,
            signature_pct=15.0,
            max_cargo_weight_lbs=23_000.0,
            max_cargo_volume_ft3=600.0,
            runway_required_ft=4_000.0,
            personnel_seats=2,
            part_break_chance_pct=5.0,
            part_break_min_lbs=200.0,
            part_break_max_lbs=800.0,
            cargo_types=[SupplyType.MUNITIONS.value, SupplyType.AVGAS.value],
        )

        # ------------------------------------------------------------------ installations
        andersen = repo.add_installation(
            sid,
            name="Andersen AFB, Guam",
            latitude=13.584,
            longitude=144.929,
            runway_length_max_ft=11_000.0,
            runway_repair_rate_ft_per_hr=100.0,
            is_target=False,
            access_type="air",
            initial_personnel=1000,
            supply_inventory=_default_supply_inventory(
                AvGas=500_000.0,
                Munitions=200_000.0,
                Food=10_000.0,
                Water=10_000.0,
                Parts=20_000.0,
                MoGas=50_000.0,
                Electricity=100_000.0,
                RunwayMaterial=80_000.0,
            ),
            consumption_rates=_default_consumption_rates(
                Food=10.0,  # lbs/hr (1000 personnel)
                Water=10.0,
                Electricity=500.0,
            ),
            alert_thresholds=_default_alert_thresholds(25.0),
        )
        kadena = repo.add_installation(
            sid,
            name="Kadena AB, Okinawa",
            latitude=26.356,
            longitude=127.768,
            runway_length_max_ft=10_000.0,
            runway_repair_rate_ft_per_hr=80.0,
            is_target=False,
            access_type="air",
            initial_personnel=500,
            supply_inventory=_default_supply_inventory(
                AvGas=200_000.0,
                Munitions=100_000.0,
                Food=5_000.0,
                Water=5_000.0,
                Parts=10_000.0,
                MoGas=20_000.0,
                Electricity=50_000.0,
                RunwayMaterial=40_000.0,
            ),
            consumption_rates=_default_consumption_rates(
                Food=5.0,
                Water=5.0,
                Electricity=250.0,
            ),
            alert_thresholds=_default_alert_thresholds(25.0),
        )
        clark = repo.add_installation(
            sid,
            name="Clark AB, Philippines",
            latitude=15.185,
            longitude=120.560,
            runway_length_max_ft=9_000.0,
            runway_repair_rate_ft_per_hr=60.0,
            is_target=False,
            access_type="air",
            initial_personnel=300,
            supply_inventory=_default_supply_inventory(
                AvGas=150_000.0,
                Munitions=80_000.0,
                Food=3_000.0,
                Water=3_000.0,
                Parts=8_000.0,
                MoGas=15_000.0,
                Electricity=30_000.0,
                RunwayMaterial=25_000.0,
            ),
            consumption_rates=_default_consumption_rates(
                Food=3.0,
                Water=3.0,
                Electricity=150.0,
            ),
            alert_thresholds=_default_alert_thresholds(25.0),
        )
        fob = repo.add_installation(
            sid,
            name="FOB Alpha (Target)",
            latitude=22.0,
            longitude=126.0,
            runway_length_max_ft=6_000.0,
            runway_repair_rate_ft_per_hr=30.0,
            is_target=True,
            access_type="air",
            initial_personnel=200,
            supply_inventory=_default_supply_inventory(
                AvGas=20_000.0,
                Munitions=5_000.0,
                Food=2_000.0,
                Water=2_000.0,
                Parts=2_000.0,
                RunwayMaterial=5_000.0,
            ),
            consumption_rates=_default_consumption_rates(
                Food=2.0,
                Water=2.0,
                Electricity=100.0,
            ),
            alert_thresholds=_default_alert_thresholds(30.0),
        )

        # ------------------------------------------------------------------ threat zones
        repo.add_threat_zone(
            sid,
            latitude=20.0, longitude=120.0,
            radius_nm=300.0,
            ground_threat_pct=10.0, sea_threat_pct=40.0, air_threat_pct=30.0,
        )
        repo.add_threat_zone(
            sid,
            latitude=24.0, longitude=122.0,
            radius_nm=150.0,
            ground_threat_pct=20.0, sea_threat_pct=15.0, air_threat_pct=50.0,
        )

        # ------------------------------------------------------------------ missile types
        cruise = repo.add_missile_type(
            sid, name="Cruise Missile",
            impact_chance_pct=75.0, supplies_hit_pct=10.0,
            runway_hit_pct=30.0, asset_hit_pct=15.0, craters_created=2,
        )
        ballistic = repo.add_missile_type(
            sid, name="Ballistic Missile",
            impact_chance_pct=60.0, supplies_hit_pct=20.0,
            runway_hit_pct=50.0, asset_hit_pct=25.0, craters_created=5,
        )

        # ------------------------------------------------------------------ attack schedule
        repo.add_attack_event(
            sid,
            time_hr=24.0,
            target_installation_id=fob.id,
            num_missiles=2,
            missile_type_id=cruise.id,
        )
        repo.add_attack_event(
            sid,
            time_hr=48.0,
            target_installation_id=fob.id,
            num_missiles=3,
            missile_type_id=ballistic.id,
        )
        repo.add_attack_event(
            sid,
            time_hr=72.0,
            target_installation_id=kadena.id,
            num_missiles=1,
            missile_type_id=cruise.id,
        )

        # ------------------------------------------------------------------ mission assets
        repo.add_mission_asset(
            sid,
            name="Globemaster 1",
            asset_type_id=c17.id,
            starting_installation_id=andersen.id,
            starting_fuel_lbs=90_000.0,
            starting_personnel=0,
            starting_supplies={},
        )
        repo.add_mission_asset(
            sid,
            name="Globemaster 2",
            asset_type_id=c17.id,
            starting_installation_id=andersen.id,
            starting_fuel_lbs=90_000.0,
            starting_personnel=0,
            starting_supplies={},
        )
        repo.add_mission_asset(
            sid,
            name="Hercules 1",
            asset_type_id=c130.id,
            starting_installation_id=clark.id,
            starting_fuel_lbs=30_000.0,
            starting_personnel=0,
            starting_supplies={},
        )
        repo.add_mission_asset(
            sid,
            name="Eagle 1",
            asset_type_id=f15.id,
            starting_installation_id=kadena.id,
            starting_fuel_lbs=20_000.0,
            starting_personnel=0,
            starting_supplies={SupplyType.MUNITIONS.value: 10_000.0},
        )
        repo.add_mission_asset(
            sid,
            name="Eagle 2",
            asset_type_id=f15.id,
            starting_installation_id=kadena.id,
            starting_fuel_lbs=20_000.0,
            starting_personnel=0,
            starting_supplies={SupplyType.MUNITIONS.value: 10_000.0},
        )

        return sid
