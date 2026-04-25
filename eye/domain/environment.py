"""LogisticsEnv — Gymnasium-compliant military logistics simulation environment.

Design principles vs v4:
  - No global state. ScenarioConfig carries everything; multiple instances can
    run in parallel for SubprocVecEnv training.
  - reset(seed=N) is fully deterministic per Gymnasium 0.26 spec.
  - Crater list is the authoritative runway state (no desync bug).
  - All geographic checks use haversine (no Euclidean lat/lon distortion).
  - Personnel is a headcount integer, not a supply weight.
  - Missile alerts are included in the observation vector.
  - action_masks() method supports MaskablePPO from sb3-contrib.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional

import numpy as np
import gymnasium as gym

from eye.config import (
    AttackEvent, MissionAssetConfig, ScenarioConfig, SUPPLY_TYPES, SupplyType,
)
from eye.domain.asset import Asset
from eye.domain.base import Base
from eye.domain.threat import ThreatZone
from eye.spaces.action import ActionBuilder
from eye.spaces.observation import ObservationBuilder
from eye.spaces.reward import RewardShaper
from eye.agents.masking import ActionMasker


class LogisticsEnv(gym.Env):
    metadata = {"render_modes": ["console"]}

    def __init__(self, scenario: ScenarioConfig, render_mode: Optional[str] = None) -> None:
        super().__init__()
        self.scenario = scenario
        self.mc = scenario.mission_config
        self.render_mode = render_mode

        # Build spaces
        self._obs_builder = ObservationBuilder(scenario)
        self._action_builder = ActionBuilder(scenario)
        self._reward_shaper = RewardShaper(scenario.reward_config)
        self._masker = ActionMasker(self._action_builder)

        self.observation_space = self._obs_builder.space
        self.action_space = self._action_builder.space

        # Index lookups built once at init
        self._at_map = {at.id: at for at in scenario.asset_types}
        self._install_map = {inst.id: inst for inst in scenario.installations}
        self._missile_map = {m.id: m for m in scenario.missile_types}
        self._attack_schedule: list[AttackEvent] = sorted(
            scenario.attack_schedule, key=lambda e: e.time_hr
        )

        # Runtime state (populated in reset())
        self.bases: list[Base] = []
        self.assets: list[Asset] = []
        self.threat_zones: list[ThreatZone] = []
        self.current_time: float = 0.0
        self._step_count: int = 0
        self._processed_attacks: set[int] = set()
        self._history: list[dict] = []

        # Seeded RNG (set properly in reset())
        self.np_random: np.random.Generator = np.random.default_rng()

    # ------------------------------------------------------------------ gym API

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.bases = self._build_bases()
        self.assets = self._build_assets()
        self.threat_zones = [ThreatZone(cfg) for cfg in self.scenario.threat_zones]
        self.current_time = 0.0
        self._step_count = 0
        self._processed_attacks = set()
        self._history = []

        self._update_missile_alerts()
        obs = self._obs_builder.build(self.bases, self.assets)
        return obs, self._get_info()

    def step(self, action: np.ndarray):
        assert not isinstance(action, type(None)), "action must not be None"

        reward = 0.0
        prev_time = self.current_time

        # 1. Decode and apply actions (loading / destination setting)
        asset_actions = self._action_builder.decode(action)
        for i, (asset, act) in enumerate(zip(self.assets, asset_actions)):
            if asset.is_destroyed or asset.in_transit:
                continue
            reward += self._execute_asset_action(asset, act, i)

        # 2. Tick ready timers — assets whose timer hits 0 may take off
        for asset in self.assets:
            if not asset.is_destroyed and not asset.in_transit:
                asset.tick_ready_timer(self.mc.step_to_hour)
                if asset.is_ready and asset.destination_installation_id and \
                        asset.destination_installation_id != asset.current_installation_id:
                    dest_cfg = self._install_map.get(asset.destination_installation_id)
                    if dest_cfg and self._base_for_id(asset.current_installation_id).can_operate(asset.type.runway_required_ft):
                        asset.takeoff(self.np_random)

        # 3. Move in-transit assets and land arrivals
        for asset in self.assets:
            if asset.is_destroyed or not asset.in_transit:
                continue
            dest_cfg = self._install_map.get(asset.destination_installation_id)
            if dest_cfg is None:
                continue
            asset.travel_step(dest_cfg.latitude, dest_cfg.longitude, self.mc.step_to_hour)
            # Reward efficient routing (shorter paths are better)
            from eye.utils.geometry import haversine_nm
            distance_nm = haversine_nm(asset.latitude, asset.longitude, dest_cfg.latitude, dest_cfg.longitude)
            reward += self._reward_shaper.efficient_routing(distance_nm)
            if asset.at_destination(dest_cfg.latitude, dest_cfg.longitude):
                dest_base = self._base_for_id(asset.destination_installation_id)
                if dest_base and dest_base.can_operate(asset.type.runway_required_ft):
                    delivered = asset.land(
                        asset.destination_installation_id,
                        dest_cfg.latitude, dest_cfg.longitude,
                    )
                    reward += self._receive_delivery(dest_base, delivered, dest_base.config.is_target)
                    # Reward timely delivery
                    reward += self._reward_shaper.timely_delivery()

        # 4. Per-base consumption and runway repair
        for base in self.bases:
            base.consume(self.mc.step_to_hour)
            base.repair_runway(self.mc.step_to_hour, self.np_random)

        # 5. Advance time and process missile strikes in this window
        self.current_time += self.mc.step_to_hour
        reward += self._process_strikes(prev_time, self.current_time)

        # 6. Threat exposure penalty for in-transit assets
        for asset in self.assets:
            if asset.in_transit and not asset.is_destroyed:
                exposure = asset.threat_exposure(self.threat_zones)
                if exposure > 0:
                    reward += self._reward_shaper.threat_exposure(exposure)
                else:
                    # Reward for avoiding threats
                    reward += self._reward_shaper.risk_avoidance()

        # 7. Asset utilization rewards
        for asset in self.assets:
            if not asset.is_destroyed and asset.in_transit:
                # Reward for keeping assets active (flying)
                reward += self._reward_shaper.asset_utilization(self.mc.step_to_hour)

        # 8. Missile alerts for next step
        self._update_missile_alerts()

        # 8. Snapshot for history / UI
        self._step_count += 1
        snapshot = self._snapshot()
        self._history.append(snapshot)

        # 9. Termination check
        terminated = self.current_time >= self.mc.mission_length_hr
        if terminated:
            reward += self._reward_shaper.mission_complete()

        obs = self._obs_builder.build(self.bases, self.assets)
        return obs, float(reward), terminated, False, self._get_info()

    def action_masks(self) -> np.ndarray:
        """Called by MaskablePPO each step to exclude invalid actions."""
        return self._masker.compute_mask(self.bases, self.assets)

    def render(self):
        if self.render_mode == "console":
            print(f"\n=== T+{self.current_time:.1f}hr ===")
            for b in self.bases:
                rh = b.runway_health_pct
                print(f"  {b.config.name}: runway {rh:.0f}%, craters={len(b.craters)}, "
                      f"munitions={b.supplies.get(SupplyType.MUNITIONS.value,0):.0f}lbs")
            for a in self.assets:
                state = "DESTROYED" if a.is_destroyed else ("transit" if a.in_transit else "ground")
                print(f"  {a.name} [{state}]: fuel={a.fuel_lbs:.0f}lbs, "
                      f"cargo={a.cargo_weight_lbs:.0f}lbs")

    # ------------------------------------------------------------------ internal helpers

    def _build_bases(self) -> list[Base]:
        return [Base(cfg, self.mc) for cfg in self.scenario.installations]

    def _build_assets(self) -> list[Asset]:
        assets = []
        for ma in self.scenario.mission_assets:
            at = self._at_map[ma.asset_type_id]
            inst = self._install_map[ma.starting_installation_id]
            assets.append(Asset(ma, at, inst.latitude, inst.longitude, self.mc))
        return assets

    def _base_for_id(self, install_id: str | None) -> Base | None:
        if install_id is None:
            return None
        for b in self.bases:
            if b.config.id == install_id:
                return b
        return None

    def _execute_asset_action(self, asset: Asset, act: dict, asset_idx: int) -> float:
        """Load fuel/supplies and set destination. Returns immediate reward."""
        reward = 0.0
        current_base = self._base_for_id(asset.current_installation_id)
        if current_base is None:
            return reward

        # Set destination
        dest_idx = act["destination_idx"]
        if 0 <= dest_idx < len(self.bases):
            dest_base = self.bases[dest_idx]
            if dest_base.config.id != asset.current_installation_id:
                asset.destination_installation_id = dest_base.config.id
                asset.destination_lat = dest_base.config.latitude
                asset.destination_lon = dest_base.config.longitude
                asset.ready_timer_hr = self.mc.load_time_hr  # loading begins

        # Load fuel
        fuel_pct = act["fuel_pct"]
        if fuel_pct > 0:
            requested_fuel = asset.type.fuel_capacity_lbs * fuel_pct / 100.0
            taken = current_base.pickup(SupplyType.AVGAS.value, requested_fuel)
            asset.load_fuel(taken)

        # Load supplies by priority
        from eye.config import SUPPLY_PRIORITIES
        for priority_name, pct in act["priority_pcts"].items():
            if pct > 0:
                # Load supplies in this priority group proportionally
                priority_supplies = SUPPLY_PRIORITIES[priority_name]
                total_available = sum(current_base.supplies.get(st.value, 0.0) for st in priority_supplies)
                if total_available > 0:
                    # Distribute the requested percentage across available supplies in this priority
                    for st in priority_supplies:
                        if st.value in (asset.type.cargo_types or [s.value for s in SUPPLY_TYPES]):
                            available = current_base.supplies.get(st.value, 0.0)
                            if available > 0:
                                # Calculate proportional share of this priority's allocation
                                priority_share = available / total_available
                                requested = available * pct / 100.0 * priority_share
                                taken = current_base.pickup(st.value, requested)
                                asset.load_supply(st.value, taken)

        return reward

    def _receive_delivery(self, base: Base, delivered: dict[str, float], is_target: bool) -> float:
        reward = 0.0
        for st, lbs in delivered.items():
            if lbs > 0:
                base.resupply(st, lbs)
                if is_target:
                    reward += self._reward_shaper.delivery(st, lbs)
        return reward

    def _process_strikes(self, t_start: float, t_end: float) -> float:
        reward = 0.0
        for idx, event in enumerate(self._attack_schedule):
            if idx in self._processed_attacks:
                continue
            if t_start < event.time_hr <= t_end:
                self._processed_attacks.add(idx)
                reward += self._handle_strike(event)
        return reward

    def _handle_strike(self, event: AttackEvent) -> float:
        reward = 0.0
        missile = self._missile_map.get(event.missile_type_id)
        if missile is None:
            return reward

        target_base = self._base_for_id(event.target_installation_id)
        if target_base is None:
            return reward

        for _ in range(event.num_missiles):
            if self.np_random.random() > missile.impact_chance_pct / 100.0:
                continue  # missile missed

            prev_runway = target_base.runway_length_ft

            # Runway damage
            if self.np_random.random() < missile.runway_hit_pct / 100.0:
                target_base.apply_runway_strike(missile.craters_created, self.np_random)
                if target_base.runway_length_ft == 0 and prev_runway > 0:
                    reward += self._reward_shaper.runway_destroyed()

            # Supply damage
            target_base.apply_supply_strike(missile.supplies_hit_pct, self.np_random)

            # Asset damage (assets on the ground at this base)
            if missile.asset_hit_pct > 0:
                for asset in self.assets:
                    if not asset.is_destroyed and not asset.in_transit \
                            and asset.current_installation_id == event.target_installation_id:
                        destroyed = asset.apply_strike(missile.asset_hit_pct, self.np_random)
                        if destroyed:
                            reward += self._reward_shaper.asset_destroyed()

        return reward

    def _update_missile_alerts(self) -> None:
        """Set missile_alert on each base: normalized time until next incoming strike."""
        for base in self.bases:
            next_strike_time = None
            for event in self._attack_schedule:
                if event.target_installation_id == base.config.id and event.time_hr > self.current_time:
                    if next_strike_time is None or event.time_hr < next_strike_time:
                        next_strike_time = event.time_hr
            if next_strike_time is not None:
                time_remaining = next_strike_time - self.current_time
                if time_remaining <= self.mc.alert_window_hr:
                    base.missile_alert = float(np.clip(time_remaining / self.mc.alert_window_hr, 0.0, 1.0))
                else:
                    base.missile_alert = 0.0
            else:
                base.missile_alert = 0.0

    def _snapshot(self) -> dict:
        return {
            "time_hr": self.current_time,
            "step": self._step_count,
            "bases": [b.status() for b in self.bases],
            "assets": [a.status() for a in self.assets],
        }

    def _get_info(self) -> dict:
        return {
            "time_hr": self.current_time,
            "step": self._step_count,
            "assets_alive": sum(1 for a in self.assets if not a.is_destroyed),
            "bases_operational": sum(1 for b in self.bases if b.runway_health_pct > 0),
            "history": self._history,
        }
