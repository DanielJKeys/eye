"""Decision Analyzer — explains agent decisions and provides interpretability.

Helps war game planners understand why the agent took specific actions,
providing confidence scores, alternatives, and reasoning.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np

from eye.domain.environment import LogisticsEnv
from eye.spaces.action import ActionBuilder


class DecisionAnalyzer:
    """Analyzes and explains agent decisions for decision support."""

    def __init__(self, env: LogisticsEnv) -> None:
        self.env = env
        self.action_builder = env._action_builder
        self.decision_history: List[Dict[str, Any]] = []

    def explain_action(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        model: Any = None,
    ) -> Dict[str, Any]:
        """
        Explain why an action was chosen.

        Args:
            obs: Current observation
            action: Chosen action
            model: RL model (for confidence/value estimation)

        Returns:
            Dictionary with explanation data
        """
        decoded_action = self.action_builder.decode(action)

        # Get current state info
        current_state = self._extract_state_info()

        # Decode action details
        action_details = self._decode_action_details(decoded_action, current_state)

        # Calculate confidence (from model if available)
        confidence = self._estimate_confidence(obs, action, model)

        # Generate reasoning
        reasoning = self._generate_reasoning(
            current_state, action_details, confidence
        )

        # Alternative actions
        alternatives = self._get_alternative_actions(
            current_state, action_details, model
        )

        explanation = {
            "chosen_action": action_details,
            "confidence": confidence,
            "reasoning": reasoning,
            "alternatives": alternatives,
            "state_context": current_state,
            "asset_impacts": self._estimate_asset_impacts(action_details, current_state),
        }

        self.decision_history.append(explanation)
        return explanation

    def _extract_state_info(self) -> Dict[str, Any]:
        """Extract relevant state information."""
        return {
            "current_time": self.env.current_time,
            "mission_time_remaining": self.env.mc.mission_length_hr - self.env.current_time,
            "assets": [
                {
                    "id": i,
                    "type": asset.type.name,
                    "in_transit": asset.in_transit,
                    "fuel_pct": (
                        asset.fuel_lbs / asset.type.fuel_capacity_lbs * 100
                        if asset.type.fuel_capacity_lbs > 0
                        else 0
                    ),
                    "cargo_pct": (
                        asset.cargo_weight_lbs / asset.type.max_cargo_weight_lbs * 100
                        if asset.type.max_cargo_weight_lbs > 0
                        else 0
                    ),
                    "current_base": asset.current_installation_id,
                    "destination": asset.destination_installation_id,
                }
                for i, asset in enumerate(self.env.assets)
                if not asset.is_destroyed
            ],
            "bases": [
                {
                    "id": base.config.id,
                    "name": base.config.name,
                    "is_target": base.config.is_target,
                    "supplies": {s: base.supplies.get(s, 0) for s in base.supplies.keys()},
                    "runway_status": f"{base.runway_length_ft}/{base.config.runway_length_max_ft}",
                }
                for base in self.env.bases
            ],
            "active_threats": [
                {
                    "zone_id": tz.config.id,
                    "threat_level": tz.peak_threat_pct(),
                    "latitude": tz.config.latitude,
                    "longitude": tz.config.longitude,
                }
                for tz in self.env.threat_zones
            ],
        }

    def _decode_action_details(
        self, decoded_action: List[Dict], state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Break down decoded actions into human-readable details."""
        details = []
        for i, asset_action in enumerate(decoded_action):
            asset = self.env.assets[i]
            if asset.is_destroyed:
                continue

            dest_base = self.env._install_map.get(
                asset_action.get("destination_idx")
            )
            dest_name = dest_base.name if dest_base else "Unknown"

            details.append(
                {
                    "asset_id": i,
                    "asset_type": asset.type.name,
                    "action": "Dispatch" if not asset.in_transit else "Update Destination",
                    "destination": dest_name,
                    "destination_idx": asset_action.get("destination_idx"),
                    "fuel_pct": asset_action.get("fuel_pct", 0),
                    "supplies_priority": asset_action.get("priority_pcts", {}),
                    "estimated_flight_time_hr": self._estimate_flight_time(
                        asset, dest_base
                    ),
                    "estimated_fuel_burn": self._estimate_fuel_burn(
                        asset, dest_base
                    ),
                }
            )
        return details

    def _estimate_confidence(
        self, obs: np.ndarray, action: np.ndarray, model: Any
    ) -> float:
        """Estimate agent confidence in the action."""
        if model is None:
            return 0.5  # Unknown confidence

        try:
            # Get policy distribution from model
            if hasattr(model, "policy"):
                dist = model.policy.get_distribution(obs)
                if hasattr(dist, "probs"):
                    max_prob = np.max(dist.probs[0].detach().cpu().numpy())
                    return float(max_prob)
        except Exception:
            pass

        return 0.5  # Default

    def _generate_reasoning(
        self, state: Dict[str, Any], actions: List[Dict], confidence: float
    ) -> str:
        """Generate human-readable reasoning for the action."""
        reasons = []

        # Analyze threats
        if state["active_threats"]:
            highest_threat = max(
                state["active_threats"],
                key=lambda t: t["threat_level"]
            )
            if highest_threat["threat_level"] > 50:
                reasons.append(f"High threat level detected ({highest_threat['threat_level']:.1f}%)")

        # Analyze fuel constraints
        low_fuel_assets = [
            a for a in state["assets"] 
            if a["fuel_pct"] < 20
        ]
        if low_fuel_assets:
            reasons.append(f"{len(low_fuel_assets)} asset(s) need refueling")

        # Analyze time pressure
        if state["mission_time_remaining"] < 12:
            reasons.append("Mission completion approaching")

        # Analyze supply needs
        critical_supplies = [
            b for b in state["bases"] 
            if b["is_target"] and b["supplies"].get("Munitions (lbs)", 0) < 500
        ]
        if critical_supplies:
            reasons.append(f"{len(critical_supplies)} target(s) need munitions")

        # Analyze cargo utilization
        loaded_assets = [a for a in state["assets"] if a["cargo_pct"] > 50]
        if loaded_assets:
            reasons.append(f"{len(loaded_assets)} asset(s) carrying significant cargo")

        confidence_text = "high" if confidence > 0.7 else "medium" if confidence > 0.4 else "low"
        
        reasoning = (
            f"Agent chose actions with {confidence_text} confidence ({confidence:.1%}). "
            f"Key factors: {'; '.join(reasons) if reasons else 'balanced scenario'}."
        )
        return reasoning

    def _get_alternative_actions(
        self, state: Dict[str, Any], action_details: List[Dict], model: Any = None
    ) -> List[Dict[str, Any]]:
        """Identify alternative actions that were considered."""
        alternatives = []

        # For each asset, suggest alternative destinations
        for asset_action in action_details:
            asset = self.env.assets[asset_action["asset_id"]]

            for base in self.env.bases:
                if base.config.id == asset_action["destination_idx"]:
                    continue  # Skip chosen destination

                alt_time = self._estimate_flight_time(asset, base.config)
                alt_fuel = self._estimate_fuel_burn(asset, base.config)

                alternatives.append(
                    {
                        "asset_id": asset_action["asset_id"],
                        "alternative_destination": base.config.name,
                        "estimated_flight_time_hr": alt_time,
                        "estimated_fuel_burn": alt_fuel,
                        "is_feasible": alt_fuel < asset.fuel_lbs * 0.9,  # 10% reserve
                        "reason_not_chosen": self._rank_alternative(
                            base, asset_action, state
                        ),
                    }
                )

        return sorted(
            alternatives,
            key=lambda x: (
                not x["is_feasible"],
                x["estimated_fuel_burn"],
            ),
        )[:5]  # Top 5 alternatives

    def _rank_alternative(
        self, base, chosen_action: Dict, state: Dict
    ) -> str:
        """Explain why an alternative wasn't chosen."""
        if base.config.is_target:
            return "Non-target base (lower priority)"
        return "Longer flight, higher fuel cost"

    def _estimate_flight_time(self, asset, dest_base) -> float:
        """Estimate flight time to destination."""
        if dest_base is None:
            return 0.0
        from eye.utils.geometry import haversine_nm
        distance_nm = haversine_nm(
            asset.latitude,
            asset.longitude,
            dest_base.latitude,
            dest_base.longitude,
        )
        return distance_nm / asset.type.speed_kts

    def _estimate_fuel_burn(self, asset, dest_base) -> float:
        """Estimate fuel burn to destination."""
        if dest_base is None:
            return 0.0
        flight_time = self._estimate_flight_time(asset, dest_base)
        return flight_time * asset.type.fuel_burn_lbs_per_hr

    def _estimate_asset_impacts(
        self, actions: List[Dict], state: Dict
    ) -> Dict[str, Any]:
        """Estimate impacts of actions on assets."""
        return {
            "assets_taking_off": sum(
                1 for a in actions if a["action"] == "Dispatch"
            ),
            "total_fuel_to_load_lbs": sum(
                self._estimate_fuel_burn(self.env.assets[a["asset_id"]], None) * 0.8
                for a in actions
            ),
            "critical_assets": [
                a["asset_id"]
                for a in actions
                if a["fuel_pct"] < 25
            ],
        }

    def get_decision_history(self) -> List[Dict[str, Any]]:
        """Get all recorded decisions."""
        return self.decision_history

    def clear_history(self) -> None:
        """Clear decision history."""
        self.decision_history = []

    def summarize_episode(self) -> Dict[str, Any]:
        """Summarize the episode's decision patterns."""
        if not self.decision_history:
            return {"error": "No decisions recorded"}

        avg_confidence = np.mean([d["confidence"] for d in self.decision_history])
        most_common_reasoning = self._extract_common_themes()

        return {
            "total_decisions": len(self.decision_history),
            "average_confidence": float(avg_confidence),
            "decision_themes": most_common_reasoning,
            "key_decision_points": self._identify_key_decisions(),
        }

    def _extract_common_themes(self) -> List[str]:
        """Extract common themes from reasoning text."""
        themes = {}
        for decision in self.decision_history:
            reasoning = decision["reasoning"]
            if "Threat zone nearby" in reasoning:
                themes["threat_avoidance"] = themes.get("threat_avoidance", 0) + 1
            if "need refueling" in reasoning:
                themes["fuel_management"] = themes.get("fuel_management", 0) + 1
            if "Mission completion approaching" in reasoning:
                themes["time_pressure"] = themes.get("time_pressure", 0) + 1
            if "need munitions" in reasoning:
                themes["supply_optimization"] = themes.get("supply_optimization", 0) + 1

        return sorted(
            themes.items(), key=lambda x: x[1], reverse=True
        )

    def _identify_key_decisions(self) -> List[Dict[str, Any]]:
        """Identify pivotal decisions that had major impact."""
        key_decisions = []
        for i, decision in enumerate(self.decision_history):
            if decision["confidence"] > 0.8 or decision["confidence"] < 0.3:
                key_decisions.append(
                    {
                        "step": i,
                        "confidence": decision["confidence"],
                        "reasoning": decision["reasoning"][:100],
                    }
                )
        return key_decisions[:5]
