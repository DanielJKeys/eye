"""
Threat Model for ACES

Implements dynamic threat zones, adversary behavior modeling, and
counter-intelligence capabilities for military logistics decision support.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
from datetime import datetime, timedelta
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import networkx as nx
from collections import defaultdict
import random

from aces.domain.threat import ThreatZone
from aces.domain.asset import Asset
from aces.utils.geometry import Point2D


@dataclass
class DynamicThreatZone:
    """Represents a dynamic threat zone."""
    id: str
    center: Point2D
    radius: float
    threat_level: float  # 0-1 scale
    threat_type: str  # 'enemy_force', 'minefield', 'air_defense', etc.
    mobility: float  # How mobile the threat is (0-1)
    detection_probability: float
    last_updated: datetime
    predicted_positions: List[Tuple[Point2D, datetime]] = field(default_factory=list)


@dataclass
class AdversaryProfile:
    """Profile of an adversary's behavior and capabilities."""
    id: str
    threat_type: str
    behavior_patterns: Dict[str, Any]
    capabilities: Dict[str, float]
    intelligence_level: float  # 0-1 scale
    adaptability: float  # How quickly they adapt to our actions
    observed_actions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class IntelligenceReport:
    """Counter-intelligence report with threat assessments."""
    report_id: str
    timestamp: datetime
    threat_zones: List[DynamicThreatZone]
    adversary_profiles: List[AdversaryProfile]
    confidence_levels: Dict[str, float]
    recommended_actions: List[str]
    risk_assessment: Dict[str, float]


@dataclass
class CounterIntelligenceMeasure:
    """Counter-intelligence measures and their effectiveness."""
    measure_type: str
    description: str
    effectiveness: float  # 0-1 scale
    cost: float
    implementation_time: float  # hours
    active: bool = False


class ThreatModel:
    """
    Advanced threat modeling system for military logistics.

    Provides dynamic threat zone tracking, adversary behavior analysis,
    and counter-intelligence capabilities for decision support.
    """

    def __init__(self):
        """Initialize the threat model."""
        self.threat_zones: Dict[str, DynamicThreatZone] = {}
        self.adversary_profiles: Dict[str, AdversaryProfile] = {}
        self.intelligence_reports: List[IntelligenceReport] = []
        self.counter_measures: Dict[str, CounterIntelligenceMeasure] = {}

        # Initialize default counter-intelligence measures
        self._initialize_counter_measures()

    def _initialize_counter_measures(self):
        """Initialize default counter-intelligence measures."""
        measures = [
            CounterIntelligenceMeasure(
                measure_type="signal_jamming",
                description="Electronic warfare jamming of enemy communications",
                effectiveness=0.7,
                cost=50.0,
                implementation_time=2.0
            ),
            CounterIntelligenceMeasure(
                measure_type="deception_operations",
                description="False flag operations to mislead enemy intelligence",
                effectiveness=0.8,
                cost=30.0,
                implementation_time=4.0
            ),
            CounterIntelligenceMeasure(
                measure_type="stealth_routing",
                description="Avoid known surveillance corridors",
                effectiveness=0.6,
                cost=10.0,
                implementation_time=1.0
            ),
            CounterIntelligenceMeasure(
                measure_type="intelligence_sharing",
                description="Share intelligence with allied forces",
                effectiveness=0.5,
                cost=5.0,
                implementation_time=0.5
            ),
            CounterIntelligenceMeasure(
                measure_type="cyber_defense",
                description="Enhanced cybersecurity measures",
                effectiveness=0.9,
                cost=40.0,
                implementation_time=6.0
            )
        ]

        for measure in measures:
            self.counter_measures[measure.measure_type] = measure

    def add_threat_zone(self, threat_zone: DynamicThreatZone):
        """
        Add or update a threat zone.

        Args:
            threat_zone: Threat zone definition
        """
        self.threat_zones[threat_zone.id] = threat_zone

    def add_adversary_profile(self, profile: AdversaryProfile):
        """
        Add or update an adversary profile.

        Args:
            profile: Adversary profile
        """
        self.adversary_profiles[profile.id] = profile

    def update_threat_zones_dynamic(
        self,
        current_time: datetime,
        asset_positions: Dict[str, Point2D],
        environmental_factors: Dict[str, Any]
    ) -> List[DynamicThreatZone]:
        """
        Update threat zones based on dynamic factors.

        Args:
            current_time: Current simulation time
            asset_positions: Current positions of friendly assets
            environmental_factors: Weather, terrain, etc.

        Returns:
            Updated list of threat zones
        """
        updated_zones = []

        for zone in self.threat_zones.values():
            # Predict threat movement based on mobility and asset positions
            new_zone = self._predict_threat_movement(zone, asset_positions, current_time)

            # Adjust threat level based on environmental factors
            new_zone = self._adjust_threat_level(new_zone, environmental_factors)

            # Update detection probability
            new_zone = self._update_detection_probability(new_zone, asset_positions)

            self.threat_zones[zone.id] = new_zone
            updated_zones.append(new_zone)

        return updated_zones

    def _predict_threat_movement(
        self,
        zone: DynamicThreatZone,
        asset_positions: Dict[str, Point2D],
        current_time: datetime
    ) -> DynamicThreatZone:
        """Predict how a threat zone might move."""
        if zone.mobility == 0:
            return zone  # Static threat

        # Simple prediction: threats move toward nearest assets with some mobility
        if asset_positions:
            nearest_asset_pos = min(
                asset_positions.values(),
                key=lambda pos: zone.center.distance_to(pos)
            )

            # Move toward nearest asset with mobility factor
            direction = (nearest_asset_pos - zone.center).normalize()
            movement_distance = zone.mobility * 2.0  # Arbitrary movement scale
            new_center = zone.center + direction * movement_distance

            # Add prediction to history
            zone.predicted_positions.append((new_center, current_time + timedelta(hours=1)))
            zone.center = new_center
            zone.last_updated = current_time

        return zone

    def _adjust_threat_level(
        self,
        zone: DynamicThreatZone,
        environmental_factors: Dict[str, Any]
    ) -> DynamicThreatZone:
        """Adjust threat level based on environmental factors."""
        base_level = zone.threat_level

        # Weather impact
        weather = environmental_factors.get('weather', 'clear')
        weather_multipliers = {
            'clear': 1.0,
            'rain': 0.8,
            'fog': 0.6,
            'storm': 0.4
        }
        weather_factor = weather_multipliers.get(weather, 1.0)

        # Time of day impact
        hour = environmental_factors.get('hour', 12)
        if 2 <= hour <= 6:  # Night time
            time_factor = 1.2  # Harder to detect at night
        else:
            time_factor = 1.0

        # Terrain impact
        terrain = environmental_factors.get('terrain', 'open')
        terrain_multipliers = {
            'open': 1.0,
            'urban': 0.7,
            'forest': 0.8,
            'mountain': 0.9
        }
        terrain_factor = terrain_multipliers.get(terrain, 1.0)

        zone.threat_level = min(base_level * weather_factor * time_factor * terrain_factor, 1.0)
        return zone

    def _update_detection_probability(
        self,
        zone: DynamicThreatZone,
        asset_positions: Dict[str, Point2D]
    ) -> DynamicThreatZone:
        """Update detection probability based on asset proximity."""
        if not asset_positions:
            zone.detection_probability = 0.1  # Base detection
            return zone

        # Detection increases with proximity to assets
        min_distance = min(zone.center.distance_to(pos) for pos in asset_positions.values())
        detection_range = zone.radius * 2  # Detection range is 2x threat radius

        if min_distance <= detection_range:
            # Exponential decay with distance
            zone.detection_probability = min(0.9, 0.8 * np.exp(-min_distance / detection_range))
        else:
            zone.detection_probability = 0.1

        return zone

    def analyze_adversary_behavior(
        self,
        observed_actions: List[Dict[str, Any]],
        time_window: timedelta = timedelta(hours=24)
    ) -> Dict[str, Any]:
        """
        Analyze adversary behavior patterns.

        Args:
            observed_actions: List of observed adversary actions
            time_window: Time window for analysis

        Returns:
            Analysis results
        """
        if not observed_actions:
            return {'patterns': [], 'confidence': 0.0}

        # Convert to DataFrame for analysis
        df = pd.DataFrame(observed_actions)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Filter by time window
        cutoff_time = datetime.now() - time_window
        recent_actions = df[df['timestamp'] > cutoff_time]

        if recent_actions.empty:
            return {'patterns': [], 'confidence': 0.0}

        # Analyze patterns
        patterns = {}

        # Action frequency
        action_counts = recent_actions['action_type'].value_counts()
        patterns['action_frequency'] = action_counts.to_dict()

        # Temporal patterns
        recent_actions['hour'] = recent_actions['timestamp'].dt.hour
        hourly_patterns = recent_actions.groupby('hour')['action_type'].count()
        patterns['hourly_activity'] = hourly_patterns.to_dict()

        # Location clustering
        if 'position' in recent_actions.columns:
            positions = np.array([pos for pos in recent_actions['position'] if pos])
            if len(positions) > 3:
                # Cluster positions to find activity hotspots
                scaler = StandardScaler()
                positions_scaled = scaler.fit_transform(positions)

                clustering = DBSCAN(eps=0.5, min_samples=2)
                clusters = clustering.fit_predict(positions_scaled)

                patterns['activity_clusters'] = len(set(clusters[clusters != -1]))

        # Predict next actions
        next_action_prediction = self._predict_adversary_actions(recent_actions)
        patterns['predicted_actions'] = next_action_prediction

        confidence = min(len(recent_actions) / 10, 1.0)  # More data = higher confidence

        return {
            'patterns': patterns,
            'confidence': confidence,
            'analysis_timestamp': datetime.now()
        }

    def _predict_adversary_actions(self, recent_actions: pd.DataFrame) -> Dict[str, float]:
        """Predict likely next actions based on patterns."""
        if recent_actions.empty:
            return {}

        # Simple frequency-based prediction
        action_probs = recent_actions['action_type'].value_counts(normalize=True).to_dict()

        # Add some randomness and recency bias
        recent_actions_sorted = recent_actions.sort_values('timestamp', ascending=False)
        recent_types = recent_actions_sorted['action_type'].head(5).tolist()

        # Boost probability of recently observed actions
        for action_type in recent_types:
            if action_type in action_probs:
                action_probs[action_type] *= 1.2

        # Normalize
        total = sum(action_probs.values())
        return {k: v/total for k, v in action_probs.items()}

    def generate_intelligence_report(
        self,
        asset_positions: Dict[str, Point2D],
        mission_objectives: List[Dict[str, Any]],
        environmental_factors: Dict[str, Any]
    ) -> IntelligenceReport:
        """
        Generate comprehensive intelligence report.

        Args:
            asset_positions: Current asset positions
            mission_objectives: Mission objectives
            environmental_factors: Environmental factors

        Returns:
            Intelligence report
        """
        current_time = datetime.now()

        # Update threat zones
        updated_zones = self.update_threat_zones_dynamic(
            current_time, asset_positions, environmental_factors
        )

        # Analyze adversary behavior
        all_observed_actions = []
        for profile in self.adversary_profiles.values():
            all_observed_actions.extend(profile.observed_actions)

        behavior_analysis = self.analyze_adversary_behavior(all_observed_actions)

        # Assess risks
        risk_assessment = self._assess_mission_risks(
            updated_zones, asset_positions, mission_objectives
        )

        # Generate recommendations
        recommendations = self._generate_threat_recommendations(
            updated_zones, risk_assessment, behavior_analysis
        )

        # Calculate confidence levels
        confidence_levels = {
            'threat_detection': np.mean([z.detection_probability for z in updated_zones]),
            'behavior_analysis': behavior_analysis.get('confidence', 0.0),
            'risk_assessment': min(len(updated_zones) / 5, 1.0)
        }

        report = IntelligenceReport(
            report_id=f"INTEL_{current_time.strftime('%Y%m%d_%H%M%S')}",
            timestamp=current_time,
            threat_zones=updated_zones,
            adversary_profiles=list(self.adversary_profiles.values()),
            confidence_levels=confidence_levels,
            recommended_actions=recommendations,
            risk_assessment=risk_assessment
        )

        self.intelligence_reports.append(report)
        return report

    def _assess_mission_risks(
        self,
        threat_zones: List[DynamicThreatZone],
        asset_positions: Dict[str, Point2D],
        mission_objectives: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Assess risks to mission objectives."""
        risks = {}

        for obj in mission_objectives:
            obj_pos = obj['position']
            obj_id = obj.get('id', f"objective_{mission_objectives.index(obj)}")

            total_risk = 0
            for zone in threat_zones:
                distance = obj_pos.distance_to(zone.center)
                if distance <= zone.radius:
                    # Risk contribution based on threat level and proximity
                    risk_contribution = zone.threat_level * (1 - distance / zone.radius)
                    total_risk += risk_contribution

            risks[obj_id] = min(total_risk, 1.0)

        # Asset-specific risks
        for asset_id, pos in asset_positions.items():
            asset_risk = 0
            for zone in threat_zones:
                distance = pos.distance_to(zone.center)
                if distance <= zone.radius:
                    risk_contribution = zone.threat_level * (1 - distance / zone.radius)
                    asset_risk += risk_contribution

            risks[f"asset_{asset_id}"] = min(asset_risk, 1.0)

        return risks

    def _generate_threat_recommendations(
        self,
        threat_zones: List[DynamicThreatZone],
        risk_assessment: Dict[str, float],
        behavior_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate threat-based recommendations."""
        recommendations = []

        # High-risk zones
        high_risk_zones = [z for z in threat_zones if z.threat_level > 0.7]
        if high_risk_zones:
            recommendations.append(
                f"Avoid {len(high_risk_zones)} high-threat zones: " +
                ", ".join([z.id for z in high_risk_zones])
            )

        # Risky assets
        risky_assets = [k for k, v in risk_assessment.items() if k.startswith('asset_') and v > 0.5]
        if risky_assets:
            recommendations.append(
                f"Relocate high-risk assets: {', '.join(risky_assets)}"
            )

        # Counter-intelligence measures
        predicted_actions = behavior_analysis.get('patterns', {}).get('predicted_actions', {})
        if predicted_actions:
            most_likely_action = max(predicted_actions, key=predicted_actions.get)
            recommendations.append(
                f"Prepare countermeasures for likely adversary action: {most_likely_action}"
            )

        # Environmental considerations
        if not recommendations:
            recommendations.append("Current threat level is manageable - maintain situational awareness")

        return recommendations

    def activate_counter_measure(self, measure_type: str) -> bool:
        """
        Activate a counter-intelligence measure.

        Args:
            measure_type: Type of counter-measure to activate

        Returns:
            Success status
        """
        if measure_type not in self.counter_measures:
            return False

        measure = self.counter_measures[measure_type]
        if measure.active:
            return True  # Already active

        measure.active = True

        # Apply effects (simplified)
        # In a real system, this would affect threat detection, adversary behavior, etc.

        return True

    def evaluate_counter_measures_effectiveness(
        self,
        threat_zones: List[DynamicThreatZone],
        asset_positions: Dict[str, Point2D]
    ) -> Dict[str, float]:
        """
        Evaluate effectiveness of active counter-measures.

        Args:
            threat_zones: Current threat zones
            asset_positions: Asset positions

        Returns:
            Effectiveness scores
        """
        effectiveness = {}

        active_measures = [m for m in self.counter_measures.values() if m.active]

        for measure in active_measures:
            # Simplified effectiveness calculation
            base_effectiveness = measure.effectiveness

            # Adjust based on threat types
            threat_type_match = any(
                z.threat_type in measure.description.lower()
                for z in threat_zones
            )
            if threat_type_match:
                base_effectiveness *= 1.2

            effectiveness[measure.measure_type] = min(base_effectiveness, 1.0)

        return effectiveness

    def export_threat_model(self, output_path: Path):
        """
        Export threat model data to JSON.

        Args:
            output_path: Path to save the export
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'threat_zones': [
                {
                    'id': z.id,
                    'center': z.center.to_dict(),
                    'radius': z.radius,
                    'threat_level': z.threat_level,
                    'threat_type': z.threat_type,
                    'mobility': z.mobility,
                    'detection_probability': z.detection_probability,
                    'last_updated': z.last_updated.isoformat(),
                    'predicted_positions': [
                        {'position': pos.to_dict(), 'time': time.isoformat()}
                        for pos, time in z.predicted_positions
                    ]
                }
                for z in self.threat_zones.values()
            ],
            'adversary_profiles': [
                {
                    'id': p.id,
                    'threat_type': p.threat_type,
                    'behavior_patterns': p.behavior_patterns,
                    'capabilities': p.capabilities,
                    'intelligence_level': p.intelligence_level,
                    'adaptability': p.adaptability,
                    'observed_actions': p.observed_actions
                }
                for p in self.adversary_profiles.values()
            ],
            'intelligence_reports': [
                {
                    'report_id': r.report_id,
                    'timestamp': r.timestamp.isoformat(),
                    'threat_zones_count': len(r.threat_zones),
                    'confidence_levels': r.confidence_levels,
                    'recommended_actions': r.recommended_actions,
                    'risk_assessment': r.risk_assessment
                }
                for r in self.intelligence_reports
            ],
            'counter_measures': [
                {
                    'measure_type': m.measure_type,
                    'description': m.description,
                    'effectiveness': m.effectiveness,
                    'cost': m.cost,
                    'implementation_time': m.implementation_time,
                    'active': m.active
                }
                for m in self.counter_measures.values()
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)