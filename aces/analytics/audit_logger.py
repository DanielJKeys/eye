"""Audit Logger — tracks all decisions for accountability and review.

Records decision trails for after-action reviews and compliance tracking.
"""
from __future__ import annotations

from typing import Any, Dict, List
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import csv
from pathlib import Path

import numpy as np


@dataclass
class DecisionRecord:
    """Single decision record."""
    timestamp: str
    step: int
    asset_id: int
    asset_type: str
    action_type: str
    destination: str
    fuel_pct: float
    cargo_pct: float
    reward: float
    reason: str
    confidence: float = 0.5


class AuditLogger:
    """Logs all decisions for audit trail and accountability."""

    def __init__(self, session_id: str = None) -> None:
        self.session_id = session_id or datetime.now().isoformat()
        self.records: List[DecisionRecord] = []
        self.deviations: List[Dict[str, Any]] = []
        self.incidents: List[Dict[str, Any]] = []

    def log_decision(
        self,
        step: int,
        asset_id: int,
        asset_type: str,
        action_type: str,
        destination: str,
        fuel_pct: float,
        cargo_pct: float,
        reward: float = 0.0,
        reason: str = "",
        confidence: float = 0.5,
    ) -> None:
        """Record a decision."""
        record = DecisionRecord(
            timestamp=datetime.now().isoformat(),
            step=step,
            asset_id=asset_id,
            asset_type=asset_type,
            action_type=action_type,
            destination=destination,
            fuel_pct=fuel_pct,
            cargo_pct=cargo_pct,
            reward=reward,
            reason=reason,
            confidence=confidence,
        )
        self.records.append(record)

        # Check for deviations from standard procedures
        self._check_deviations(record)

    def _check_deviations(self, record: DecisionRecord) -> None:
        """Detect deviations from standard procedures."""
        issues = []

        # Low fuel dispatch
        if record.action_type == "Dispatch" and record.fuel_pct < 15:
            issues.append("Dispatch with critically low fuel")

        # High confidence low performance
        if record.confidence > 0.9 and record.reward < 0:
            issues.append("High confidence decision resulted in negative reward")

        # Overloaded cargo
        if record.cargo_pct > 95:
            issues.append("Asset at maximum cargo capacity")

        if issues:
            self.deviations.append(
                {
                    "step": record.step,
                    "asset_id": record.asset_id,
                    "issues": issues,
                    "timestamp": record.timestamp,
                }
            )

    def flag_incident(
        self, step: int, incident_type: str, description: str, severity: str = "medium"
    ) -> None:
        """Flag an incident for review."""
        self.incidents.append(
            {
                "step": step,
                "type": incident_type,
                "description": description,
                "severity": severity,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def generate_after_action_review(self) -> Dict[str, Any]:
        """Generate comprehensive after-action review."""
        if not self.records:
            return {"error": "No decisions recorded"}

        total_decisions = len(self.records)
        avg_confidence = np.mean([r.confidence for r in self.records])
        avg_reward = np.mean([r.reward for r in self.records])

        decision_types = {}
        for record in self.records:
            key = record.action_type
            decision_types[key] = decision_types.get(key, 0) + 1

        asset_performance = {}
        for record in self.records:
            if record.asset_id not in asset_performance:
                asset_performance[record.asset_id] = {
                    "decisions": 0,
                    "avg_fuel_pct": [],
                    "avg_cargo_pct": [],
                    "total_reward": 0,
                }
            asset_performance[record.asset_id]["decisions"] += 1
            asset_performance[record.asset_id]["avg_fuel_pct"].append(record.fuel_pct)
            asset_performance[record.asset_id]["avg_cargo_pct"].append(record.cargo_pct)
            asset_performance[record.asset_id]["total_reward"] += record.reward

        # Calculate averages
        for asset_id in asset_performance:
            perf = asset_performance[asset_id]
            perf["avg_fuel_pct"] = np.mean(perf["avg_fuel_pct"])
            perf["avg_cargo_pct"] = np.mean(perf["avg_cargo_pct"])

        return {
            "session_id": self.session_id,
            "total_decisions": total_decisions,
            "average_confidence": float(avg_confidence),
            "average_reward_per_decision": float(avg_reward),
            "decision_breakdown": decision_types,
            "asset_performance": asset_performance,
            "deviations_detected": len(self.deviations),
            "incidents_flagged": len(self.incidents),
            "deviations": self.deviations,
            "incidents": self.incidents,
        }

    def export_csv(self, filepath: Path | str) -> None:
        """Export audit trail to CSV."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "step", "asset_id", "asset_type", "action_type",
                "destination", "fuel_pct", "cargo_pct", "reward", "reason", "confidence"
            ])
            writer.writeheader()
            for record in self.records:
                writer.writerow(asdict(record))

    def export_json(self, filepath: Path | str) -> None:
        """Export audit trail to JSON."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "session_id": self.session_id,
            "records": [asdict(r) for r in self.records],
            "deviations": self.deviations,
            "incidents": self.incidents,
            "aar": self.generate_after_action_review(),
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "total_decisions": len(self.records),
            "avg_confidence": float(np.mean([r.confidence for r in self.records])),
            "avg_fuel_pct": float(np.mean([r.fuel_pct for r in self.records])),
            "avg_cargo_pct": float(np.mean([r.cargo_pct for r in self.records])),
            "avg_reward": float(np.mean([r.reward for r in self.records])),
        }
