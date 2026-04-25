"""ThreatZone — haversine-based geographic threat geometry."""
from __future__ import annotations

from eye.config import AttackEvent, ThreatZoneConfig
from eye.utils.geometry import haversine_nm, in_radius_nm


class ThreatZone:
    def __init__(self, config: ThreatZoneConfig) -> None:
        self.config = config

    def contains(self, lat: float, lon: float) -> bool:
        return in_radius_nm(lat, lon, self.config.latitude, self.config.longitude, self.config.radius_nm)

    def peak_threat_pct(self) -> float:
        return max(self.config.ground_threat_pct, self.config.sea_threat_pct, self.config.air_threat_pct)

    def threat_level(self, lat: float, lon: float) -> float:
        """Normalized threat [0, 1] — increases toward center."""
        dist = haversine_nm(lat, lon, self.config.latitude, self.config.longitude)
        if dist >= self.config.radius_nm:
            return 0.0
        proximity = 1.0 - dist / self.config.radius_nm
        return proximity * self.peak_threat_pct() / 100.0

    def as_dict(self) -> dict:
        return {
            "id": self.config.id,
            "latitude": self.config.latitude,
            "longitude": self.config.longitude,
            "radius_nm": self.config.radius_nm,
            "peak_threat_pct": self.peak_threat_pct(),
        }
