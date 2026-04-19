"""All geographic calculations use haversine great-circle math.

Threat radii, travel distances, and arrival checks are all in nautical miles.
Never use Euclidean lat/lon distance — it distorts at non-equatorial latitudes.
"""
import math
from typing import Tuple

EARTH_RADIUS_NM = 3440.065  # nautical miles


def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in nautical miles."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_NM * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing from point 1 to point 2 in degrees [0, 360)."""
    lat1_r, lon1_r, lat2_r, lon2_r = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2_r - lon1_r
    x = math.sin(dlon) * math.cos(lat2_r)
    y = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def move_point(lat: float, lon: float, bearing: float, dist_nm: float) -> Tuple[float, float]:
    """New (lat, lon) after traveling dist_nm nautical miles along bearing."""
    if dist_nm <= 0:
        return lat, lon
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    bearing_r = math.radians(bearing)
    d = dist_nm / EARTH_RADIUS_NM
    new_lat_r = math.asin(
        math.sin(lat_r) * math.cos(d)
        + math.cos(lat_r) * math.sin(d) * math.cos(bearing_r)
    )
    new_lon_r = lon_r + math.atan2(
        math.sin(bearing_r) * math.sin(d) * math.cos(lat_r),
        math.cos(d) - math.sin(lat_r) * math.sin(new_lat_r),
    )
    new_lat = math.degrees(new_lat_r)
    new_lon = (math.degrees(new_lon_r) + 540) % 360 - 180  # normalize to [-180, 180]
    return new_lat, new_lon


def in_radius_nm(
    lat: float, lon: float, center_lat: float, center_lon: float, radius_nm: float
) -> bool:
    """True if (lat, lon) is within radius_nm nautical miles of center."""
    return haversine_nm(lat, lon, center_lat, center_lon) <= radius_nm
