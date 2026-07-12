"""Geometry helpers for RobotCar sun-direction labels."""

from __future__ import annotations

import csv
import math
from bisect import bisect_left
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


WGS84_A = 6378137.0
WGS84_E2 = 0.00669437999014
UTM_K0 = 0.9996


def timestamp_us_to_datetime(timestamp_us: int) -> datetime:
    return datetime.fromtimestamp(timestamp_us / 1_000_000, tz=timezone.utc)


def utm_to_latlon(
    easting: float,
    northing: float,
    zone_number: int = 30,
    northern: bool = True,
) -> tuple[float, float]:
    """Convert WGS84 UTM coordinates to latitude/longitude degrees."""
    x = easting - 500000.0
    y = northing if northern else northing - 10000000.0

    lon_origin = math.radians((zone_number - 1) * 6 - 180 + 3)
    e = math.sqrt(WGS84_E2)
    e1 = (1.0 - math.sqrt(1.0 - WGS84_E2)) / (1.0 + math.sqrt(1.0 - WGS84_E2))
    e1sq = WGS84_E2 / (1.0 - WGS84_E2)

    m = y / UTM_K0
    mu = m / (
        WGS84_A
        * (
            1.0
            - WGS84_E2 / 4.0
            - 3.0 * WGS84_E2**2 / 64.0
            - 5.0 * WGS84_E2**3 / 256.0
        )
    )

    fp = (
        mu
        + (3.0 * e1 / 2.0 - 27.0 * e1**3 / 32.0) * math.sin(2.0 * mu)
        + (21.0 * e1**2 / 16.0 - 55.0 * e1**4 / 32.0) * math.sin(4.0 * mu)
        + (151.0 * e1**3 / 96.0) * math.sin(6.0 * mu)
        + (1097.0 * e1**4 / 512.0) * math.sin(8.0 * mu)
    )

    sin_fp = math.sin(fp)
    cos_fp = math.cos(fp)
    tan_fp = math.tan(fp)

    c1 = e1sq * cos_fp**2
    t1 = tan_fp**2
    n1 = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_fp**2)
    r1 = WGS84_A * (1.0 - WGS84_E2) / (1.0 - WGS84_E2 * sin_fp**2) ** 1.5
    d = x / (n1 * UTM_K0)

    lat = fp - (n1 * tan_fp / r1) * (
        d**2 / 2.0
        - (5.0 + 3.0 * t1 + 10.0 * c1 - 4.0 * c1**2 - 9.0 * e1sq) * d**4 / 24.0
        + (
            61.0
            + 90.0 * t1
            + 298.0 * c1
            + 45.0 * t1**2
            - 252.0 * e1sq
            - 3.0 * c1**2
        )
        * d**6
        / 720.0
    )
    lon = lon_origin + (
        d
        - (1.0 + 2.0 * t1 + c1) * d**3 / 6.0
        + (5.0 - 2.0 * c1 + 28.0 * t1 - 3.0 * c1**2 + 8.0 * e1sq + 24.0 * t1**2)
        * d**5
        / 120.0
    ) / cos_fp

    return math.degrees(lat), math.degrees(lon)


def sun_vector_enu(timestamp_us: int, lat_deg: float, lon_deg: float) -> np.ndarray:
    """Approximate unit sun direction in local ENU coordinates [east, north, up]."""
    dt = timestamp_us_to_datetime(timestamp_us)
    day = dt.timetuple().tm_yday
    minutes = dt.hour * 60.0 + dt.minute + dt.second / 60.0 + dt.microsecond / 60_000_000.0
    hour = minutes / 60.0

    gamma = 2.0 * math.pi / 365.0 * (day - 1.0 + (hour - 12.0) / 24.0)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2.0 * gamma)
        - 0.040849 * math.sin(2.0 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2.0 * gamma)
        + 0.000907 * math.sin(2.0 * gamma)
        - 0.002697 * math.cos(3.0 * gamma)
        + 0.00148 * math.sin(3.0 * gamma)
    )

    true_solar_time = (minutes + eqtime + 4.0 * lon_deg) % 1440.0
    hour_angle = math.radians(true_solar_time / 4.0 - 180.0)
    lat = math.radians(lat_deg)

    east = -math.cos(decl) * math.sin(hour_angle)
    north = math.cos(lat) * math.sin(decl) - math.sin(lat) * math.cos(decl) * math.cos(hour_angle)
    up = math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(hour_angle)
    vec = np.array([east, north, up], dtype=np.float32)
    return vec / max(float(np.linalg.norm(vec)), 1e-8)


def load_location_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as f:
        rows = [
            {
                "timestamp": int(row["timestamp"]),
                "northing": float(row["northing"]),
                "easting": float(row["easting"]),
            }
            for row in csv.DictReader(f)
        ]
    rows.sort(key=lambda row: row["timestamp"])
    return rows


def estimate_heading_enu(timestamp_us: int, location_rows: list[dict[str, float]]) -> np.ndarray:
    """Estimate horizontal vehicle heading [east, north] from neighbouring locations."""
    if len(location_rows) < 2:
        return np.array([0.0, 1.0], dtype=np.float32)

    timestamps = [int(row["timestamp"]) for row in location_rows]
    idx = bisect_left(timestamps, timestamp_us)
    if idx >= len(location_rows):
        idx = len(location_rows) - 1
    if idx > 0 and abs(timestamps[idx - 1] - timestamp_us) < abs(timestamps[idx] - timestamp_us):
        idx -= 1

    prev_idx = max(idx - 1, 0)
    next_idx = min(idx + 1, len(location_rows) - 1)
    if prev_idx == next_idx:
        next_idx = min(idx + 1, len(location_rows) - 1)
        prev_idx = max(idx - 1, 0)

    de = location_rows[next_idx]["easting"] - location_rows[prev_idx]["easting"]
    dn = location_rows[next_idx]["northing"] - location_rows[prev_idx]["northing"]
    heading = np.array([de, dn], dtype=np.float32)
    norm = float(np.linalg.norm(heading))
    if norm < 1e-6:
        return np.array([0.0, 1.0], dtype=np.float32)
    return heading / norm


def enu_to_car_frame(sun_enu: np.ndarray, heading_enu: np.ndarray) -> np.ndarray:
    """Rotate ENU sun vector into approximate car frame [right, forward, up]."""
    forward = heading_enu / max(float(np.linalg.norm(heading_enu)), 1e-8)
    right = np.array([forward[1], -forward[0]], dtype=np.float32)
    horizontal = sun_enu[:2]
    car = np.array(
        [
            float(np.dot(horizontal, right)),
            float(np.dot(horizontal, forward)),
            float(sun_enu[2]),
        ],
        dtype=np.float32,
    )
    return car / max(float(np.linalg.norm(car)), 1e-8)
