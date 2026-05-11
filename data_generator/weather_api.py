"""
data_generator/weather_api.py

Fetches real meteorological data from Open-Meteo (free, no API key required).
Used to correlate outdoor temperature with data center cooling load (PUE impact).

Open-Meteo docs: https://open-meteo.com/en/docs
"""

from __future__ import annotations
import math 
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import httpx
from loguru import logger


OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Default location: Florianópolis, SC, Brazil
DEFAULT_LAT =  -27.5954
DEFAULT_LON =  -48.5480
DEFAULT_LOCATION = "Florianopolis-SC"


@dataclass
class WeatherReading:
    """Meteorological reading at a point in time."""
    event_id:              str
    location:              str
    latitude:              float
    longitude:             float
    timestamp_utc:         str

    # Primary metrics (direct PUE impact)
    temperature_c:         float
    relative_humidity_pct: float
    dew_point_c:           float
    apparent_temp_c:       float

    # Cooling load drivers
    direct_radiation_wm2:  float   # Solar irradiance on DC roof
    diffuse_radiation_wm2: float
    wind_speed_10m_ms:     float
    wind_direction_deg:    float

    # Precipitation (chiller efficiency)
    precipitation_mm:      float
    cloud_cover_pct:       float

    # Derived
    enthalpy_kj_kg:        float   # Moist air enthalpy — key for economizer
    wet_bulb_temp_c:       float   # Chiller approach temperature


def _wet_bulb(temp_c: float, rh_pct: float) -> float:
    """Stull (2011) wet-bulb approximation. Accurate to ±0.3°C for typical ranges."""
    rh = rh_pct
    return (
        temp_c * math.atan(0.151977 * (rh + 8.313659) ** 0.5)
        + math.atan(temp_c + rh)
        - math.atan(rh - 1.676331)
        + 0.00391838 * rh ** 1.5 * math.atan(0.023101 * rh)
        - 4.686035
    )


def _enthalpy(temp_c: float, rh_pct: float) -> float:
    """
    Specific enthalpy of moist air [kJ/kg dry air].
    h = cp*T + W*(hg + cp_v*T)  where W = humidity ratio
    """
    import math
    # Saturation pressure (Antoine eq.)
    psat = 0.61078 * math.exp(17.27 * temp_c / (temp_c + 237.3))
    rh = rh_pct / 100.0
    p_atm = 101.325  # kPa
    W = 0.622 * (rh * psat) / (p_atm - rh * psat)
    return round(1.006 * temp_c + W * (2501 + 1.86 * temp_c), 3)


class WeatherClient:
    """
    Fetches current + forecast weather from Open-Meteo.
    Implements a simple local cache to avoid hammering the free API.
    """

    HOURLY_VARS = [
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "apparent_temperature",
        "precipitation",
        "cloud_cover",
        "wind_speed_10m",
        "wind_direction_10m",
        "direct_radiation",
        "diffuse_radiation",
    ]

    def __init__(
        self,
        lat: float = DEFAULT_LAT,
        lon: float = DEFAULT_LON,
        location_name: str = DEFAULT_LOCATION,
        cache_ttl_seconds: int = 900,   # 15-minute cache
    ):
        self.lat = lat
        self.lon = lon
        self.location_name = location_name
        self.cache_ttl = cache_ttl_seconds
        self._cache: dict = {}
        self._cache_ts: float = 0.0

    def _fetch_raw(self) -> dict:
        """Calls Open-Meteo API and returns raw JSON."""
        params = {
            "latitude":        self.lat,
            "longitude":       self.lon,
            "hourly":          ",".join(self.HOURLY_VARS),
            "timezone":        "America/Sao_Paulo",
            "forecast_days":   2,
        }
        with httpx.Client(timeout=15) as client:
            resp = client.get(OPEN_METEO_URL, params=params)
            resp.raise_for_status()
        logger.info(f"Open-Meteo API fetched for {self.location_name}")
        return resp.json()

    def _get_data(self) -> dict:
        """Returns cached or freshly fetched API data."""
        if time.time() - self._cache_ts > self.cache_ttl or not self._cache:
            self._cache = self._fetch_raw()
            self._cache_ts = time.time()
        return self._cache

    def get_current(self) -> Optional[WeatherReading]:
        """Returns the WeatherReading closest to the current time."""
        import uuid

        try:
            data = self._get_data()
            times = data["hourly"]["time"]
            now = datetime.now(timezone.utc)

            # Find the closest hourly timestamp
            idx = 0
            min_diff = float("inf")
            for i, ts in enumerate(times):
                t = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
                diff = abs((t - now).total_seconds())
                if diff < min_diff:
                    min_diff = diff
                    idx = i

            h = data["hourly"]
            temp_c = h["temperature_2m"][idx]
            rh_pct = h["relative_humidity_2m"][idx]

            return WeatherReading(
                event_id=str(uuid.uuid4()),
                location=self.location_name,
                latitude=self.lat,
                longitude=self.lon,
                timestamp_utc=now.isoformat(),
                temperature_c=temp_c,
                relative_humidity_pct=rh_pct,
                dew_point_c=h["dew_point_2m"][idx],
                apparent_temp_c=h["apparent_temperature"][idx],
                direct_radiation_wm2=h["direct_radiation"][idx] or 0.0,
                diffuse_radiation_wm2=h["diffuse_radiation"][idx] or 0.0,
                wind_speed_10m_ms=h["wind_speed_10m"][idx] or 0.0,
                wind_direction_deg=h["wind_direction_10m"][idx] or 0.0,
                precipitation_mm=h["precipitation"][idx] or 0.0,
                cloud_cover_pct=h["cloud_cover"][idx] or 0.0,
                enthalpy_kj_kg=_enthalpy(temp_c, rh_pct),
                wet_bulb_temp_c=round(
                    _wet_bulb(temp_c, rh_pct) if rh_pct > 0 else temp_c - 5, 2
                ),
            )
        except Exception as e:
            logger.error(f"WeatherClient error: {e}")
            return None

    def get_forecast_dataframe(self):
        """Returns a full 48h forecast as a pandas DataFrame. Useful for ML feature tables."""
        import pandas as pd

        data = self._get_data()
        df = pd.DataFrame(data["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        df["enthalpy_kj_kg"] = df.apply(
            lambda r: _enthalpy(r["temperature_2m"], r["relative_humidity_2m"]), axis=1
        )
        df["location"] = self.location_name
        df["latitude"]  = self.lat
        df["longitude"] = self.lon
        df.rename(columns={
            "time":                   "timestamp_local",
            "temperature_2m":         "temperature_c",
            "relative_humidity_2m":   "relative_humidity_pct",
            "dew_point_2m":           "dew_point_c",
            "apparent_temperature":   "apparent_temp_c",
            "wind_speed_10m":         "wind_speed_ms",
            "wind_direction_10m":     "wind_direction_deg",
        }, inplace=True)
        return df


# ---------------------------------------------------------------------------
# Quick standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import math
    from dataclasses import asdict
    import json

    client = WeatherClient()
    reading = client.get_current()
    if reading:
        print(json.dumps(asdict(reading), indent=2))

    df = client.get_forecast_dataframe()
    print(f"\n48h Forecast shape: {df.shape}")
    print(df[["timestamp_local", "temperature_c", "relative_humidity_pct", "enthalpy_kj_kg"]].head(10))
