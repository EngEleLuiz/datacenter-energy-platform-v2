"""
data_generator/external_data_fetcher.py
========================================
Fetches external real-world data to enrich the LSTM PUE forecasting model:
  - Outdoor temperature (OpenWeatherMap API — free tier)
  - Energy spot price (CCEE PLD — public, or realistic mock)
  - Solar irradiance (Open-Meteo API — free, no API key required)

Usage:
    fetcher = ExternalDataFetcher(
        owm_api_key="YOUR_KEY_HERE",   # https://openweathermap.org/api
        lat=-27.60, lon=-48.55,        # Florianopolis, Brazil
        city_name="Florianopolis"
    )
    df = fetcher.get_merged_features(hours=48)
"""

import os
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────
OWM_CURRENT_URL  = "https://api.openweathermap.org/data/2.5/weather"
OWM_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
OPEN_METEO_URL   = "https://api.open-meteo.com/v1/forecast"
CCEE_PLD_URL     = "https://www.ccee.org.br/ccee/documentos/CCEE_653023"


class WeatherFetcher:
    """Fetches temperature, humidity, wind speed and cloud cover
    via OpenWeatherMap API (free tier — 60 calls/min)."""

    def __init__(self, api_key: str, lat: float, lon: float):
        self.api_key = api_key
        self.lat = lat
        self.lon = lon

    def get_current(self) -> dict:
        """Returns current weather conditions."""
        params = {
            "lat": self.lat, "lon": self.lon,
            "appid": self.api_key, "units": "metric",
        }
        r = requests.get(OWM_CURRENT_URL, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        return {
            "timestamp_utc": datetime.now(timezone.utc),
            "temp_c":        data["main"]["temp"],
            "humidity_pct":  data["main"]["humidity"],
            "wind_ms":       data["wind"]["speed"],
            "clouds_pct":    data["clouds"]["all"],
            "weather_main":  data["weather"][0]["main"],
        }

    def get_forecast_48h(self) -> pd.DataFrame:
        """Returns 48-hour forecast in 3-hour intervals (16 data points)."""
        params = {
            "lat": self.lat, "lon": self.lon,
            "appid": self.api_key, "units": "metric",
            "cnt": 16,
        }
        r = requests.get(OWM_FORECAST_URL, params=params, timeout=10)
        r.raise_for_status()
        items = r.json()["list"]
        rows = []
        for item in items:
            rows.append({
                "timestamp_utc": pd.to_datetime(item["dt"], unit="s", utc=True),
                "temp_c":        item["main"]["temp"],
                "humidity_pct":  item["main"]["humidity"],
                "wind_ms":       item["wind"]["speed"],
                "clouds_pct":    item["clouds"]["all"],
            })
        return pd.DataFrame(rows)


class SolarFetcher:
    """Fetches solar irradiance via Open-Meteo (no API key required)."""

    def __init__(self, lat: float, lon: float):
        self.lat = lat
        self.lon = lon

    def get_hourly(self, hours: int = 48) -> pd.DataFrame:
        """Returns hourly solar irradiance (W/m²)."""
        end   = datetime.now(timezone.utc)
        start = end - timedelta(hours=hours)
        params = {
            "latitude":   self.lat,
            "longitude":  self.lon,
            "hourly":     "shortwave_radiation,direct_radiation,diffuse_radiation",
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date":   end.strftime("%Y-%m-%d"),
            "timezone":   "UTC",
        }
        r = requests.get(OPEN_METEO_URL, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()["hourly"]
        df = pd.DataFrame({
            "timestamp_utc":     pd.to_datetime(data["time"], utc=True),
            "solar_ghi_wm2":     data["shortwave_radiation"],   # Global Horizontal Irradiance
            "solar_direct_wm2":  data["direct_radiation"],
            "solar_diffuse_wm2": data["diffuse_radiation"],
        })
        # Normalize to 0-1 range (physical maximum ~1000 W/m²)
        df["solar_norm"] = (df["solar_ghi_wm2"] / 1000.0).clip(0, 1)
        return df


class EnergyPriceFetcher:
    """Fetches or simulates electricity spot price.

    Attempt order:
      1. CCEE PLD (public CSV) — Brazilian energy market
      2. Realistic mock based on real hourly pattern
    """

    # Hourly average price profile (based on real CCEE 2023 data)
    # Index = hour of day (0-23), value = multiplier over base price
    HOURLY_PROFILE = [
        0.72, 0.68, 0.65, 0.63, 0.64, 0.70,  # 00-05h: overnight (low)
        0.80, 0.92, 1.05, 1.12, 1.18, 1.20,  # 06-11h: morning (rising)
        1.22, 1.25, 1.28, 1.30, 1.35, 1.40,  # 12-17h: afternoon (peak)
        1.45, 1.50, 1.38, 1.20, 1.00, 0.82,  # 18-23h: evening (falling)
    ]

    def __init__(self, base_price_brl_mwh: float = 120.0, noise_std: float = 8.0):
        """
        Args:
            base_price_brl_mwh: Base price in BRL/MWh (PLD avg 2023 ~ BRL 100-140)
            noise_std: Standard deviation of random noise (price volatility)
        """
        self.base_price = base_price_brl_mwh
        self.noise_std  = noise_std

    def get_historical_mock(self, hours: int = 48,
                             end: Optional[datetime] = None) -> pd.DataFrame:
        """Generates a realistic mock historical energy price series."""
        if end is None:
            end = datetime.now(timezone.utc)
        timestamps = [end - timedelta(minutes=15 * i) for i in range(hours * 4)][::-1]

        prices = []
        rng = np.random.default_rng(seed=42)
        for ts in timestamps:
            hour    = ts.hour
            weekday = ts.weekday()  # 0=Monday, 6=Sunday
            # Weekend discount: ~20% cheaper
            wd_factor = 0.80 if weekday >= 5 else 1.0
            profile_v = self.HOURLY_PROFILE[hour]
            noise     = rng.normal(0, self.noise_std)
            price     = self.base_price * profile_v * wd_factor + noise
            prices.append(max(price, 10.0))  # floor at BRL 10

        df = pd.DataFrame({
            "timestamp_utc":  pd.to_datetime(timestamps, utc=True),
            "price_brl_mwh":  prices,
            "price_norm":     np.array(prices) / (self.base_price * 1.6),
        })
        return df

    def get_ccee_pld(self) -> Optional[pd.DataFrame]:
        """Attempts to fetch real PLD data from CCEE (public CSV).
        Returns None on failure (falls back to mock)."""
        try:
            r = requests.get(CCEE_PLD_URL, timeout=15)
            r.raise_for_status()
            from io import StringIO
            df = pd.read_csv(StringIO(r.text), sep=";", decimal=",", encoding="latin1")
            logger.info("CCEE PLD fetched successfully")
            return df
        except Exception as e:
            logger.warning(f"CCEE PLD unavailable ({e}), falling back to mock")
            return None


class ExternalDataFetcher:
    """Main orchestrator — fetches and aligns all external data sources."""

    def __init__(
        self,
        owm_api_key: Optional[str] = None,
        lat: float = -27.60,
        lon: float = -48.55,
        city_name: str = "Florianopolis",
        energy_base_price: float = 120.0,
    ):
        self.owm_api_key = owm_api_key or os.getenv("OWM_API_KEY", "")
        self.lat = lat
        self.lon = lon
        self.city_name = city_name

        self.solar   = SolarFetcher(lat, lon)
        self.price   = EnergyPriceFetcher(energy_base_price)
        self.weather = WeatherFetcher(self.owm_api_key, lat, lon) if self.owm_api_key else None

    def _get_weather_df(self, hours: int) -> pd.DataFrame:
        """Returns weather DataFrame — real if API key is provided, mock otherwise."""
        if self.weather and self.owm_api_key:
            try:
                return self.weather.get_forecast_48h()
            except Exception as e:
                logger.warning(f"OWM failed ({e}), falling back to weather mock")

        # Synthetic weather mock based on typical Florianopolis climate
        end = datetime.now(timezone.utc)
        timestamps = pd.date_range(end=end, periods=hours * 4, freq="15min", tz="UTC")
        rng = np.random.default_rng(seed=99)
        temp_base = 24.0 + 6.0 * np.sin(
            2 * np.pi * (np.arange(len(timestamps)) / (24 * 4) + 0.3)
        )
        return pd.DataFrame({
            "timestamp_utc": timestamps,
            "temp_c":        temp_base + rng.normal(0, 1.5, len(timestamps)),
            "humidity_pct":  np.clip(65 + 15 * rng.standard_normal(len(timestamps)), 30, 100),
            "wind_ms":       np.abs(3.5 + rng.normal(0, 1.2, len(timestamps))),
            "clouds_pct":    np.clip(40 + 30 * rng.standard_normal(len(timestamps)), 0, 100),
        })

    def _get_solar_df(self, hours: int) -> pd.DataFrame:
        """Returns solar DataFrame — real or synthetic mock."""
        try:
            return self.solar.get_hourly(hours)
        except Exception as e:
            logger.warning(f"Open-Meteo failed ({e}), falling back to solar mock")
            end = datetime.now(timezone.utc)
            timestamps = pd.date_range(end=end, periods=hours * 4, freq="15min", tz="UTC")
            hour_arr = np.array([t.hour + t.minute / 60 for t in timestamps])
            solar = np.maximum(0, np.sin(np.pi * (hour_arr - 6) / 12)) ** 1.5 * 850
            rng   = np.random.default_rng(seed=77)
            solar += rng.normal(0, 30, len(timestamps))
            solar  = np.clip(solar, 0, 1000)
            return pd.DataFrame({
                "timestamp_utc": timestamps,
                "solar_ghi_wm2": solar,
                "solar_norm":    solar / 1000,
            })

    def get_merged_features(self, hours: int = 48) -> pd.DataFrame:
        """
        Returns a DataFrame with all external features aligned at 15-minute frequency.

        Returned columns:
            timestamp_utc, temp_c, humidity_pct, wind_ms, clouds_pct,
            solar_ghi_wm2, solar_norm, price_brl_mwh, price_norm,
            cooling_load_factor, time_sin, time_cos, weekday_sin, weekday_cos, is_peak
        """
        df_weather = self._get_weather_df(hours)
        df_solar   = self._get_solar_df(hours)
        df_price   = self.price.get_historical_mock(hours)

        # Temporal alignment — resample everything to 15-min frequency
        end   = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        start = end - timedelta(hours=hours)
        idx   = pd.date_range(start, end, freq="15min", tz="UTC")

        def _resample(df: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
            df = df.set_index("timestamp_utc").sort_index()
            df = df.select_dtypes(include="number")
            return df.reindex(idx, method="nearest", tolerance="30min").ffill().bfill()

        df_w = _resample(df_weather, idx)
        df_s = _resample(df_solar,   idx)
        df_p = _resample(df_price,   idx)

        merged = pd.concat([df_w, df_s, df_p], axis=1).reset_index()
        merged.rename(columns={"index": "timestamp_utc"}, inplace=True)

        # ── Derived features ──────────────────────────────────────────────
        # Cooling load factor: high outdoor temp → more cooling → higher PUE
        merged["cooling_load_factor"] = (
            (merged["temp_c"] - 18.0).clip(lower=0) / 20.0 +
            merged["humidity_pct"] / 200.0
        ).clip(0, 1)

        # Cyclical time encoding — preserves periodicity for the LSTM
        hour_of_day  = merged["timestamp_utc"].dt.hour + merged["timestamp_utc"].dt.minute / 60
        day_of_week  = merged["timestamp_utc"].dt.dayofweek
        merged["time_sin"]    = np.sin(2 * np.pi * hour_of_day / 24)
        merged["time_cos"]    = np.cos(2 * np.pi * hour_of_day / 24)
        merged["weekday_sin"] = np.sin(2 * np.pi * day_of_week / 7)
        merged["weekday_cos"] = np.cos(2 * np.pi * day_of_week / 7)

        # Peak hour flag (5pm-9pm, weekdays)
        is_peak_hour   = (hour_of_day >= 17) & (hour_of_day < 21)
        is_working_day = day_of_week < 5
        merged["is_peak"] = (is_peak_hour & is_working_day).astype(float)

        return merged.dropna()


# ── Feature list for the multivariate LSTM model ─────────────────────────
EXTERNAL_FEATURE_COLS = [
    "temp_c",
    "humidity_pct",
    "wind_ms",
    "clouds_pct",
    "solar_norm",
    "price_norm",
    "cooling_load_factor",
    "time_sin",
    "time_cos",
    "weekday_sin",
    "weekday_cos",
    "is_peak",
]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fetcher = ExternalDataFetcher(
        owm_api_key=os.getenv("OWM_API_KEY", ""),
        lat=-27.60, lon=-48.55,
    )
    df = fetcher.get_merged_features(hours=48)
    print(f"Shape: {df.shape}")
    print(df[["timestamp_utc"] + EXTERNAL_FEATURE_COLS].tail(8).to_string())
