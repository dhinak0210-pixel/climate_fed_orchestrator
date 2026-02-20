"""
live_carbon_api.py — Multi-Source Live Carbon Intensity Client
══════════════════════════════════════════════════════════════════════
Implements a unified, production-grade carbon intensity data layer with:
  • ElectricityMaps API v3 (primary, commercial)
  • WattTime API v2 (secondary, MOER-based)
  • UK National Grid Carbon Intensity API (free, no auth)
  • Sinusoidal Simulation Fallback (offline / dev mode)
  • LRU TTL cache (5-minute freshness guarantee)
  • Automatic failover across providers
  • Async-first (aiohttp); sync wrapper available

GDPR / AI Act compliance:
  All API calls are logged with data source, timestamp, and freshness
  flag for full auditability.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import aiohttp

log = logging.getLogger("climate_fed.carbon_api")


# ──────────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class LiveCarbonData:
    """Normalised carbon intensity snapshot from any provider."""

    carbon_intensity_g_kwh: float  # g CO₂/kWh
    renewable_percentage: float  # 0–100 %
    fossil_percentage: float  # 0–100 %
    renewable_score: float  # 0.0–1.0 (derived)
    timestamp: datetime
    zone: str
    data_source: str
    is_estimated: bool = False
    is_simulated: bool = False

    @classmethod
    def from_intensity(
        cls,
        intensity_g_kwh: float,
        zone: str,
        source: str,
        renewable_pct: float = 0.0,
        fossil_pct: float = 0.0,
        estimated: bool = False,
    ) -> "LiveCarbonData":
        """
        Construct from raw intensity value.
        Renewable score: linear scale 0–800 g/kWh → 1.0–0.0
        """
        score = max(0.0, min(1.0, 1.0 - intensity_g_kwh / 800.0))
        return cls(
            carbon_intensity_g_kwh=intensity_g_kwh,
            renewable_percentage=renewable_pct,
            fossil_percentage=fossil_pct,
            renewable_score=score,
            timestamp=datetime.now(timezone.utc),
            zone=zone,
            data_source=source,
            is_estimated=estimated,
        )


# ──────────────────────────────────────────────────────────────────────────────
# TTL Cache
# ──────────────────────────────────────────────────────────────────────────────


class _TTLCache:
    def __init__(self, ttl_seconds: int = 300):
        self._store: Dict[str, Tuple[LiveCarbonData, float]] = {}
        self._ttl = ttl_seconds

    def get(self, key: str) -> Optional[LiveCarbonData]:
        if key in self._store:
            data, ts = self._store[key]
            if time.monotonic() - ts < self._ttl:
                return data
            del self._store[key]
        return None

    def set(self, key: str, value: LiveCarbonData) -> None:
        self._store[key] = (value, time.monotonic())

    def info(self) -> Dict:
        return {k: v[0].data_source for k, v in self._store.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Provider 1: Electricity Maps API v3
# ──────────────────────────────────────────────────────────────────────────────


class ElectricityMapsClient:
    """
    Electricity Maps API v3 (commercial + free tier: co2signal.com).
    Covers 100+ countries / zones with real-time data.
    """

    BASE_URL = "https://api.electricitymap.org/v3"
    CO2_SIGNAL_URL = "https://api.co2signal.com/v1/latest"  # free tier

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ELECTRICITY_MAPS_API_KEY")

    async def get_carbon_intensity(
        self,
        zone: str,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> LiveCarbonData:
        headers = {}
        if self.api_key:
            headers["auth-token"] = self.api_key

        # Try commercial API first, then free CO₂ Signal
        url = f"{self.BASE_URL}/carbon-intensity/latest"
        params = {"zone": zone}
        if lat is not None and lon is not None:
            params = {"lat": str(lat), "lon": str(lon)}

        close_session = session is None
        if close_session:
            session = aiohttp.ClientSession()

        try:
            async with session.get(
                url,
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=8),
            ) as r:
                r.raise_for_status()
                data = await r.json()
                intensity = float(
                    data.get(
                        "carbonIntensity",
                        data.get("data", {}).get("carbonIntensity", 450),
                    )
                )
                renew_pct = float(data.get("renewablePercentage", 50.0))
                fossil_pct = float(data.get("fossilFuelPercentage", 50.0))
                return LiveCarbonData.from_intensity(
                    intensity_g_kwh=intensity,
                    zone=zone,
                    source="ElectricityMaps v3",
                    renewable_pct=renew_pct,
                    fossil_pct=fossil_pct,
                    estimated=data.get("isEstimated", False),
                )
        except Exception as err:
            raise RuntimeError(f"ElectricityMaps failed ({zone}): {err}") from err
        finally:
            if close_session:
                await session.close()


# ──────────────────────────────────────────────────────────────────────────────
# Provider 2: WattTime API v2
# ──────────────────────────────────────────────────────────────────────────────


class WattTimeClient:
    """
    WattTime API v2 — Marginal Operating Emissions Rate (MOER).
    Provides real-time + 24 h forecast.  Unit: lbs CO₂/MWh → g CO₂/kWh × 0.453592.
    """

    BASE_URL = "https://api2.watttime.org/v2"
    _LBS_MWH_TO_G_KWH = 0.453592

    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        self.username = username or os.environ.get("WATTTIME_USERNAME", "")
        self.password = password or os.environ.get("WATTTIME_PASSWORD", "")
        self._token: Optional[str] = None

    async def _ensure_token(self, session: aiohttp.ClientSession) -> None:
        if self._token:
            return
        async with session.post(
            f"{self.BASE_URL}/login",
            auth=aiohttp.BasicAuth(self.username, self.password),
            timeout=aiohttp.ClientTimeout(total=8),
        ) as r:
            r.raise_for_status()
            self._token = (await r.json())["token"]

    async def get_realtime_emissions(
        self,
        lat: float,
        lon: float,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> LiveCarbonData:
        close_session = session is None
        if close_session:
            session = aiohttp.ClientSession()
        try:
            await self._ensure_token(session)
            headers = {"Authorization": f"Bearer {self._token}"}
            params = {"latitude": str(lat), "longitude": str(lon)}
            async with session.get(
                f"{self.BASE_URL}/index",
                headers=headers,
                params=params,
                timeout=aiohttp.ClientTimeout(total=8),
            ) as r:
                r.raise_for_status()
                data = await r.json()
                moer = float(data.get("moer", 500)) * self._LBS_MWH_TO_G_KWH
                pct_fossil = float(data.get("percent", 50))
                return LiveCarbonData.from_intensity(
                    intensity_g_kwh=moer,
                    zone=f"{lat:.2f},{lon:.2f}",
                    source="WattTime v2 (MOER)",
                    renewable_pct=100.0 - pct_fossil,
                    fossil_pct=pct_fossil,
                )
        except Exception as err:
            raise RuntimeError(f"WattTime failed ({lat},{lon}): {err}") from err
        finally:
            if close_session:
                await session.close()

    async def get_forecast(
        self,
        lat: float,
        lon: float,
        hours_ahead: int = 24,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> List[LiveCarbonData]:
        """24 h marginal carbon forecast for predictive scheduling."""
        close_session = session is None
        if close_session:
            session = aiohttp.ClientSession()
        try:
            await self._ensure_token(session)
            headers = {"Authorization": f"Bearer {self._token}"}
            params = {
                "latitude": str(lat),
                "longitude": str(lon),
                "horizon_hours": str(hours_ahead),
            }
            async with session.get(
                f"{self.BASE_URL}/forecast",
                headers=headers,
                params=params,
                timeout=aiohttp.ClientTimeout(total=12),
            ) as r:
                r.raise_for_status()
                rows = (await r.json()).get("forecast", [])
                return [
                    LiveCarbonData.from_intensity(
                        intensity_g_kwh=float(row["value"]) * self._LBS_MWH_TO_G_KWH,
                        zone=f"{lat},{lon}",
                        source="WattTime Forecast",
                        estimated=True,
                    )
                    for row in rows
                ]
        except Exception as err:
            raise RuntimeError(f"WattTime forecast failed: {err}") from err
        finally:
            if close_session:
                await session.close()


# ──────────────────────────────────────────────────────────────────────────────
# Provider 3: UK National Grid Carbon Intensity (free, no auth)
# ──────────────────────────────────────────────────────────────────────────────


class CarbonIntensityUKClient:
    """
    UK National Grid ESO Carbon Intensity API.
    Free, no API key required.  Covers Great Britain only.
    """

    BASE_URL = "https://api.carbonintensity.org.uk"
    _RENEWABLE_FUELS = {"biomass", "hydro", "solar", "wind"}

    async def get_national_intensity(
        self,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> LiveCarbonData:
        close_session = session is None
        if close_session:
            session = aiohttp.ClientSession()
        try:
            async with session.get(
                f"{self.BASE_URL}/intensity",
                timeout=aiohttp.ClientTimeout(total=8),
            ) as r:
                r.raise_for_status()
                data = (await r.json())["data"][0]
                intensity = float(
                    data["intensity"].get("actual")
                    or data["intensity"].get("forecast", 250)
                )
                mix = data.get("generationmix", [])
                renew = sum(
                    item["perc"]
                    for item in mix
                    if item.get("fuel", "").lower() in self._RENEWABLE_FUELS
                )
                fossil = sum(
                    item["perc"]
                    for item in mix
                    if item.get("fuel", "").lower() in {"gas", "coal", "oil"}
                )
                return LiveCarbonData.from_intensity(
                    intensity_g_kwh=intensity,
                    zone="GB",
                    source="UK National Grid",
                    renewable_pct=renew,
                    fossil_pct=fossil,
                )
        except Exception as err:
            raise RuntimeError(f"UK Carbon API failed: {err}") from err
        finally:
            if close_session:
                await session.close()


# ──────────────────────────────────────────────────────────────────────────────
# Fallback: Sinusoidal Simulation (offline / CI)
# ──────────────────────────────────────────────────────────────────────────────


class SimulationFallback:
    """
    Physics-inspired renewable simulation used when all APIs fail.
    Reproduces diurnal solar cycles + Weibull wind distribution.
    """

    def __init__(
        self,
        base_carbon_g_kwh: float = 250.0,
        solar_capacity: float = 0.3,
        wind_capacity: float = 0.4,
        lat: float = 0.0,
        seed: int = 42,
    ):
        self.base_carbon = base_carbon_g_kwh
        self.solar_cap = solar_capacity
        self.wind_cap = wind_capacity
        self.lat = lat
        import numpy as np

        self._rng = np.random.default_rng(seed)

    def get(self, zone: str) -> LiveCarbonData:
        import numpy as np

        hour = datetime.now(timezone.utc).hour
        # Diurnal solar factor (peak at noon UTC)
        solar = self.solar_cap * max(0, math.sin(math.pi * (hour - 6) / 12))
        # Weibull wind  (k=2, λ=0.9)
        wind = self.wind_cap * float(self._rng.weibull(2) * 0.9)
        wind = min(wind, self.wind_cap)
        renew = min(1.0, solar + wind)
        intensity = self.base_carbon * (1 - 0.7 * renew) + self._rng.normal(0, 10)
        intensity = max(10.0, intensity)
        return LiveCarbonData(
            carbon_intensity_g_kwh=intensity,
            renewable_percentage=renew * 100,
            fossil_percentage=(1 - renew) * 100,
            renewable_score=max(0.0, min(1.0, 1.0 - intensity / 800.0)),
            timestamp=datetime.now(timezone.utc),
            zone=zone,
            data_source="Simulation (fallback)",
            is_estimated=True,
            is_simulated=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Unified Carbon API Manager
# ──────────────────────────────────────────────────────────────────────────────


class CarbonAPIManager:
    """
    Production-grade multi-source carbon intensity aggregator.

    Priority order (best data quality first):
        1. Electricity Maps v3
        2. WattTime v2
        3. UK National Grid (GB nodes only)
        4. Simulation Fallback (always succeeds)

    Features:
        - 5-minute TTL cache (freshness guarantee for AI Act compliance)
        - Automatic failover with structured error logging
        - Async-first design with sync bridge (`get_sync`)
        - Full auditability via LiveCarbonData.data_source

    Configuration via environment variables:
        ELECTRICITY_MAPS_API_KEY
        WATTTIME_USERNAME / WATTTIME_PASSWORD
    """

    def __init__(
        self,
        electricity_maps_key: Optional[str] = None,
        watttime_username: Optional[str] = None,
        watttime_password: Optional[str] = None,
        cache_ttl_seconds: int = 300,
        simulation_configs: Optional[Dict[str, Dict]] = None,
    ):
        self._cache = _TTLCache(ttl_seconds=cache_ttl_seconds)
        self._em = ElectricityMapsClient(api_key=electricity_maps_key)
        self._wt = WattTimeClient(
            username=watttime_username, password=watttime_password
        )
        self._uk = CarbonIntensityUKClient()
        self._sim_configs = simulation_configs or {}
        self._api_stats: Dict[str, int] = {
            "electricity_maps": 0,
            "watttime": 0,
            "uk_grid": 0,
            "simulation": 0,
            "cache_hits": 0,
            "failures": 0,
        }

    async def get_current_intensity(
        self,
        zone: str,
        lat: float,
        lon: float,
        force_refresh: bool = False,
    ) -> LiveCarbonData:
        """
        Fetch live carbon intensity for a geographic location.

        Tries providers in priority order, caches result.

        Args:
            zone:          Grid zone code (e.g. "NO", "AU-VIC", "GB").
            lat, lon:      Geographic coordinates.
            force_refresh: Bypass cache for real-time accuracy.

        Returns:
            LiveCarbonData with renewable_score 0.0–1.0.
        """
        cache_key = f"{zone}|{lat:.3f}|{lon:.3f}"

        if not force_refresh:
            cached = self._cache.get(cache_key)
            if cached:
                self._api_stats["cache_hits"] += 1
                log.debug(f"[CarbonAPI] Cache HIT {cache_key} ({cached.data_source})")
                return cached

        async with aiohttp.ClientSession() as session:
            # 1. Electricity Maps
            try:
                data = await self._em.get_carbon_intensity(
                    zone=zone, lat=lat, lon=lon, session=session
                )
                self._api_stats["electricity_maps"] += 1
                self._cache.set(cache_key, data)
                log.info(
                    f"[CarbonAPI] EM → {zone}: {data.carbon_intensity_g_kwh:.1f}g/kWh (score={data.renewable_score:.2f})"
                )
                return data
            except Exception as e:
                log.warning(f"[CarbonAPI] ElectricityMaps failed: {e}")

            # 2. WattTime
            try:
                data = await self._wt.get_realtime_emissions(
                    lat=lat, lon=lon, session=session
                )
                self._api_stats["watttime"] += 1
                self._cache.set(cache_key, data)
                log.info(
                    f"[CarbonAPI] WT → {zone}: {data.carbon_intensity_g_kwh:.1f}g/kWh (score={data.renewable_score:.2f})"
                )
                return data
            except Exception as e:
                log.warning(f"[CarbonAPI] WattTime failed: {e}")

            # 3. UK Grid (only for GB-ish zones)
            if "GB" in zone or "UK" in zone:
                try:
                    data = await self._uk.get_national_intensity(session=session)
                    self._api_stats["uk_grid"] += 1
                    self._cache.set(cache_key, data)
                    log.info(
                        f"[CarbonAPI] UK → {zone}: {data.carbon_intensity_g_kwh:.1f}g/kWh (score={data.renewable_score:.2f})"
                    )
                    return data
                except Exception as e:
                    log.warning(f"[CarbonAPI] UK Grid failed: {e}")

        # 4. Simulation fallback (always succeeds)
        self._api_stats["simulation"] += 1
        sim_cfg = self._sim_configs.get(zone, {})
        sim = SimulationFallback(
            base_carbon_g_kwh=sim_cfg.get("base_carbon", 250.0),
            solar_capacity=sim_cfg.get("solar_capacity", 0.3),
            wind_capacity=sim_cfg.get("wind_capacity", 0.4),
            lat=lat,
        )
        data = sim.get(zone)
        self._cache.set(cache_key, data)
        log.warning(
            f"[CarbonAPI] SIMULATED → {zone}: {data.carbon_intensity_g_kwh:.1f}g/kWh"
        )
        return data

    def get_sync(
        self,
        zone: str,
        lat: float,
        lon: float,
        force_refresh: bool = False,
    ) -> LiveCarbonData:
        """
        Synchronous bridge for use in non-async training loops.
        Creates a new event loop if necessary.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        lambda: asyncio.run(
                            self.get_current_intensity(zone, lat, lon, force_refresh)
                        )
                    )
                    return future.result(timeout=15)
            else:
                return loop.run_until_complete(
                    self.get_current_intensity(zone, lat, lon, force_refresh)
                )
        except Exception:
            return asyncio.run(
                self.get_current_intensity(zone, lat, lon, force_refresh)
            )

    def api_health_report(self) -> Dict:
        """Return API call statistics for monitoring dashboards."""
        total = sum(self._api_stats.values())
        return {
            "total_calls": total,
            "provider_stats": self._api_stats,
            "cache_info": self._cache.info(),
            "dominant_source": max(
                (k for k in self._api_stats if k != "cache_hits"),
                key=lambda k: self._api_stats[k],
                default="simulation",
            ),
        }
