"""
api_setup_test.py — Live Carbon API Setup & Connectivity Tester
════════════════════════════════════════════════════════════════════
Run this script to:
  1. Test which APIs are reachable from your environment
  2. Validate your API keys before running the full experiment
  3. See real-time carbon intensity for all 3 node locations

Usage:
  python3 -m climate_fed_orchestrator.api_setup_test

  With real keys:
  ELECTRICITY_MAPS_API_KEY=your_key python3 -m climate_fed_orchestrator.api_setup_test
  WATTTIME_USERNAME=user WATTTIME_PASSWORD=pass python3 -m climate_fed_orchestrator.api_setup_test
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))

import aiohttp


# ── Terminal Colours ──────────────────────────────────────────────────────────
G = "\033[32m"  # green
R = "\033[31m"  # red
Y = "\033[33m"  # yellow
B = "\033[36m"  # cyan
W = "\033[0m"  # reset
BLD = "\033[1m"


def ok(msg):
    print(f"  {G}✅ {msg}{W}")


def fail(msg):
    print(f"  {R}❌ {msg}{W}")


def warn(msg):
    print(f"  {Y}⚠️  {msg}{W}")


def info(msg):
    print(f"  {B}ℹ  {msg}{W}")


NODE_LOCATIONS = [
    {"name": "Oslo, Norway", "lat": 59.9, "lon": 10.7, "zone": "NO"},
    {"name": "Melbourne, Australia", "lat": -37.8, "lon": 144.9, "zone": "AU-VIC"},
    {"name": "San José, Costa Rica", "lat": 9.7, "lon": -84.0, "zone": "CR"},
]


# ──────────────────────────────────────────────────────────────────────────────
# 1. UK National Grid (FREE, no auth, GB only)
# ──────────────────────────────────────────────────────────────────────────────


async def test_uk_grid(session: aiohttp.ClientSession) -> bool:
    print(f"\n{BLD}── Test 1: UK National Grid Carbon Intensity API (FREE){W}")
    info("URL: https://api.carbonintensity.org.uk/intensity")
    info("No API key needed. Coverage: Great Britain (GB) only.")
    try:
        t = time.monotonic()
        async with session.get(
            "https://api.carbonintensity.org.uk/intensity",
            timeout=aiohttp.ClientTimeout(total=8),
        ) as r:
            latency = int((time.monotonic() - t) * 1000)
            if r.status != 200:
                fail(f"HTTP {r.status}")
                return False
            data = (await r.json())["data"][0]
            intensity = data["intensity"]
            actual = intensity.get("actual") or intensity.get("forecast", "?")
            index = intensity.get("index", "?")
            mix = data.get("generationmix", [])
            renewables = ["biomass", "hydro", "solar", "wind"]
            renew_pct = sum(m["perc"] for m in mix if m.get("fuel", "") in renewables)

            ok(f"Connected! Latency: {latency}ms")
            print(
                f"  {'':4}Current GB Carbon Intensity:  {BLD}{actual} g CO₂/kWh{W}  [{index}]"
            )
            print(f"  {'':4}Renewable Mix (GB right now):  {BLD}{renew_pct:.1f}%{W}")
            print(f"\n  {'':4}Generation Breakdown:")
            for src in sorted(mix, key=lambda x: -x.get("perc", 0))[:6]:
                bar = "█" * int(src["perc"] / 2)
                print(f"  {'':6}{src['fuel']:12s} {src['perc']:5.1f}%  {G}{bar}{W}")
            return True
    except asyncio.TimeoutError:
        fail("Timeout — no internet access to external APIs in this environment.")
        return False
    except Exception as e:
        fail(f"Error: {e}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# 2. Electricity Maps v3
# ──────────────────────────────────────────────────────────────────────────────


async def test_electricity_maps(session: aiohttp.ClientSession) -> bool:
    key = os.environ.get("ELECTRICITY_MAPS_API_KEY", "")
    print(f"\n{BLD}── Test 2: Electricity Maps API v3{W}")
    info("URL: https://api.electricitymap.org/v3")
    info("Coverage: 100+ countries / zones globally.")

    if not key or key == "your_electricity_maps_key_here":
        warn("ELECTRICITY_MAPS_API_KEY not set.")
        print(
            f"""
  {BLD}How to get a FREE key:{W}
  1. Visit:   {B}https://app.electricitymaps.com/map{W}
  2. Click:   "API" → "Get API Access"
  3. Choose:  Free tier (commercial map-based access)
     OR use:  {B}https://co2signal.com{W} for a free hobbyist token
  4. Set env: {Y}export ELECTRICITY_MAPS_API_KEY=your_key_here{W}
  5. Rerun:   python3 -m climate_fed_orchestrator.api_setup_test
        """
        )
        return False

    try:
        t = time.monotonic()
        headers = {"auth-token": key}
        async with session.get(
            "https://api.electricitymap.org/v3/carbon-intensity/latest",
            params={"zone": "NO"},
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=8),
        ) as r:
            latency = int((time.monotonic() - t) * 1000)
            if r.status == 401:
                fail("API key invalid or expired. Check your key.")
                return False
            if r.status == 403:
                fail("Free tier does not cover this zone. Try zone=US-CAL-CISO or DE.")
                return False
            if r.status != 200:
                fail(f"HTTP {r.status}: {await r.text()}")
                return False
            data = await r.json()
            ok(f"Connected! Key valid. Latency: {latency}ms")
            print(
                f"  {'':4}Norway (NO) Carbon Intensity: {BLD}{data.get('carbonIntensity','?')} g CO₂/kWh{W}"
            )
            return True
    except asyncio.TimeoutError:
        fail("Timeout — no internet access in this environment.")
        return False
    except Exception as e:
        fail(f"Error: {e}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# 3. WattTime v2
# ──────────────────────────────────────────────────────────────────────────────


async def test_watttime(session: aiohttp.ClientSession) -> bool:
    username = os.environ.get("WATTTIME_USERNAME", "")
    password = os.environ.get("WATTTIME_PASSWORD", "")
    print(f"\n{BLD}── Test 3: WattTime API v2 (MOER — Marginal Carbon){W}")
    info("URL: https://api2.watttime.org/v2")
    info("Coverage: USA + select global. Free public tier available.")

    if not username or username == "your_watttime_username":
        warn("WATTTIME_USERNAME / WATTTIME_PASSWORD not set.")
        print(
            f"""
  {BLD}How to get a FREE WattTime account:{W}
  1. Register: {B}https://www.watttime.org/api-documentation/#register-new-user{W}
     (Or use the curl command below)

  {BLD}Quick Registration via curl:{W}
  {Y}curl -X POST "https://api2.watttime.org/v2/register" \\
       -H "Content-Type: application/json" \\
       -d '{{"username":"YOUR_USERNAME","password":"YOUR_PASSWORD",
            "email":"YOUR_EMAIL","org":"climate-fed-research"}}'
  {W}
  2. Set envs: {Y}export WATTTIME_USERNAME=your_username{W}
               {Y}export WATTTIME_PASSWORD=your_password{W}
  3. Rerun:    python3 -m climate_fed_orchestrator.api_setup_test
        """
        )
        return False

    try:
        # Authenticate
        t = time.monotonic()
        async with session.post(
            "https://api2.watttime.org/v2/login",
            auth=aiohttp.BasicAuth(username, password),
            timeout=aiohttp.ClientTimeout(total=8),
        ) as r:
            if r.status != 200:
                fail(
                    f"Authentication failed (HTTP {r.status}). Check username/password."
                )
                return False
            token = (await r.json())["token"]

        # Get real-time MOER
        headers = {"Authorization": f"Bearer {token}"}
        async with session.get(
            "https://api2.watttime.org/v2/index",
            headers=headers,
            params={"latitude": "59.9", "longitude": "10.7"},  # Oslo
            timeout=aiohttp.ClientTimeout(total=8),
        ) as r:
            latency = int((time.monotonic() - t) * 1000)
            if r.status != 200:
                fail(f"MOER fetch failed (HTTP {r.status})")
                return False
            data = await r.json()
            moer_g = data.get("moer", 0) * 0.453592  # lbs/MWh → g/kWh
            ok(f"Connected! Key valid. Latency: {latency}ms")
            print(f"  {'':4}Oslo (59.9°N,10.7°E) MOER: {BLD}{moer_g:.1f} g CO₂/kWh{W}")
            print(f"  {'':4}Grid %:  {data.get('percent', '?')}%")
            return True
    except asyncio.TimeoutError:
        fail("Timeout — no internet access in this environment.")
        return False
    except Exception as e:
        fail(f"Error: {e}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# 4. Simulation Fallback
# ──────────────────────────────────────────────────────────────────────────────


def test_simulation() -> bool:
    print(f"\n{BLD}── Test 4: Simulation Fallback (always available, no key){W}")
    from climate_fed_orchestrator.core.live_carbon_api import SimulationFallback

    try:
        for loc in NODE_LOCATIONS:
            sim = SimulationFallback(
                base_carbon_g_kwh=200.0,
                solar_capacity=0.35,
                wind_capacity=0.45,
                lat=loc["lat"],
            )
            d = sim.get(loc["zone"])
            ok(
                f"{loc['name']:25s} → {d.carbon_intensity_g_kwh:6.1f} g CO₂/kWh | "
                f"Renewable Score: {d.renewable_score:.3f} | Source: {d.data_source}"
            )
        return True
    except Exception as e:
        fail(f"Simulation error: {e}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# 5. Setup Instructions
# ──────────────────────────────────────────────────────────────────────────────


def print_setup_guide(uk_ok: bool, em_ok: bool, wt_ok: bool):
    print(f"\n{'═'*68}")
    print(f"{BLD}  📋 SETUP GUIDE & NEXT STEPS{W}")
    print(f"{'═'*68}\n")

    avail = [
        ("UK National Grid (free)", uk_ok, "GB / Europe only", "No key needed"),
        ("Electricity Maps v3", em_ok, "100+ countries", "ELECTRICITY_MAPS_API_KEY"),
        ("WattTime v2 (MOER)", wt_ok, "USA + select", "WATTTIME_USERNAME/PASSWORD"),
        ("Simulation Fallback", True, "Worldwide", "No key needed"),
    ]

    print(f"  {'Provider':<28} {'Status':<12} {'Coverage':<20} {'Credential'}")
    print(f"  {'-'*28} {'-'*12} {'-'*20} {'-'*24}")
    for name, status, cov, cred in avail:
        sym = f"{G}✅ Live{W}" if status else f"{Y}⚠️  Offline{W}"
        print(f"  {name:<28} {sym:<20} {cov:<20} {cred}")

    print(f"\n{BLD}  Step 1: Copy the env template{W}")
    print(f"  {Y}cp climate_fed_orchestrator/.env.dp.example .env.dp{W}")

    print(f"\n{BLD}  Step 2: Fill in your keys{W}")
    print(f"  {Y}nano .env.dp{W}")

    print(f"\n{BLD}  Step 3: Export the variables{W}")
    print(f"  {Y}set -a && source .env.dp && set +a{W}")

    print(f"\n{BLD}  Step 4: Run the DP Orchestrator with live carbon data{W}")
    print(f"  {Y}python3 -m climate_fed_orchestrator.dp_main \\{W}")
    print(f"  {Y}    --mode dp_oracle --rounds 20 --epsilon 1.0 --live-carbon{W}")

    print(f"\n{BLD}  Step 5: Run without any API key (simulation mode){W}")
    print(f"  {Y}python3 -m climate_fed_orchestrator.dp_main \\{W}")
    print(f"  {Y}    --mode full --rounds 20 --epsilon 1.0{W}")

    print(f"\n  {'─'*64}")
    total_live = sum([uk_ok, em_ok, wt_ok])
    if total_live == 0:
        warn("No live APIs reachable. Running in simulation mode (100% functional).")
        info(
            "Simulation produces scientifically valid diurnal solar + Weibull wind curves."
        )
    elif total_live == 1:
        ok(f"1 live API available — partial live data enabled.")
    else:
        ok(f"{total_live}/3 live APIs available — full real-time data enabled!")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


async def main():
    print(f"\n{BLD}{'═'*68}{W}")
    print(f"{BLD}  🌍🔒  CLIMATE-FED API CONNECTIVITY TESTER{W}")
    print(f"{BLD}{'═'*68}{W}\n")
    print(f"  Testing all carbon intensity providers...\n")

    async with aiohttp.ClientSession() as session:
        uk_ok = await test_uk_grid(session)
        em_ok = await test_electricity_maps(session)
        wt_ok = await test_watttime(session)

    sim_ok = test_simulation()
    print_setup_guide(uk_ok, em_ok, wt_ok)


if __name__ == "__main__":
    asyncio.run(main())
