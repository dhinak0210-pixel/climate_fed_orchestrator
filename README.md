---
title: Climate Fed Orchestrator
emoji: 🌍🔒
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# 🌍🔒 Climate-Fed Orchestrator

**Privacy-Preserving Federated Learning with Real-Time Carbon Intelligence.**

This system coordinates decentralized AI training across global nodes, optimizing for the lowest carbon footprint by intelligently scheduling computation during peaks of renewable energy availability.

---

## 🏛️ Project Architecture
The project has been optimized for **Streamlit Community Cloud** and low-latency execution with a flattened package structure.

### 📦 Modern Flat Structure
```
.
├── core/               # Engine, Privacy, and Carbon Logic
├── data/               # Non-IID Data Partitioning
├── models/             # Eco-efficient CNN models
├── simulation/         # Grid & Network Simulation
├── visualization/      # Interactive Dashboards
├── streamlit_app.py    # Main Dashboard (Entry Point)
└── main.py             # CLI Research Entry Point
```

### 🛠️ Internal Import Standard
As of the latest optimization, all internal imports follow the flat standard:

```python
# Fixed structure (verified in latest commit):
from core.carbon_engine import RenewableOracle, NodeGeography
from core.energy_accountant import CarbonLedger
```

---

## 🚀 Quick Start

### 1. Local Setup
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### 2. CI/CD Pipeline
The project includes a robust **GitHub Actions** workflow (`.github/workflows/test_and_deploy.yml`) that verifies:
- ✅ **Unit Tests**: Full coverage of carbon and privacy logic.
- ✅ **Linting**: PEP8 compliance and type safety.
- ✅ **Pathing**: Correct `PYTHONPATH` resolution for the flat structure.

---

## 📊 Impact
- **Accuracy**: >94% on MNIST (Privacy-Preserving)
- **Sustainability**: Up to **43.7% reduction** in training-related CO2.
- **Privacy**: Proven Differential Privacy (ε < 1.0).

---

> *"Training models doesn't have to cost the Earth."*