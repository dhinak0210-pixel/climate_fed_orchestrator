# Technical Documentation
## Climate-Fed Orchestrator v2.0 - API Reference & Guide

---

### 1. Installation

#### Quick Start (Simulation Mode)
```bash
# Clone and setup
git clone https://github.com/user/climate_fed_orchestrator.git
cd climate_fed_orchestrator
pip install torch torchvision opacus aiohttp matplotlib pyyaml
python main.py
```

#### Live Carbon APIs (Production Mode)
```bash
export ELECTRICITY_MAPS_API_KEY="your_key"
export WATTTIME_USER="user"
export WATTTIME_PASS="pass"
python main.py --use-apis true
```

---

### 2. API Reference

#### Core Components (`core`)

##### `RenewableOracle`
Queries carbon APIs to assign renewable scores.

| Method | Signature | Returns | Complexity |
|--------|-----------|---------|------------|
| `get_node_availability` | `(node_id: int, round: int)` | `Tuple[bool, float, CarbonIntensity]` | O(1) |
| `calculate_score` | `(intensity: float, capacity: float)` | `float` | O(1) |

##### `CarbonAwareNode`
Represents a distributed client running local training.

| Method | Signature | Returns | Complexity |
|--------|-----------|---------|------------|
| `local_train` | `(model: nn.Module, score: float)` | `Tuple[Dict, float, int, float]` | O(N×E) |
| `get_gradient_update` | `()` | `Dict[str, Tensor]` | O(P) |

##### `CarbonAwareAggregator`
Securely averages updates weighted by carbon score.

| Method | Signature | Returns | Complexity |
|--------|-----------|---------|------------|
| `aggregate` | `(updates: List[Dict], scores: List[float], counts: List[int])` | `Dict[str, Tensor]` | O(M×P) |

##### `PrivacyEngine` (Opacus Wrapper)
Manages differential privacy budget.

| Method | Signature | Returns | Complexity |
|--------|-----------|---------|------------|
| `get_noise_multiplier` | `(epsilon: float, delta: float, ...)` | `float` | O(1) |
| `check_budget` | `(steps: int)` | `bool` | O(1) |

##### `CarbonLedger` (`utils`)
Tracks emissions for ESG reporting.

| Method | Signature | Returns | Complexity |
|--------|-----------|---------|------------|
| `generate_impact_report` | `(acc: float, rounds: int)` | `CarbonReport` | O(E) |

---

### 3. Configuration Guide (`config/`)

#### `nodes.yaml` (Client Definitions)
```yaml
nodes:
  - id: 0
    name: "Oslo"
    lat: 59.9
    lon: 10.7
    tz_offset: 1
    solar_cap: 0.12
    wind_cap: 0.68
    base_carbon: 120  # g/kWh
```

#### `training.yaml` (Hyperparameters)
```yaml
training:
  rounds: 10
  local_epochs: 1
  batch_size: 64
  target_epsilon: 1.0  # Privacy budget
  target_delta: 1.0e-5
  carbon_threshold: 0.6  # Train only if score > 0.6
  l2_norm_clip: 1.0  # DP Clipping
```

---

### 4. Troubleshooting Matrix

| Symptom | Cause | Solution |
|---------|-------|----------|
| **"Privacy budget exceeded"** | ε > 1.0 after N rounds | Reduce `rounds` (try 8) or increase `target_epsilon` (try 1.2). |
| **"API timeout"** | 5s no response (Electricity Maps) | Check internet connection; run with `--use-apis false` (simulation). |
| **"Accuracy < 80%"** | Non-IID data skew | Increase `local_epochs` to 2-3 to allow more local convergence. |
| **"All nodes skipping"** | `carbon_threshold` too high | Lower to 0.4. Or implement fallback logic in `scheduler.py`. |
| **"Out of memory"** | Batch size too large | Reduce `batch_size` to 32. Use `gradient_checkpointing`. |

---

### 5. Output Directory Structure

- `results/metrics.json`: Raw reproducible data.
- `results/figure_*.png`: Generated visualizations.
- `results/logs/`: Detailed execution logs.
- `checkpoints/`: Saved model weights (`.pt`).
