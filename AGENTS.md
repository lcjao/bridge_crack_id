# AGENTS.md - Bridge Crack Identification Project

## Project Overview

**Purpose**: Bridge health monitoring using vehicle-bridge coupled dynamics simulation to generate CPDV (Contact Point Displacement Variation) signals for crack position/depth prediction via neural networks.

**Core Concept**: CPDV = AP_damaged - AP_intact (damage state minus healthy state displacement)

---

## Build & Test Commands

### Dependencies
```bash
pip install -r requirements.txt
```

### Pipeline Execution
```bash
# Step 1: Generate training data
python scripts/01_generate_data.py --n_samples 10000 --output outputs/data/training_data.npz

# Step 2a: Train BP neural network
python scripts/02_train_model.py --data outputs/data/training_data.npz --model outputs/models/model.json

# Step 2b: Train LSTM model (PyTorch)
python scripts/02_train_lstm_model.py --data outputs/data/training_data.npz --model outputs/models/lstm_model.pth

# Step 3a: Run BP inference
python scripts/03_run_inference.py --model outputs/models/model.json --input outputs/data/verify_data.npz --plot

# Step 3b: Run LSTM inference
python scripts/03_run_lstm_inference.py --model outputs/models/lstm_model.pth --input outputs/data/verify_data.npz --plot

python scripts/04_cpdv_analysis.py --depths 0.05 0.25 0.3 --distances 5 15 25 --prefix extreme
python scripts/04_cpdv_analysis.py --depths 0.1 0.2 0.3 --distances 5 15 25 --prefix set1v2
```

### Run Single Test (Verify Module Imports)
```bash
python -c "from simulation.enhanced_system import BridgeVehicleSystem; print('OK')"
python -c "from data_pipeline.generator import DataGenerator, DataPipeline; print('OK')"
python -c "from model.nn import NeuralNetwork; print('OK')"
python -c "from model.lstm import LSTMRegressor; print('OK')"
```

---

## Code Style Guidelines

### Import Conventions
```python
# Standard library
import os, sys, json, argparse

# Third-party (alphabetical)
import numpy as np
import yaml
from scipy import linalg

# Local modules
from simulation.enhanced_system import BridgeVehicleSystem
from data_pipeline.generator import DataPipeline
```

### Formatting
- **Line length**: 100 characters max
- **Indentation**: 4 spaces
- **Blank lines**: 2 between top-level definitions, 1 between functions

### Naming Conventions
| Element | Convention | Example |
|---------|------------|---------|
| Classes | PascalCase | `BridgeVehicleSystem` |
| Functions/Methods | snake_case | `calculate_cpdv` |
| Constants | UPPER_SNAKE_CASE | `MAX_ITERATIONS` |
| Variables | snake_case | `crack_position` |
| Private Methods | `_snake_case` | `_validate_params` |

### Types & Type Hints
- Use Python type hints for function signatures
- Common: `Dict`, `List`, `Tuple`, `Optional`, `np.ndarray`
- Example: `def calculate(self, params: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:`

### Documentation
- **Class docstring**: Chinese description + Args/Returns sections
- **Function docstring**: Chinese description, English for parameters and return values

### Error Handling
- Validate inputs at entry points
- Raise descriptive Chinese error messages
- Check for NaN/Inf in simulation results

```python
if self.uc_healthy is None:
    raise ValueError("请先运行健康状态分析")
```

### Numerical Computing Guidelines
- Use `np.nan_to_num()`, `np.clip()` for numerical stability
- Add regularization: `K + 1e-6 * np.eye(n)`
- Clip activation inputs: `np.clip(x, -500, 500)`

---

## Project Structure
```
bridge_crack_id/
├── simulation/           # Vehicle-bridge coupling core
│   └── enhanced_system.py  # BridgeVehicleSystem, EnhancedBridgeVehicleSystem
├── data_pipeline/        # Data generation
│   └── generator.py       # DataGenerator, DataPipeline, DataProcessor
├── model/                # Neural networks
│   ├── nn.py             # NeuralNetwork (from-scratch BP)
│   └── lstm.py           # LSTMRegressor (PyTorch)
├── visualization/        # Plotting
│   └── plots.py
├── scripts/              # Execution pipelines
│   ├── 01_generate_data.py
│   ├── 02_train_model.py
│   ├── 02_train_lstm_model.py
│   ├── 03_run_inference.py
│   └── 03_run_lstm_inference.py
├── config/               # Parameter configurations
│   └── params.yaml       # Main config (simulation, data, model sections)
└── requirements.txt
```

---

## Key Dependencies
- `numpy >= 1.21.0` - Core scientific computing
- `scipy >= 1.7.0` - Linear algebra
- `matplotlib >= 3.4.0` - Visualization
- `pyyaml >= 5.4.0` - Configuration
- `torch >= 2.0.0` - LSTM model (optional)

---

## Parameter Configuration

### YAML Structure (config/params.yaml)
```yaml
simulation:    # Passed to BridgeVehicleSystem (flattened, no "simulation" prefix)
  mv: 5000     # Vehicle mass (kg)
  kv: 100000   # Suspension stiffness (N/m)
  cv: 5000     # Damping (N/(m/s))
  V: 10        # Vehicle speed (m/s)
  L: 30        # Bridge length (m)
  E: 3.0e10    # Elastic modulus (Pa)
  I: 0.1       # Moment of inertia (m^4)
  m: 400       # Unit length mass (kg/m)
  EL: 20       # Element count
  depth: 0.8   # Beam depth (m)
  width: 0.25  # Beam width (m)
  n_modes: 3   # Modal count
  kexi: 0.1    # Damping ratio
  deltat: 0.005  # Time step (s)
  gamma: 0.5   # Newmark-beta gamma
  beta: 0.25   # Newmark-beta beta
  road_type: 'b'

data:          # Data generation parameters
model:         # Neural network parameters
```

### Parameter Flow
1. YAML loads with nested structure (`simulation.mv`)
2. Script extracts: `sim_params = config.get("simulation", {})` 
3. Passes flattened dict: `{"mv": 5000, ...}` to BridgeVehicleSystem

---

## Success Criteria
- Training samples >= 1000
- Position MAE < 1.0m, Depth MAE < 5%
- Deterministic training (fixed random seeds)

---

## Notes for Agents
1. **Scientific simulation project** - Not a web application
2. **No formal test framework** - Verify via module imports
3. **Method**: Newmark-β method for dynamic analysis
4. **CPDV**: Contact Point Displacement Variation = AP_damaged - AP_intact
5. **Language**: Chinese comments for domain terminology, code in English
6. **Algorithm**: 6-step iterative algorithm for vehicle-bridge coupling (see system.md)
