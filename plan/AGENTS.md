# AGENTS.md - Bridge Crack Identification Project

## Project Overview

桥梁裂缝识别系统（BCIS）- 利用车-桥耦合动力学仿真生成CPDV信号，使用神经网络预测裂缝位置和深度。

**Project Path**: `D:\python\pythonProject\数据分析\书\结构监测\Bridge-health-monitoring\车桥耦合\for CPDV\bridge_crack_id`

**核心概念**: CPDV (Contact Point Displacement Variation) = 健康-损伤状态位移差

## Build & Test Commands

### Dependencies
```bash
pip install -r requirements.txt
```

### Run Scripts (Pipeline)
```bash
# Step 1: Generate training data
python scripts/01_generate_data.py --n_samples 10000 --output outputs/data/training_data.npz

# Step 2a: Train BP model
python scripts/02_train_model.py --data outputs/data/training_data.npz --model outputs/models/model.json

# Step 2b: Train LSTM model (PyTorch)
python scripts/02_train_lstm_model.py --data outputs/data/training_data.npz --model outputs/models/lstm_model.pth

# Step 3a: Run BP inference
python scripts/03_run_inference.py --model outputs/models/model.json --input outputs/data/verify_data.npz

# Step 3b: Run LSTM inference
python scripts/03_run_lstm_inference.py --model outputs/models/lstm_model.pth --input outputs/data/verify_data.npz
```

### Single Test
```bash
python -c "from simulation.enhanced_system import BridgeVehicleSystem; print('OK')"
python -c "from data_pipeline.generator import DataGenerator; print('OK')"
python -c "from model.nn import NeuralNetwork; print('OK')"
python -c "from model.lstm import LSTMRegressor; print('OK')"
```

## Architecture

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ 参数空间 &   │──▶│  仿真引擎     │──▶│  数据预处理   │──▶│  神经网络    │
│  场景生成器  │   │BridgeVehicle │   │ (归一化/划分) │   │ (训练/推理)  │
└──────────────┘   │   System)    │   └──────────────┘   └──────┬───────┘
                  └──────────────┘                                 │
                              ▲                                    │
                              │ (position, depth)                 │
                    ┌─────────┴─────────┐                          │
                    │   评估 & 可视化   │◀─────────────────────────┘
                    └──────────────────┘
```

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
from data_pipeline.generator import DataGenerator
from model.nn import NeuralNetwork
```

### Formatting
- Line length: 100 chars max, Indentation: 4 spaces
- Blank lines: 2 between top-level, 1 between functions

### Naming Conventions
| Element | Convention | Example |
|---------|------------|---------|
| Classes | PascalCase | `BridgeVehicleSystem` |
| Functions/methods | snake_case | `calculate_cpdv` |
| Constants | UPPER_SNAKE_CASE | `MAX_ITERATIONS` |
| Variables | snake_case | `crack_position` |

### Documentation
- Class docstring: 中文描述 + Args/Returns
- Function docstring: 中文描述，参数和返回值用英文

### Error Handling
- Validate inputs at entry, raise descriptive Chinese errors
- Check NaN/Inf in simulation results

```python
if self.uc_healthy is None:
    raise ValueError("请先运行健康状态分析")
```

### Numerical Computing
- Use `np.nan_to_num()`, `np.clip()` for stability
- Add regularization: `K + 1e-6 * np.eye(n)`

## Project Structure
```
bridge_crack_id/
├── simulation/           # 车-桥耦合核心
│   ├── system.py        # BridgeVehicleSystem类
│   └── enhanced_system.py
├── data_pipeline/       # 数据生成
│   └── generator.py    # DataGenerator, SequenceWindowDataset
├── model/              # 神经网络
│   ├── nn.py           # NeuralNetwork (从头实现BP)
│   └── lstm.py         # LSTMRegressor (PyTorch)
├── visualization/       # 可视化
│   └── plots.py
├── scripts/            # 执行脚本
│   ├── 01_generate_data.py
│   ├── 02_train_model.py
│   ├── 02_train_lstm_model.py
│   ├── 03_run_inference.py
│   └── 03_run_lstm_inference.py
├── config/             # 参数配置
│   ├── params.yaml
│   └── lstm_params.yaml
└── requirements.txt
```

## Key Dependencies
- numpy >= 1.21.0, scipy >= 1.7.0, matplotlib >= 3.4.0, pyyaml >= 5.4.0
- torch >= 2.0.0 (LSTM模型)

## Success Criteria
- 训练集 >= 1000 样本
- 位置 MAE < 1.0米, 深度 MAE < 5%
- 训练过程确定性（固定随机种子）

## Output Format Specification (Unified)

All training and inference scripts use a unified output format for consistency.

### Training Output Format
```
==================================================
[模型名称] 模型训练 - 评估结果
==================================================
最佳验证损失: X.XXXXXX
--------------------------------------------------
评估指标
--------------------------------------------------
位置 MAE:  X.XXXX m
位置 RMSE: X.XXXX m
深度 MAE:  X.XXXX
深度 RMSE: X.XXXX
R2 Score: X.XXXX
--------------------------------------------------
成功标准对比
--------------------------------------------------
  位置 MAE < 1.0m: [PASS/FAIL]
  深度 MAE < 0.05: [PASS/FAIL]
==================================================
```

### Inference Output Format
```
==================================================
[模型名称] 模型推理 - 评估结果
==================================================
推理样本数: XXX
--------------------------------------------------
评估指标
--------------------------------------------------
位置 MAE:  X.XXXX m
位置 RMSE: X.XXXX m
深度 MAE:  X.XXXX
深度 RMSE: X.XXXX
R2 Score: X.XXXX
--------------------------------------------------
成功标准对比
--------------------------------------------------
  位置 MAE < 1.0m: [PASS/FAIL]
  深度 MAE < 0.05: [PASS/FAIL]
==================================================

--------------------------------------------------
预测示例
--------------------------------------------------
  样本1: 位置=X.XXXXm, 深度=X.XXXX
  样本2: 位置=X.XXXXm, 深度=X.XXXX
  ...
==================================================
```

### Supported Scripts
| Script | Model Type | Supports --plot |
|--------|------------|-----------------|
| `02_train_model.py` | Neural Network (BP) | ✅ Yes |
| `02_train_lstm_model.py` | LSTM (PyTorch) | ✅ Yes |
| `03_run_inference.py` | Neural Network (BP) | ✅ Yes |
| `03_run_lstm_inference.py` | LSTM (PyTorch) | ✅ Yes |

### Notes for Agents
1. 科学仿真项目，非Web应用
2. 无正式测试框架，通过import验证
3. 使用Newmark-β法进行动态分析
4. CPDV = 健康-损伤状态位移差
5. 中文注释适用于领域术语
