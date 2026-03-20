# Plan Generated: lstm_cpdv_plan (Metis-Updated)

This plan reflects the Metis review feedback and sets explicit conventions for Scheme B: two targets (position, depth) with a windowed LSTM model training pipeline integrated with the existing data pipeline.

## Key Changes (Metis)
- Target semantics: horizon mapping defined; ground-truth [position, depth] corresponds to the LAST timestep of each window.
- Windowing: seq_len = 64, stride = 32; reset hidden state at episode boundaries.
- Coordinate semantics: position and depth defined in meters (m); both stored as-is in data pipeline.
- Normalization: inputs normalized with feature-wise mean/std; outputs stored raw.
- Architecture: LSTMRegressor, 2-layer LSTM, hidden_size=128, dropout=0.2, batch_first=True; single Linear head to 2 outputs.
- Loss: MSELoss on [position, depth]; optional per-dimension weighting if scales differ.
- Training: 70/15/15 split by episodes; seed=42; Adam lr=0.001, weight_decay=1e-5; ReduceLROnPlateau; max_norm=1.0 gradient clipping; early stopping patience=5; metrics RMSE per dimension, MAE, joint MSE.
- Edges: skip windows with NaN; padding/masking future extension; device-agnostic code.
- Reproducibility: pinned PyTorch version; seeds logged in outputs.
- Commit Strategy: simplified to 5 commits as described below.

## Context
### Original Request
- Implement an LSTM model in the model folder for training on CPDV data to predict crack location and depth; integrate with data_pipeline/generator.py; follow existing training/inference script patterns.
- Horizon mapping and windowing are explicitly defined per Metis.

### Commit Plan (4–5 commits)
- Commit 1: model/lstm.py + __init__.py
- Commit 2: data_pipeline/generator.py: SequenceWindowDataset and windowing utilities
- Commit 3: scripts/02_train_lstm_model.py
- Commit 4: scripts/03_run_lstm_inference.py
- Commit 5: config/lstm_params.yaml + pinned requirements + tests

## Work Objectives
### Core Objective
- Build an LSTM-based regressor with 2 outputs: [position, depth].

### Deliverables
- model/lstm.py
- data_pipeline/generator.py: SequenceWindowDataset, to_sequences, horizon mapping logic
- scripts/02_train_lstm_model.py
- scripts/03_run_lstm_inference.py
- config/lstm_params.yaml
- requirements.txt (PyTorch version pinned)
- Tests: windowing shapes, forward pass, NaN skip behavior, end-to-end sanity on synthetic data

### Plan for Implementation (High-level)
- Implement horizon-aware windowing: windows of length 64, labels at end of window.
- Normalize inputs; outputs raw.
- Build LSTM with 2 layers, 128 hidden units, 0.2 dropout; 2-dim output head.
- Training with MSE loss on 2 outputs; optional weighting if required by observed scale differences.
- 70/15/15 split by episodes; seeds; LR scheduler; gradient clipping; early stopping.
- Tests covering: windowing, forward pass, missing data skip, device placement.

## Verification & Acceptance
- Forward pass shape: (batch, 2)
- Loss computed on [position, depth]
- Windows without NaN processed; NaN causes skip
- Reproducibility: seeds logged; PyTorch pinned

## Commit Summary (Atomic)
- Commit 1: model/lstm.py + __init__.py
- Commit 2: data_pipeline/generator.py: SequenceWindowDataset + window utilities
- Commit 3: scripts/02_train_lstm_model.py
- Commit 4: scripts/03_run_lstm_inference.py
- Commit 5: config/lstm_params.yaml + requirements.txt + tests
