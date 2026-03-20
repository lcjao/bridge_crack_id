# Draft: LSTM CPDV Training Plan

## Requirements (confirmed)
- Implement an LSTM-based model in the model/ directory to train on CPDV data and predict two outputs: position and depth_ratio (per sample).
- Integrate the LSTM model with the existing data pipeline in data_pipeline/generator.py so that sequence data can be produced and consumed by the trainer.
- Provide a training script that follows the pattern of scripts/02_train_model.py but tailored for an LSTM model (e.g., scripts/02_train_lstm_model.py).
- Provide an inference script that mirrors the style of scripts/03_run_inference.py for LSTM (e.g., scripts/03_run_lstm_inference.py).
- Use PyTorch as the core deep learning framework for the LSTM implementation, with a clear path to TensorFlow as a future option if needed.
- Handle sequence data properly with windowing (seq_len, stride) to create (X_seq, y_seq) samples for training.
- Maintain code style: PascalCase for classes, snake_case for functions, and Chinese domain terms in comments.

- data flow: CPDV features X shape (n_features, n_samples); targets y shape (2, n_samples) initially with [position, depth], transformed to [position, depth_ratio] for training. The LSTM will use inputs of shape (batch, seq_len, input_dim) and outputs (batch, 2).
- Training targets: [position, depth_ratio]. Depth_ratio is computed as depth / max_depth in the training data; max_depth will be stored to be used during inference.

- Evaluation: use MSE loss for both outputs; report RMSE for each dimension and overall RMSE.

## Technical Decisions
- Framework: PyTorch (default). TensorFlow alternative noted for future work.
- Model: LSTM-based regressor for 2D outputs (position, depth_ratio).
- Data handling: Implement a SequenceDataset that builds (seq_len) windows from the raw CPDV features and yields (X_seq, y_seq).
- Normalization: Use the existing data_pipeline normalization (fit on training data) and store stats for inference.
- Output dimension: 2 (position, depth_ratio).
- Loss: MSELoss with potentially per-dimension weighting if needed in the future.
- Optimizer: Adam with learning_rate from config/params.yaml.
- Checkpointing: save model state_dict and training stats to outputs/models and outputs/stats, respectively.

## Filenames / Code Structure Changes
- Add: model/lstm.py
- Extend: data_pipeline/generator.py to support sequence windowing and optional depth_ratio output.
- Add: scripts/02_train_lstm_model.py
- Add: scripts/03_run_lstm_inference.py
- Add: config/params.yaml (LSTM-specific settings: seq_len, stride, hidden_dim, num_layers, dropout, lr, batch_size, epochs)
- Add: requirements.txt entry for PyTorch (torch==2.x) and related dependencies.
- Update: .gitignore to exclude large model artifacts and training logs if necessary.

## Class/Function Definitions
- model/lstm.py
  - class LSTMRegressor(nn.Module):
      - __init__(input_dim, hidden_dim, num_layers, output_dim=2, dropout=0.0)
      - forward(self, x): returns (batch, output_dim)
  - function initialize_lstm_from_config(config) -> LSTMRegressor

- data_pipeline/generator.py (extension)
  - class SequenceWindowDataset(Dataset):
      - __init__(X, y, seq_len, stride, normalize=True, to_depth_ratio=True)
      - __len__(), __getitem__(idx)
  - function to_sequences(X, y, seq_len, stride) -> (X_seq, y_seq)
  - function transform_depth_to_ratio(y, max_depth) -> depth_ratio

- scripts/02_train_lstm_model.py
  - Load data from outputs/data/training_data.npz or build via data_pipeline.
  - Build DataLoader with SequenceWindowDataset.
  - Instantiate LSTMRegressor with config.
  - Train with MSELoss and Adam.
  - Validate on a held-out validation set.
  - Save model state_dict to outputs/models/lstm_cracknet.pth and stats to outputs/stats/lstm_stats.npz.

- scripts/03_run_lstm_inference.py
  - Load saved model and stats.
  - Load test data, construct sequences, run inference, and produce a report with RMSE per dimension.
  - Optionally generate plots for predictions vs truth.

## Integration Points with Existing Pipeline
- The new LSTM will plug into the same data flow as BP NN by consuming the CPDV feature vectors from data_pipeline/generator.py, but in windowed sequence form.
- The depth value uses depth_ratio rather than raw depth for the second target to align with the requested labeling scheme.
- Inference results should be compatible with existing evaluation tooling, enabling side-by-side comparison with the BP NN baseline in future tasks.

## Training and Inference Workflows (TDD-oriented)
- Tests for data processing: ensure SequenceWindowDataset yields correct shapes for a small synthetic dataset.
- Tests for model: forward pass returns correct shape (batch, 2).
- Training loop test: single training epoch on a tiny dataset runs without errors.
- Inference test: loading a fake/trained model can produce outputs of shape (batch, 2).

## Dependencies
- PyTorch (torch) for model and training.
- numpy, pandas for data handling (likely already in project).
- torchvision is optional (not required for this task).
- Matplotlib/Seaborn for optional plotting during inference or training visualization.

## Atomic Commit Strategy
- Commit 1: Add model/lstm.py implementing LSTMRegressor class and initialization helper.
- Commit 2: Extend data_pipeline/generator.py with SequenceWindowDataset and utility functions for windowing and depth_ratio transformation.
- Commit 3: Add training script scripts/02_train_lstm_model.py and ensure it can run with existing data format.
- Commit 4: Add inference script scripts/03_run_lstm_inference.py and sample evaluation routine.
- Commit 5: Add config/params.yaml with LSTM hyperparameters.
- Commit 6: Update requirements.txt with PyTorch dependency and any related changes.

## Open Questions (Decisions Needed)
- [DECISION NEEDED] Depth target: keep depth_ratio as label or revert to depth and compute ratio via a post-process? Default: use depth_ratio as label to satisfy requirements.
- [DECISION NEEDED] Windowing parameters: seq_len and stride; propose defaults, but please confirm if project data distribution requires alternative.
- [DECISION NEEDED] Data split strategy for train/val/test with windowed data; propose 70/15/15 split; confirm.

## Next Action
- If you approve, I will generate the final plan file at .sisyphus/plans/lstm_cpdv_plan.md based on this draft and proceed with the Metis/Momus review as per process.

Would you like me to proceed with creating the formal plan file (.sisyphus/plans/lstm_cpdv_plan.md) and lock in PyTorch as the preferred framework? Also, do you want to enforce depth_ratio as the target by default and base windowing on a seq_len of 32 with stride 16, or would you prefer different defaults? 
