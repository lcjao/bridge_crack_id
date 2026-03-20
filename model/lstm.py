try:
    import torch
    import torch.nn as nn

    class LSTMRegressor(nn.Module):
        """LSTM回归器 - 用于CPDV序列预测损伤位置和深度"""

        def __init__(
            self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2, output_dim=2
        ):
            super(LSTMRegressor, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers

            self.lstm = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
            )
            self.fc = nn.Linear(hidden_dim, output_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            # x shape: (batch, seq_len, input_dim)
            lstm_out, _ = self.lstm(x)
            # 取最后一个时间步的输出
            out = self.fc(lstm_out[:, -1, :])
            return out
except ImportError:
    # PyTorch 未安装时的兜底实现，避免直接导入报错
    class LSTMRegressor:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch 未安装，无法构建 LSTMRegressor。请安装 PyTorch 以使用该模型。"
            )
