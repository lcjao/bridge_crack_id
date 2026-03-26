#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LSTM模型推理脚本"""

import os
import sys
import argparse
import numpy as np
import torch

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.lstm import LSTMRegressor

# 可视化模块
try:
    from visualization.plots import TrainingPlotter, Evaluator

    PLOT_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入可视化模块 - {e}")
    PLOT_AVAILABLE = False
    TrainingPlotter = None
    Evaluator = None


class SequenceWindowDataset(torch.utils.data.Dataset):
    """序列窗口数据集"""

    def __init__(self, X, y, seq_len=20, stride=10):
        self.X = X
        self.y = y
        self.seq_len = seq_len
        self.stride = stride
        self.valid_indices = [
            i
            for i in range(0, len(X) - seq_len + 1, stride)
            if not np.isnan(X[i : i + seq_len]).any()
        ]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        return torch.FloatTensor(self.X[i : i + self.seq_len]), torch.FloatTensor(
            self.y[i + self.seq_len - 1]
        )


# LSTM 训练使用的默认参数（用于旧 checkpoint 的 fallback）
_TRAIN_DEFAULTS = dict(
    seq_len=8, stride=2, hidden_dim=128, num_layers=2, dropout=0.2, input_dim=None
)


def load_model(model_path, input_dim, args):
    """加载训练好的模型"""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = LSTMRegressor(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        output_dim=2,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    X_mean = checkpoint.get("X_mean", checkpoint.get("mean", None))
    X_std = checkpoint.get("X_std", checkpoint.get("std", None))
    return model, X_mean, X_std


def inference(args):
    """推理主函数"""
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    args_ckpt = checkpoint.get("args", None)
    # 合并：checkpoint args 优先，命令行其次，回退到训练默认值
    defaults = {**_TRAIN_DEFAULTS, "input_dim": None}
    if args_ckpt is not None:
        for k in defaults:
            if not hasattr(args, k) or getattr(args, k) is None:
                setattr(args, k, getattr(args_ckpt, k, defaults[k]))
    else:
        for k in defaults:
            if not hasattr(args, k) or getattr(args, k) is None:
                setattr(args, k, defaults[k])

    # 获取模型参数
    input_dim = args.input_dim if args.input_dim else None
    model, X_mean, X_std = load_model(args.model, input_dim or 801, args)

    print("\n" + "=" * 50)
    print("桥梁裂纹智能检测系统 - LSTM推理")
    print("=" * 50)
    print(f"模型: {args.model}")

    if args.input:
        data = np.load(args.input)
        keys = list(data.keys())

        # 支持多种格式: 原始格式(X, y), 预处理格式(X_test/y_test), 验证格式(X_val/y_val)
        if "X" in keys:
            X = data["X"]
            y_true = data.get("y", None)
        elif "X_test" in keys:
            X = data["X_test"]
            y_true = data.get("y_test", None)
        elif "X_val" in keys:
            X = data["X_val"]
            y_true = data.get("y_val", None)
        else:
            raise ValueError(f"Unknown data format in {args.input}. Keys: {keys}")

        # 转置: (n_features, n_samples) -> (n_samples, n_features)
        if X.shape[0] < X.shape[1]:
            X = X.T
        if y_true is not None and len(y_true.shape) > 1:
            if y_true.shape[0] < y_true.shape[1]:
                y_true = y_true.T

        input_dim = X.shape[1]

        # 归一化
        if X_mean is not None:
            if X_mean.shape[0] == X.shape[1]:
                X_mean = X_mean.T
                X_std = X_std.T
            X_norm = (X - X_mean) / X_std
        else:
            X_norm = X

        # 反归一化参数
        y_mean = data.get("y_mean")
        y_std = data.get("y_std")
        if y_mean is not None and y_mean.shape[0] == 2:
            y_mean = y_mean[:, 0]
        if y_std is not None and y_std.shape[0] == 2:
            y_std = y_std[:, 0]

        print(f"数据: {args.input}")
        print("-" * 50)

        # 创建序列数据集
        dataset = SequenceWindowDataset(
            X_norm, np.zeros((len(X), 2)), args.seq_len, args.stride
        )

        # 推理
        predictions = []
        with torch.no_grad():
            for i in range(len(dataset)):
                X_seq, _ = dataset[i]
                X_seq = X_seq.unsqueeze(0)
                pred = model(X_seq)
                predictions.append(pred.numpy())

        predictions = np.array(predictions).squeeze()

        # 计算评估指标（基于反归一化后的真实值和预测值）
        metrics = {}
        y_true_valid = None
        if y_true is not None and len(predictions) > 0:
            # 使用与 dataset 相同的 valid_indices
            valid_indices = dataset.valid_indices
            y_true_valid = np.array(
                [y_true[i + args.seq_len - 1] for i in valid_indices]
            )

            # 反归一化
            if y_mean is not None and y_std is not None:
                predictions_denorm = predictions * y_std + y_mean
                y_true_denorm = y_true_valid * y_std + y_mean
            else:
                predictions_denorm = predictions
                y_true_denorm = y_true_valid

            rmse_pos = np.sqrt(
                np.mean((predictions_denorm[:, 0] - y_true_denorm[:, 0]) ** 2)
            )
            rmse_depth = np.sqrt(
                np.mean((predictions_denorm[:, 1] - y_true_denorm[:, 1]) ** 2)
            )
            mae_pos = np.mean(np.abs(predictions_denorm[:, 0] - y_true_denorm[:, 0]))
            mae_depth = np.mean(np.abs(predictions_denorm[:, 1] - y_true_denorm[:, 1]))

            # 计算 R2 Score（基于反归一化值）
            ss_res_pos = np.sum((y_true_denorm[:, 0] - predictions_denorm[:, 0]) ** 2)
            ss_tot_pos = np.sum(
                (y_true_denorm[:, 0] - np.mean(y_true_denorm[:, 0])) ** 2
            )
            r2_pos = 1 - (ss_res_pos / ss_tot_pos) if ss_tot_pos > 0 else 0.0

            ss_res_depth = np.sum((y_true_denorm[:, 1] - predictions_denorm[:, 1]) ** 2)
            ss_tot_depth = np.sum(
                (y_true_denorm[:, 1] - np.mean(y_true_denorm[:, 1])) ** 2
            )
            r2_depth = 1 - (ss_res_depth / ss_tot_depth) if ss_tot_depth > 0 else 0.0

            r2_score = (r2_pos + r2_depth) / 2

            metrics = {
                "rmse_position": rmse_pos,
                "rmse_depth": rmse_depth,
                "mae_position": mae_pos,
                "mae_depth": mae_depth,
                "r2_score": r2_score,
            }
            predictions_denorm_out = predictions_denorm
            y_true_denorm_out = y_true_denorm
            # 裁剪到物理有效范围
            predictions_denorm_out[:, 0] = np.clip(
                predictions_denorm_out[:, 0], 0.0, None
            )
            predictions_denorm_out[:, 1] = np.clip(
                predictions_denorm_out[:, 1], 0.0, 1.0
            )
        else:
            predictions_denorm_out = np.copy(predictions)
            predictions_denorm_out[:, 0] = np.clip(
                predictions_denorm_out[:, 0], 0.0, None
            )
            predictions_denorm_out[:, 1] = np.clip(
                predictions_denorm_out[:, 1], 0.0, 1.0
            )
            y_true_denorm_out = y_true_valid

        # 输出结果 - 统一格式
        print("\n" + "=" * 51)
        print("LSTM模型推理 - 评估结果")
        print("=" * 51)
        print(f"推理样本数: {len(predictions)}")
        print("-" * 51)

        if metrics:
            print("评估指标")
            print("-" * 51)
            print(f"  位置 MAE:  {metrics['mae_position']:.4f} m")
            print(f"  位置 RMSE: {metrics['rmse_position']:.4f} m")
            print(f"  深度 MAE:  {metrics['mae_depth']:.4f}")
            print(f"  深度 RMSE: {metrics['rmse_depth']:.4f}")
            print(f"  R2 Score: {metrics['r2_score']:.4f}")
            print("-" * 51)
            print("成功标准对比")
            print("-" * 51)
            print(
                f"    位置 MAE < 1.0m: {'[PASS]' if metrics['mae_position'] < 1.0 else '[FAIL]'}"
            )
            print(
                f"    深度 MAE < 0.05: {'[PASS]' if metrics['mae_depth'] < 0.05 else '[FAIL]'}"
            )
            print("=" * 51)

        # 预测示例（紧凑格式，仅在有真实值时显示）
        if metrics and y_true_denorm_out is not None:
            print("-" * 51)
            print("预测示例")
            print("-" * 51)
            for i, pred in enumerate(predictions_denorm_out[:5]):
                if i < len(y_true_denorm_out):
                    print(
                        f"  样本{i + 1}: "
                        f"位置 - 真实={y_true_denorm_out[i, 0]:.4f}m, 预测={pred[0]:.4f}m | "
                        f"深度 - 真实={y_true_denorm_out[i, 1]:.4f}, 预测={pred[1]:.4f}"
                    )
                else:
                    print(
                        f"  样本{i + 1}: "
                        f"位置 - 预测={pred[0]:.4f}m | "
                        f"深度 - 预测={pred[1]:.4f}"
                    )
        print("=" * 51)

        if args.output:
            np.save(args.output, predictions_denorm_out)
            print(f"结果已保存: {args.output}")

        # 绘制结果
        if args.plot and PLOT_AVAILABLE:
            figure_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "outputs/figures",
            )
            if not os.path.exists(figure_dir):
                os.makedirs(figure_dir)

            stats = {"y_mean": y_mean, "y_std": y_std}
            y_true_norm = y_true_valid
            y_pred_norm = predictions

            # 预测散点图
            scatter_path = os.path.join(figure_dir, "lstm_prediction_scatter.png")
            TrainingPlotter.plot_prediction_scatter(
                y_true_norm, y_pred_norm, stats, save_path=scatter_path
            )

            # 误差分布
            error_path = os.path.join(figure_dir, "lstm_error_distribution.png")
            Evaluator.plot_error_distribution(
                y_true_norm, y_pred_norm, stats, save_path=error_path
            )
    else:
        print("数据: (未提供)")
        print("-" * 50)
        print("请使用 --input 指定CPDV数据文件")
        print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM模型推理")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--input", type=str, help="输入CPDV数据路径")
    parser.add_argument("--output", type=str, help="输出预测结果路径")
    parser.add_argument("--plot", action="store_true", help="是否绘制结果")
    parser.add_argument("--seq_len", type=int, default=6, help="序列长度")
    parser.add_argument("--stride", type=int, default=2, help="滑动步长")
    parser.add_argument("--hidden_dim", type=int, default=128, help="隐藏层维度")
    parser.add_argument("--num_layers", type=int, default=2, help="LSTM层数")
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout率")
    parser.add_argument("--input_dim", type=int, help="输入维度")
    args = parser.parse_args()
    inference(args)
