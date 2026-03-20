#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LSTM模型训练脚本"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.lstm import LSTMRegressor
from scripts.data_loader import load_data
from scripts.metrics import compute_metrics, print_metrics
from visualization.plots import TrainingPlotter


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


def train(args):
    torch.manual_seed(42)
    np.random.seed(42)

    print("\n" + "=" * 50)
    print("桥梁裂缝识别系统 - LSTM模型训练")
    print("=" * 50)
    print(f"数据: {args.data}")
    print(f"模型保存: {args.model}")
    print("-" * 50)

    # 加载数据
    data = load_data(args.data)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]

    print(f"[数据] 训练:{X_train.shape}, 验证:{X_val.shape}")

    # 数据集
    train_dataset = SequenceWindowDataset(X_train, y_train, args.seq_len, args.stride)
    val_dataset = SequenceWindowDataset(X_val, y_val, args.seq_len, args.stride)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    print(f"[数据集] 训练集:{len(train_dataset)} 窗口, 验证集:{len(val_dataset)} 窗口")

    # 模型
    model = LSTMRegressor(
        input_dim=X_train.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        output_dim=2,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"[设备] {device}")
    print("-" * 50)

    # 训练
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                val_loss += criterion(model(X_batch), y_batch).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1}/{args.epochs} - Train:{train_loss:.4f}, Val:{val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(os.path.dirname(args.model), exist_ok=True)
            torch.save(
                {"model_state_dict": model.state_dict(), "args": args}, args.model
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[早停] 触发于 epoch {epoch + 1}")
                break

    # 评估
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            all_preds.append(model(X_batch.to(device)).cpu().numpy())
            all_targets.append(y_batch.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    metrics = compute_metrics(all_targets, all_preds)
    print_metrics(metrics, "验证集评估结果")

    # 绘制训练曲线
    if args.plot:
        print("\n[绘图]")
        history = {"train_loss": train_losses, "val_loss": val_losses}
        TrainingPlotter.plot_loss_curve(history)

        # 绘制预测散点图
        if len(all_preds) > 0:
            # 需要反标准化，这里简化处理
            stats = {"y_mean": np.array([15, 0.15]), "y_std": np.array([8.6, 0.086])}
            TrainingPlotter.plot_prediction_scatter(all_targets, all_preds, stats)

    print(f"\n模型已保存: {args.model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练LSTM模型")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--plot", action="store_true", help="训练完成后绘制图表")
    train(parser.parse_args())
