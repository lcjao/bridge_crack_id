#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""神经网络模型训练"""

import os
import sys
import argparse
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.nn import NeuralNetwork
from scripts.data_loader import load_data
from scripts.metrics import compute_metrics, print_metrics
from visualization.plots import TrainingPlotter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def train(args):
    print("\n" + "=" * 50)
    print("桥梁裂缝识别系统 - 神经网络训练")
    print("=" * 50)
    print(f"数据: {args.data}")
    print(f"模型保存: {args.model}")
    print("-" * 50)

    # 加载数据 (for_nn=True: NeuralNetwork 需要 n_features, n_samples 格式)
    data = load_data(args.data, for_nn=True)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    print(f"[数据] 训练:{X_train.shape}, 验证:{X_val.shape}, 测试:{X_test.shape}")

    # 模型 - input_dim 使用 n_features (shape[0]), not n_samples (shape[1])
    model = NeuralNetwork(
        input_dim=X_train.shape[0],
        hidden_dims=args.hidden_dims,
        output_dim=2,
        activation=args.activation,
        lr=args.lr,
        lambda_reg=args.reg,
        seed=42,
    )

    # 训练
    print("[训练]")
    history = model.train(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping_patience=args.patience,
        verbose=True,
    )

    # 评估
    metrics = model.evaluate(X_test, y_test)
    print_metrics(metrics, "测试集评估结果")

    # 绘制训练曲线
    if args.plot:
        print("\n[绘图]")
        # 保存路径
        plot_dir = (
            os.path.dirname(args.model).replace("models", "plots")
            if args.model
            else "outputs/plots"
        )
        os.makedirs(plot_dir, exist_ok=True)

        # 绘制损失曲线
        loss_path = os.path.join(plot_dir, "loss_curve.png")
        TrainingPlotter.plot_loss_curve(history, save_path=loss_path)

        # 获取预测结果用于绘制散点图
        y_pred = model.predict(X_test)
        # 构建stats字典用于反标准化
        if data["y_mean"] is not None and data["y_std"] is not None:
            stats = {"y_mean": data["y_mean"], "y_std": data["y_std"]}
            scatter_path = os.path.join(plot_dir, "prediction_scatter.png")
            TrainingPlotter.plot_prediction_scatter(
                y_test, y_pred, stats, save_path=scatter_path
            )

    # 保存
    # os.makedirs(os.path.dirname(args.model), exist_ok=True)
    # model.save(args.model)

    # print(f"\n模型已保存: {args.model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练神经网络")
    parser.add_argument("--data", type=str, default="outputs/data/training_data.npz")
    parser.add_argument("--model", type=str, default="outputs/models/cracknet.json")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[128, 64, 32])
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--reg", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--plot", action="store_true", help="训练完成后绘制图表")
    train(parser.parse_args())
