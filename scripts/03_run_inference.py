#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 3: 运行推理
Run Inference

Usage:
    python 03_run_inference.py --model outputs/models/cracknet.json --data outputs/data/training_data.npz
"""

import os
import sys
import argparse
import numpy as np

# 使用更可靠的方式获取项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 假设项目根目录是 scripts 的父目录
project_root = os.path.normpath(os.path.join(script_dir, ".."))
sys.path.insert(0, project_root)

# 尝试直接导入模块
sys.path.insert(0, os.path.join(project_root, "model"))
sys.path.insert(0, os.path.join(project_root, "visualization"))
sys.path.insert(0, os.path.join(project_root, "simulation"))
sys.path.insert(0, os.path.join(project_root, "data_pipeline"))

from model.nn import NeuralNetwork

try:
    from visualization.plots import TrainingPlotter, Evaluator
except ImportError as e:
    print(f"警告: 无法导入可视化模块 - {e}")
    TrainingPlotter = None
    Evaluator = None


def main():
    parser = argparse.ArgumentParser(description="运行推理")
    parser.add_argument(
        "--model", type=str, default="outputs/models/cracknet.json", help="模型路径"
    )
    parser.add_argument(
        "--stats", type=str, default=None, help="统计量文件路径（默认从模型目录加载）"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/data/verify_data.npz",
        help="测试数据路径",
    )
    parser.add_argument("--plot", action="store_true", help="是否绘制结果")
    parser.add_argument("--n_samples", type=int, default=100, help="可视化样本数量")

    args = parser.parse_args()

    # 确保路径相对于项目根目录
    if not os.path.isabs(args.model):
        args.model = os.path.join(project_root, args.model)
    if args.stats and not os.path.isabs(args.stats):
        args.stats = os.path.join(project_root, args.stats)
    if not os.path.isabs(args.input):
        args.input = os.path.join(project_root, args.input)

    print("\n" + "=" * 50)
    print("桥梁裂纹智能检测系统 - 神经网络推理")
    print("=" * 50)

    # 加载模型（load 会恢复所有维度，无需先初始化）
    print("\n加载模型...")
    net = NeuralNetwork(input_dim=1, hidden_dims=[1], output_dim=1)
    net.load(args.model)

    # 加载测试数据及统计量（从数据文件获取 X_mean/X_std/y_mean/y_std）
    print("加载测试数据...")
    data = np.load(args.input)

    X_test = data["X_test"]
    y_test = data["y_test"]

    # 加载统计量
    if args.stats:
        stats_data = np.load(args.stats)
        stats = {
            "X_mean": stats_data.get("X_mean"),
            "X_std": stats_data.get("X_std"),
            "y_mean": stats_data.get("y_mean"),
            "y_std": stats_data.get("y_std"),
        }
        print(f"统计量: {args.stats}")
    else:
        stats = {
            "X_mean": data.get("X_mean"),
            "X_std": data.get("X_std"),
            "y_mean": data.get("y_mean"),
            "y_std": data.get("y_std"),
        }
        print(f"统计量: 从数据文件 {args.input} 加载")

    # 评估
    print("评估模型...")
    metrics = net.evaluate(X_test, y_test)

    # 反归一化参数
    y_mean = stats["y_mean"]
    y_std = stats["y_std"]
    if y_mean is not None and y_mean.shape[0] == 2:
        y_mean = y_mean[:, 0]
    if y_std is not None and y_std.shape[0] == 2:
        y_std = y_std[:, 0]

    # 反归一化真实值和预测值
    y_true_denorm = metrics["y_true"].T * y_std + y_mean  # (n, 2)
    y_pred_denorm = metrics["y_pred"].T * y_std + y_mean  # (n, 2)

    # 裁剪到物理有效范围
    y_pred_denorm[:, 0] = np.clip(y_pred_denorm[:, 0], 0.0, None)  # 位置 >= 0
    y_pred_denorm[:, 1] = np.clip(y_pred_denorm[:, 1], 0.0, 1.0)  # 深度 [0, 1]

    # 基于物理空间重新计算评估指标
    errors = y_pred_denorm - y_true_denorm
    mae_pos = float(np.mean(np.abs(errors[:, 0])))
    rmse_pos = float(np.sqrt(np.mean(errors[:, 0] ** 2)))
    mae_depth = float(np.mean(np.abs(errors[:, 1])))
    rmse_depth = float(np.sqrt(np.mean(errors[:, 1] ** 2)))

    ss_res_pos = np.sum((y_true_denorm[:, 0] - y_pred_denorm[:, 0]) ** 2)
    ss_tot_pos = np.sum((y_true_denorm[:, 0] - np.mean(y_true_denorm[:, 0])) ** 2)
    r2_pos = 1 - (ss_res_pos / ss_tot_pos) if ss_tot_pos > 0 else 0.0
    ss_res_depth = np.sum((y_true_denorm[:, 1] - y_pred_denorm[:, 1]) ** 2)
    ss_tot_depth = np.sum((y_true_denorm[:, 1] - np.mean(y_true_denorm[:, 1])) ** 2)
    r2_depth = 1 - (ss_res_depth / ss_tot_depth) if ss_tot_depth > 0 else 0.0
    r2_score = (r2_pos + r2_depth) / 2

    # 打印统一格式评估报告
    n_samples = metrics["y_true"].shape[1]
    print("\n" + "=" * 51)
    print("BP神经网络模型推理 - 评估结果")
    print("=" * 51)
    print(f"推理样本数: {n_samples}")
    print("-" * 51)
    print("评估指标")
    print("-" * 51)
    print(f"  位置 MAE:  {mae_pos:.4f} m")
    print(f"  位置 RMSE: {rmse_pos:.4f} m")
    print(f"  深度 MAE:  {mae_depth:.4f}")
    print(f"  深度 RMSE: {rmse_depth:.4f}")
    print(f"  R2 Score: {r2_score:.4f}")
    print("-" * 51)
    print("成功标准对比")
    print("-" * 51)
    print(f"    位置 MAE < 1.0m: {'[PASS]' if mae_pos < 1.0 else '[FAIL]'}")
    print(f"    深度 MAE < 0.05: {'[PASS]' if mae_depth < 0.05 else '[FAIL]'}")
    print("=" * 51)

    # 预测示例（紧凑格式）
    print("-" * 51)
    print("预测示例")
    print("-" * 51)

    n_examples = min(5, X_test.shape[1])
    indices = np.random.choice(X_test.shape[1], n_examples, replace=False)

    for i, idx in enumerate(indices):
        y_t = y_true_denorm[idx]
        y_p = y_pred_denorm[idx]
        print(
            f"  样本{i + 1}: "
            f"位置 - 真实={y_t[0]:.4f}m, 预测={y_p[0]:.4f}m | "
            f"深度 - 真实={y_t[1]:.4f}, 预测={y_p[1]:.4f}"
        )

    print("=" * 51)

    # 绘制结果
    if args.plot:
        figure_dir = os.path.join(project_root, "outputs/figures")
        if not os.path.exists(figure_dir):
            os.makedirs(figure_dir)

        # 预测散点图
        scatter_path = os.path.join(figure_dir, "prediction_scatter.png")
        TrainingPlotter.plot_prediction_scatter(
            metrics["y_true"], metrics["y_pred"], stats, save_path=scatter_path
        )

        # 误差分布
        error_path = os.path.join(figure_dir, "error_distribution.png")
        Evaluator.plot_error_distribution(
            metrics["y_true"], metrics["y_pred"], stats, save_path=error_path
        )

    print("推理完成!")


if __name__ == "__main__":
    main()
