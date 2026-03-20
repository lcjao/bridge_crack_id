#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成推理验证数据
Generate Verification Data for Inference

生成独立的测试数据集，用于模型推理验证。
保存格式与 training_data.npz 兼容。

Usage:
    python scripts/generate_verify_data.py --n_samples 100 --output outputs/data/verify_data.npz
"""

import os
import sys
import argparse
import yaml
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.generator import DataGenerator


def main():
    parser = argparse.ArgumentParser(description="生成推理验证数据")
    parser.add_argument(
        "--config", type=str, default="config/params.yaml", help="配置文件路径"
    )
    parser.add_argument("--n_samples", type=int, default=100, help="样本数量")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/data/verify_data.npz",
        help="输出文件路径",
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="随机种子（与训练数据不同）"
    )
    parser.add_argument("--road_type", type=str, default="a", help="路面等级 (a/b/c/d)")

    args = parser.parse_args()

    # 加载配置
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.config)
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    sim_params = config.get("simulation", {})

    print("=" * 50)
    print("桥梁裂纹智能检测系统 - 推理数据生成")
    print("=" * 50)
    print(f"样本数量: {args.n_samples}")
    print(f"随机种子: {args.seed}")
    print(f"路面等级: {args.road_type}")
    print(f"输出路径: {args.output}")
    print("=" * 50)

    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 生成数据
    generator = DataGenerator(sim_params, args.seed)
    X, y = generator.generate_dataset(
        args.n_samples, sim_params, road_type=args.road_type
    )

    # 划分训练/验证/测试集
    rng = np.random.RandomState(args.seed)
    n = X.shape[1]
    indices = rng.permutation(n)

    train_ratio = 0.7
    val_ratio = 0.15
    train_split = int(n * train_ratio)
    val_split = int(n * (train_ratio + val_ratio))

    train_idx = indices[:train_split]
    val_idx = indices[train_split:val_split]
    test_idx = indices[val_split:]

    X_train = X[:, train_idx]
    y_train = y[:, train_idx]
    X_val = X[:, val_idx]
    y_val = y[:, val_idx]
    X_test = X[:, test_idx]
    y_test = y[:, test_idx]

    # 标准化（基于训练集统计量）
    X_mean = np.mean(X_train, axis=1, keepdims=True)
    X_std = np.std(X_train, axis=1, keepdims=True) + 1e-8
    y_mean = np.mean(y_train, axis=1, keepdims=True)
    y_std = np.std(y_train, axis=1, keepdims=True) + 1e-8

    X_train_norm = (X_train - X_mean) / X_std
    y_train_norm = (y_train - y_mean) / y_std
    X_val_norm = (X_val - X_mean) / X_std
    y_val_norm = (y_val - y_mean) / y_std
    X_test_norm = (X_test - X_mean) / X_std
    y_test_norm = (y_test - y_mean) / y_std

    print(f"\n数据集划分:")
    print(f"  训练集: {X_train_norm.shape[1]} 样本")
    print(f"  验证集: {X_val_norm.shape[1]} 样本")
    print(f"  测试集: {X_test_norm.shape[1]} 样本")

    # 保存
    np.savez(
        args.output,
        X_train=X_train_norm,
        y_train=y_train_norm,
        X_val=X_val_norm,
        y_val=y_val_norm,
        X_test=X_test_norm,
        y_test=y_test_norm,
        X_mean=X_mean,
        X_std=X_std,
        y_mean=y_mean,
        y_std=y_std,
    )

    print(f"\n数据已保存至: {args.output}")
    print("\n数据统计:")
    print(f"  X_test 范围: [{X_test_norm.min():.4f}, {X_test_norm.max():.4f}]")
    print(f"  y_test 范围: [{y_test_norm.min():.4f}, {y_test_norm.max():.4f}]")
    print("=" * 50)


if __name__ == "__main__":
    main()
