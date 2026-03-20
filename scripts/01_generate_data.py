#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 1: 生成训练数据
Generate Training Data

Usage:
    python 01_generate_data.py --n_samples 10000 --output outputs/data/training_data.npz
"""

import os
import sys
import argparse
import yaml
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.generator import DataPipeline


def main():
    parser = argparse.ArgumentParser(description="生成训练数据")
    parser.add_argument(
        "--config", type=str, default="config/params.yaml", help="配置文件路径"
    )
    parser.add_argument("--n_samples", type=int, default=None, help="样本数量")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/data/training_data.npz",
        help="输出文件路径",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    # 加载配置
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.config)
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # 获取参数
    sim_params = config.get("simulation", {})
    data_params = config.get("data", {})

    n_samples = (
        args.n_samples if args.n_samples else data_params.get("n_samples", 10000)
    )
    train_ratio = data_params.get("train_ratio", 0.7)
    val_ratio = data_params.get("val_ratio", 0.15)

    print("=" * 50)
    print("桥梁裂纹智能检测系统 - 数据生成")
    print("=" * 50)
    print(f"样本数量: {n_samples}")
    print(f"训练集比例: {train_ratio}")
    print(f"验证集比例: {val_ratio}")
    print(f"随机种子: {args.seed}")
    print("=" * 50)

    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 运行数据流水线
    pipeline = DataPipeline(
        system_params=sim_params,
        n_samples=n_samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=args.seed,
    )

    data = pipeline.run(verbose=True)

    # 保存数据
    pipeline.save(args.output)

    print("\n数据生成完成!")
    print(f"数据已保存至: {args.output}")


if __name__ == "__main__":
    main()
