#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CPDV特征分析脚本 - 生成峰值位置/峰值与深度关系图

Usage:
    python scripts/04_cpdv_analysis.py --config config/params.yaml --output outputs/figures/
"""

import os
import sys
import argparse
import yaml
import numpy as np

# 添加项目路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(os.path.join(script_dir, ".."))
sys.path.insert(0, project_root)

from simulation.enhanced_system import BridgeVehicleSystem
from visualization.plots import SignalPlotter


def analyze_peak_vs_distance(system, depths, positions):
    """
    分析不同深度下CPDV峰值位置 vs 距离

    Args:
        system: BridgeVehicleSystem实例
        depths: 裂纹深度列表
        positions: 固定的位置列表

    Returns:
        peak_positions: {深度标签: [峰值位置数组]}
    """
    # 确保健康状态已分析
    if system.uc_healthy is None:
        system.run_analysis()

    peak_positions = {}

    for depth in depths:
        depth_peak_positions = []
        for pos in positions:
            results = system.analyze_damage(pos, depth)
            cpdv = system.calculate_cpdv(results["uc"])
            # 峰值位置 = 最大绝对值对应的时间/位置
            peak_idx = np.argmax(np.abs(cpdv))
            depth_peak_positions.append(float(np.abs(cpdv[peak_idx])))

        depth_label = f"深度 {depth}"
        peak_positions[depth_label] = depth_peak_positions

    return peak_positions


def analyze_peak_vs_depth(system, depths, positions):
    """
    分析不同位置CPDV峰值 vs 裂纹深度

    Args:
        system: BridgeVehicleSystem实例
        depths: 裂纹深度列表
        positions: 位置列表

    Returns:
        peak_values: {位置标签: [峰值数组]}
    """
    # 确保健康状态已分析
    if system.uc_healthy is None:
        system.run_analysis()

    peak_values = {}

    for pos in positions:
        pos_peak_values = []
        for depth in depths:
            results = system.analyze_damage(pos, depth)
            cpdv = system.calculate_cpdv(results["uc"])
            # 峰值 = 最大绝对值
            peak_val = float(np.max(np.abs(cpdv)))
            pos_peak_values.append(peak_val)

        pos_label = f"位置 {pos}m"
        peak_values[pos_label] = pos_peak_values

    return peak_values


def main():
    parser = argparse.ArgumentParser(description="CPDV特征分析")
    parser.add_argument(
        "--config",
        type=str,
        default="config/params.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/figures/",
        help="输出目录",
    )
    parser.add_argument(
        "--bridge_length",
        type=float,
        default=None,
        help="桥梁长度(m)，若指定则覆盖配置文件",
    )
    parser.add_argument(
        "--depths",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3],
        help="裂纹深度列表，如: --depths 0.1 0.2 0.3",
    )
    parser.add_argument(
        "--distances",
        type=float,
        nargs="+",
        default=[7.5, 15.0, 22.5],
        help="裂纹位置列表，如: --distances 5 10 15 20",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="输出文件前缀，如: --prefix comparison1",
    )
    args = parser.parse_args()

    # 加载配置
    config_path = os.path.join(project_root, args.config)
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        sim_params = config.get("simulation", {})
    else:
        sim_params = {}

    # 如果命令行指定了bridge_length，覆盖配置
    if args.bridge_length is not None:
        sim_params["L"] = args.bridge_length

    # 确保输出目录存在
    os.makedirs(args.output, exist_ok=True)

    # 创建桥梁系统（使用配置文件中的参数）
    system = BridgeVehicleSystem(params=sim_params)

    # 分析参数（从命令行获取）
    depths = args.depths
    distances = args.distances

    print("=" * 50)
    print("CPDV特征分析")
    print("=" * 50)
    print(f"depths: {depths}")
    print(f"distances: {distances}")

    print("=" * 50)
    print("CPDV特征分析")
    print("=" * 50)

    # 图1: 不同深度下CPDV峰值位置 vs 距离
    print("\n[1/2] 分析峰值位置 vs 距离...")
    peak_positions = analyze_peak_vs_distance(system, depths, distances)

    prefix = args.prefix + "_" if args.prefix else ""
    save_path1 = os.path.join(
        args.output, f"cpdv_{prefix}peak_position_vs_distance.png"
    )
    SignalPlotter.plot_peak_position_vs_distance(
        peak_positions,
        title=f"不同深度下CPDV峰值位置 (depths={depths})",
        save_path=save_path1,
        distances=distances,
    )

    # 图2: 不同位置CPDV峰值 vs 裂纹深度
    print("\n[2/2] 分析峰值 vs 裂纹深度...")
    peak_values = analyze_peak_vs_depth(system, depths, distances)

    save_path2 = os.path.join(args.output, f"cpdv_{prefix}peak_value_vs_depth.png")
    SignalPlotter.plot_peak_value_vs_depth(
        peak_values,
        depths=depths,
        title=f"不同位置CPDV峰值 vs 裂纹深度",
        save_path=save_path2,
    )

    print("\n" + "=" * 50)
    print("分析完成!")
    print(f"图表已保存至: {args.output}")
    print("=" * 50)


if __name__ == "__main__":
    main()
