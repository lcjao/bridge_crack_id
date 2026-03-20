#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统一评估指标模块"""

import numpy as np


def compute_metrics(y_true, y_pred):
    """计算评估指标

    Args:
        y_true: 真实值 (n_samples, 2) - [position, depth]
        y_pred: 预测值 (n_samples, 2) - [position, depth]

    Returns:
        dict: 包含各项指标
    """
    # 位置 (position)
    mae_pos = np.mean(np.abs(y_pred[:, 0] - y_true[:, 0]))
    rmse_pos = np.sqrt(np.mean((y_pred[:, 0] - y_true[:, 0]) ** 2))

    # 深度 (depth)
    mae_depth = np.mean(np.abs(y_pred[:, 1] - y_true[:, 1]))
    rmse_depth = np.sqrt(np.mean((y_pred[:, 1] - y_true[:, 1]) ** 2))

    # R2 Score
    ss_res_pos = np.sum((y_true[:, 0] - y_pred[:, 0]) ** 2)
    ss_tot_pos = np.sum((y_true[:, 0] - np.mean(y_true[:, 0])) ** 2)
    r2_pos = 1 - (ss_res_pos / ss_tot_pos) if ss_tot_pos > 0 else 0.0

    ss_res_depth = np.sum((y_true[:, 1] - y_pred[:, 1]) ** 2)
    ss_tot_depth = np.sum((y_true[:, 1] - np.mean(y_true[:, 1])) ** 2)
    r2_depth = 1 - (ss_res_depth / ss_tot_depth) if ss_tot_depth > 0 else 0.0

    r2_score = (r2_pos + r2_depth) / 2

    return {
        "position_mae": mae_pos,
        "position_rmse": rmse_pos,
        "depth_mae": mae_depth,
        "depth_rmse": rmse_depth,
        "r2_score": r2_score,
    }


def print_metrics(metrics, title="评估结果"):
    """打印评估指标

    Args:
        metrics: 指标字典
        title: 标题
    """
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)
    print(f"位置 MAE:  {metrics['position_mae']:.4f} m")
    print(f"位置 RMSE: {metrics['position_rmse']:.4f} m")
    print(f"深度 MAE:  {metrics['depth_mae']:.4f}")
    print(f"深度 RMSE: {metrics['depth_rmse']:.4f}")
    print(f"R2 Score: {metrics['r2_score']:.4f}")
    print("-" * 50)
    print(
        f"  位置 MAE < 1.0m: {'[PASS]' if metrics['position_mae'] < 1.0 else '[FAIL]'}"
    )
    print(f"  深度 MAE < 0.05: {'[PASS]' if metrics['depth_mae'] < 0.05 else '[FAIL]'}")
    print("=" * 50)
