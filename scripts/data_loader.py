#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统一数据加载模块"""

import numpy as np


def load_data(data_path, for_nn=False):
    """加载训练数据

    Args:
        data_path: npz 文件路径
        for_nn: 如果为True，返回 (n_features, n_samples) 格式给 NeuralNetwork
                如果为False，返回 (n_samples, n_features) 格式给 LSTM

    Returns:
        dict: 包含训练/验证/测试数据和统计信息
    """
    data = np.load(data_path)

    # 获取数据
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"] if "X_test" in data.files else None
    y_test = data["y_test"] if "y_test" in data.files else None

    # 原数据格式: (n_features, n_samples) = (801, 7000)
    # NeuralNetwork 需要: (n_features, n_samples)
    # LSTM 需要: (n_samples, n_features)

    if for_nn:
        # NeuralNetwork 格式: 保持原样 (n_features, n_samples) 和 (2, n_samples)
        pass
    else:
        # LSTM 格式: 转置 (n_samples, n_features) 和 (n_samples, 2)
        if X_train.shape[0] < X_train.shape[1]:
            X_train = X_train.T
            X_val = X_val.T
            if X_test is not None:
                X_test = X_test.T

        # 转置 y: (2, n_samples) -> (n_samples, 2)
        if y_train.shape[0] < y_train.shape[1]:
            y_train = y_train.T
            y_val = y_val.T
            if y_test is not None:
                y_test = y_test.T

    # 归一化参数
    X_mean = data.get("X_mean", None)
    X_std = data.get("X_std", None)
    y_mean = data.get("y_mean", None)
    y_std = data.get("y_std", None)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "X_mean": X_mean,
        "X_std": X_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }
