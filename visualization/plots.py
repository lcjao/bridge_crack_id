"""
可视化模块
Visualization Module

用于绘制：
- CPDV信号
- 训练曲线
- 预测结果对比
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict
import os


# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class SignalPlotter:
    """信号可视化"""

    @staticmethod
    def plot_cpdv(
        t: np.ndarray,
        cpdv: np.ndarray,
        title: str = "CPDV Signal",
        save_path: Optional[str] = None,
    ):
        """
        绘制CPDV信号

        Args:
            t: 时间序列
            cpdv: CPDV数据
            title: 图表标题
            save_path: 保存路径
        """
        plt.figure(figsize=(12, 6))
        plt.plot(t, cpdv, linewidth=2, color="blue")
        plt.xlabel("Time (s)")
        plt.ylabel("CPDV (m)")
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"图表已保存至: {save_path}")

        plt.show()

    @staticmethod
    def plot_road_profile(
        t: np.ndarray,
        road: np.ndarray,
        road_type: str = "a",
        save_path: Optional[str] = None,
    ):
        """
        绘制路面轮廓

        Args:
            t: 时间序列
            road: 路面数据
            road_type: 路面等级
            save_path: 保存路径
        """
        plt.figure(figsize=(12, 6))
        plt.plot(t, road, linewidth=2, color="purple")
        plt.xlabel("Time (s)")
        plt.ylabel("Road Roughness (m)")
        plt.title(f"Road Profile - Class {road_type.upper()}")
        plt.grid(True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)

        plt.show()

    @staticmethod
    def plot_multiple_cpdv(
        t: np.ndarray, cpdv_dict: Dict, save_path: Optional[str] = None
    ):
        """
        绘制多条CPDV曲线

        Args:
            t: 时间序列
            cpdv_dict: {label: cpdv} 字典
            save_path: 保存路径
        """
        plt.figure(figsize=(12, 6))
        colors = ["r", "b", "g", "m", "c", "orange"]

        for i, (label, cpdv) in enumerate(cpdv_dict.items()):
            plt.plot(t, cpdv, linewidth=2, color=colors[i % len(colors)], label=label)

        plt.xlabel("Time (s)")
        plt.ylabel("CPDV (m)")
        plt.title("CPDV Comparison")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)

        plt.show()

    @staticmethod
    def plot_peak_position_vs_distance(
        peak_positions: Dict,
        title: str = "不同深度下CPDV峰值位置",
        save_path: Optional[str] = None,
        distances: list = None,
    ):
        """
        绘制不同深度下CPDV峰值位置 vs 距离

        Args:
            peak_positions: {深度标签: [峰值位置数组]} 字典
            title: 图表标题
            save_path: 保存路径
            distances: 距离列表，若为None则根据数据长度自动生成
        """
        plt.figure(figsize=(10, 6))

        # 自动生成 distances 若未提供
        if distances is None:
            first_values = list(peak_positions.values())[0] if peak_positions else []
            distances = list(range(1, len(first_values) + 1))

        colors = ["turquoise", "darkblue", "lightblue", "red", "orange"]

        for i, (depth_label, values) in enumerate(peak_positions.items()):
            plt.plot(
                distances[: len(values)],
                values,
                marker="o",
                linewidth=2,
                markersize=8,
                color=colors[i % len(colors)],
                label=depth_label,
            )

        plt.xlabel("距离 (m)")
        plt.ylabel("CPDV峰值")
        plt.title(title)

        # 显示数据值
        for i, (depth_label, values) in enumerate(peak_positions.items()):
            for j, (x, y) in enumerate(zip(distances[: len(values)], values)):
                plt.annotate(
                    f"{y:.4f}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha="center",
                    fontsize=8,
                )

        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"图表已保存至: {save_path}")

        plt.show()

    @staticmethod
    def plot_peak_value_vs_depth(
        peak_values: Dict,
        depths: list,
        title: str = "不同位置CPDV峰值 vs 裂纹深度",
        save_path: Optional[str] = None,
    ):
        """
        绘制不同位置CPDV峰值 vs 裂纹深度

        Args:
            peak_values: {位置标签: [峰值数组]} 字典
            depths: 裂纹深度列表
            title: 图表标题
            save_path: 保存路径
        """
        plt.figure(figsize=(10, 6))

        n_positions = len(peak_values)
        n_depths = len(depths)
        # 自动计算合适的柱状图宽度
        if n_positions == 1:
            bar_width = 0.6 / n_positions  # 单系列时较宽
        elif n_positions <= 3:
            bar_width = 0.5 / n_positions  # 2-3系列适中
        else:
            bar_width = 0.4 / n_positions  # 多系列时较窄

        colors = ["darkblue", "blue", "teal", "red", "orange"]
        positions = list(peak_values.keys())

        for i, (pos_label, values) in enumerate(peak_values.items()):
            x = np.arange(n_depths) + i * bar_width
            plt.bar(
                x,
                values,
                bar_width,
                label=pos_label,
                color=colors[i % len(colors)],
            )

        plt.xlabel("裂纹深度")
        plt.ylabel("CPDV峰值")
        plt.title(title)
        plt.xticks(np.arange(n_depths) + bar_width, depths)
        plt.legend()
        plt.grid(True, axis="y")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"图表已保存至: {save_path}")

        plt.show()


class TrainingPlotter:
    """训练过程可视化"""

    @staticmethod
    def plot_loss_curve(history: Dict, save_path: Optional[str] = None):
        """
        绘制损失曲线

        Args:
            history: 训练历史 {'train_loss': [...], 'val_loss': [...]}
            save_path: 保存路径
        """
        plt.figure(figsize=(10, 6))

        epochs = range(1, len(history["train_loss"]) + 1)
        plt.plot(
            epochs, history["train_loss"], "b-", linewidth=2, label="Training Loss"
        )

        if "val_loss" in history and history["val_loss"]:
            plt.plot(
                epochs, history["val_loss"], "r-", linewidth=2, label="Validation Loss"
            )

        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.title("Training Progress")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"损失曲线已保存至: {save_path}")

        plt.show()

    @staticmethod
    def plot_prediction_scatter(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        stats: Dict,
        save_path: Optional[str] = None,
    ):
        """
        绘制预测值 vs 真实值散点图

        Args:
            y_true: 真实标签 (2, n_samples)
            y_pred: 预测标签 (2, n_samples)
            stats: 统计信息（用于反标准化）
            save_path: 保存路径
        """
        # 反标准化
        y_true_denorm = y_true * stats["y_std"] + stats["y_mean"]
        y_pred_denorm = y_pred * stats["y_std"] + stats["y_mean"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 位置预测
        ax1 = axes[0]
        ax1.scatter(y_true_denorm[0, :], y_pred_denorm[0, :], alpha=0.5, s=20, c="blue")

        # 理想预测线
        min_val = min(y_true_denorm[0, :].min(), y_pred_denorm[0, :].min())
        max_val = max(y_true_denorm[0, :].max(), y_pred_denorm[0, :].max())
        ax1.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)

        ax1.set_xlabel("True Position (m)")
        ax1.set_ylabel("Predicted Position (m)")
        ax1.set_title("Crack Position Prediction")
        ax1.grid(True)

        # 深度预测
        ax2 = axes[1]
        ax2.scatter(
            y_true_denorm[1, :], y_pred_denorm[1, :], alpha=0.5, s=20, c="green"
        )

        min_val = min(y_true_denorm[1, :].min(), y_pred_denorm[1, :].min())
        max_val = max(y_true_denorm[1, :].max(), y_pred_denorm[1, :].max())
        ax2.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)

        ax2.set_xlabel("True Depth Ratio")
        ax2.set_ylabel("Predicted Depth Ratio")
        ax2.set_title("Crack Depth Prediction")
        ax2.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"预测散点图已保存至: {save_path}")

        plt.show()


class Evaluator:
    """模型评估可视化"""

    @staticmethod
    def plot_error_distribution(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        stats: Dict,
        save_path: Optional[str] = None,
    ):
        """
        绘制误差分布直方图

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            stats: 统计信息
            save_path: 保存路径
        """
        # 反标准化
        y_true_denorm = y_true * stats["y_std"] + stats["y_mean"]
        y_pred_denorm = y_pred * stats["y_std"] + stats["y_mean"]

        errors = y_pred_denorm - y_true_denorm

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 位置误差分布
        ax1 = axes[0]
        ax1.hist(errors[0, :], bins=50, color="blue", alpha=0.7, edgecolor="black")
        ax1.axvline(x=0, color="r", linestyle="--", linewidth=2)
        ax1.set_xlabel("Position Error (m)")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Position Error Distribution")
        ax1.grid(True)

        # 深度误差分布
        ax2 = axes[1]
        ax2.hist(errors[1, :], bins=50, color="green", alpha=0.7, edgecolor="black")
        ax2.axvline(x=0, color="r", linestyle="--", linewidth=2)
        ax2.set_xlabel("Depth Ratio Error")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Depth Error Distribution")
        ax2.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)

        plt.show()

    @staticmethod
    def print_evaluation_report(metrics: Dict):
        """
        打印评估报告（统一格式）

        Args:
            metrics: 评估指标字典
        """
        print("\n" + "=" * 51)
        print("BP神经网络模型推理 - 评估结果")
        print("=" * 51)
        print(f"推理样本数: {metrics.get('n_samples', 'N/A')}")
        print("-" * 51)
        print("评估指标")
        print("-" * 51)
        print(f"  位置 MAE:  {metrics['position_mae']:.4f} m")
        print(f"  位置 RMSE: {metrics['position_rmse']:.4f} m")
        print(f"  深度 MAE:  {metrics['depth_mae']:.4f}")
        print(f"  深度 RMSE: {metrics['depth_rmse']:.4f}")
        print(f"  R2 Score: {metrics['r2_score']:.4f}")
        print("-" * 51)
        print("成功标准对比")
        print("-" * 51)
        print(
            f"    位置 MAE < 1.0m: {'[PASS]' if metrics['position_mae'] < 1.0 else '[FAIL]'}"
        )
        print(
            f"    深度 MAE < 0.05: {'[PASS]' if metrics['depth_mae'] < 0.05 else '[FAIL]'}"
        )
        print("=" * 51)
