"""
数据生成与预处理模块
Data Generation and Preprocessing
"""

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation.enhanced_system import BridgeVehicleSystem


class DataGenerator:
    """
    数据生成器

    用于批量生成训练数据：
    - 输入: CPDV序列
    - 输出: 裂纹位置、深度
    """

    def __init__(self, params=None, seed=42):
        """
        初始化数据生成器

        Args:
            params: 系统参数字典
            seed: 随机种子
        """
        self.params = params or {}
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # 默认参数范围
        self.crack_position_range = self.params.get("crack_position_range", (0, 30))
        self.crack_depth_range = self.params.get("crack_depth_range", (0.01, 0.3))
        self.road_type = self.params.get("road_type", "a")  # 单一路面等级

    def generate_sample(self, system):
        """
        生成单个样本

        Args:
            system: BridgeVehicleSystem实例

        Returns:
            X: CPDV序列 (特征)
            y: [位置, 深度比] (标签)
        """
        # 随机生成裂纹参数
        pos = self.rng.uniform(*self.crack_position_range)
        depth = self.rng.uniform(*self.crack_depth_range)

        # 使用单一路面等级
        system.road_type = self.road_type

        # 分析健康状态（如果尚未分析）
        if system.uc_healthy is None:
            system.run_analysis()

        # 分析损伤状态
        damaged_results = system.analyze_damage(pos, depth)

        # 计算CPDV
        cpdv = system.calculate_cpdv(damaged_results["uc"])

        X = cpdv
        y = np.array([pos, depth])

        return X, y

    def generate_dataset(self, n_samples, system_params=None, road_type=None):
        """
        批量生成数据集

        Args:
            n_samples: 样本数量
            system_params: 系统参数字典
            road_type: 路面等级，若为None则从配置中获取（单次生成使用相同等级）

        Returns:
            X: CPDV数据矩阵 (n_features, n_samples)
            y: 标签矩阵 (2, n_samples)
        """
        X_list = []
        y_list = []

        # 如果未指定路面类型，使用配置中的单一路面等级
        if road_type is None:
            road_type = self.road_type

        print(f"使用路面等级: {road_type}")

        # 创建基础系统实例用于健康状态分析
        base_system = BridgeVehicleSystem(system_params)
        base_system.road_type = road_type

        # 预计算健康状态
        print("正在计算健康状态基准...")
        base_system.run_analysis()

        if base_system.uc_healthy is None or not np.all(
            np.isfinite(base_system.uc_healthy)
        ):
            raise ValueError("健康状态分析失败")

        uc_healthy = base_system.uc_healthy.copy()

        print(f"正在生成 {n_samples} 个样本...")
        for i in range(n_samples):
            if (i + 1) % 500 == 0:
                print(f"  已完成 {i + 1}/{n_samples}")

            # 每个样本创建新的系统实例
            system = BridgeVehicleSystem(system_params)
            system.uc_healthy = uc_healthy
            system.road_type = road_type  # 使用相同的路面等级

            # 随机生成裂纹参数
            pos = self.rng.uniform(*self.crack_position_range)
            depth = self.rng.uniform(*self.crack_depth_range)

            try:
                # 分析损伤状态
                damaged_results = system.analyze_damage(pos, depth)

                # 计算CPDV
                cpdv = system.calculate_cpdv(damaged_results["uc"])

                # 检查CPDV有效性
                if cpdv is not None and len(cpdv) > 0 and np.all(np.isfinite(cpdv)):
                    X_list.append(cpdv)
                    y_list.append([pos, depth])
                else:
                    # 无效数据，跳过
                    continue
            except Exception as e:
                # 发生错误，跳过该样本
                continue

        X = np.array(X_list).T  # 转置为 (n_features, n_samples)
        y = np.array(y_list).T  # 转置为 (2, n_samples)

        print(f"数据生成完成 - X: {X.shape}, y: {y.shape}")

        return X, y


class DataProcessor:
    """
    数据预处理器

    功能：
    - 数据集划分（训练/验证/测试）
    - 归一化/标准化
    - 数据统计
    """

    def __init__(self, train_ratio=0.7, val_ratio=0.15, seed=42):
        """
        初始化数据处理器

        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            seed: 随机种子
        """
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # 归一化统计量
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None

    def split_data(self, X, y):
        """
        划分数据集

        Args:
            X: 特征矩阵 (n_features, n_samples)
            y: 标签矩阵 (2, n_samples)

        Returns:
            划分后的数据字典
        """
        n = X.shape[1]
        indices = self.rng.permutation(n)

        train_split = int(n * self.train_ratio)
        val_split = int(n * (self.train_ratio + self.val_ratio))

        train_idx = indices[:train_split]
        val_idx = indices[train_split:val_split]
        test_idx = indices[val_split:]

        data = {
            "X_train": X[:, train_idx],
            "y_train": y[:, train_idx],
            "X_val": X[:, val_idx],
            "y_val": y[:, val_idx],
            "X_test": X[:, test_idx],
            "y_test": y[:, test_idx],
        }

        print(f"数据集划分:")
        print(f"  训练集: {data['X_train'].shape[1]} 样本")
        print(f"  验证集: {data['X_val'].shape[1]} 样本")
        print(f"  测试集: {data['X_test'].shape[1]} 样本")

        return data

    def normalize(self, X, y, fit=True):
        """
        标准化数据

        Args:
            X: 特征矩阵
            y: 标签矩阵
            fit: 是否拟合统计量（训练模式）

        Returns:
            标准化后的X, y, 统计量字典
        """
        if fit:
            self.X_mean = np.mean(X, axis=1, keepdims=True)
            self.X_std = np.std(X, axis=1, keepdims=True) + 1e-8
            self.y_mean = np.mean(y, axis=1, keepdims=True)
            self.y_std = np.std(y, axis=1, keepdims=True) + 1e-8

        X_norm = (X - self.X_mean) / self.X_std
        y_norm = (y - self.y_mean) / self.y_std

        stats = {
            "X_mean": self.X_mean,
            "X_std": self.X_std,
            "y_mean": self.y_mean,
            "y_std": self.y_std,
        }

        return X_norm, y_norm, stats

    def normalize_with_stats(self, X, y, stats):
        """
        使用给定统计量标准化数据

        Args:
            X: 特征矩阵
            y: 标签矩阵
            stats: 统计量字典

        Returns:
            标准化后的X, y
        """
        X_norm = (X - stats["X_mean"]) / stats["X_std"]
        y_norm = (y - stats["y_mean"]) / stats["y_std"]
        return X_norm, y_norm

    def denormalize(self, y_norm, stats):
        """
        反标准化

        Args:
            y_norm: 标准化后的标签
            stats: 统计量字典

        Returns:
            原始尺度的标签
        """
        return y_norm * stats["y_std"] + stats["y_mean"]

    def compute_statistics(self, X, y):
        """
        计算数据统计信息

        Args:
            X: 特征矩阵
            y: 标签矩阵

        Returns:
            统计信息字典
        """
        stats = {
            "X_shape": X.shape,
            "y_shape": y.shape,
            "X_min": np.min(X, axis=1),
            "X_max": np.max(X, axis=1),
            "X_mean": np.mean(X, axis=1),
            "X_std": np.std(X, axis=1),
            "y_position_min": np.min(y[0, :]),
            "y_position_max": np.max(y[0, :]),
            "y_depth_min": np.min(y[1, :]),
            "y_depth_max": np.max(y[1, :]),
        }
        return stats


class DataPipeline:
    """
    完整数据流水线

    整合数据生成、划分和预处理
    """

    def __init__(
        self,
        system_params=None,
        n_samples=10000,
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42,
        road_type=None,
    ):
        """
        初始化数据流水线

        Args:
            system_params: 系统参数
            n_samples: 样本数量
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            seed: 随机种子
            road_type: 路面等级，若为None则使用默认配置
        """
        self.system_params = system_params
        self.n_samples = n_samples
        self.seed = seed
        self.road_type = road_type

        self.generator = DataGenerator(system_params, seed)
        self.processor = DataProcessor(train_ratio, val_ratio, seed)

        self.raw_data = None
        self.processed_data = None
        self.stats = None

    def run(self, verbose=True):
        """
        运行完整数据流水线

        Returns:
            处理后的数据字典
        """
        # 生成数据
        if verbose:
            print("=" * 50)
            print("步骤1: 生成数据")
            print("=" * 50)

        X, y = self.generator.generate_dataset(
            self.n_samples, self.system_params, self.road_type
        )

        # 划分数据集
        if verbose:
            print("\n" + "=" * 50)
            print("步骤2: 划分数据集")
            print("=" * 50)

        data = self.processor.split_data(X, y)

        # 标准化
        if verbose:
            print("\n" + "=" * 50)
            print("步骤3: 数据标准化")
            print("=" * 50)

        X_train_norm, y_train_norm, stats = self.processor.normalize(
            data["X_train"], data["y_train"], fit=True
        )
        X_val_norm, y_val_norm = self.processor.normalize_with_stats(
            data["X_val"], data["y_val"], stats
        )
        X_test_norm, y_test_norm = self.processor.normalize_with_stats(
            data["X_test"], data["y_test"], stats
        )

        self.stats = stats
        self.processed_data = {
            "X_train": X_train_norm,
            "y_train": y_train_norm,
            "X_val": X_val_norm,
            "y_val": y_val_norm,
            "X_test": X_test_norm,
            "y_test": y_test_norm,
            "stats": stats,
        }

        if verbose:
            print(f"特征统计:")
            print(
                f"  X_mean 范围: [{stats['X_mean'].min():.6f}, {stats['X_mean'].max():.6f}]"
            )
            print(
                f"  X_std 范围: [{stats['X_std'].min():.6f}, {stats['X_std'].max():.6f}]"
            )
            print(f"  y_mean: {stats['y_mean'].flatten()}")
            print(f"  y_std: {stats['y_std'].flatten()}")

        return self.processed_data

    def save(self, filepath):
        """
        保存处理后的数据

        Args:
            filepath: 保存路径
        """
        np.savez(
            filepath,
            X_train=self.processed_data["X_train"],
            y_train=self.processed_data["y_train"],
            X_val=self.processed_data["X_val"],
            y_val=self.processed_data["y_val"],
            X_test=self.processed_data["X_test"],
            y_test=self.processed_data["y_test"],
            X_mean=self.stats["X_mean"],
            X_std=self.stats["X_std"],
            y_mean=self.stats["y_mean"],
            y_std=self.stats["y_std"],
        )
        print(f"数据已保存至: {filepath}")

    def load(self, filepath):
        """
        加载预处理后的数据

        Args:
            filepath: 数据文件路径

        Returns:
            处理后的数据字典
        """
        data = np.load(filepath)

        self.processed_data = {
            "X_train": data["X_train"],
            "y_train": data["y_train"],
            "X_val": data["X_val"],
            "y_val": data["y_val"],
            "X_test": data["X_test"],
            "y_test": data["y_test"],
        }

        self.stats = {
            "X_mean": data["X_mean"],
            "X_std": data["X_std"],
            "y_mean": data["y_mean"],
            "y_std": data["y_std"],
        }

        print(f"数据已从 {filepath} 加载")

        return self.processed_data, self.stats
