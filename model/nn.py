"""
神经网络模块 - 从零实现的BP神经网络
Neural Network Module - From Scratch BP Network
"""

import numpy as np
import json
from typing import Tuple, Dict, Optional


# ==================== 激活函数 ====================


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU激活函数"""
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """ReLU导数"""
    return (x > 0).astype(float)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid激活函数"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """Sigmoid导数"""
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x: np.ndarray) -> np.ndarray:
    """Tanh激活函数"""
    return np.tanh(x)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    """Tanh导数"""
    return 1 - np.tanh(x) ** 2


def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Leaky ReLU激活函数"""
    return np.where(x > 0, x, alpha * x)


def leaky_relu_derivative(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Leaky ReLU导数"""
    return np.where(x > 0, 1, alpha)


# ==================== 神经网络类 ====================


class NeuralNetwork:
    """
    从零实现的BP神经网络

    支持：
    - 单隐藏层（可扩展为多层）
    - 多种激活函数
    - L2正则化
    - 小批量梯度下降
    - 早停机制
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = None,
        output_dim: int = 2,
        activation: str = "relu",
        lr: float = 0.01,
        lambda_reg: float = 0.0,
        seed: int = 42,
        clip_grad: float = 1.0,
    ):
        """
        初始化神经网络

        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表，如[64, 32]表示两层
            output_dim: 输出维度（默认2：位置和深度）
            activation: 激活函数类型 ('relu', 'sigmoid', 'tanh', 'leaky_relu')
            lr: 学习率
            lambda_reg: L2正则化系数
            seed: 随机种子
            clip_grad: 梯度裁剪阈值
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [64]
        self.output_dim = output_dim
        self.activation_name = activation
        self.lr = lr
        self.lambda_reg = lambda_reg
        self.seed = seed
        self.clip_grad = clip_grad

        # 设置随机种子
        self.rng = np.random.RandomState(seed)

        # 选择激活函数
        self._setup_activation(activation)

        # 初始化权重
        self._init_weights()

        # 训练历史
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "epochs": 0,
        }

    def _setup_activation(self, activation: str):
        """设置激活函数"""
        activations = {
            "relu": (relu, relu_derivative),
            "sigmoid": (sigmoid, sigmoid_derivative),
            "tanh": (tanh, tanh_derivative),
            "leaky_relu": (leaky_relu, leaky_relu_derivative),
        }

        if activation not in activations:
            raise ValueError(f"不支持的激活函数: {activation}")

        self.activation, self.activation_derivative = activations[activation]

    def _init_weights(self):
        """初始化权重和偏置"""
        self.weights = []
        self.biases = []

        # 构建网络结构
        dims = [self.input_dim] + self.hidden_dims + [self.output_dim]

        for i in range(len(dims) - 1):
            # Xavier初始化
            fan_in = dims[i]
            fan_out = dims[i + 1]
            limit = np.sqrt(6 / (fan_in + fan_out))

            W = self.rng.uniform(-limit, limit, (fan_out, fan_in))
            b = np.zeros((fan_out, 1))

            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, list]:
        """
        前向传播

        Args:
            X: 输入矩阵 (input_dim, n_samples)

        Returns:
            output: 网络输出
            cache: 中间结果缓存（用于反向传播）
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.X = X
        self.z = []  # 线性组合
        self.a = [X]  # 激活值

        # 隐藏层
        for i in range(len(self.weights) - 1):
            z = self.weights[i] @ self.a[-1] + self.biases[i]
            self.z.append(z)
            a = self.activation(z)
            self.a.append(a)

        # 输出层（线性激活）
        z_out = self.weights[-1] @ self.a[-1] + self.biases[-1]
        self.z.append(z_out)
        self.a.append(z_out)

        return self.a[-1], self.z

    def backward(self, y_true: np.ndarray):
        """
        反向传播

        Args:
            y_true: 真实标签 (output_dim, n_samples)
        """
        m = y_true.shape[1]  # 样本数

        # 输出层误差
        delta = self.a[-1] - y_true

        # 反向传播
        for i in range(len(self.weights) - 1, -1, -1):
            # 计算梯度
            dW = (delta @ self.a[i].T) / m + self.lambda_reg * self.weights[i]
            db = np.sum(delta, axis=1, keepdims=True) / m

            # 存储梯度
            if i == len(self.weights) - 1:
                self.dW = [dW]
                self.db = [db]
            else:
                self.dW.insert(0, dW)
                self.db.insert(0, db)

            # 计算前一层的误差
            if i > 0:
                delta = (self.weights[i].T @ delta) * self.activation_derivative(
                    self.z[i - 1]
                )

    def update(self):
        """更新权重和偏置"""
        # 梯度裁剪
        for i in range(len(self.dW)):
            self.dW[i] = np.clip(self.dW[i], -self.clip_grad, self.clip_grad)
            self.db[i] = np.clip(self.db[i], -self.clip_grad, self.clip_grad)

        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * self.dW[i]
            self.biases[i] -= self.lr * self.db[i]

    def train_step(self, X_batch: np.ndarray, y_batch: np.ndarray):
        """
        单步训练

        Args:
            X_batch: 批次输入
            y_batch: 批次标签
        """
        self.forward(X_batch)
        self.backward(y_batch)
        self.update()

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        计算均方误差损失

        Args:
            X: 输入
            y: 真实标签

        Returns:
            loss: MSE损失
        """
        y_pred, _ = self.forward(X)
        mse_loss = 0.5 * np.mean((y_pred - y) ** 2)

        # L2正则化
        reg_loss = 0
        for W in self.weights:
            reg_loss += np.sum(W**2)
        reg_loss = 0.5 * self.lambda_reg * reg_loss

        return mse_loss + reg_loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测

        Args:
            X: 输入矩阵

        Returns:
            predictions: 预测值
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        pred, _ = self.forward(X)
        return pred

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 200,
        batch_size: int = 32,
        early_stopping_patience: int = 20,
        verbose: bool = True,
    ) -> Dict:
        """
        训练神经网络

        Args:
            X_train: 训练数据
            y_train: 训练标签
            X_val: 验证数据
            y_val: 验证标签
            epochs: 训练轮数
            batch_size: 批量大小
            early_stopping_patience: 早停耐心值
            verbose: 是否打印训练过程

        Returns:
            history: 训练历史
        """
        m = X_train.shape[1]
        best_val_loss = float("inf")
        patience_counter = 0
        best_weights = None
        best_biases = None

        for epoch in range(epochs):
            # 打乱数据
            indices = self.rng.permutation(m)
            X_shuffled = X_train[:, indices]
            y_shuffled = y_train[:, indices]

            # 小批量训练
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[:, i : i + batch_size]
                y_batch = y_shuffled[:, i : i + batch_size]
                self.train_step(X_batch, y_batch)

            # 计算损失
            train_loss = self.compute_loss(X_train, y_train)
            self.history["train_loss"].append(train_loss)

            if X_val is not None and y_val is not None:
                val_loss = self.compute_loss(X_val, y_val)
                self.history["val_loss"].append(val_loss)

                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # 保存最佳权重
                    best_weights = [W.copy() for W in self.weights]
                    best_biases = [b.copy() for b in self.biases]
                else:
                    patience_counter += 1

                if verbose and (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                    )

                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"早停触发于 Epoch {epoch + 1}")
                    break
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}")

        # 恢复最佳权重
        if best_weights is not None:
            self.weights = best_weights
            self.biases = best_biases

        self.history["epochs"] = epoch + 1

        return self.history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        评估模型

        Args:
            X_test: 测试数据
            y_test: 测试标签

        Returns:
            metrics: 评估指标
        """
        y_pred = self.predict(X_test)

        # 计算误差
        errors = y_pred - y_test

        # 位置误差
        position_mae = np.mean(np.abs(errors[0, :]))
        position_rmse = np.sqrt(np.mean(errors[0, :] ** 2))

        # 深度误差
        depth_mae = np.mean(np.abs(errors[1, :]))
        depth_rmse = np.sqrt(np.mean(errors[1, :] ** 2))

        # R²分数
        ss_res = np.sum(errors**2)
        ss_tot = np.sum((y_test - np.mean(y_test, axis=1, keepdims=True)) ** 2)
        r2_score = 1 - ss_res / (ss_tot + 1e-8)

        # 总损失
        test_loss = self.compute_loss(X_test, y_test)

        metrics = {
            "test_loss": test_loss,
            "position_mae": position_mae,
            "position_rmse": position_rmse,
            "depth_mae": depth_mae,
            "depth_rmse": depth_rmse,
            "r2_score": r2_score,
            "y_true": y_test,
            "y_pred": y_pred,
        }

        return metrics

    def save(self, filepath: str):
        """
        保存模型

        Args:
            filepath: 保存路径
        """
        model_data = {
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "output_dim": self.output_dim,
            "activation": self.activation_name,
            "lr": self.lr,
            "lambda_reg": self.lambda_reg,
            "seed": self.seed,
            "weights": [W.tolist() for W in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "history": self.history,
        }

        with open(filepath, "w") as f:
            json.dump(model_data, f, indent=2)

        print(f"模型已保存至: {filepath}")

    def load(self, filepath: str):
        """
        加载模型

        Args:
            filepath: 模型文件路径
        """
        with open(filepath, "r") as f:
            model_data = json.load(f)

        self.input_dim = model_data["input_dim"]
        self.hidden_dims = model_data["hidden_dims"]
        self.output_dim = model_data["output_dim"]
        self.activation_name = model_data["activation"]
        self.lr = model_data["lr"]
        self.lambda_reg = model_data["lambda_reg"]
        self.seed = model_data["seed"]

        # 重新设置激活函数
        self._setup_activation(self.activation_name)

        # 加载权重
        self.weights = [np.array(W) for W in model_data["weights"]]
        self.biases = [np.array(b) for b in model_data["biases"]]

        self.history = model_data["history"]

        print(f"模型已从 {filepath} 加载")

    def summary(self):
        """打印网络结构"""
        print("=" * 50)
        print("神经网络结构")
        print("=" * 50)

        dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        print(f"输入层: {self.input_dim}")

        for i, dim in enumerate(self.hidden_dims):
            print(f"隐藏层 {i + 1}: {dim} 神经元, 激活函数: {self.activation_name}")

        print(f"输出层: {self.output_dim} (位置, 深度)")
        print("-" * 50)

        # 计算参数量
        total_params = 0
        for i in range(len(dims) - 1):
            params = dims[i] * dims[i + 1] + dims[i + 1]
            total_params += params
            print(
                f"层{i + 1}参数: {dims[i]} x {dims[i + 1]} + {dims[i + 1]} = {params}"
            )

        print("-" * 50)
        print(f"总参数量: {total_params}")
        print(f"学习率: {self.lr}, L2正则化: {self.lambda_reg}")
        print("=" * 50)
