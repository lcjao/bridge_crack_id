"""
车-桥耦合仿真系统 - 基于system.md方法论
Bridge-Vehicle Coupled System Simulation (Based on system.md Methodology)

该模块实现system.md中描述的核心方法：
1. 非耦合迭代程序（6步算法）进行车辆-桥梁相互作用模拟
2. 从车辆加速度反算AP（表观剖面）
3. CPDV计算：AP_damaged - AP_intact

核心参考：system.md - "求解接触点位移（CPDV）的具体数值方法"
"""

import numpy as np
from scipy import linalg
import math
from typing import Dict, List, Tuple, Optional


class BridgeVehicleSystem:
    """
    车-桥耦合系统类 - 基于system.md方法论

    实现核心概念：
    - CPDV (Contact Point Displacement Variation): 接触点位移变化
    - AP (Apparent Profile): 表观剖面 = 车辆-桥梁接触点位移时程 + 路面不平度
    - 6步迭代算法进行车辆-桥梁耦合分析
    """

    def __init__(self, params: Optional[Dict] = None):
        """
        初始化车-桥耦合系统

        Args:
            params: 参数字典，若为None则使用默认参数
        """
        self.params = params or {}

        # 参数类型转换辅助函数
        def to_float(val, default):
            if val is None:
                return default
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):  # 处理YAML解析的科学计数法字符串
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return default
            return default

        def to_int(val, default):
            if val is None:
                return default
            if isinstance(val, (int, float)):
                return int(val)
            if isinstance(val, str):  # 处理YAML解析的字符串
                try:
                    return int(val)
                except (ValueError, TypeError):
                    return default
            return default

        # 路面等级参数
        self.road_type = self.params.get("road_type", "a")

        # 车辆参数 (1/4车模型)
        self.mv = to_float(self.params.get("mv", 1000), 1000)  # 车辆质量 (kg)
        self.kv = to_float(self.params.get("kv", 170000), 170000)  # 车辆悬架刚度 (N/m)
        self.cv = to_float(self.params.get("cv", 1000), 1000)  # 车辆悬架阻尼 (N/(m/s))
        self.k_a = to_float(self.params.get("k_a", 170000), 170000)  # 车轴刚度 (N/m)
        self.V = to_float(self.params.get("V", 10), 5)  # 车辆速度 (m/s)
        self.g = 9.8  # 重力加速度 (m/s^2)
        self.fv = np.sqrt(self.kv / self.mv) / (2 * np.pi)  # 车辆固有频率 (Hz)

        # 桥梁参数
        self.L = to_float(self.params.get("L", 30), 30)  # 桥梁长度 (m)
        self.E = to_float(self.params.get("E", 2.75e10), 2.75e10)  # 桥梁弹性模量 (Pa)
        self.I = to_float(self.params.get("I", 0.175), 0.175)  # 桥梁惯性矩 (m^4)
        self.m = to_float(self.params.get("m", 1000), 1000)  # 单位长度质量 (kg/m)
        self.EL = to_int(self.params.get("EL", 30), 30)  # 桥梁离散单元数量
        self.depth = to_float(self.params.get("depth", 1.0), 1.0)  # 梁深 (m)
        self.width = to_float(self.params.get("width", 0.3), 0.3)  # 梁宽 (m)

        # 模态参数
        self.n_modes = to_int(self.params.get("n_modes", 4), 4)
        self.kexi = to_float(self.params.get("kexi", 0.02), 0.02)  # 阻尼比

        # 瑞利阻尼系数 [C_b] = α[M_b] + β[K_b]
        self.alpha = to_float(self.params.get("alpha", 0.0), 0.0)  # 质量矩阵系数
        self.beta_damping = to_float(
            self.params.get("beta_damping", 0.0), 0.0
        )  # 刚度矩阵系数

        # 分析参数
        self.ttotal = self.L / self.V  # 总分析时间 (s)
        self.deltat = to_float(self.params.get("deltat", 0.01), 0.01)  # 时间步长 (s)
        self.t = np.arange(0, self.ttotal + self.deltat, self.deltat)  # 时间向量
        self.tstep = len(self.t)

        # Newmark-β法参数
        self.gamma = to_float(self.params.get("gamma", 0.5), 0.5)
        self.beta_nb = to_float(
            self.params.get("beta", 0.25), 0.25
        )  # 避免与阻尼beta混淆
        self._setup_newmark_params()

        # 迭代收敛参数 (system.md Step 6)
        self.max_iterations = to_int(self.params.get("max_iterations", 50), 50)
        self.convergence_tol = to_float(
            self.params.get("convergence_tol", 0.01), 0.01
        )  # 1%收敛准则

        # 裂纹损伤参数
        self.cracks: List[Dict] = []
        self.has_crack = False

        # 存储结果
        self.results = {}
        self.healthy_results = {}
        self.damaged_results = {}
        self.uc = []  # 接触点位移响应 (AP)
        self.CPDV = []  # 接触点位移变化
        self.uc_healthy = None  # 健康状态AP
        self.vehicle_acc = None  # 车辆加速度时程

        # 随机种子
        self.rng = np.random.RandomState(42)

    def _setup_newmark_params(self):
        """设置Newmark-β法参数"""
        dt = self.deltat
        beta = self.beta_nb
        gamma = self.gamma

        self.Alpha_0 = 1 / (beta * dt**2)
        self.Alpha_1 = gamma / (beta * dt)
        self.Alpha_2 = 1 / (beta * dt)
        self.Alpha_3 = 1 / (2 * beta) - 1
        self.Alpha_4 = gamma / beta - 1
        self.Alpha_5 = dt / 2 * (gamma / beta - 2)
        self.Alpha_6 = dt * (1 - gamma)
        self.Alpha_7 = gamma * dt

    def add_crack(self, position: float, depth_ratio: float, width: float = 0.3):
        """添加裂纹"""
        self.cracks.append(
            {"position": position, "depth_ratio": depth_ratio, "width": width}
        )
        self.has_crack = True

    def clear_cracks(self):
        """清除所有裂纹"""
        self.cracks = []
        self.has_crack = False

    def beam_km0_f(
        self, m: float, L: float, E: float, I: float, EL: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        桥梁有限元质量矩阵和刚度矩阵生成函数

        参考: system.md 边界条件处理 - 简支梁两端节点竖向位移为零

        Args:
            m: 单位长度质量 (kg/m)
            L: 桥梁长度 (m)
            E: 桥梁弹性模量 (Pa)
            I: 桥梁惯性矩 (m^4)
            EL: 桥梁离散单元数量

        Returns:
            M_global: 全局质量矩阵
            K_global: 全局刚度矩阵
        """
        Le = L / EL

        # 单元质量矩阵 (一致质量矩阵)
        mb = (m * Le / 420) * np.array(
            [
                [156, 22 * Le, 54, -13 * Le],
                [22 * Le, 4 * Le**2, 13 * Le, -3 * Le**2],
                [54, 13 * Le, 156, 22 * Le],
                [-13 * Le, -3 * Le**2, 22 * Le, 4 * Le**2],
            ]
        )

        # 单元刚度矩阵
        kb = (E * I / Le**3) * np.array(
            [
                [12, 6 * Le, -12, 6 * Le],
                [6 * Le, 4 * Le**2, -6 * Le, 2 * Le**2],
                [-12, -6 * Le, 12, -6 * Le],
                [6 * Le, 2 * Le**2, -6 * Le, 4 * Le**2],
            ]
        )

        n_dof = 2 * (EL + 1)
        K_global = np.zeros((n_dof, n_dof))
        M_global = np.zeros((n_dof, n_dof))

        # 多裂纹处理
        if self.has_crack and len(self.cracks) > 0:
            for crack in self.cracks:
                crack_position = crack["position"]
                crack_depth_ratio = crack["depth_ratio"]
                crack_width = crack["width"]

                crack_elem = int(np.floor(crack_position / Le)) + 1
                if crack_elem > EL:
                    crack_elem = EL
                xi_j = crack_position - (crack_elem - 1) * Le

                d = self.depth
                w = crack_width
                I_0 = w * d**3 / 12
                d_cj = crack_depth_ratio * d
                I_cj = w * (d - d_cj) ** 3 / 12

                l_c = 1.5 * self.depth
                xi_j_norm = xi_j / Le

                # 刚度降低矩阵元素
                k11 = (12 * E * (I_0 - I_cj) / Le**4) * (
                    (2 * l_c**3) / (Le**2) + 3 * l_c * ((2 * xi_j_norm - 1) ** 2)
                )
                k12 = (12 * E * (I_0 - I_cj) / Le**3) * (
                    (l_c**3) / (Le**2) + l_c * (2 - 7 * xi_j_norm + 6 * xi_j_norm**2)
                )
                k14 = (12 * E * (I_0 - I_cj) / Le**3) * (
                    (l_c**3) / (Le**2) + l_c * (1 - 5 * xi_j_norm + 6 * xi_j_norm**2)
                )
                k22 = (12 * E * (I_0 - I_cj) / Le**2) * (
                    (3 * l_c**3) / (Le**2) + 2 * l_c * ((3 * xi_j_norm - 2) ** 2)
                )
                k24 = (12 * E * (I_0 - I_cj) / Le**2) * (
                    (3 * l_c**3) / (Le**2)
                    + 2 * l_c * (2 - 9 * xi_j_norm + 9 * xi_j_norm**2)
                )
                k44 = (12 * E * (I_0 - I_cj) / Le**2) * (
                    (3 * l_c**3) / (Le**2) + 2 * l_c * ((3 * xi_j_norm - 1) ** 2)
                )

                K_crack = np.array(
                    [
                        [k11, k12, -k11, k14],
                        [k12, k22, -k12, k24],
                        [-k11, -k12, k11, -k14],
                        [k14, k24, -k14, k44],
                    ]
                )

                # 应用裂纹刚度降低
                K_e = kb.copy()
                K_e_crack = K_e - K_crack
                K_e_crack = (K_e_crack + K_e_crack.T) / 2
                K_global[
                    2 * (crack_elem - 1) : 2 * (crack_elem - 1) + 4,
                    2 * (crack_elem - 1) : 2 * (crack_elem - 1) + 4,
                ] += K_e_crack

        # 组装全局矩阵（无裂纹或未损伤单元）
        for i in range(EL):
            # 跳过已应用裂纹的单元
            if self.has_crack:
                skip = False
                for crack in self.cracks:
                    crack_elem = int(np.floor(crack["position"] / Le)) + 1
                    if (i + 1) == crack_elem:
                        skip = True
                        break
                if skip:
                    continue

            K_global[2 * i : 2 * i + 4, 2 * i : 2 * i + 4] += kb
            M_global[2 * i : 2 * i + 4, 2 * i : 2 * i + 4] += mb

        # 应用边界条件（简支梁）
        # system.md: "两端节点的竖向位移约束为零"
        keep_indices = list(range(n_dof))
        keep_indices.remove(0)  # 左端竖向位移
        keep_indices.remove(n_dof - 2)  # 右端竖向位移
        M_global = M_global[np.ix_(keep_indices, keep_indices)]
        K_global = K_global[np.ix_(keep_indices, keep_indices)]

        return M_global, K_global

    def setup_damping(self, M: np.ndarray, K: np.ndarray) -> np.ndarray:
        """
        设置瑞利阻尼 [C_b] = α[M_b] + β[K_b]

        参考: system.md - "阻尼处理：桥梁阻尼采用经典的粘性瑞利阻尼模型"

        Args:
            M: 质量矩阵
            K: 刚度矩阵

        Returns:
            C: 阻尼矩阵
        """
        if self.alpha == 0 and self.beta_damping == 0:
            # 自动计算瑞利阻尼系数
            if hasattr(self, "Frequ") and len(self.Frequ) >= 2:
                f_1 = self.Frequ[0]
                f_2 = self.Frequ[1]
                w1 = 2 * np.pi * f_1
                w2 = 2 * np.pi * f_2

                # 基于阻尼比计算
                coefficient = 2 * w1 * w2 / (w1**2 - w2**2)
                matrix = np.array([[w1, -w2], [-1 / w1, 1 / w2]])
                a0, a1 = np.dot(matrix, [self.kexi, self.kexi]) * coefficient
                self.alpha = a0
                self.beta_damping = a1

        C = self.alpha * M + self.beta_damping * K
        return C

    def psd_r(
        self, roadtype: str, L: float, V: float, deltat: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        路面不平顺度生成函数（ISO 8608标准）

        Args:
            roadtype: 路面等级 'a', 'b', 'c'
            L: 桥梁长度 (m)
            V: 车辆速度 (m/s)
            deltat: 时间步长 (s)

        Returns:
            RX: 路面不平顺序列
            dRX: 路面不平顺一阶导数
        """
        ts = 5 * deltat

        road_dict = {
            "a": 0.001 * 1e-6,
            "b": 8 * 1e-6,
            "c": 16 * 1e-6,
        }
        Gd_n0 = road_dict.get(roadtype.lower(), 0.001 * 1e-6)

        n_max = 2.83
        n_min = 0.011
        n0 = 0.1
        N = max(100, int(L / V / ts))

        t_total = L / V
        num_points = int(t_total / deltat) + 1
        t = np.linspace(0, t_total, num_points)

        theta = self.rng.uniform(0, 2 * np.pi, N)

        delta_n = (n_max - n_min) / N
        n_k = n_min + np.arange(N) * delta_n

        Gd_n = Gd_n0 * (n_k / n0) ** (-2)

        RX = np.zeros(num_points)
        dRX = np.zeros(num_points)
        amplitude = np.sqrt(2 * Gd_n * delta_n)

        for i in range(num_points):
            phase = 2 * np.pi * n_k * V * t[i] + theta
            RX[i] = np.sum(amplitude * np.cos(phase))
            dRX[i] = -2 * np.pi * V * np.sum(n_k * amplitude * np.sin(phase))

        if len(RX) > self.tstep:
            return RX[: self.tstep], dRX[: self.tstep]
        else:
            RX_full = np.zeros(self.tstep)
            dRX_full = np.zeros(self.tstep)
            RX_full[: len(RX)] = RX
            dRX_full[: len(dRX)] = dRX
            return RX_full, dRX_full

    def get_shape_function(self, xc: float, Le: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取形函数及其导数

        参考: system.md Step 4 - 使用形函数向量 {N_b}_i 将接触点外力转换到桥梁节点

        Args:
            xc: 车辆在单元内的位置 (m)
            Le: 单元长度 (m)

        Returns:
            N: 形函数向量 [N1, N2, N3, N4]
            dN: 形函数导数向量
        """
        if Le <= 0:
            return np.zeros(4), np.zeros(4)

        zeta = xc / Le

        N1 = 1 - 3 * zeta**2 + 2 * zeta**3
        N2 = Le * (zeta - 2 * zeta**2 + zeta**3)
        N3 = 3 * zeta**2 - 2 * zeta**3
        N4 = Le * (-(zeta**2) + zeta**3)

        N = np.array([N1, N2, N3, N4])

        dN1_dx = (-6 * zeta + 6 * zeta**2) / Le
        dN2_dx = 1 - 4 * zeta + 3 * zeta**2
        dN3_dx = (6 * zeta - 6 * zeta**2) / Le
        dN4_dx = -2 * zeta + 3 * zeta**2

        dN = np.array([dN1_dx, dN2_dx, dN3_dx, dN4_dx])

        return N, dN

    def get_element_location_vector(self, elem_num: int, EL: int) -> np.ndarray:
        """
        获取单元定位向量

        参考: system.md Step 4 - 使用单元定位向量 [L_i] 将力转换到全局节点

        Args:
            elem_num: 单元编号 (1-based)
            EL: 单元总数

        Returns:
            L_vec: 单元定位向量
        """
        n_dof = 2 * (EL + 1)
        L_vec = np.zeros(n_dof)

        if elem_num < 1 or elem_num > EL:
            return L_vec

        # 单元的4个自由度
        L_vec[2 * (elem_num - 1)] = 1
        L_vec[2 * (elem_num - 1) + 1] = 1
        L_vec[2 * elem_num] = 1
        L_vec[2 * elem_num + 1] = 1

        # 移除边界条件对应的索引
        keep_indices = list(range(n_dof))
        keep_indices.remove(0)
        keep_indices.remove(n_dof - 2)

        return L_vec[keep_indices]

    def solve_vehicle_newmark(
        self,
        f_v: float,
        u_prev: float,
        du_prev: float,
        ddu_prev: float,
    ) -> Tuple[float, float, float]:
        """
        使用Newmark-β法求解车辆运动方程

        参考: system.md Step 3 - 使用Newmark-β法求解车辆加速度

        Args:
            f_v: 车辆所受外力 (scalar)
            u_prev: 上一时刻车辆位移
            du_prev: 上一时刻车辆速度
            ddu_prev: 上一时刻车辆加速度

        Returns:
            u_new: 当前时刻位移
            du_new: 当前时刻速度
            ddu_new: 当前时刻加速度
        """
        # 车辆质量矩阵 (1 DOF)
        M_v = self.mv

        # 车辆刚度
        K_v = self.kv

        # 车辆阻尼
        C_v = self.cv

        # Newmark-β有效刚度
        Keff = K_v + self.Alpha_0 * M_v + self.Alpha_1 * C_v

        # 防止除零
        if abs(Keff) < 1e-15:
            Keff = 1e-15

        # 计算有效力
        term1 = self.Alpha_0 * u_prev + self.Alpha_2 * du_prev + self.Alpha_3 * ddu_prev
        term2 = self.Alpha_1 * u_prev + self.Alpha_4 * du_prev + self.Alpha_5 * ddu_prev
        Feff = f_v + M_v * term1 + C_v * term2

        # 求解
        u_new = Feff / Keff

        # 计算速度和加速度
        ddu_new = (
            self.Alpha_0 * (u_new - u_prev)
            - self.Alpha_2 * du_prev
            - self.Alpha_3 * ddu_prev
        )
        du_new = du_prev + self.Alpha_6 * ddu_prev + self.Alpha_7 * ddu_new

        # 数值稳定性检查
        if not np.isfinite(u_new):
            u_new = u_prev
        if not np.isfinite(du_new):
            du_new = du_prev
        if not np.isfinite(ddu_new):
            ddu_new = ddu_prev

        # 裁剪异常值
        max_val = 1e10
        u_new = np.clip(u_new, -max_val, max_val)
        du_new = np.clip(du_new, -max_val, max_val)
        ddu_new = np.clip(ddu_new, -max_val, max_val)

        return u_new, du_new, ddu_new

    def solve_bridge_newmark(
        self,
        f_b: np.ndarray,
        u_prev: np.ndarray,
        du_prev: np.ndarray,
        ddu_prev: np.ndarray,
        M: np.ndarray,
        K: np.ndarray,
        C: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        使用Newmark-β法求解桥梁运动方程

        参考: system.md Step 5 - 使用Newmark-β法求解桥梁位移

        Args:
            f_b: 桥梁所受外力向量
            u_prev: 上一时刻桥梁位移
            du_prev: 上一时刻桥梁速度
            ddu_prev: 上一时刻桥梁加速度
            M: 桥梁质量矩阵
            K: 桥梁刚度矩阵
            C: 桥梁阻尼矩阵

        Returns:
            u_new: 当前时刻位移
            du_new: 当前时刻速度
            ddu_new: 当前时刻加速度
        """
        # Newmark-β有效刚度
        Keff = K + self.Alpha_0 * M + self.Alpha_1 * C

        # 添加正则化防止奇异
        Keff = Keff + 1e-6 * np.eye(Keff.shape[0])

        # 计算有效力
        term1 = self.Alpha_0 * u_prev + self.Alpha_2 * du_prev + self.Alpha_3 * ddu_prev
        term2 = self.Alpha_1 * u_prev + self.Alpha_4 * du_prev + self.Alpha_5 * ddu_prev
        Feff = f_b + M @ term1 + C @ term2

        # 检查有效力和矩阵的数值稳定性
        if not np.all(np.isfinite(Feff)):
            Feff = np.nan_to_num(Feff, nan=0.0, posinf=1e10, neginf=-1e10)

        if not np.all(np.isfinite(Keff)):
            Keff = np.eye(Keff.shape[0])  # 回退到单位矩阵

        # 求解
        try:
            u_new = linalg.solve(Keff, Feff)
        except Exception:
            # 求解失败时回退到上一步
            u_new = u_prev.copy()

        # 计算速度和加速度
        ddu_new = (
            self.Alpha_0 * (u_new - u_prev)
            - self.Alpha_2 * du_prev
            - self.Alpha_3 * ddu_prev
        )
        du_new = du_prev + self.Alpha_6 * ddu_prev + self.Alpha_7 * ddu_new

        # 数值稳定性检查和裁剪
        max_val = 1e10

        if not np.all(np.isfinite(u_new)):
            u_new = u_prev.copy()
        else:
            u_new = np.clip(u_new, -max_val, max_val)

        if not np.all(np.isfinite(du_new)):
            du_new = du_prev.copy()
        else:
            du_new = np.clip(du_new, -max_val, max_val)

        if not np.all(np.isfinite(ddu_new)):
            ddu_new = ddu_prev.copy()
        else:
            ddu_new = np.clip(ddu_new, -max_val, max_val)

        return u_new.flatten(), du_new.flatten(), ddu_new.flatten()

    def analyze_iterative(self):
        """
        执行车-桥耦合分析 - 6步迭代算法

        参考: system.md 第一部分 - "迭代求解步骤（6步算法）"

        步骤1：初始化。假设每个时间步接触点（CP）位移为零。
        步骤2：计算车辆轴上的接触力。f_va_i = k_a * (w_bi + r_ci)
        步骤3：求解车辆加速度。使用Newmark-β法。
        步骤4：计算作用于桥梁的全局力向量。
        步骤5：求解桥梁全局位移。使用Newmark-β法。
        步骤6：收敛性判断。相对误差 < 1%。
        """
        # 桥梁有限元模型
        M, K = self.beam_km0_f(self.m, self.L, self.E, self.I, self.EL)

        # 保存矩阵供后续使用
        self.M = M
        self.K = K

        # 确保矩阵对称
        K = (K + K.T) / 2
        M = (M + M.T) / 2

        # 添加正则化防止数值问题
        K_reg = K + 1e-6 * np.eye(K.shape[0])
        M_reg = M + 1e-6 * np.eye(M.shape[0])

        # 特征值分析获取频率
        if np.all(np.linalg.eigvals(K_reg) > 0) and np.all(
            np.linalg.eigvals(M_reg) > 0
        ):
            eigvals, eigvecs = linalg.eig(K_reg, M_reg)
            real_mask = np.isreal(eigvals)

            if np.sum(real_mask) >= self.n_modes:
                eigvals = np.real(eigvals[real_mask])
                eigvecs = np.real(eigvecs[:, real_mask])

                sorted_indices = np.argsort(eigvals)
                eigvals = eigvals[sorted_indices]
                eigvecs = eigvecs[:, sorted_indices]

                self.phi = eigvecs[:, : self.n_modes]
                self.Omega = np.sqrt(eigvals[: self.n_modes])
                self.Frequ = self.Omega / (2 * np.pi)
            else:
                self.Frequ = self._compute_theoretical_freq()
        else:
            self.Frequ = self._compute_theoretical_freq()

        # 设置瑞利阻尼
        C = self.setup_damping(M, K)
        self.C = C

        # 路面激励
        RX, dRX = self.psd_r(self.road_type, self.L, self.V, self.deltat)

        # 初始化变量
        n_dof = M.shape[0]
        Le = self.L / self.EL

        # 车辆变量
        u_v = np.zeros(self.tstep)  # 车辆位移
        du_v = np.zeros(self.tstep)  # 车辆速度
        ddu_v = np.zeros(self.tstep)  # 车辆加速度

        # 接触点变量
        w_bi = np.zeros(self.tstep)  # 桥梁在接触点的位移

        # 桥梁变量
        u_b = np.zeros((n_dof, self.tstep))
        du_b = np.zeros((n_dof, self.tstep))
        ddu_b = np.zeros((n_dof, self.tstep))

        # 时间步循环
        for k in range(1, self.tstep):
            xp = self.V * k * self.deltat  # 车辆位置

            if xp > self.L:
                # 车辆已离开桥梁，保持上一时刻状态
                u_v[k] = u_v[k - 1]
                du_v[k] = du_v[k - 1]
                ddu_v[k] = ddu_v[k - 1]
                continue

            # 获取当前单元信息
            s = int(np.floor(xp / Le)) + 1
            if s > self.EL:
                s = self.EL
            xc = xp - (s - 1) * Le  # 车辆在单元内的位置

            N, dN = self.get_shape_function(xc, Le)
            L_vec = self.get_element_location_vector(s, self.EL)

            # 路面不平度
            r_c = RX[k] if k < len(RX) else 0

            # ==================== 6步迭代算法 ====================

            # 步骤1: 初始化 - 假设接触点位移为零 (第一次迭代)
            if k == 1:
                w_bi[k] = 0

            # 迭代求解
            iteration = 0
            converged = False
            diverged = False  # 发散检测
            w_bi_new = 0  # 初始化

            while iteration < self.max_iterations and not converged and not diverged:
                try:
                    # 步骤2: 计算车辆轴上的接触力
                    # f_va_i = k_a * (w_bi + r_ci)
                    f_va = self.k_a * (w_bi[k] + r_c)

                    # 车辆运动方程右端力 (考虑重力)
                    f_vehicle = f_va - self.mv * self.g

                    # 步骤3: 求解车辆加速度 (Newmark-β)
                    u_v_prev = u_v[k - 1]
                    du_v_prev = du_v[k - 1]
                    ddu_v_prev = ddu_v[k - 1]

                    u_v_new, du_v_new, ddu_v_new = self.solve_vehicle_newmark(
                        f_vehicle, u_v_prev, du_v_prev, ddu_v_prev
                    )

                    u_v[k] = u_v_new
                    du_v[k] = du_v_new
                    ddu_v[k] = ddu_v_new

                    # 步骤4: 计算作用于桥梁的全局力向量
                    # 接触点力 = 车辆重力 + 惯性力
                    # 惯性力 f_inertia = -m_v * a_v
                    f_inertia = -self.mv * ddu_v[k]
                    f_total = f_va + f_inertia

                    # 使用形函数和定位向量组装全局力向量
                    f_b_global = f_total * L_vec

                    # 步骤5: 求解桥梁全局位移 (Newmark-β)
                    u_b_prev = u_b[:, k - 1]
                    du_b_prev = du_b[:, k - 1]
                    ddu_b_prev = ddu_b[:, k - 1]

                    u_b_new, du_b_new, ddu_b_new = self.solve_bridge_newmark(
                        f_b_global, u_b_prev, du_b_prev, ddu_b_prev, M, K, C
                    )

                    u_b[:, k] = u_b_new
                    du_b[:, k] = du_b_new
                    ddu_b[:, k] = ddu_b_new

                    # 计算新的接触点位移 (使用形函数插值)
                    if len(N) >= 4 and len(u_b_new) >= 4:
                        w_bi_new = N @ u_b_new[:4]
                    else:
                        w_bi_new = 0

                    # 发散检测：如果值超过阈值，停止迭代
                    if abs(w_bi_new) > 1e8 or not np.isfinite(w_bi_new):
                        diverged = True
                        # 回退到上一时刻的值
                        w_bi_new = w_bi[k - 1] if k > 1 else 0
                        break

                    # 步骤6: 收敛性判断
                    # 收敛准则: 接触点位移相对误差 < 1%
                    if abs(w_bi[k]) > 1e-10:
                        rel_error = abs(w_bi_new - w_bi[k]) / abs(w_bi[k])
                    else:
                        rel_error = abs(w_bi_new - w_bi[k])

                    if rel_error < self.convergence_tol:
                        converged = True
                        w_bi[k] = w_bi_new
                    else:
                        # 更新接触点位移，开始下一轮迭代
                        w_bi[k] = w_bi_new
                        iteration += 1

                except Exception as e:
                    # 发生异常时回退
                    diverged = True
                    w_bi_new = w_bi[k - 1] if k > 1 else 0
                    w_bi[k] = w_bi_new
                    break

            # 如果发散，回退到上一时刻的值
            if diverged:
                u_v[k] = u_v[k - 1]
                du_v[k] = du_v[k - 1]
                ddu_v[k] = ddu_v[k - 1]
                u_b[:, k] = u_b[:, k - 1]
                du_b[:, k] = du_b[:, k - 1]
                ddu_b[:, k] = ddu_b[:, k - 1]
                w_bi[k] = w_bi[k - 1] if k > 1 else 0

        # 存储结果
        self.u = np.vstack([u_v.reshape(1, -1), u_b])
        self.du = np.vstack([du_v.reshape(1, -1), du_b])
        self.ddu = np.vstack([ddu_v.reshape(1, -1), ddu_b])

        self.zv_DIS = u_v
        self.zv_VEL = du_v
        self.zv_ACC = ddu_v

        # 接触点位移 (AP) = 桥梁接触点位移 + 路面不平度
        # 参考: system.md - "AP是车辆-桥梁接触点的位移时程与路面不平度之和"
        self.uc = w_bi + RX[: self.tstep]

        # 数值稳定性处理
        self.uc = np.clip(self.uc, -1e6, 1e6)
        self.uc = np.nan_to_num(self.uc, nan=0.0, posinf=1e6, neginf=-1e6)

        # 存储中间桥梁位移
        self.u_b = u_b

    def _compute_theoretical_freq(self) -> np.ndarray:
        """计算理论频率"""
        freq = np.zeros(self.n_modes)
        for n in range(1, self.n_modes + 1):
            freq[n - 1] = (
                (n**2) / (2 * self.L**2) * np.sqrt(self.E * self.I / (np.pi * self.m))
            )
        return freq

    def analyze(self):
        """
        执行车-桥耦合分析 - 使用6步迭代算法
        """
        self.analyze_iterative()

    def run_analysis(self) -> Dict:
        """
        运行完整分析流程：健康状态 -> 损伤状态 -> CPDV
        """
        # 健康状态分析
        self.clear_cracks()
        self.analyze()
        self.healthy_results = {
            "zv_DIS": self.zv_DIS.copy(),
            "zv_VEL": self.zv_VEL.copy(),
            "zv_ACC": self.zv_ACC.copy(),
            "uc": self.uc.copy(),
            "Frequ": self.Frequ.copy() if hasattr(self, "Frequ") else None,
            "K_matrix": self.K.copy() if hasattr(self, "K") else None,
        }
        self.uc_healthy = self.uc.copy()

        return self.healthy_results

    def analyze_damage(self, crack_position: float, crack_depth_ratio: float) -> Dict:
        """
        分析指定损伤状态

        Args:
            crack_position: 裂纹位置 (m)
            crack_depth_ratio: 裂纹深度比 (0-1)

        Returns:
            损伤状态结果字典
        """
        # 设置裂纹
        self.clear_cracks()
        self.add_crack(crack_position, crack_depth_ratio)

        # 分析
        self.analyze()

        return {
            "zv_DIS": self.zv_DIS.copy(),
            "zv_VEL": self.zv_VEL.copy(),
            "zv_ACC": self.zv_ACC.copy(),
            "uc": self.uc.copy(),
            "Frequ": self.Frequ.copy() if hasattr(self, "Frequ") else None,
            "K_matrix": self.K.copy() if hasattr(self, "K") else None,
        }

    def calculate_cpdv(self, damaged_uc: np.ndarray) -> np.ndarray:
        """
        计算CPDV（接触点位移变化）

        参考: system.md - "CPDV = AP_damaged - AP_intact"

        Args:
            damaged_uc: 损伤状态接触点位移 (AP)

        Returns:
            CPDV序列
        """
        if self.uc_healthy is None:
            raise ValueError("请先运行健康状态分析")

        # CPDV = AP_damaged - AP_intact (注意方向与原实现相反)
        cpdv = damaged_uc - self.uc_healthy

        # 数值稳定性处理 - 使用更合理的裁剪范围
        # 桥梁位移通常在毫米级 (10^-3 m)，裁剪范围设为 ±1m 足够
        cpdv = np.clip(cpdv, -1.0, 1.0)
        cpdv = np.nan_to_num(cpdv, nan=0.0, posinf=0.0, neginf=0.0)

        return cpdv

    def normalize_cpdv(self, cpdv: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        数据归一化 - 最小最大缩放至[0,1]区间

        参考: system.md - "数据预处理：在将CPDV数据输入BP神经网络前，
        进行了归一化处理（最小-最大缩放至[0,1]区间）"

        Args:
            cpdv: CPDV数据

        Returns:
            归一化后的CPDV, 最小值, 最大值
        """
        cpdv_min = np.min(cpdv)
        cpdv_max = np.max(cpdv)

        if cpdv_max - cpdv_min < 1e-10:
            return np.zeros_like(cpdv), cpdv_min, cpdv_max

        cpdv_norm = (cpdv - cpdv_min) / (cpdv_max - cpdv_min)

        return cpdv_norm, cpdv_min, cpdv_max

    def denormalize_cpdv(
        self, cpdv_norm: np.ndarray, cpdv_min: float, cpdv_max: float
    ) -> np.ndarray:
        """
        反归一化

        Args:
            cpdv_norm: 归一化后的CPDV
            cpdv_min: 最小值
            cpdv_max: 最大值

        Returns:
            原始尺度的CPDV
        """
        return cpdv_norm * (cpdv_max - cpdv_min) + cpdv_min


class EnhancedBridgeVehicleSystem(BridgeVehicleSystem):
    """
    增强版车-桥耦合系统类

    继承自BridgeVehicleSystem，增加多裂纹支持等功能
    """

    def __init__(self, params: Optional[Dict] = None):
        super().__init__(params)

        # 多车辆支持
        self.multi_vehicle_mode = self.params.get("multi_vehicle", False)
        self.vehicle_spacing = self.params.get("vehicle_spacing", 5.0)

    def analyze_multi_cracks(self, crack_list: List[Tuple[float, float]]) -> Dict:
        """
        分析多裂纹损伤状态

        Args:
            crack_list: 裂纹列表 [(位置1, 深度比1), (位置2, 深度比2), ...]

        Returns:
            损伤状态结果字典
        """
        self.clear_cracks()
        for pos, depth in crack_list:
            self.add_crack(pos, depth)

        self.analyze()

        return {
            "zv_DIS": self.zv_DIS.copy(),
            "zv_VEL": self.zv_VEL.copy(),
            "zv_ACC": self.zv_ACC.copy(),
            "uc": self.uc.copy(),
            "Frequ": self.Frequ.copy() if hasattr(self, "Frequ") else None,
            "K_matrix": self.K.copy() if hasattr(self, "K") else None,
            "cracks": self.cracks.copy(),
        }


def create_system(params: Optional[Dict] = None) -> BridgeVehicleSystem:
    """
    工厂函数：创建车-桥耦合系统

    Args:
        params: 系统参数字典

    Returns:
        BridgeVehicleSystem实例
    """
    return BridgeVehicleSystem(params)


def create_enhanced_system(
    params: Optional[Dict] = None,
) -> EnhancedBridgeVehicleSystem:
    """
    工厂函数：创建增强版车-桥耦合系统

    Args:
        params: 系统参数字典

    Returns:
        EnhancedBridgeVehicleSystem实例
    """
    return EnhancedBridgeVehicleSystem(params)
